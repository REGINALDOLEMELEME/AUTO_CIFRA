from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .alignment import align_chords_by_word_time
from .alignment_asr import align_words
from .audio import normalize_audio
from .beats import detect_beats_key
from .chords import detect_chords, normalize_chord_vocabulary, smooth_to_beats, write_json
from .config import Settings
from .jobs import Job, JobRepo
from .models import release_whisper, release_wav2vec
from .separation import separate_vocals
from .structure import label_sections
from .transcription import transcribe_audio, write_transcription_json


class PipelineCancelled(Exception):
    """Raised when a running job is cancelled by the user between stages."""


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _check_cancelled(job_id: str, repo: JobRepo) -> None:
    current = repo.get(job_id)
    if current and current.stage == "cancelled":
        raise PipelineCancelled(f"job {job_id} cancelled by user")


def run(job: Job, repo: JobRepo, settings: Settings) -> Path:
    job_tmp = settings.tmp_dir / job.id
    job_tmp.mkdir(parents=True, exist_ok=True)
    input_audio = settings.input_dir / job.filename
    warnings: list[str] = []

    _check_cancelled(job.id, repo)
    repo.advance(job.id, "separating", progress=0.05)
    repo.heartbeat(job.id)
    if settings.separation.enabled:
        vocals, warn = separate_vocals(input_audio, job_tmp, model_name=settings.separation.model)
        if warn:
            warnings.append(warn)
    else:
        vocals = input_audio
        warnings.append("separation disabled in settings")

    _check_cancelled(job.id, repo)
    repo.advance(job.id, "transcribing", progress=0.25)
    repo.heartbeat(job.id)
    normalized = normalize_audio(vocals, job_tmp / f"{vocals.stem}.norm.wav")
    asr = transcribe_audio(
        normalized,
        job_tmp,
        language=settings.asr.language,
        model_size=settings.asr.model,
        compute_type=settings.asr.compute_type,
        device=settings.asr.device,
    )
    write_transcription_json(asr, job_tmp / "transcription.json")
    release_whisper()

    _check_cancelled(job.id, repo)
    repo.advance(job.id, "aligning", progress=0.55)
    repo.heartbeat(job.id)
    if settings.alignment.enabled:
        aligned_asr, align_warn = align_words(
            normalized_audio=normalized,
            transcription=asr,
            language=settings.asr.language,
            model_name=settings.alignment.model,
        )
        if align_warn:
            warnings.append(align_warn)
    else:
        aligned_asr = asr
    release_wav2vec()

    _check_cancelled(job.id, repo)
    repo.advance(job.id, "chords", progress=0.70)
    repo.heartbeat(job.id)
    beats = detect_beats_key(input_audio)
    raw_chords = detect_chords(input_audio=input_audio, tmp_dir=job_tmp, key=beats.get("key", ""))
    normalized_chords = normalize_chord_vocabulary(raw_chords)
    smoothed = smooth_to_beats(normalized_chords, beats)
    write_json(smoothed, job_tmp / "chords.json")
    _write_json(beats, job_tmp / "beats.json")

    _check_cancelled(job.id, repo)
    repo.advance(job.id, "structure", progress=0.85)
    repo.heartbeat(job.id)
    sections = (
        label_sections(
            input_audio=input_audio,
            beats=beats,
            transcription=aligned_asr,
            n_segments=settings.structure.n_segments,
            confidence_threshold=settings.structure.confidence_threshold,
        )
        if settings.structure.enabled
        else []
    )
    _write_json({"sections": sections}, job_tmp / "sections.json")

    _check_cancelled(job.id, repo)
    repo.advance(job.id, "rendering", progress=0.95)
    aligned = align_chords_by_word_time(
        transcription=aligned_asr,
        chords=smoothed,
        sections=sections,
    )
    aligned.setdefault("warnings", [])
    for w in warnings:
        if w and w not in aligned["warnings"]:
            aligned["warnings"].append(w)
    out = job_tmp / "aligned.json"
    _write_json(aligned, out)

    repo.advance(job.id, "ready_for_review", progress=1.0)
    return out

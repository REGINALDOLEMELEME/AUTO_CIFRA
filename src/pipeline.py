from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .alignment import align_chords_by_word_time
from .alignment_asr import align_words
from .audio import normalize_audio
from .beats import detect_beats_key
from .chord_reclassifier import reclassify_with_chroma
from .chords import (
    detect_chords,
    filter_out_of_key_flickers,
    normalize_chord_vocabulary,
    refine_chords_to_key,
    smooth_to_beats,
    write_json,
)
from .config import Settings
from .jobs import Job, JobRepo
from .lexicon import apply_lexicon
from .models import release_whisper, release_wav2vec
from .separation import separate_vocals
from .structure import label_sections
from .transcription import transcribe_audio, transcribe_clip, write_transcription_json


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
        initial_prompt=settings.asr.initial_prompt or None,
        hotwords=settings.asr.hotwords or None,
    )

    # Tail-fallback: if Demucs over-suppressed the track's quiet tail and the
    # primary pass emitted no segments beyond `tail_fallback_min_gap_s` before
    # the end of the audio, re-run Whisper on the ORIGINAL mix over that
    # window and merge. Segments returned by transcribe_clip already passed
    # the hallucination filters.
    #
    # Use last *word's* end, not segment.end: faster-whisper pads segment.end
    # with trailing silence/noise when the next segment starts late. We saw a
    # segment end stretch from the actual last-word time (~115 s) out to 131 s
    # of near-silence on São Francisco, which hid the real final refrain at
    # ~118–123 s from the tail-fallback.
    def _true_segment_end(seg: dict[str, Any]) -> float:
        words = seg.get("words") or []
        if words:
            return float(words[-1].get("end") or seg.get("end") or 0.0)
        return float(seg.get("end") or 0.0)

    # Helper: run transcribe_clip + dedup merge. Shared by head & tail paths.
    def _probe_and_merge(window_start: float, window_end: float, note: str) -> None:
        nonlocal asr
        clip_segs = transcribe_clip(
            input_audio=input_audio,
            tmp_dir=job_tmp,
            start_s=window_start,
            end_s=window_end,
            language=settings.asr.language,
            model_size=settings.asr.model,
            compute_type=settings.asr.compute_type,
            device=settings.asr.device,
            initial_prompt=settings.asr.initial_prompt or None,
            hotwords=settings.asr.hotwords or None,
        )
        import re as _re
        def _tok(s: str) -> set[str]:
            return set(t for t in _re.findall(r"\w+", (s or "").lower()) if len(t) > 1)
        existing_now = list(asr.get("segments") or [])
        existing_tokens = [_tok(s.get("text", "")) for s in existing_now]
        new_segs: list[dict[str, Any]] = []
        for s in clip_segs:
            st = _tok(s.get("text", ""))
            if not st:
                continue
            duplicate = False
            for et in existing_tokens:
                if et and len(st & et) / len(st) >= 0.6:
                    duplicate = True
                    break
            if not duplicate:
                new_segs.append(s)
        if new_segs:
            merged = existing_now + new_segs
            merged.sort(key=lambda s: float(s.get("start", 0.0)))
            asr["segments"] = merged
            warnings.append(
                f"{note}: recovered {len(new_segs)} segment(s) from original mix "
                f"(probed {window_start:.1f}s→{window_end:.1f}s)"
            )

    # GAP-fallback: detect any gap between consecutive segments longer than
    # `gap_fallback_min_s` and re-probe that window on the ORIGINAL mix with
    # chunked transcribe_clip. Covers three regression cases we've hit:
    #   - Head gap (song opens with a filtered hallucination, e.g. the
    #     30s "A CIDADE NO BRASIL" that hid Legião Urbana's first verse).
    #   - Tail gap (Demucs over-suppressed the quiet end, e.g. São Francisco's
    #     final refrain) — covered by tail_fallback below too, but this is
    #     stricter.
    #   - Mid-song gap (a spurious long hallucination got filtered somewhere
    #     in the middle of the track).
    if settings.asr.gap_fallback_enabled:
        try:
            duration = float(asr.get("duration") or 0.0)
            existing = sorted(
                list(asr.get("segments") or []),
                key=lambda s: float(s.get("start", 0.0)),
            )
            # Walk the timeline; collect gap windows worth probing.
            gaps: list[tuple[float, float]] = []
            prev_end = 0.0
            for seg in existing:
                s_start = float(seg.get("start", 0.0))
                if s_start - prev_end >= settings.asr.gap_fallback_min_s:
                    gaps.append((prev_end, s_start))
                prev_end = max(prev_end, _true_segment_end(seg))
            # We intentionally skip the tail here — the existing tail-fallback
            # below has its own min_gap + probe_last_s logic.
            for (gstart, gend) in gaps:
                _probe_and_merge(max(gstart - 1.0, 0.0), min(gend + 1.0, duration), note="gap_fallback")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"gap_fallback failed: {exc}")

    if settings.asr.tail_fallback_enabled:
        try:
            duration = float(asr.get("duration") or 0.0)
            existing = list(asr.get("segments") or [])
            last_end = max((_true_segment_end(s) for s in existing), default=0.0)
            gap = duration - last_end
            if duration > 0 and gap >= settings.asr.tail_fallback_min_gap_s:
                # Probe at least `tail_fallback_probe_last_s` seconds of the
                # track, even when Whisper's last word claims to end later.
                # Guards against word-end stretching into silence windows.
                min_probe_start = max(0.0, duration - settings.asr.tail_fallback_probe_last_s)
                window_start = min(max(last_end - 2.0, 0.0), min_probe_start)
                _probe_and_merge(window_start, duration, note="tail_fallback")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"tail_fallback failed: {exc}")

    if settings.asr.lexicon_correction:
        apply_lexicon(asr)
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
    normalized_chords = normalize_chord_vocabulary(
        raw_chords,
        simplify_to_triads=settings.chords.simplify_to_triads,
    )
    if settings.chords.refine_to_key and beats.get("key"):
        normalized_chords = refine_chords_to_key(normalized_chords, beats["key"])
    if settings.chords.filter_out_of_key_flickers and beats.get("key"):
        normalized_chords = filter_out_of_key_flickers(normalized_chords, beats["key"])
    if settings.chords.chroma_reclassify and beats.get("key"):
        # Second-opinion: cross-validate each segment's label against the
        # audio's chroma profile. Replaces out-of-key Chordino picks when
        # the in-key chroma candidate scores meaningfully higher.
        normalized_chords = reclassify_with_chroma(
            normalized_chords,
            audio_path=input_audio,
            key=beats["key"],
            margin=settings.chords.chroma_reclassify_margin,
            only_out_of_key=settings.chords.chroma_reclassify_only_out_of_key,
        )
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

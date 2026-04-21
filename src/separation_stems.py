"""Stem-removal pipeline (ADR-SR-003 cache-on-sha, ADR-SR-004 channel preservation,
ADR-SR-006 pydub MP3 encoding).

Given an input audio file and a remove-mask (subset of {drums, bass, vocals, other}),
produce an MP3 CBR containing only the non-removed stems. Reuses Demucs model via
``src.models.get_demucs``; caches the 4 full stems under
``data/stems_cache/{sha256}/`` so changing the mask on the same input skips the
expensive Demucs pass (AT-004).
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import Settings
from .models import get_demucs, release_demucs
from .stems_jobs import StemsJob, StemsJobRepo

STEM_NAMES: tuple[str, ...] = (
    "drums", "bass", "vocals", "other", "guitar", "piano",
)


class StemRemoverError(Exception):
    """Base exception for the stems pipeline."""


class DemucsUnavailable(StemRemoverError):
    """Raised when Demucs cannot be loaded. Never silently fallback (AT-010)."""


# ---------------------------------------------------------------------------
# Pure helpers (easy to unit-test)
# ---------------------------------------------------------------------------


def hash_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def slugify_filename(stem: str) -> str:
    """'Será - Legião Urbana (youtube)' -> 'sera-legiao-urbana-youtube'."""
    nfkd = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", nfkd).strip("-").lower()
    return slug or "audio"


def compute_cache_key(
    sha: str, remove: tuple[str, ...], model: str, bitrate: int
) -> str:
    mask = "-".join(sorted(remove)) or "none"
    return f"{sha}|{mask}|{model}|{bitrate}"


def deterministic_output_path(
    output_root: Path, job_id: str, input_name: str, remove: tuple[str, ...]
) -> Path:
    slug = slugify_filename(Path(input_name).stem)
    removed = "-".join(sorted(remove)) or "none"
    return output_root / job_id / f"{slug}.no-{removed}.mp3"


def remix(
    stems: dict[str, np.ndarray],
    remove: set[str],
    input_was_mono: bool,
    headroom: float = 0.99,
) -> np.ndarray:
    """Sum kept stems, peak-normalise if they'd clip, optionally downmix to mono.

    Summing stems from a mastered track routinely yields peaks > 1.0.
    Hard-clipping at ±1.0 destroys transient info and sounds like digital
    distortion (the classic "muddy bass / crunchy guitar" artefact).
    Instead, if the summed peak exceeds ``headroom`` we scale the whole
    mix down proportionally — dynamics are preserved, only the overall
    level drops.

    Arrays are ``[channels, samples]`` float32. Returns same shape with
    ``channels=1`` when ``input_was_mono`` else 2.
    """
    kept = [arr for name, arr in stems.items() if name not in remove]
    if not kept:
        # Caller should have validated; defensive only.
        raise StemRemoverError("All stems removed — refusing to output silence.")
    mix = np.sum(np.stack(kept, axis=0), axis=0)

    peak = float(np.max(np.abs(mix)))
    if peak > headroom:
        mix = mix * (headroom / peak)

    if input_was_mono:
        mix = mix.mean(axis=0, keepdims=True)
    return mix.astype(np.float32)


def encode_mp3(
    array: np.ndarray, sr: int, target: Path, bitrate: int = 320
) -> None:
    """Encode ``[channels, samples]`` float32 to MP3 CBR via pydub+ffmpeg."""
    from pydub import AudioSegment

    import os as _os

    target.parent.mkdir(parents=True, exist_ok=True)
    # pydub wants a real file, not a buffer, for reliable cross-platform MP3
    # encoding via ffmpeg. Close the fd immediately — on Windows an open
    # handle blocks the final unlink.
    fd, tmp_name = tempfile.mkstemp(suffix=".wav")
    _os.close(fd)
    tmp = Path(tmp_name)
    try:
        sf.write(str(tmp), array.T, sr, subtype="PCM_16")
        AudioSegment.from_wav(str(tmp)).export(
            str(target),
            format="mp3",
            bitrate=f"{int(bitrate)}k",
            codec="libmp3lame",
        )
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# ffprobe helpers — private; tests monkeypatch these or use real fixtures.
# ---------------------------------------------------------------------------


def _ffprobe_json(path: Path, *args: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", *args, "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def probe_channels(path: Path) -> int:
    data = _ffprobe_json(
        path, "-select_streams", "a:0", "-show_entries", "stream=channels"
    )
    return int(data["streams"][0]["channels"])


def probe_sample_rate(path: Path) -> int:
    data = _ffprobe_json(
        path, "-select_streams", "a:0", "-show_entries", "stream=sample_rate"
    )
    return int(data["streams"][0]["sample_rate"])


def probe_duration_sec(path: Path) -> float | None:
    try:
        data = _ffprobe_json(path, "-show_entries", "format=duration")
        dur = float(data["format"]["duration"])
        # Require at least one audio stream.
        astream = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=True,
        )
        if "audio" not in astream.stdout:
            return None
        return dur
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Demucs-backed 4-stem extraction with on-disk cache.
# ---------------------------------------------------------------------------


def extract_all_stems(
    input_audio: Path, cache_dir: Path, model_name: str = "htdemucs_ft"
) -> dict[str, np.ndarray]:
    """Return ``{name: [channels, samples] float32}`` for all 4 stems.

    Reads from ``cache_dir/{name}.wav`` when all 4 exist; otherwise runs
    Demucs and writes them for next time.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    wavs = {n: cache_dir / f"{n}.wav" for n in STEM_NAMES}

    if all(p.exists() for p in wavs.values()):
        loaded: dict[str, np.ndarray] = {}
        for n, p in wavs.items():
            data, _sr = sf.read(str(p), dtype="float32", always_2d=True)
            # soundfile returns [samples, channels]; we want [channels, samples]
            loaded[n] = data.T.astype(np.float32, copy=False)
        return loaded

    model, err = get_demucs(model_name)
    if err or model is None:
        raise DemucsUnavailable(
            f"Demucs model not found ({err}). "
            "Run `scripts/prefetch_models.py`."
        )

    try:
        import torch
        import torchaudio
        from demucs.apply import apply_model
    except Exception as exc:  # noqa: BLE001
        raise DemucsUnavailable(
            f"Demucs runtime deps missing: {exc}. "
            "Run `scripts/setup.ps1`."
        ) from exc

    try:
        waveform, sr = torchaudio.load(str(input_audio))
        if sr != model.samplerate:
            waveform = torchaudio.functional.resample(
                waveform, sr, model.samplerate
            )
            sr = model.samplerate
        # htdemucs_ft expects stereo; mirror src/separation.py channel handling.
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2]

        with torch.no_grad():
            sources = apply_model(
                model,
                waveform.unsqueeze(0),
                device="cpu",
                progress=False,
                split=True,
                overlap=0.1,
            )

        out: dict[str, np.ndarray] = {}
        for i, name in enumerate(model.sources):
            arr = sources[0, i].cpu().numpy()  # [2, T] float32
            # Cache on disk — soundfile wants [samples, channels]
            sf.write(str(wavs[name]), arr.T, sr, subtype="FLOAT")
            out[name] = arr

        # Defensive: ensure all 4 canonical names exist (htdemucs_ft always has them).
        for name in STEM_NAMES:
            if name not in out:
                out[name] = np.zeros_like(sources[0, 0].cpu().numpy())
        return out
    finally:
        release_demucs()


# ---------------------------------------------------------------------------
# Orchestrator called by the worker.
# ---------------------------------------------------------------------------


def process_job(
    job: StemsJob, repo: StemsJobRepo, settings: Settings
) -> None:
    """End-to-end: separate (or cache-hit) → remix → encode MP3 → advance to ready."""
    stems_cfg = settings.stems
    input_path = settings.input_dir / "stems" / job.id / job.filename
    cache_dir = settings.project_root / "data" / "stems_cache" / job.input_sha256
    output_root = settings.project_root / "data" / "output" / "stems"
    output_path = deterministic_output_path(
        output_root, job.id, job.filename, job.remove_mask
    )

    # Short-circuit: identical job already produced this output (AT-003).
    if output_path.exists():
        repo.set_output_path(job.id, str(output_path))
        repo.advance(job.id, "ready", progress=1.0)
        return

    repo.advance(job.id, "separating", progress=0.2)
    stems = extract_all_stems(input_path, cache_dir, stems_cfg.model)

    repo.advance(job.id, "encoding", progress=0.8)
    channels = probe_channels(input_path)
    sr = probe_sample_rate(input_path)
    mix = remix(stems, set(job.remove_mask), input_was_mono=channels == 1)
    encode_mp3(mix, sr=sr, target=output_path, bitrate=job.bitrate)

    repo.set_output_path(job.id, str(output_path))
    repo.advance(job.id, "ready", progress=1.0)

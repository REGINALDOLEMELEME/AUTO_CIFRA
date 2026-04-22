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
import math
import os
import re
import subprocess
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import Settings, resolve_preset
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
    sha: str,
    remove: tuple[str, ...],
    model: str,
    bitrate: int,
    shifts: int = 1,
    overlap: float = 0.1,
) -> str:
    mask = "-".join(sorted(remove)) or "none"
    return f"{sha}|{mask}|{model}|{bitrate}|s{shifts}|ov{int(overlap * 100)}"


def quality_cache_subdir(model: str, shifts: int, overlap: float) -> str:
    """Disk directory name for the per-quality stem cache.

    Used as the first path segment under ``data/stems_cache/`` so stems from
    different quality presets coexist on disk and don't pollute each other.
    """
    return f"{model}_s{shifts}_ov{int(overlap * 100)}"


def deterministic_output_path(
    output_root: Path,
    job_id: str,
    input_name: str,
    remove: tuple[str, ...],
    output_format: str = "mp3",
) -> Path:
    slug = slugify_filename(Path(input_name).stem)
    removed = "-".join(sorted(remove)) or "none"
    ext = output_format.lower().lstrip(".")
    return output_root / job_id / f"{slug}.no-{removed}.{ext}"


def low_shelf(
    signal: np.ndarray,
    sample_rate: int,
    freq_hz: float,
    gain_db: float,
    q: float = 0.707,
) -> np.ndarray:
    """Apply an RBJ low-shelf biquad to ``[channels, samples]`` audio.

    Returns the filtered signal at the same dtype/shape. When ``gain_db``
    is 0 or negative, returns the input unchanged (no-op fast path).
    """
    if gain_db <= 0 or signal.size == 0:
        return signal
    # RBJ audio-EQ cookbook — low-shelf biquad
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq_hz / sample_rate
    cos_w = math.cos(w0)
    sin_w = math.sin(w0)
    alpha = sin_w / (2.0 * q)
    two_sqrtA_alpha = 2.0 * math.sqrt(A) * alpha

    b0 = A * ((A + 1) - (A - 1) * cos_w + two_sqrtA_alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * cos_w)
    b2 = A * ((A + 1) - (A - 1) * cos_w - two_sqrtA_alpha)
    a0 = (A + 1) + (A - 1) * cos_w + two_sqrtA_alpha
    a1 = -2 * ((A - 1) + (A + 1) * cos_w)
    a2 = (A + 1) + (A - 1) * cos_w - two_sqrtA_alpha

    from scipy.signal import sosfilt, tf2sos
    sos = tf2sos([b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0])

    # sosfilt along the sample axis (axis=-1). Works for mono or stereo.
    out = sosfilt(sos, signal, axis=-1).astype(signal.dtype, copy=False)
    return out


def remix(
    stems: dict[str, np.ndarray],
    remove: set[str],
    input_was_mono: bool,
    sample_rate: int = 44100,
    headroom: float = 0.95,
) -> np.ndarray:
    """Sum kept stems, peak-limit to prevent clipping, optionally downmix to mono.

    Summing stems from a mastered track routinely yields peaks > 1.0.
    Instead of global scaling (which causes 'ducking'), we use a SoftLimiter
    to squash only the offending peaks.

    Arrays are ``[channels, samples]`` float32. Returns same shape with
    ``channels=1`` when ``input_was_mono`` else 2.
    """
    kept = [arr for name, arr in stems.items() if name not in remove]
    if not kept:
        # Caller should have validated; defensive only.
        raise StemRemoverError("All stems removed — refusing to output silence.")
    mix = np.sum(np.stack(kept, axis=0), axis=0)

    mix = _peak_limit(mix, sample_rate=sample_rate, headroom=headroom)

    if input_was_mono:
        mix = mix.mean(axis=0, keepdims=True)
    return mix.astype(np.float32)


def _match_length(array: np.ndarray, samples: int) -> np.ndarray:
    """Trim or zero-pad ``[channels, samples]`` audio to a target length."""
    if array.shape[-1] == samples:
        return array
    if array.shape[-1] > samples:
        return array[..., :samples]
    pad = samples - array.shape[-1]
    return np.pad(array, ((0, 0), (0, pad)), mode="constant")


def _match_channels(array: np.ndarray, channels: int) -> np.ndarray:
    if array.shape[0] == channels:
        return array
    if channels == 1:
        return array.mean(axis=0, keepdims=True)
    if array.shape[0] == 1:
        return np.repeat(array, channels, axis=0)
    return array[:channels]


class SoftLimiter:
    """A look-ahead peak limiter to prevent digital clipping without 'ducking'.

    Instead of scaling the whole song down when a drum transient peaks, this
    squashes only the peaks that exceed the threshold, using a short attack/
    release window to keep the rest of the mix at its original volume.
    """

    def __init__(
        self,
        sample_rate: int,
        threshold: float = 0.95,
        attack_ms: float = 2.0,
        release_ms: float = 50.0,
        lookahead_ms: float = 5.0,
    ):
        self.sr = sample_rate
        self.threshold = threshold
        self.attack_samples = int(attack_ms * sample_rate / 1000)
        self.release_samples = int(release_ms * sample_rate / 1000)
        self.lookahead_samples = int(lookahead_ms * sample_rate / 1000)

    def limit(self, signal: np.ndarray) -> np.ndarray:
        """Apply limiting to [channels, samples] float32 audio."""
        if signal.size == 0:
            return signal
        # Use the max envelope across all channels for linked-stereo limiting.
        envelope = np.max(np.abs(signal), axis=0)

        # Calculate required gain reduction (0.0 to 1.0)
        # target_gain = threshold / envelope where envelope > threshold
        target_gain = np.ones_like(envelope)
        mask = envelope > self.threshold
        target_gain[mask] = self.threshold / envelope[mask]

        # Smooth the gain reduction using a simple attack/release filter
        # In a real-time limiter we'd use lookahead; here we can just use
        # a rolling minimum / smoothed envelope.
        smoothed_gain = np.ones_like(target_gain)
        current_gain = 1.0

        # We'll use a slightly simpler but effective offline approach:
        # 1. Take the target gain
        # 2. Apply a minimum-filter to account for lookahead
        # 3. Smooth with attack/release logic
        from scipy.ndimage import minimum_filter1d
        smoothed_gain = minimum_filter1d(target_gain, size=self.lookahead_samples)

        # Further smooth to avoid fast gain-change distortion (clicks)
        # Simple IIR filter for release; attack is usually handled by the min-filter.
        alpha_rel = math.exp(-1.0 / self.release_samples)
        for i in range(1, len(smoothed_gain)):
            # If we are releasing (gain increasing), use release constant
            if smoothed_gain[i] > current_gain:
                current_gain = current_gain * alpha_rel + smoothed_gain[i] * (1 - alpha_rel)
                smoothed_gain[i] = current_gain
            else:
                # Attack is instant in this offline min-filter model
                current_gain = smoothed_gain[i]

        return (signal * smoothed_gain).astype(np.float32)


def _peak_limit(
    array: np.ndarray, sample_rate: int, headroom: float = 0.95
) -> np.ndarray:
    """Prevent clipping. Uses SoftLimiter to avoid ducking the whole mix."""
    limiter = SoftLimiter(sample_rate=sample_rate, threshold=headroom)
    return limiter.limit(array)


def load_original_mix(
    input_audio: Path,
    sample_rate: int,
    input_was_mono: bool,
) -> np.ndarray:
    """Decode the original mix as float32 at the Demucs stem sample rate.

    Using ffmpeg keeps this helper independent from torchaudio in tests and
    supports every upload format accepted by the API.
    """
    channels = 1 if input_was_mono else 2
    proc = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(input_audio),
            "-ac", str(channels), "-ar", str(int(sample_rate)),
            "-f", "f32le", "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    data = np.frombuffer(proc.stdout, dtype=np.float32)
    if data.size == 0:
        raise StemRemoverError("Decoded audio is empty.")
    frames = data.size // channels
    data = data[:frames * channels]
    return data.reshape(frames, channels).T.astype(np.float32, copy=False)


def subtract_removed_stems(
    original_mix: np.ndarray,
    stems: dict[str, np.ndarray],
    remove: set[str],
    sample_rate: int = 44100,
    strength: float = 1.0,
    headroom: float = 0.95,
) -> np.ndarray:
    """Remove stems by subtracting their estimate from the original mix.

    This preserves the original mastering, ambience and stereo image better
    than reconstructing the whole song from the predicted remaining stems.
    """
    removed = [stems[name] for name in remove if name in stems]
    if not removed:
        raise StemRemoverError("No known stems selected for removal.")
    channels = original_mix.shape[0]
    samples = original_mix.shape[-1]
    aligned = [
        _match_channels(_match_length(arr, samples), channels)
        for arr in removed
    ]
    removed_sum = np.sum(np.stack(aligned, axis=0), axis=0)
    mix = original_mix - removed_sum * float(strength)
    return _peak_limit(mix, sample_rate=sample_rate, headroom=headroom)


def add_bass_lift(
    mix: np.ndarray,
    bass_stem: np.ndarray,
    sample_rate: int,
    gain_db: float,
    freq_hz: float = 120.0,
    q: float = 0.707,
    headroom: float = 0.95,
) -> np.ndarray:
    """Add only the boosted bass-stem delta to the mix.

    This makes bass more audible without applying a blanket low-shelf EQ to
    vocals, guitars, room tone, or residual kick.
    """
    if gain_db <= 0:
        return mix.astype(np.float32, copy=False)
    bass = _match_channels(_match_length(bass_stem, mix.shape[-1]), mix.shape[0])
    boosted = low_shelf(
        bass,
        sample_rate,
        gain_db=gain_db,
        freq_hz=freq_hz,
        q=q,
    )
    return _peak_limit(mix + (boosted - bass), sample_rate=sample_rate, headroom=headroom)


def encode_mp3(
    array: np.ndarray, sr: int, target: Path, bitrate: int = 320
) -> None:
    """Encode ``[channels, samples]`` float32 to MP3 CBR via pydub+ffmpeg."""
    from pydub import AudioSegment

    target.parent.mkdir(parents=True, exist_ok=True)
    # pydub wants a real file, not a buffer, for reliable cross-platform MP3
    # encoding via ffmpeg. Close the fd immediately — on Windows an open
    # handle blocks the final unlink.
    fd, tmp_name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        sf.write(str(tmp), array.T, sr, subtype="PCM_24")
        AudioSegment.from_wav(str(tmp)).export(
            str(target),
            format="mp3",
            bitrate=f"{int(bitrate)}k",
            codec="libmp3lame",
        )
    finally:
        tmp.unlink(missing_ok=True)


def encode_audio(
    array: np.ndarray,
    sr: int,
    target: Path,
    output_format: str = "mp3",
    bitrate: int = 320,
) -> None:
    fmt = output_format.lower()
    target.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "mp3":
        encode_mp3(array, sr=sr, target=target, bitrate=bitrate)
        return
    if fmt == "wav":
        sf.write(str(target), array.T, sr, subtype="PCM_24")
        return
    if fmt == "flac":
        sf.write(str(target), array.T, sr, format="FLAC", subtype="PCM_24")
        return
    raise StemRemoverError(f"Unsupported output format: {output_format}")


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
    input_audio: Path,
    cache_dir: Path,
    model_name: str = "htdemucs_ft",
    shifts: int = 1,
    overlap: float = 0.1,
) -> tuple[dict[str, np.ndarray], int]:
    """Return ``({name: [channels, samples] float32}, sample_rate)``.

    Reads from ``cache_dir/{name}.wav`` when all expected WAVs exist;
    otherwise runs Demucs with the given ``shifts``/``overlap`` and writes
    the full stem set for next time. The caller is responsible for
    supplying a cache_dir that incorporates ``model_name``/``shifts``/
    ``overlap`` so stems from different quality presets never collide.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    wavs = {n: cache_dir / f"{n}.wav" for n in STEM_NAMES}

    if all(p.exists() for p in wavs.values()):
        loaded: dict[str, np.ndarray] = {}
        sample_rate = 0
        for n, p in wavs.items():
            data, _sr = sf.read(str(p), dtype="float32", always_2d=True)
            sample_rate = sample_rate or int(_sr)
            # soundfile returns [samples, channels]; we want [channels, samples]
            loaded[n] = data.T.astype(np.float32, copy=False)
        return loaded, sample_rate

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
                shifts=max(1, int(shifts)),
                overlap=float(overlap),
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
                sf.write(str(wavs[name]), out[name].T, sr, subtype="FLOAT")
        return out, int(sr)
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
    job_quality = getattr(job, "quality", None) or stems_cfg.quality
    output_format = (
        getattr(job, "output_format", None)
        or getattr(stems_cfg, "output_format", "mp3")
    ).lower()
    shifts, overlap = resolve_preset(
        job_quality, stems_cfg.shifts, stems_cfg.overlap
    )
    cache_subdir = quality_cache_subdir(stems_cfg.model, shifts, overlap)
    cache_dir = (
        settings.project_root / "data" / "stems_cache"
        / cache_subdir / job.input_sha256
    )
    output_root = settings.project_root / "data" / "output" / "stems"
    output_path = deterministic_output_path(
        output_root, job.id, job.filename, job.remove_mask, output_format
    )

    # Short-circuit: identical job already produced this output (AT-003).
    if output_path.exists():
        repo.set_output_path(job.id, str(output_path))
        repo.advance(job.id, "ready", progress=1.0)
        return

    repo.advance(job.id, "separating", progress=0.2)
    extracted = extract_all_stems(
        input_path, cache_dir, stems_cfg.model, shifts=shifts, overlap=overlap
    )
    if isinstance(extracted, tuple):
        stems, stem_sr = extracted
    else:
        # Backward-compatible for tests/plugins monkeypatching older helpers.
        stems = extracted
        stem_sr = probe_sample_rate(input_path)

    repo.advance(job.id, "encoding", progress=0.8)
    channels = probe_channels(input_path)
    original = load_original_mix(input_path, stem_sr, input_was_mono=channels == 1)
    stem_samples = (
        min(arr.shape[-1] for arr in stems.values())
        if stems else original.shape[-1]
    )
    original = _match_length(original, stem_samples)
    mix = subtract_removed_stems(
        original,
        stems,
        set(job.remove_mask),
        sample_rate=stem_sr,
        strength=getattr(stems_cfg, "removal_strength", 1.0),
    )
    if "bass" in stems and "bass" not in set(job.remove_mask):
        mix = add_bass_lift(
            mix,
            stems["bass"],
            stem_sr,
            gain_db=getattr(stems_cfg, "bass_boost_db", 0.0),
            freq_hz=getattr(stems_cfg, "bass_boost_freq_hz", 120.0),
            q=getattr(stems_cfg, "bass_boost_q", 0.707),
        )
    encode_audio(
        mix,
        sr=stem_sr,
        target=output_path,
        output_format=output_format,
        bitrate=job.bitrate,
    )

    repo.set_output_path(job.id, str(output_path))
    repo.advance(job.id, "ready", progress=1.0)

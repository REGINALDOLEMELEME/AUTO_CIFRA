from __future__ import annotations

from pathlib import Path
import subprocess


def normalize_audio(
    input_audio: Path,
    output_audio: Path,
    sample_rate: int = 16000,
) -> Path:
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_audio),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-af",
        "loudnorm=I=-16:LRA=11:TP=-1.5",
        str(output_audio),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_audio


def ensure_mono_16k_wav(input_audio: Path, output_dir: Path) -> Path:
    output = output_dir / f"{input_audio.stem}.norm.wav"
    return normalize_audio(input_audio, output, sample_rate=16000)

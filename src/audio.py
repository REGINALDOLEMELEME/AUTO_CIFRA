from __future__ import annotations

from pathlib import Path
import subprocess


def normalize_audio(input_audio: Path, output_audio: Path) -> Path:
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_audio),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_audio),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_audio

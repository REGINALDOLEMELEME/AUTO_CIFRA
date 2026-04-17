from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any

from .audio import normalize_audio


@dataclass
class LyricSegment:
    start: float
    end: float
    text: str


def transcribe_audio(
    input_audio: Path,
    tmp_dir: Path,
    language: str = "pt",
    model_size: str = "small",
    use_vad: bool = False,
) -> dict[str, Any]:
    normalized_audio = tmp_dir / f"{input_audio.stem}.normalized.wav"
    normalize_audio(input_audio=input_audio, output_audio=normalized_audio)

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Run setup script first."
        ) from exc

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        str(normalized_audio),
        language=language,
        vad_filter=use_vad,
        word_timestamps=False,
    )

    items: list[LyricSegment] = []
    for segment in segments:
        text = (segment.text or "").strip()
        if not text:
            continue
        items.append(
            LyricSegment(
                start=round(float(segment.start), 3),
                end=round(float(segment.end), 3),
                text=text,
            )
        )

    return {
        "source_file": input_audio.name,
        "normalized_audio": str(normalized_audio),
        "language": getattr(info, "language", language),
        "language_probability": float(getattr(info, "language_probability", 0.0)),
        "duration": float(getattr(info, "duration", 0.0)),
        "segments": [asdict(item) for item in items],
    }


def write_transcription_json(result: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path

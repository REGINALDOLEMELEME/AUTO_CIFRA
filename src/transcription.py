from __future__ import annotations

from dataclasses import dataclass, asdict, field
import json
from pathlib import Path
from typing import Any

from .audio import normalize_audio
from .models import get_whisper


@dataclass
class LyricSegment:
    start: float
    end: float
    text: str
    words: list[dict[str, Any]] = field(default_factory=list)


def transcribe_audio(
    input_audio: Path,
    tmp_dir: Path,
    language: str = "pt",
    model_size: str = "large-v3",
    use_vad: bool = False,
    compute_type: str = "int8",
    device: str = "cpu",
) -> dict[str, Any]:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    normalized_audio = tmp_dir / f"{input_audio.stem}.normalized.wav"
    normalize_audio(input_audio=input_audio, output_audio=normalized_audio)

    model, err = get_whisper(model_size=model_size, device=device, compute_type=compute_type)
    if model is None or err:
        raise RuntimeError(err or "faster-whisper is not available")

    # Deterministic settings:
    #   - temperature=0.0 and a single-temperature list force greedy decoding
    #     (no fallback sampling pass that reintroduces randomness).
    #   - condition_on_previous_text=False prevents an early segment-boundary
    #     decision from cascading into a completely different segmentation on
    #     later runs; with it enabled, faster-whisper had been producing
    #     16 vs 7 segments across byte-identical re-runs of the same audio.
    #   - no_speech_threshold=0.7 trims the silent pre-roll that Demucs leaves
    #     at the head of the vocals track (otherwise Whisper invents tokens
    #     like "Amém." from near-silence).
    segments, info = model.transcribe(
        str(normalized_audio),
        language=language,
        vad_filter=use_vad,
        word_timestamps=True,
        beam_size=5,
        temperature=[0.0],
        condition_on_previous_text=False,
        no_speech_threshold=0.7,
    )

    items: list[LyricSegment] = []
    for segment in segments:
        text = (segment.text or "").strip()
        if not text:
            continue
        words: list[dict[str, Any]] = []
        for w in (getattr(segment, "words", None) or []):
            wt = (getattr(w, "word", "") or "").strip()
            if not wt:
                continue
            w_start = float(getattr(w, "start", segment.start) or segment.start)
            w_end = float(getattr(w, "end", w_start) or w_start)
            words.append(
                {"start": round(w_start, 3), "end": round(w_end, 3), "word": wt}
            )
        items.append(
            LyricSegment(
                start=round(float(segment.start), 3),
                end=round(float(segment.end), 3),
                text=text,
                words=words,
            )
        )

    return {
        "source_file": input_audio.name,
        "normalized_audio": str(normalized_audio),
        "language": getattr(info, "language", language),
        "language_probability": float(getattr(info, "language_probability", 0.0)),
        "duration": float(getattr(info, "duration", 0.0)),
        "segments": [asdict(item) for item in items],
        "mode": "real",
        "model_size": model_size,
    }


def write_transcription_json(result: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path

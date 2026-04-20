"""WhisperX forced alignment wrapper (ADR-004). Falls back to Whisper's native
word timestamps on any failure — AT-006."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import get_wav2vec_aligner


def align_words(
    normalized_audio: Path,
    transcription: dict[str, Any],
    language: str = "pt",
    model_name: str | None = None,
) -> tuple[dict[str, Any], str]:
    """
    Return (transcription_with_precise_words, warning). On failure, the input
    transcription is returned unchanged (its native word timestamps are kept).
    """
    try:
        import whisperx
    except Exception as exc:  # noqa: BLE001
        return transcription, f"WhisperX unavailable: {exc}. Using native Whisper word timestamps."

    model, meta, err = get_wav2vec_aligner(language=language, model_name=model_name, device="cpu")
    if err or model is None:
        return transcription, err or "wav2vec aligner failed to load"

    wx_segments = [
        {
            "start": float(s.get("start", 0.0)),
            "end": float(s.get("end", s.get("start", 0.0))),
            "text": str(s.get("text", "")),
        }
        for s in transcription.get("segments", [])
    ]
    if not wx_segments:
        return transcription, ""

    try:
        aligned = whisperx.align(
            wx_segments,
            model,
            meta,
            str(normalized_audio),
            device="cpu",
            return_char_alignments=False,
        )
    except Exception as exc:  # noqa: BLE001
        return transcription, f"WhisperX align failed: {exc}. Keeping native word timestamps."

    original_segments = transcription.get("segments", []) or []
    new_segments: list[dict[str, Any]] = []
    aligned_segments = aligned.get("segments") or []
    for orig, new in zip(original_segments, aligned_segments):
        words_aligned = new.get("words") or []
        words_out: list[dict[str, Any]] = []
        for w in words_aligned:
            text = str(w.get("word") or w.get("text") or "").strip()
            if not text:
                continue
            start = float(w.get("start") or orig.get("start", 0.0))
            end = float(w.get("end") or start)
            words_out.append(
                {"word": text, "start": round(start, 3), "end": round(end, 3)}
            )
        new_segments.append(
            {
                "start": round(float(new.get("start", orig.get("start", 0.0))), 3),
                "end": round(float(new.get("end", orig.get("end", 0.0))), 3),
                "text": str(new.get("text") or orig.get("text", "")).strip(),
                "words": words_out,
            }
        )

    return (
        {
            **transcription,
            "segments": new_segments,
            "alignment_source": "whisperx-wav2vec2",
        },
        "",
    )

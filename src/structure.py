"""Section segmentation via librosa + heuristic labeling (ADR-009).

Output schema:
[
  {"start": float, "end": float, "label": "Intro|Verso 1|Refrão|Ponte|Solo|Outro|Seção N",
   "confidence": float}
]
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def _segments_from_boundaries(bounds: list[float], duration: float) -> list[tuple[float, float]]:
    xs = sorted({round(float(b), 3) for b in bounds})
    if not xs or xs[0] > 0.0:
        xs = [0.0] + xs
    if xs[-1] < duration:
        xs.append(duration)
    return [(xs[i], xs[i + 1]) for i in range(len(xs) - 1) if xs[i + 1] - xs[i] >= 1.0]


def _lyric_density(start: float, end: float, lines: list[dict[str, Any]]) -> float:
    if end <= start:
        return 0.0
    total = 0
    span = max(end - start, 1e-3)
    for line in lines:
        s = float(line.get("start", 0.0))
        e = float(line.get("end", s))
        if e >= start and s <= end:
            overlap = max(0.0, min(e, end) - max(s, start))
            total += int(overlap * len((line.get("lyric_line") or "").split()))
    return total / span


def _segment_fingerprint(start: float, end: float, lines: list[dict[str, Any]]) -> str:
    """Join a few words from each lyric line that overlaps the segment. Stable
    enough to detect repeated chorus/verses across the song."""
    chunks: list[str] = []
    for line in lines:
        s = float(line.get("start", 0.0))
        e = float(line.get("end", s))
        if s >= start and e <= end:
            text = (line.get("lyric_line") or "").lower().strip()
            words = text.split()
            if words:
                chunks.append(" ".join(words[:5]))
    return " | ".join(chunks)


def _label_segments(
    segments: list[tuple[float, float]],
    lines: list[dict[str, Any]],
    duration: float,
) -> list[dict[str, Any]]:
    if not segments:
        return []

    fingerprints = [_segment_fingerprint(s, e, lines) for s, e in segments]
    counts: dict[str, int] = {}
    for fp in fingerprints:
        if fp:
            counts[fp] = counts.get(fp, 0) + 1

    verse_idx = 0
    labels: list[dict[str, Any]] = []
    for i, (s, e) in enumerate(segments):
        fp = fingerprints[i]
        density = _lyric_density(s, e, lines)
        label = f"Seção {i + 1}"
        conf = 0.5
        is_last = i == len(segments) - 1

        if i == 0 and density < 0.3:
            label, conf = "Intro", 0.7
        elif is_last and density < 0.3:
            label, conf = "Outro", 0.7
        elif fp and counts.get(fp, 1) >= 2:
            label, conf = "Refrão", 0.8
        elif density < 0.3 and not is_last:
            label, conf = "Solo", 0.6
        else:
            verse_idx += 1
            label, conf = f"Verso {verse_idx}", 0.7
        labels.append(
            {
                "start": round(s, 3),
                "end": round(e, 3),
                "label": label,
                "confidence": round(conf, 3),
            }
        )
    return labels


def label_sections(
    input_audio: Path,
    beats: dict[str, Any],
    transcription: dict[str, Any],
    n_segments: int = 6,
    confidence_threshold: float = 0.6,
) -> list[dict[str, Any]]:
    segments = transcription.get("segments", []) or []
    lines = [
        {
            "start": float(s.get("start", 0.0)),
            "end": float(s.get("end", s.get("start", 0.0))),
            "lyric_line": str(s.get("text", "")).strip(),
        }
        for s in segments
    ]

    try:
        import librosa
        import numpy as np
    except Exception:
        total = float(lines[-1]["end"]) if lines else 0.0
        if total <= 0.0:
            return []
        step = total / max(n_segments, 1)
        bounds = [i * step for i in range(1, n_segments)]
        sections = _segments_from_boundaries(bounds, total)
        return _label_segments(sections, lines, total)

    try:
        y, sr = librosa.load(str(input_audio), sr=22050, mono=True)
        duration = float(len(y) / sr) if sr else 0.0
        if duration < 5.0:
            return []
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        bounds_frames = librosa.segment.agglomerative(mfcc, k=max(2, n_segments))
        bounds_times = librosa.frames_to_time(bounds_frames, sr=sr).tolist()
        sections = _segments_from_boundaries(bounds_times, duration)
        labeled = _label_segments(sections, lines, duration)
        return [s for s in labeled if s["confidence"] >= confidence_threshold] or labeled
    except Exception:
        total = float(lines[-1]["end"]) if lines else 0.0
        step = total / max(n_segments, 1) if total else 1.0
        bounds = [i * step for i in range(1, n_segments)]
        sections = _segments_from_boundaries(bounds, total)
        return _label_segments(sections, lines, total)

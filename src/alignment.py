from __future__ import annotations

import re
from typing import Any


def _collect_chord_events(
    chord_segments: list[dict[str, Any]],
    start: float,
    end: float,
) -> list[dict[str, Any]]:
    events = [
        {
            "start": float(c.get("start", 0.0)),
            "chord": str(c.get("chord", "")).strip(),
        }
        for c in chord_segments
        if float(c.get("start", 0.0)) >= start and float(c.get("start", 0.0)) <= end
    ]
    events = [e for e in events if e["chord"]]
    if events:
        return events

    previous = [
        {
            "start": float(c.get("start", 0.0)),
            "chord": str(c.get("chord", "")).strip(),
        }
        for c in chord_segments
        if float(c.get("start", 0.0)) <= start and str(c.get("chord", "")).strip()
    ]
    if previous:
        return [previous[-1]]
    return []


def _word_positions(
    lyric_text: str,
    words: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    spans = list(re.finditer(r"\S+", lyric_text))
    if not spans:
        return []
    if not words:
        return [{"index": s.start(), "start": None} for s in spans]

    out: list[dict[str, Any]] = []
    limit = min(len(spans), len(words))
    for i in range(limit):
        out.append(
            {
                "index": spans[i].start(),
                "start": float(words[i].get("start", 0.0)),
            }
        )
    for i in range(limit, len(spans)):
        out.append({"index": spans[i].start(), "start": None})
    return out


def _target_index_for_event(
    event_start: float,
    seg_start: float,
    seg_end: float,
    word_pos: list[dict[str, Any]],
    lyric_len: int,
) -> int:
    timed_words = [w for w in word_pos if w.get("start") is not None]
    if timed_words:
        # nearest word start in time
        best = min(timed_words, key=lambda w: abs(float(w["start"]) - event_start))
        return int(best["index"])

    duration = max(seg_end - seg_start, 1e-6)
    ratio = min(1.0, max(0.0, (event_start - seg_start) / duration))
    return int(round(ratio * max(0, lyric_len - 1)))


def _build_chord_line(
    lyric_text: str,
    events: list[dict[str, Any]],
    seg_start: float,
    seg_end: float,
    words: list[dict[str, Any]],
) -> str:
    if not events:
        return ""

    lyric_len = max(len(lyric_text), 1)
    buffer = [" "] * (lyric_len + 40)
    word_pos = _word_positions(lyric_text, words)

    last_chord = ""
    for e in events:
        chord = str(e.get("chord", "")).strip()
        if not chord or chord == last_chord:
            continue
        last_chord = chord

        pos = _target_index_for_event(
            event_start=float(e.get("start", seg_start)),
            seg_start=seg_start,
            seg_end=seg_end,
            word_pos=word_pos,
            lyric_len=lyric_len,
        )
        pos = max(0, min(pos, len(buffer) - len(chord)))

        # Avoid overlap by shifting right
        while pos < len(buffer) - len(chord):
            left_ok = (pos == 0) or (buffer[pos - 1] == " ")
            right_idx = pos + len(chord)
            right_ok = (right_idx >= len(buffer)) or (buffer[right_idx] == " ")
            if all(buffer[pos + i] == " " for i in range(len(chord))) and left_ok and right_ok:
                break
            pos += 1
        for i, ch in enumerate(chord):
            if pos + i < len(buffer):
                buffer[pos + i] = ch

    return "".join(buffer).rstrip()


def align_chords_to_lyrics(transcription: dict[str, Any], chords: dict[str, Any]) -> dict[str, Any]:
    lyric_segments = transcription.get("segments", [])
    chord_segments = chords.get("segments", [])
    lines: list[dict[str, Any]] = []

    for lyric in lyric_segments:
        start = float(lyric.get("start", 0.0))
        end = float(lyric.get("end", start))
        text = str(lyric.get("text", "")).strip()
        if not text:
            continue

        words = lyric.get("words", []) or []
        events = _collect_chord_events(chord_segments=chord_segments, start=start, end=end)
        unique_chords: list[str] = []
        for e in events:
            c = str(e.get("chord", "")).strip()
            if c and c not in unique_chords:
                unique_chords.append(c)
        chord_line = _build_chord_line(
            lyric_text=text,
            events=events,
            seg_start=start,
            seg_end=end,
            words=words,
        )

        lines.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "chords": unique_chords,
                "chord_line": chord_line if chord_line else " ".join(unique_chords),
                "lyric_line": text,
            }
        )

    return {
        "source_file": transcription.get("source_file", ""),
        "transcription_mode": transcription.get("mode", "real"),
        "chord_mode": chords.get("mode", "real"),
        "warnings": [w for w in [transcription.get("warning", ""), chords.get("warning", "")] if w],
        "lines": lines,
    }

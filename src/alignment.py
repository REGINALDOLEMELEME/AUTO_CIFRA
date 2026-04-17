from __future__ import annotations

from typing import Any


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

        in_window = [
            c.get("chord", "")
            for c in chord_segments
            if float(c.get("start", 0.0)) >= start and float(c.get("start", 0.0)) <= end
        ]
        if not in_window:
            previous = [
                c.get("chord", "")
                for c in chord_segments
                if float(c.get("start", 0.0)) <= start
            ]
            if previous:
                in_window = [previous[-1]]

        unique_chords: list[str] = []
        for chord in in_window:
            chord_text = str(chord).strip()
            if chord_text and chord_text not in unique_chords:
                unique_chords.append(chord_text)

        lines.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "chords": unique_chords,
                "chord_line": " ".join(unique_chords),
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

from __future__ import annotations

from typing import Any


def _word_interval(word: dict[str, Any]) -> tuple[float, float]:
    start = float(word.get("start") or 0.0)
    end = float(word.get("end") or start)
    if end < start:
        end = start
    return start, end


def _segment_words(segment: dict[str, Any]) -> list[dict[str, Any]]:
    words = segment.get("words") or []
    out: list[dict[str, Any]] = []
    for w in words:
        text = str(w.get("word") or w.get("text") or "").strip()
        if not text:
            continue
        start, end = _word_interval(w)
        out.append({"text": text, "start": round(start, 3), "end": round(end, 3)})
    if not out:
        fallback_text = str(segment.get("text") or "").strip()
        if fallback_text:
            s = float(segment.get("start") or 0.0)
            e = float(segment.get("end") or s)
            span = max(e - s, 1e-3)
            tokens = fallback_text.split()
            step = span / max(len(tokens), 1)
            for i, t in enumerate(tokens):
                out.append(
                    {
                        "text": t,
                        "start": round(s + i * step, 3),
                        "end": round(s + (i + 1) * step, 3),
                    }
                )
    return out


def _chord_at(chord_segments: list[dict[str, Any]], t: float) -> str | None:
    for seg in chord_segments:
        s = float(seg.get("start", 0.0))
        e = float(seg.get("end", s))
        if s <= t < e:
            chord = str(seg.get("chord") or "").strip()
            return chord or None
    return None


def _unique_chord_at_word(
    chord_segments: list[dict[str, Any]], word: dict[str, Any], last_chord: str | None
) -> str | None:
    start, end = word["start"], word["end"]
    chord_at_start = _chord_at(chord_segments, start)
    if chord_at_start and chord_at_start != last_chord:
        return chord_at_start
    # chord changed inside the word window -> still place at this word
    for seg in chord_segments:
        cs = float(seg.get("start", 0.0))
        if start <= cs < end:
            c = str(seg.get("chord") or "").strip()
            if c and c != last_chord:
                return c
    return None


def _attach_section(
    line_start: float,
    line_end: float,
    sections: list[dict[str, Any]],
) -> str | None:
    for s in sections:
        if float(s.get("start", 0.0)) <= line_start < float(s.get("end", 0.0)):
            label = str(s.get("label") or "").strip()
            return label or None
    # Fall back to whichever section overlaps the line center.
    mid = (line_start + line_end) / 2.0
    for s in sections:
        if float(s.get("start", 0.0)) <= mid < float(s.get("end", 0.0)):
            label = str(s.get("label") or "").strip()
            return label or None
    return None


def align_chords_by_word_time(
    transcription: dict[str, Any],
    chords: dict[str, Any],
    sections: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Attach each chord change to the word whose interval contains the chord's start.
    - Word timestamps come from WhisperX (preferred) or Whisper's native word times.
    - Chord segments have already been vocabulary-normalized and beat-smoothed.
    """
    sections = sections or []
    segments = transcription.get("segments", []) or []
    chord_segments = chords.get("segments", []) or []

    lines: list[dict[str, Any]] = []
    last_chord: str | None = None
    emitted_sections: set[str] = set()

    for seg in segments:
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        words = _segment_words(seg)
        if not words:
            continue
        line_start = words[0]["start"]
        line_end = words[-1]["end"]

        decorated_words: list[dict[str, Any]] = []
        for w in words:
            chord_here = _unique_chord_at_word(chord_segments, w, last_chord)
            if chord_here:
                last_chord = chord_here
            decorated_words.append({**w, "chord": chord_here})

        # If the first word inherits a chord that was already playing at the line
        # start but not emitted yet, surface it so musicians see the entry chord.
        if decorated_words and not decorated_words[0].get("chord"):
            running = _chord_at(chord_segments, line_start)
            if running and running != _prev_emitted(lines):
                decorated_words[0]["chord"] = running
                last_chord = running

        section_label = _attach_section(line_start, line_end, sections)
        show_section = None
        if section_label and section_label not in emitted_sections:
            show_section = section_label
            emitted_sections.add(section_label)

        lines.append(
            {
                "section": show_section,
                "start": round(line_start, 3),
                "end": round(line_end, 3),
                "words": decorated_words,
                "lyric_line": " ".join(w["text"] for w in decorated_words),
                "chord_line": _legacy_chord_line(decorated_words),
                "chords": [w["chord"] for w in decorated_words if w.get("chord")],
            }
        )

    return {
        "source_file": transcription.get("source_file", ""),
        "transcription_mode": transcription.get("mode", "real"),
        "chord_mode": chords.get("mode", "real"),
        "warnings": [
            w
            for w in (transcription.get("warning", ""), chords.get("warning", ""))
            if w
        ],
        "lines": lines,
    }


def _prev_emitted(lines: list[dict[str, Any]]) -> str | None:
    for line in reversed(lines):
        for w in reversed(line.get("words", [])):
            c = w.get("chord")
            if c:
                return c
    return None


def _legacy_chord_line(words: list[dict[str, Any]]) -> str:
    """Flat, space-separated list of chords for CLI/JSON consumers. The DOCX
    exporter bypasses this and uses the per-word `words[].chord` directly."""
    chords: list[str] = []
    for w in words:
        c = w.get("chord")
        if c and (not chords or chords[-1] != c):
            chords.append(c)
    return " ".join(chords)

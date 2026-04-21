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


def _chord_changes(chord_segments: list[dict[str, Any]]) -> list[tuple[float, str]]:
    """Collapse a dense chord-segment list into (start_time, label) tuples
    at each change point. Keeps absolute timestamps."""
    out: list[tuple[float, str]] = []
    for seg in chord_segments:
        label = str(seg.get("chord") or "").strip()
        if not label:
            continue
        start = float(seg.get("start", 0.0))
        if not out or out[-1][1] != label:
            out.append((start, label))
    return out


def _assign_chords_nearest_word(
    words: list[dict[str, Any]],
    chord_segments: list[dict[str, Any]],
    line_start: float,
    line_end: float,
    last_chord: str | None,
    tolerance_s: float = 0.5,
) -> list[dict[str, Any]]:
    """For each chord change that falls in or near this line's time window,
    assign it to the word whose start timestamp is closest.

    Rationale: the previous algorithm placed a chord on whichever word was
    *playing* when the chord started. Because ASR word-onset timestamps and
    Chordino segment boundaries each carry ~100-300 ms of independent jitter,
    the chord routinely landed one word early or late. Nearest-start matching
    with a tolerance window is more forgiving of that jitter and matches how
    humans read chord sheets — the chord sits over the syllable it *starts*
    on, not the syllable that happens to be sustaining when the chord arrives.
    """
    decorated = [{**w, "chord": None} for w in words]
    if not decorated:
        return decorated

    # Chord changes whose onset falls in [line_start - tol, line_end + tol].
    changes = [
        (t, c) for (t, c) in _chord_changes(chord_segments)
        if line_start - tolerance_s <= t <= line_end + tolerance_s
    ]
    used_word_indices: set[int] = set()
    running = last_chord
    for t, c in changes:
        if c == running:
            continue
        # Pick the closest word by start time that hasn't already been assigned.
        best_idx = -1
        best_dist = float("inf")
        for i, w in enumerate(decorated):
            if i in used_word_indices:
                continue
            dist = abs(float(w["start"]) - t)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx < 0:
            break
        # Only accept the match if it's within tolerance OR if the chord's
        # onset is inside the word's span (word stretched past the chord's
        # nominal onset).
        w = decorated[best_idx]
        inside_word = float(w["start"]) <= t < float(w["end"])
        if best_dist <= tolerance_s or inside_word:
            decorated[best_idx]["chord"] = c
            used_word_indices.add(best_idx)
            running = c
    return decorated


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


def _chord_changes_in_window(
    chord_segments: list[dict[str, Any]], t_start: float, t_end: float
) -> list[str]:
    """Ordered, deduped chord labels whose segments overlap [t_start, t_end)."""
    if t_end <= t_start:
        return []
    out: list[str] = []
    for seg in chord_segments:
        s = float(seg.get("start", 0.0))
        e = float(seg.get("end", s))
        if e <= t_start or s >= t_end:
            continue
        label = str(seg.get("chord") or "").strip()
        if not label:
            continue
        if not out or out[-1] != label:
            out.append(label)
    return out


def _instrumental_line(
    start: float,
    end: float,
    chord_labels: list[str],
    section: str | None,
) -> dict[str, Any]:
    return {
        "section": section,
        "start": round(start, 3),
        "end": round(end, 3),
        "words": [],
        "lyric_line": "",
        "chord_line": " ".join(chord_labels),
        "chords": chord_labels,
    }


def align_chords_by_word_time(
    transcription: dict[str, Any],
    chords: dict[str, Any],
    sections: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Attach each chord change to the word whose interval contains the chord's start.
    - Word timestamps come from WhisperX (preferred) or Whisper's native word times.
    - Chord segments have already been vocabulary-normalized and beat-smoothed.
    - Chord changes that fall in the gap between vocal lines (intros, solos,
      instrumental breaks, outros) are emitted as chord-only "instrumental"
      lines so musicians reading the sheet see the full chord progression.
    """
    sections = sections or []
    segments = transcription.get("segments", []) or []
    chord_segments = chords.get("segments", []) or []

    vocal_lines: list[dict[str, Any]] = []
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

        decorated_words = _assign_chords_nearest_word(
            words=words,
            chord_segments=chord_segments,
            line_start=line_start,
            line_end=line_end,
            last_chord=last_chord,
        )
        # Track the running chord for the next line: last chord *label* placed.
        for dw in decorated_words:
            if dw.get("chord"):
                last_chord = dw["chord"]

        # If the first word inherits a chord that was already playing at the line
        # start but not emitted yet, surface it so musicians see the entry chord.
        if decorated_words and not decorated_words[0].get("chord"):
            running = _chord_at(chord_segments, line_start)
            if running and running != _prev_emitted(vocal_lines):
                decorated_words[0]["chord"] = running
                last_chord = running

        vocal_lines.append(
            {
                "_line_start": line_start,
                "_line_end": line_end,
                "words": decorated_words,
            }
        )

    if not vocal_lines and not chord_segments:
        return {
            "source_file": transcription.get("source_file", ""),
            "transcription_mode": transcription.get("mode", "real"),
            "chord_mode": chords.get("mode", "real"),
            "warnings": [
                w
                for w in (transcription.get("warning", ""), chords.get("warning", ""))
                if w
            ],
            "lines": [],
        }

    song_end = max(
        (float(s.get("end", 0.0)) for s in chord_segments),
        default=(vocal_lines[-1]["_line_end"] if vocal_lines else 0.0),
    )

    lines: list[dict[str, Any]] = []

    def _emit_section_tag(start: float, end: float) -> str | None:
        label = _attach_section(start, end, sections)
        if label and label not in emitted_sections:
            emitted_sections.add(label)
            return label
        return None

    def _last_emitted_chord() -> str | None:
        for line in reversed(lines):
            chords_seq = line.get("chords") or []
            if chords_seq:
                return chords_seq[-1]
        return None

    prev_end = 0.0
    for vl in vocal_lines:
        line_start = vl["_line_start"]
        line_end = vl["_line_end"]
        gap_labels = _chord_changes_in_window(chord_segments, prev_end, line_start)
        last_seen = _last_emitted_chord()
        if gap_labels and gap_labels[0] == last_seen:
            gap_labels = gap_labels[1:]
        if gap_labels:
            section_tag = _emit_section_tag(prev_end, line_start)
            lines.append(_instrumental_line(prev_end, line_start, gap_labels, section_tag))

        decorated_words = vl["words"]
        section_tag = _emit_section_tag(line_start, line_end)
        lines.append(
            {
                "section": section_tag,
                "start": round(line_start, 3),
                "end": round(line_end, 3),
                "words": decorated_words,
                "lyric_line": " ".join(w["text"] for w in decorated_words),
                "chord_line": _legacy_chord_line(decorated_words),
                "chords": [w["chord"] for w in decorated_words if w.get("chord")],
            }
        )
        prev_end = line_end

    outro_labels = _chord_changes_in_window(chord_segments, prev_end, song_end)
    last_seen = _last_emitted_chord()
    if outro_labels and outro_labels[0] == last_seen:
        outro_labels = outro_labels[1:]
    if outro_labels:
        section_tag = _emit_section_tag(prev_end, song_end)
        lines.append(_instrumental_line(prev_end, song_end, outro_labels, section_tag))

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

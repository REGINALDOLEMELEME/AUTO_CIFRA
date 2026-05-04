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
    for seg in chord_segments:
        cs = float(seg.get("start", 0.0))
        if start <= cs < end:
            c = str(seg.get("chord") or "").strip()
            if c and c != last_chord:
                return c
    return None


def _chord_changes(chord_segments: list[dict[str, Any]]) -> list[tuple[float, str]]:
    """Collapse a dense chord-segment list into (start_time, label) tuples."""
    out: list[tuple[float, str]] = []
    for seg in chord_segments:
        label = str(seg.get("chord") or "").strip()
        if not label:
            continue
        start = float(seg.get("start", 0.0))
        if not out or out[-1][1] != label:
            out.append((start, label))
    return out


def _word_attack_time(word: dict[str, Any]) -> float:
    """Return a better visual anchor than the raw word start."""
    start = float(word["start"])
    end = float(word["end"])
    dur = max(0.0, end - start)
    return start + min(0.12, dur * 0.35)


def _nearest_word_index(words: list[dict[str, Any]], t: float) -> int:
    """Map a chord onset to the nearest word-attack slot on the line."""
    if not words:
        return -1
    attacks = [_word_attack_time(w) for w in words]
    if len(attacks) == 1:
        return 0
    for i in range(len(attacks) - 1):
        boundary = (attacks[i] + attacks[i + 1]) / 2.0
        if t < boundary:
            return i
    return len(attacks) - 1


def _musical_word_index(words: list[dict[str, Any]], t: float) -> int:
    """Map a chord onset to the word where a musician expects to read it.

    A chord that lands in the silence before the next lyric usually belongs
    above that next word, not above the previous sustained word. This differs
    from pure nearest-neighbor timing and produces cifra layouts closer to
    hand-written charts.
    """
    if not words:
        return -1
    if len(words) == 1:
        return 0

    first_start = float(words[0]["start"])
    if t <= first_start:
        return 0

    for i, word in enumerate(words):
        start = float(word["start"])
        end = float(word["end"])
        if start <= t < end:
            return i
        if i + 1 >= len(words):
            continue

        next_start = float(words[i + 1]["start"])
        if end <= t < next_start:
            gap = max(0.0, next_start - end)
            if gap <= 0.08:
                return _nearest_word_index(words, t)
            # Bias toward the next word once the chord is meaningfully inside
            # the pre-word gap. This fixes the common "one word late/early"
            # cifra problem without inventing extra visual columns.
            handoff = end + gap * 0.35
            return i + 1 if t >= handoff else i

    return _nearest_word_index(words, t)


def _assign_chords_nearest_word(
    words: list[dict[str, Any]],
    chord_segments: list[dict[str, Any]],
    line_start: float,
    line_end: float,
    last_chord: str | None,
    tolerance_s: float = 0.5,
) -> list[dict[str, Any]]:
    """Assign each chord change to the nearest word-attack anchor."""
    decorated = [{**w, "chord": None} for w in words]
    if not decorated:
        return decorated

    changes = [
        (t, c)
        for (t, c) in _chord_changes(chord_segments)
        if line_start - tolerance_s <= t <= line_end + tolerance_s
    ]
    used_word_indices: set[int] = set()
    running = last_chord
    for t, c in changes:
        if c == running:
            continue
        best_idx = _musical_word_index(decorated, t)
        if best_idx < 0 or best_idx in used_word_indices:
            continue
        candidate_idx = best_idx
        w = decorated[candidate_idx]
        dist = abs(_word_attack_time(w) - t)
        inside_word = float(w["start"]) <= t < float(w["end"])
        if dist <= tolerance_s or inside_word or candidate_idx == 0:
            decorated[candidate_idx]["chord"] = c
            decorated[candidate_idx]["chord_time"] = round(float(t), 3)
            used_word_indices.add(candidate_idx)
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
    - Chord changes that fall in the gap between vocal lines are emitted as
      chord-only instrumental lines.
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
        for dw in decorated_words:
            if dw.get("chord"):
                last_chord = dw["chord"]

        if decorated_words and not decorated_words[0].get("chord"):
            running = _chord_at(chord_segments, line_start)
            if running and running != _prev_emitted(vocal_lines):
                decorated_words[0]["chord"] = running
                decorated_words[0]["chord_time"] = round(float(line_start), 3)
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
    """Flat, space-separated list of chords for CLI/JSON consumers."""
    chords: list[str] = []
    for w in words:
        c = w.get("chord")
        if c and (not chords or chords[-1] != c):
            chords.append(c)
    return " ".join(chords)

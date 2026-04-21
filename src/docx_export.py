from __future__ import annotations

from pathlib import Path
from typing import Any

from .transpose import effective_semitones, shift_chord


def _set_font(run, name: str, size_pt: int, bold: bool = False) -> None:
    from docx.shared import Pt

    run.font.name = name
    run.font.size = Pt(size_pt)
    run.bold = bold


def _set_cell_font(cell, text: str, name: str, size_pt: int, bold: bool = False) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_before = 0
    paragraph.paragraph_format.space_after = 0
    run = paragraph.add_run(text)
    _set_font(run, name=name, size_pt=size_pt, bold=bold)


def _remove_table_borders(table) -> None:
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    tbl_pr = table._element.find(qn("w:tblPr"))
    if tbl_pr is None:
        tbl_pr = OxmlElement("w:tblPr")
        table._element.insert(0, tbl_pr)
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "nil")
        borders.append(el)
    existing = tbl_pr.find(qn("w:tblBorders"))
    if existing is not None:
        tbl_pr.remove(existing)
    tbl_pr.append(borders)


def _split_line_on_punctuation(
    words: list[dict[str, Any]], max_words: int = 12
) -> list[list[dict[str, Any]]]:
    """Split a long line into readable sub-lines. Break AFTER a word whose
    text ends in comma/semicolon/period, preferring breaks close to the
    midpoint when a single segment runs longer than `max_words`.

    Why this exists: Whisper sometimes emits a whole verse as one segment.
    Rendered verbatim, that becomes a 30-column chord table that no musician
    can read. Splitting on the punctuation Whisper already inserted gives
    us natural phrase boundaries without guessing at prosody.
    """
    if len(words) <= max_words:
        return [words]
    # Gather indices of break points (word ends with ,;.:!?).
    break_chars = (",", ";", ".", ":", "!", "?")
    break_idx = [
        i for i, w in enumerate(words)
        if str(w.get("text", "")).strip().endswith(break_chars)
    ]
    if not break_idx:
        # No punctuation in a long line — hard-split every `max_words` words.
        return [words[i:i + max_words] for i in range(0, len(words), max_words)]

    # Greedy: walk through, cut after the last break that keeps the sub-line
    # at most `max_words` long. Never emit a sub-line > max_words even if it
    # means cutting at a non-punctuation boundary as a last resort.
    chunks: list[list[dict[str, Any]]] = []
    start = 0
    while start < len(words):
        # Candidate break points within [start, start + max_words)
        limit = min(start + max_words, len(words))
        candidates = [i for i in break_idx if start <= i < limit]
        if not candidates and limit == len(words):
            chunks.append(words[start:limit])
            break
        if not candidates:
            # No punctuation in this window — hard cut at max_words - 1.
            chunks.append(words[start:limit])
            start = limit
            continue
        cut = candidates[-1] + 1  # inclusive of the punctuation word
        chunks.append(words[start:cut])
        start = cut
    return [c for c in chunks if c]


def _write_chord_word_table(
    document,
    words: list[dict[str, Any]],
    semitones: int,
    prefer_flats: bool,
    body_font: str,
    chord_font: str,
    body_size_pt: int,
    chord_size_pt: int,
) -> None:
    if not words:
        return
    table = document.add_table(rows=2, cols=len(words))
    table.autofit = True
    _remove_table_borders(table)

    for i, w in enumerate(words):
        chord = w.get("chord")
        chord_txt = shift_chord(chord, semitones, prefer_flats=prefer_flats) if chord else ""
        _set_cell_font(
            table.cell(0, i), chord_txt, name=chord_font, size_pt=chord_size_pt, bold=True
        )
        _set_cell_font(
            table.cell(1, i), w.get("text", ""), name=body_font, size_pt=body_size_pt
        )


def export_aligned_chord_docx(
    arrangement: dict[str, Any],
    output_path: Path,
    title: str,
    transpose_semitones: int = 0,
    capo_fret: int = 0,
    prefer_flats: bool = True,
    body_font: str = "Calibri",
    chord_font: str = "Calibri",
    body_size_pt: int = 11,
    chord_size_pt: int = 11,
) -> Path:
    try:
        from docx import Document
        from docx.shared import Pt
    except ImportError as exc:
        raise RuntimeError("python-docx is not installed.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = Document()

    heading = document.add_heading(title, level=1)
    heading.alignment = 0

    semitones = effective_semitones(transpose_semitones, capo_fret)

    meta = document.add_paragraph()
    meta_run = meta.add_run(
        f"Source: {arrangement.get('source_file', '-')}"
        + (f"  |  Transpose: {transpose_semitones:+d}" if transpose_semitones else "")
        + (f"  |  Capo: {capo_fret}" if capo_fret else "")
    )
    _set_font(meta_run, name=body_font, size_pt=10)
    meta.paragraph_format.space_after = Pt(6)

    warnings = arrangement.get("warnings", []) or []
    if warnings:
        warn_p = document.add_paragraph()
        warn_h = warn_p.add_run("Notes:")
        _set_font(warn_h, name=body_font, size_pt=10, bold=True)
        for warning in warnings:
            wp = document.add_paragraph()
            _set_font(wp.add_run(f"- {warning}"), name=body_font, size_pt=10)

    current_section: str | None = None
    for line in arrangement.get("lines", []) or []:
        section = line.get("section")
        if section and section != current_section:
            current_section = section
            sh = document.add_paragraph()
            sh.paragraph_format.space_before = Pt(10)
            sh.paragraph_format.space_after = Pt(2)
            _set_font(sh.add_run(f"[{section}]"), name=body_font, size_pt=12, bold=True)

        words = line.get("words") or []
        if not words:
            text = str(line.get("lyric_line") or "").strip()
            if text:
                p = document.add_paragraph()
                _set_font(p.add_run(text), name=body_font, size_pt=body_size_pt)
            continue

        # Split long lines at punctuation for readability (see helper docstring).
        for chunk in _split_line_on_punctuation(words, max_words=12):
            _write_chord_word_table(
                document=document,
                words=chunk,
                semitones=semitones,
                prefer_flats=prefer_flats,
                body_font=body_font,
                chord_font=chord_font,
                body_size_pt=body_size_pt,
                chord_size_pt=chord_size_pt,
            )
        spacer = document.add_paragraph()
        spacer.paragraph_format.space_after = Pt(4)

    document.save(str(output_path))
    return output_path


def export_transcription_docx(transcription: dict[str, Any], output_path: Path, title: str) -> Path:
    """Simple lyrics-only export used by the non-arrangement endpoint."""
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError("python-docx is not installed.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = Document()
    document.add_heading(title, level=1)
    document.add_paragraph(f"Source: {transcription.get('source_file', '-')}")

    for segment in transcription.get("segments", []):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        start = float(segment.get("start", 0.0) or 0.0)
        document.add_paragraph(f"[{start:0.2f}] {text}")

    document.save(str(output_path))
    return output_path

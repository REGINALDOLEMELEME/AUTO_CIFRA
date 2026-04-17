from __future__ import annotations

from pathlib import Path


def _set_run_font(run, name: str, size_pt: int, bold: bool = False) -> None:
    from docx.shared import Pt

    run.font.name = name
    run.font.size = Pt(size_pt)
    run.bold = bold


def export_transcription_docx(transcription: dict, output_path: Path, title: str) -> Path:
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError("python-docx is not installed.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = Document()
    document.add_heading(title, level=1)
    document.add_paragraph(f"Source: {transcription.get('source_file', '-')}")
    document.add_paragraph("")

    for segment in transcription.get("segments", []):
        start = segment.get("start", 0.0)
        text = segment.get("text", "").strip()
        if not text:
            continue
        document.add_paragraph(f"[{start:0.2f}] {text}")

    document.save(str(output_path))
    return output_path


def export_aligned_chord_docx(arrangement: dict, output_path: Path, title: str) -> Path:
    try:
        from docx import Document
        from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
        from docx.shared import Pt
    except ImportError as exc:
        raise RuntimeError("python-docx is not installed.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = Document()
    heading = document.add_heading(title, level=1)
    heading.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT

    info = document.add_paragraph()
    info_run = info.add_run("AUTO_CIFRA - Chord Sheet")
    _set_run_font(info_run, name="Calibri", size_pt=11, bold=True)

    meta = document.add_paragraph()
    meta_run = meta.add_run(
        f"Source: {arrangement.get('source_file', '-')} | "
        f"Transcription mode: {arrangement.get('transcription_mode', 'real')} | "
        f"Chord mode: {arrangement.get('chord_mode', 'real')}"
    )
    _set_run_font(meta_run, name="Calibri", size_pt=10, bold=False)
    meta.paragraph_format.space_after = Pt(8)

    warnings = arrangement.get("warnings", [])
    if warnings:
        warn_p = document.add_paragraph()
        warn_h = warn_p.add_run("Warnings:")
        _set_run_font(warn_h, name="Calibri", size_pt=10, bold=True)
        for warning in warnings:
            wp = document.add_paragraph()
            wr = wp.add_run(f"- {warning}")
            _set_run_font(wr, name="Calibri", size_pt=10, bold=False)
        document.add_paragraph("")

    for line in arrangement.get("lines", []):
        chord_line = str(line.get("chord_line", "")).strip()
        lyric_line = str(line.get("lyric_line", "")).strip()
        if chord_line:
            chord_p = document.add_paragraph()
            chord_p.paragraph_format.space_before = Pt(4)
            chord_p.paragraph_format.space_after = Pt(0)
            chord_r = chord_p.add_run(chord_line)
            _set_run_font(chord_r, name="Consolas", size_pt=12, bold=True)
        if lyric_line:
            lyric_p = document.add_paragraph()
            lyric_p.paragraph_format.space_before = Pt(0)
            lyric_p.paragraph_format.space_after = Pt(6)
            lyric_r = lyric_p.add_run(lyric_line)
            _set_run_font(lyric_r, name="Consolas", size_pt=11, bold=False)

    document.save(str(output_path))
    return output_path

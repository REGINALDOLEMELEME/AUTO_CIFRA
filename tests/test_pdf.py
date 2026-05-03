from __future__ import annotations

from pathlib import Path

from src.docx_export import export_aligned_chord_pdf


def test_pdf_export_writes_valid_header(sample_aligned: dict, tmp_path: Path):
    out = tmp_path / "song.pdf"
    export_aligned_chord_pdf(
        arrangement=sample_aligned,
        output_path=out,
        title="song",
        transpose_semitones=0,
        capo_fret=0,
    )
    data = out.read_bytes()
    assert data.startswith(b"%PDF-")
    assert len(data) > 500


"""Re-export a DOCX from an existing aligned.json.

Fast iteration loop for formatting fixes: change docx_export.py or the aligned
data structure, rerun this, open the resulting DOCX. No 8-minute pipeline
re-run required.

Usage:
    python scripts/reexport_docx.py <job_id> [--out <path>]
    python scripts/reexport_docx.py data/tmp/<job_id>/aligned.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import get_settings
from src.docx_export import export_aligned_chord_docx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("job_or_path", help="job_id or path to aligned.json")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--transpose", type=int, default=0)
    ap.add_argument("--capo", type=int, default=0)
    args = ap.parse_args()

    settings = get_settings()
    p = Path(args.job_or_path)
    if p.is_file():
        aligned_path = p
    else:
        aligned_path = settings.tmp_dir / args.job_or_path / "aligned.json"
    if not aligned_path.exists():
        raise SystemExit(f"aligned.json not found: {aligned_path}")
    aligned = json.loads(aligned_path.read_text(encoding="utf-8"))
    source = aligned.get("source_file", "song.wav")
    stem = Path(source).stem
    title = args.title or stem
    out_path = args.out or (aligned_path.parent / f"{stem}.reexport.docx")

    export_aligned_chord_docx(
        arrangement=aligned,
        output_path=out_path,
        title=title,
        transpose_semitones=args.transpose,
        capo_fret=args.capo,
        prefer_flats=True,
        body_font=settings.docx.body_font,
        chord_font=settings.docx.chord_font,
        body_size_pt=settings.docx.body_size_pt,
        chord_size_pt=settings.docx.chord_size_pt,
    )
    print(f"Exported: {out_path}")


if __name__ == "__main__":
    main()

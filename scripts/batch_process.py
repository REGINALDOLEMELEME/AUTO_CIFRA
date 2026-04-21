"""Batch-process a folder of audio files through the AUTO_CIFRA pipeline.

Usage:
    python scripts/batch_process.py path/to/folder [--pdfs path/to/pdfs]

The folder should contain song audio files (.mp3/.wav/.m4a/.flac/.ogg).
If --pdfs is provided, each audio file's matching PDF (by stem) will be used
as ground truth for lyric WER measurement; results land in a CSV next to the
DOCX outputs so lexicon expansion can be driven by the worst offenders.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import get_settings
from src.docx_export import export_aligned_chord_docx
from src.jobs import Job, JobRepo
from src.pipeline import run as run_pipeline


AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}


def _normalize_for_wer(text: str) -> list[str]:
    """Lowercase, strip accents and punctuation, split on whitespace. Lossy by
    design — we want WER on *what the listener hears*, not on formatting."""
    t = unicodedata.normalize("NFD", text or "")
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = t.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return [w for w in t.split() if w]


def _wer(ref_tokens: list[str], hyp_tokens: list[str]) -> float:
    if not ref_tokens:
        return 1.0 if hyp_tokens else 0.0
    # Classic Levenshtein on tokens.
    n, m = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m] / n


def _read_pdf_text(pdf_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore[no-redef]
        except Exception:
            return ""
    reader = PdfReader(str(pdf_path))
    return "\n".join((p.extract_text() or "") for p in reader.pages)


def _load_aligned(job_tmp: Path) -> dict:
    import json

    path = job_tmp / "aligned.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _hyp_text_from_aligned(aligned: dict) -> str:
    lines = []
    for line in aligned.get("lines", []) or []:
        text = line.get("lyric_line") or ""
        if text:
            lines.append(text)
    return " ".join(lines)


def process_one(audio: Path, repo: JobRepo, settings) -> tuple[str, Path, dict]:
    job = repo.create(audio.name)
    repo.advance(job.id, "queued", progress=0.0)
    job = repo.get(job.id)  # type: ignore[assignment]
    # Ensure the file is in the input dir (repo.create only names it).
    dest = settings.input_dir / audio.name
    if not dest.exists():
        dest.write_bytes(audio.read_bytes())
    run_pipeline(job, repo, settings)
    job_tmp = settings.tmp_dir / job.id
    aligned = _load_aligned(job_tmp)

    out_dir = settings.output_dir / job.id
    out_dir.mkdir(parents=True, exist_ok=True)
    docx_path = out_dir / f"{audio.stem}.docx"
    export_aligned_chord_docx(
        arrangement=aligned,
        output_path=docx_path,
        title=audio.stem,
        transpose_semitones=0,
        capo_fret=0,
        prefer_flats=True,
        body_font=settings.docx.body_font,
        chord_font=settings.docx.chord_font,
        body_size_pt=settings.docx.body_size_pt,
        chord_size_pt=settings.docx.chord_size_pt,
    )
    return job.id, docx_path, aligned


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", type=Path, help="Folder with audio files")
    ap.add_argument("--pdfs", type=Path, default=None, help="Folder with matching chord PDFs")
    ap.add_argument("--report", type=Path, default=None, help="Output CSV path (default: folder/batch_report.csv)")
    args = ap.parse_args()

    settings = get_settings()
    repo = JobRepo(settings.db_path)

    audio_files = sorted(
        p for p in args.folder.iterdir() if p.suffix.lower() in AUDIO_EXTS
    )
    if not audio_files:
        print(f"No audio files found in {args.folder}")
        return

    report_path = args.report or (args.folder / "batch_report.csv")
    rows: list[dict[str, str]] = []
    for audio in audio_files:
        print(f"[{audio.name}] processing...")
        t0 = time.time()
        try:
            job_id, docx_path, aligned = process_one(audio, repo, settings)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {exc!r}")
            rows.append({"audio": audio.name, "job_id": "", "status": f"fail: {exc!r}",
                         "elapsed_s": "", "wer": "", "docx": ""})
            continue
        elapsed = time.time() - t0

        wer_str = ""
        if args.pdfs:
            pdf = args.pdfs / f"{audio.stem}.pdf"
            if pdf.exists():
                ref_text = _read_pdf_text(pdf)
                hyp_text = _hyp_text_from_aligned(aligned)
                ref_tok = _normalize_for_wer(ref_text)
                hyp_tok = _normalize_for_wer(hyp_text)
                wer_str = f"{_wer(ref_tok, hyp_tok):.3f}"
        rows.append({
            "audio": audio.name,
            "job_id": job_id,
            "status": "ok",
            "elapsed_s": f"{elapsed:.1f}",
            "wer": wer_str,
            "docx": str(docx_path),
        })
        print(f"  done in {elapsed:.1f}s  wer={wer_str or 'n/a'}  -> {docx_path.name}")

    with report_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["audio", "job_id", "status", "elapsed_s", "wer", "docx"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()

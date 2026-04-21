"""Evaluate the pipeline against the regression fixtures in
tests/regression/fixtures/.

For each song:
- Run the full pipeline.
- Compute WER against lyrics.txt.
- Compute chord-multiset accuracy against chords.txt.
- Compare to tests/regression/baseline.json; fail (non-zero exit) if WER goes
  up by > wer_tolerance or chord-accuracy drops by > chord_tolerance.

Usage:
    python scripts/eval_regression.py
    python scripts/eval_regression.py --update-baseline
    python scripts/eval_regression.py --only=<slug>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import get_settings
from src.jobs import JobRepo
from src.pipeline import run as run_pipeline


FIXTURES_DIR = ROOT / "tests" / "regression" / "fixtures"
BASELINE_PATH = ROOT / "tests" / "regression" / "baseline.json"
AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".flac", ".ogg")

WER_TOLERANCE = 0.03
CHORD_TOLERANCE = 0.05


def _fold(text: str) -> list[str]:
    t = unicodedata.normalize("NFD", text or "")
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = t.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return [w for w in t.split() if w]


def _wer(ref: list[str], hyp: list[str]) -> float:
    if not ref:
        return 1.0 if hyp else 0.0
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m] / n


def _chord_accuracy(ref_chords: list[str], hyp_chords: list[str]) -> float:
    if not ref_chords:
        return 1.0 if not hyp_chords else 0.0
    ref_counter = Counter(c.strip() for c in ref_chords if c.strip())
    hyp_counter = Counter(c.strip() for c in hyp_chords if c.strip())
    overlap = sum((ref_counter & hyp_counter).values())
    return overlap / sum(ref_counter.values())


def _find_audio(fix_dir: Path) -> Path | None:
    for ext in AUDIO_EXTS:
        cand = fix_dir / f"audio{ext}"
        if cand.exists():
            return cand
        # Any file with a known audio extension.
        for c in fix_dir.iterdir():
            if c.suffix.lower() == ext:
                return c
    return None


def _hyp_lyrics_from_aligned(aligned: dict) -> str:
    return " ".join(
        (line.get("lyric_line") or "") for line in (aligned.get("lines") or [])
    )


def _hyp_chords_from_aligned(aligned: dict) -> list[str]:
    chords: list[str] = []
    for line in aligned.get("lines") or []:
        chord_line = (line.get("chord_line") or "").split()
        chords.extend(chord_line)
    return chords


def run_one(slug: str, audio: Path, settings, repo: JobRepo) -> dict:
    job = repo.create(audio.name)
    repo.advance(job.id, "queued", progress=0.0)
    job = repo.get(job.id)  # type: ignore[assignment]
    dest = settings.input_dir / audio.name
    if not dest.exists():
        dest.write_bytes(audio.read_bytes())
    t0 = time.time()
    run_pipeline(job, repo, settings)
    elapsed = time.time() - t0
    job_tmp = settings.tmp_dir / job.id
    aligned = json.loads((job_tmp / "aligned.json").read_text(encoding="utf-8"))

    fix_dir = audio.parent
    ref_lyrics = (fix_dir / "lyrics.txt").read_text(encoding="utf-8")
    ref_chords = [c.strip() for c in (fix_dir / "chords.txt").read_text(encoding="utf-8").splitlines() if c.strip()]

    hyp_lyrics = _hyp_lyrics_from_aligned(aligned)
    hyp_chords = _hyp_chords_from_aligned(aligned)

    wer = _wer(_fold(ref_lyrics), _fold(hyp_lyrics))
    chord_acc = _chord_accuracy(ref_chords, hyp_chords)
    return {
        "slug": slug,
        "job_id": job.id,
        "elapsed_s": round(elapsed, 1),
        "wer": round(wer, 4),
        "chord_accuracy": round(chord_acc, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--only", type=str, default=None, help="Run only this slug")
    args = ap.parse_args()

    if not FIXTURES_DIR.exists():
        print(f"No fixtures directory at {FIXTURES_DIR} — see tests/regression/README.md")
        return

    settings = get_settings()
    repo = JobRepo(settings.db_path)

    results: list[dict] = []
    for fix_dir in sorted(FIXTURES_DIR.iterdir()):
        if not fix_dir.is_dir():
            continue
        if args.only and fix_dir.name != args.only:
            continue
        audio = _find_audio(fix_dir)
        if not audio:
            print(f"[skip] {fix_dir.name}: no audio")
            continue
        if not (fix_dir / "lyrics.txt").exists() or not (fix_dir / "chords.txt").exists():
            print(f"[skip] {fix_dir.name}: missing lyrics.txt or chords.txt")
            continue
        print(f"[run ] {fix_dir.name}...")
        results.append(run_one(fix_dir.name, audio, settings, repo))

    if not results:
        print("No fixtures processed.")
        return

    print("\nResults:")
    print(f"{'slug':<30} {'wer':>8} {'chord_acc':>10} {'elapsed':>9}")
    for r in results:
        print(f"{r['slug']:<30} {r['wer']:>8.3f} {r['chord_accuracy']:>10.3f} {r['elapsed_s']:>7.1f}s")

    baseline = {}
    if BASELINE_PATH.exists():
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    if args.update_baseline:
        new_baseline = {r["slug"]: {"wer": r["wer"], "chord_accuracy": r["chord_accuracy"]} for r in results}
        BASELINE_PATH.write_text(json.dumps(new_baseline, indent=2), encoding="utf-8")
        print(f"\nBaseline updated: {BASELINE_PATH}")
        return

    regressed = False
    for r in results:
        base = baseline.get(r["slug"])
        if not base:
            print(f"[warn] {r['slug']}: no baseline entry — run with --update-baseline once this song is acceptable")
            continue
        if r["wer"] > base["wer"] + WER_TOLERANCE:
            print(f"[FAIL] {r['slug']}: WER {r['wer']:.3f} > baseline {base['wer']:.3f} + {WER_TOLERANCE}")
            regressed = True
        if r["chord_accuracy"] < base["chord_accuracy"] - CHORD_TOLERANCE:
            print(f"[FAIL] {r['slug']}: chord_acc {r['chord_accuracy']:.3f} < baseline {base['chord_accuracy']:.3f} - {CHORD_TOLERANCE}")
            regressed = True

    if regressed:
        sys.exit(1)
    print("\nAll fixtures within tolerance.")


if __name__ == "__main__":
    main()

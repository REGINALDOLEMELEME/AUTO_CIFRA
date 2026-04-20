"""S-3: Fraction of chord changes that land within ±150 ms of the correct word.

Ground truth: <name>.alignment.json with schema:
{"chord_events": [{"t": float, "word_start": float}, ...]}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/regression")
    ap.add_argument("--tolerance-ms", type=int, default=150)
    args = ap.parse_args()

    tol = args.tolerance_ms / 1000.0
    root = Path(args.dir)
    scores: list[tuple[str, float]] = []
    for gt_path in sorted(root.glob("*.alignment.json")):
        name = gt_path.name.removesuffix(".alignment.json")
        aligned_path = root / f"{name}.aligned.json"
        if not aligned_path.exists():
            print(f"[skip] {name}: no aligned.json")
            continue
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        pred = json.loads(aligned_path.read_text(encoding="utf-8"))
        predicted_words = [
            w
            for line in pred.get("lines", [])
            for w in line.get("words", [])
            if w.get("chord")
        ]
        if not predicted_words:
            scores.append((name, 0.0))
            continue

        hits = 0
        total = 0
        for event in gt.get("chord_events", []):
            total += 1
            t = float(event.get("t", 0.0))
            word_start = float(event.get("word_start", t))
            closest = min(
                predicted_words,
                key=lambda w: abs(float(w.get("start", 0.0)) - t),
                default=None,
            )
            if closest and abs(float(closest.get("start", 0.0)) - word_start) <= tol:
                hits += 1
        score = hits / total if total else 0.0
        scores.append((name, score))
        print(f"{name:<30} accuracy = {score:.3f}  ({hits}/{total})")

    if not scores:
        print("No regression alignment files found.")
        return 1
    avg = sum(s for _, s in scores) / len(scores)
    print(f"\nAverage = {avg:.3f} over {len(scores)} tracks (target ≥ 0.95)")
    return 0 if avg >= 0.95 else 2


if __name__ == "__main__":
    raise SystemExit(main())

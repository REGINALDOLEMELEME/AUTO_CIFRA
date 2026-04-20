"""S-2: Chord F1 (major/minor triads, beat-resolution) on the regression set.

Ground truth format: <name>.chords.lab — each line `start end chord` (MIREX .lab).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.transpose import shift_chord


def _parse_lab(path: Path) -> list[tuple[float, float, str]]:
    out: list[tuple[float, float, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        try:
            start = float(parts[0])
            end = float(parts[1])
        except ValueError:
            continue
        chord = " ".join(parts[2:]).strip()
        out.append((start, end, chord))
    return out


def _load_predicted(path: Path) -> list[tuple[float, float, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[tuple[float, float, str]] = []
    for s in data.get("segments", []):
        out.append((float(s["start"]), float(s["end"]), str(s["chord"])))
    return out


def _canonical_triad(label: str) -> str:
    if not label or label in {"N", "X"}:
        return "N"
    if len(label) >= 2 and label[1] in {"#", "b"}:
        root = label[:2]
        rest = label[2:]
    else:
        root = label[:1]
        rest = label[1:]
    # Use enharmonic-normalized sharp root for comparison stability.
    root = shift_chord(root, 0, prefer_flats=False)
    if rest.startswith("m") and not rest.startswith("maj"):
        return f"{root}m"
    return root


def _sampled_labels(
    segments: list[tuple[float, float, str]], duration: float, step: float = 0.5
) -> list[str]:
    labels: list[str] = []
    t = 0.0
    idx = 0
    while t < duration:
        while idx + 1 < len(segments) and segments[idx][1] <= t:
            idx += 1
        s, e, c = segments[idx] if segments else (0.0, 0.0, "N")
        labels.append(_canonical_triad(c) if s <= t < e else "N")
        t += step
    return labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/regression")
    ap.add_argument("--step", type=float, default=0.5)
    args = ap.parse_args()

    root = Path(args.dir)
    scores: list[tuple[str, float]] = []
    for lab in sorted(root.glob("*.chords.lab")):
        name = lab.name.removesuffix(".chords.lab")
        pred = root / f"{name}.chords.json"
        if not pred.exists():
            print(f"[skip] {name}: no {pred.name}")
            continue
        gt = _parse_lab(lab)
        pr = _load_predicted(pred)
        duration = max(gt[-1][1] if gt else 0.0, pr[-1][1] if pr else 0.0)
        gt_labels = _sampled_labels(gt, duration, args.step)
        pr_labels = _sampled_labels(pr, duration, args.step)
        tp = sum(1 for g, p in zip(gt_labels, pr_labels) if g == p and g != "N")
        fp = sum(1 for g, p in zip(gt_labels, pr_labels) if g == "N" and p != "N")
        fn = sum(1 for g, p in zip(gt_labels, pr_labels) if g != "N" and p == "N")
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append((name, f1))
        print(f"{name:<30} F1 = {f1:.3f}  (P={precision:.3f} R={recall:.3f})")

    if not scores:
        print("No samples evaluated.")
        return 1
    avg = sum(s for _, s in scores) / len(scores)
    print(f"\nAverage F1 = {avg:.3f} over {len(scores)} tracks (target ≥ 0.75)")
    return 0 if avg >= 0.75 else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""S-1: Word Error Rate on the regression set.

Expects the AUTO_CIFRA/data/regression/ directory laid out as:
  <name>.mp3              input audio
  <name>.lyrics.txt       ground-truth lyrics (utf-8, one line per stanza line)
  <name>.aligned.json     predicted aligned JSON from a finished pipeline run
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path


def _tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text.lower())
    return text.split()


def wer(ref: list[str], hyp: list[str]) -> float:
    m, n = len(ref), len(hyp)
    if m == 0:
        return 1.0 if n else 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n] / m


def _flatten_lyrics(aligned: dict) -> str:
    return " ".join(line.get("lyric_line") or "" for line in aligned.get("lines", []))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/regression", help="Regression directory")
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.exists():
        print(f"No regression directory: {root}")
        return 1

    totals: list[tuple[str, float]] = []
    for txt in sorted(root.glob("*.lyrics.txt")):
        name = txt.name.removesuffix(".lyrics.txt")
        aligned = root / f"{name}.aligned.json"
        if not aligned.exists():
            print(f"[skip] {name}: no {aligned.name}")
            continue
        ref = _tokenize(txt.read_text(encoding="utf-8"))
        hyp = _tokenize(_flatten_lyrics(json.loads(aligned.read_text(encoding="utf-8"))))
        score = wer(ref, hyp)
        totals.append((name, score))
        print(f"{name:<30} WER = {score:.3f}")

    if not totals:
        print("No samples evaluated.")
        return 1
    avg = sum(s for _, s in totals) / len(totals)
    print(f"\nAverage WER = {avg:.3f} over {len(totals)} tracks (target ≤ 0.06)")
    return 0 if avg <= 0.06 else 2


if __name__ == "__main__":
    raise SystemExit(main())

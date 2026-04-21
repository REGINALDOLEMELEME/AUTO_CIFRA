"""End-to-end evaluation of AUTO_CIFRA output for "Será" (Legião Urbana)
against the Cifra Club ground truth.

Produces a go/no-go readiness report measuring:
  - Chord sequence accuracy (strict + root-only)
  - Lyric WER
  - Section label accuracy
  - Key detection
  - Edit-budget (single-cell edits to match reference)

Usage:
  python scripts/eval_sera.py \
    --ground-truth data/regression/sera_legiao_urbana/ground_truth.json \
    --lyrics data/regression/sera_legiao_urbana/lyrics.txt \
    --aligned data/tmp/<job_id>/aligned.json \
    --chords-json data/tmp/<job_id>/chords.json \
    --beats-json data/tmp/<job_id>/beats.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path


def _strip_slash(chord: str) -> str:
    return chord.split("/", 1)[0] if chord else chord


def _normalize_for_compare(chord: str) -> str:
    """Canonical form used for comparison: root + quality.
    Strips bass notes (/X), unifies enharmonics to sharp.
    """
    if not chord or chord in {"N", "X"}:
        return ""
    chord = _strip_slash(chord)
    enh = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
    if len(chord) >= 2 and chord[:2] in enh:
        chord = enh[chord[:2]] + chord[2:]
    chord = re.sub(r"(maj|min|m|dim|aug|sus[24]?|\d+|b\d+)?$", lambda m: m.group(0) or "", chord)
    return chord


def flatten_ground_truth(gt: dict) -> list[str]:
    seq: list[str] = []
    for section in gt.get("sections", []):
        for ch in section.get("chord_sequence", []):
            if not seq or seq[-1] != ch:
                seq.append(ch)
    return seq


def flatten_aligned_chords(aligned: dict) -> list[str]:
    """Flatten chords from aligned.json in order, collapsing consecutive duplicates.

    aligned.json stores the same chord info twice (words[].chord and line.chords);
    we use words[].chord and fall back to line.chords only when the line has no
    word-level chord annotations.
    """
    seq: list[str] = []
    for line in aligned.get("lines", []):
        word_chords = [w.get("chord") for w in line.get("words", []) if w.get("chord")]
        if word_chords:
            for ch in word_chords:
                if not seq or seq[-1] != ch:
                    seq.append(ch)
        else:
            for ch in line.get("chords") or []:
                label = ch.get("chord") if isinstance(ch, dict) else ch
                if label and (not seq or seq[-1] != label):
                    seq.append(label)
    return seq


def flatten_chords_json(chords: dict | list) -> list[str]:
    segs = chords.get("segments") if isinstance(chords, dict) else chords
    seq: list[str] = []
    for s in segs or []:
        ch = s.get("chord") if isinstance(s, dict) else s
        if ch and (not seq or seq[-1] != ch):
            seq.append(ch)
    return seq


def levenshtein(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


def _tokenize_lyrics(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text.lower())
    return text.split()


def wer(ref: list[str], hyp: list[str]) -> tuple[float, int]:
    m, n = len(ref), len(hyp)
    if m == 0:
        return (1.0 if n else 0.0, n)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return (dp[m][n] / m, dp[m][n])


def section_match(gt: dict, aligned: dict) -> tuple[int, int, list[str], list[str]]:
    gt_sections = [s["name"] for s in gt.get("sections", []) if s["name"] not in ("Instrumental 1", "Instrumental 2", "Instrumental 3")]
    pred_sections: list[str] = []
    for line in aligned.get("lines", []):
        name = line.get("section") or ""
        if not pred_sections or pred_sections[-1] != name:
            pred_sections.append(name)

    def _norm(s: str) -> str:
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()
        return re.sub(r"[^a-z0-9]+", "", s)

    gt_norm = [_norm(s) for s in gt_sections]
    pred_norm = [_norm(s) for s in pred_sections]
    matches = sum(1 for s in set(gt_norm) if s and s in set(pred_norm))
    return (matches, len(set(gt_norm) - {""}), gt_sections, pred_sections)


def sequence_accuracy(ref: list[str], hyp: list[str], strip_slash: bool = False) -> dict:
    if strip_slash:
        ref = [_strip_slash(c) for c in ref]
        hyp = [_strip_slash(c) for c in hyp]
    dist = levenshtein(ref, hyp)
    acc = 1.0 - (dist / max(len(ref), 1))
    return {
        "ref_len": len(ref),
        "hyp_len": len(hyp),
        "edit_distance": dist,
        "accuracy": max(acc, 0.0),
    }


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground-truth", required=True, type=Path)
    ap.add_argument("--lyrics", required=True, type=Path)
    ap.add_argument("--aligned", required=True, type=Path)
    ap.add_argument("--chords-json", type=Path, default=None)
    ap.add_argument("--beats-json", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="Write metrics JSON to this path")
    args = ap.parse_args()

    gt = load_json(args.ground_truth)
    aligned = load_json(args.aligned)

    ref_seq = flatten_ground_truth(gt)
    hyp_seq_from_aligned = flatten_aligned_chords(aligned)
    hyp_seq_from_chords = flatten_chords_json(load_json(args.chords_json)) if args.chords_json and args.chords_json.exists() else []

    strict = sequence_accuracy(ref_seq, hyp_seq_from_aligned, strip_slash=False)
    root_only = sequence_accuracy(ref_seq, hyp_seq_from_aligned, strip_slash=True)
    strict_raw = sequence_accuracy(ref_seq, hyp_seq_from_chords, strip_slash=False) if hyp_seq_from_chords else None
    root_raw = sequence_accuracy(ref_seq, hyp_seq_from_chords, strip_slash=True) if hyp_seq_from_chords else None

    ref_tokens = _tokenize_lyrics(args.lyrics.read_text(encoding="utf-8"))
    hyp_lyrics = " ".join(line.get("lyric_line") or "" for line in aligned.get("lines", []))
    hyp_tokens = _tokenize_lyrics(hyp_lyrics)
    wer_val, wer_edits = wer(ref_tokens, hyp_tokens)

    sec_hit, sec_total, gt_sections, pred_sections = section_match(gt, aligned)

    gt_key = gt.get("song", {}).get("key")
    pred_key = ""
    if args.beats_json and args.beats_json.exists():
        pred_key = (load_json(args.beats_json) or {}).get("key", "")
    key_match = (pred_key.upper().strip() == (gt_key or "").upper().strip()) if gt_key else False

    edit_budget = strict["edit_distance"] + wer_edits

    report = {
        "song": gt.get("song", {}),
        "chord_mode": aligned.get("chord_mode"),
        "transcription_mode": aligned.get("transcription_mode"),
        "warnings": aligned.get("warnings", []),
        "reference_sequence_len": len(ref_seq),
        "hypothesis_sequence_len_aligned": len(hyp_seq_from_aligned),
        "hypothesis_sequence_len_raw": len(hyp_seq_from_chords),
        "chord_strict": strict,
        "chord_root_only": root_only,
        "chord_strict_raw": strict_raw,
        "chord_root_only_raw": root_raw,
        "lyric_wer": {
            "wer": wer_val,
            "ref_tokens": len(ref_tokens),
            "hyp_tokens": len(hyp_tokens),
            "edits": wer_edits,
        },
        "sections": {
            "matched": sec_hit,
            "expected_total": sec_total,
            "ground_truth_order": gt_sections,
            "predicted_order": pred_sections,
        },
        "key": {"expected": gt_key, "predicted": pred_key, "match": key_match},
        "edit_budget_cells": edit_budget,
        "reference_chord_sequence": ref_seq,
        "hypothesis_chord_sequence_aligned": hyp_seq_from_aligned,
        "hypothesis_chord_sequence_raw": hyp_seq_from_chords,
    }

    print("=" * 70)
    print(f" AUTO_CIFRA regression — {gt['song']['title']} / {gt['song']['artist']}")
    print("=" * 70)
    print(f" chord_mode           : {report['chord_mode']}")
    print(f" transcription_mode   : {report['transcription_mode']}")
    print(f" warnings             : {report['warnings']}")
    print()
    print(f" Reference chord seq  : {len(ref_seq):>4} symbols")
    print(f" Aligned chord seq    : {len(hyp_seq_from_aligned):>4} symbols")
    print(f" Raw chord seq        : {len(hyp_seq_from_chords):>4} symbols")
    print()
    print(" Chord accuracy (aligned, strict)    : "
          f"{strict['accuracy']*100:5.1f}%  edits={strict['edit_distance']}")
    print(" Chord accuracy (aligned, root-only) : "
          f"{root_only['accuracy']*100:5.1f}%  edits={root_only['edit_distance']}")
    if strict_raw is not None:
        print(" Chord accuracy (raw,     strict)    : "
              f"{strict_raw['accuracy']*100:5.1f}%  edits={strict_raw['edit_distance']}")
        print(" Chord accuracy (raw,     root-only) : "
              f"{root_raw['accuracy']*100:5.1f}%  edits={root_raw['edit_distance']}")
    print()
    print(f" Lyric WER            : {wer_val*100:5.2f}%  ({wer_edits} edits, ref={len(ref_tokens)} tokens, hyp={len(hyp_tokens)} tokens)")
    print(f" Sections matched     : {sec_hit}/{sec_total}   expected={gt_sections}")
    print(f"                         predicted={pred_sections}")
    print(f" Key                  : expected={gt_key!r}  predicted={pred_key!r}  match={key_match}")
    print()
    print(f" Total edit-budget    : {edit_budget} cells (chord edits + lyric edits)")
    print()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f" Metrics JSON written to: {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

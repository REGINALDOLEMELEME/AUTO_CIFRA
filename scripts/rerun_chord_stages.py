"""Re-run only the post-chord stages of an AUTO_CIFRA job.

Skips Demucs + Whisper + wav2vec2 alignment. Reads the existing raw Chordino
CSV, applies the current normalizer (honouring config settings such as
simplify_to_triads), re-smooths with the stored beats, and re-aligns chords
against the existing aligned.json's word timings. Writes the result back to
aligned.json so eval_sera.py can re-score without a full pipeline run.

Usage:
    python scripts/rerun_chord_stages.py --job <job_id>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.alignment import align_chords_by_word_time
from src.chords import (
    _parse_chord_csv,
    normalize_chord_vocabulary,
    refine_chords_to_key,
    smooth_to_beats,
)
from src.config import get_settings


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _reconstruct_aligned_asr(existing_aligned: dict) -> dict:
    """Rebuild the flat transcription structure that align_chords_by_word_time
    consumes, from a previously written aligned.json."""
    segments = []
    for line in existing_aligned.get("lines", []):
        words = [
            {
                "word": w.get("text") or w.get("word") or "",
                "start": float(w.get("start", 0.0)),
                "end": float(w.get("end", 0.0)),
            }
            for w in line.get("words", [])
        ]
        if not words:
            continue
        segments.append(
            {
                "text": line.get("lyric_line") or "",
                "start": float(line.get("start", words[0]["start"])),
                "end": float(line.get("end", words[-1]["end"])),
                "words": words,
            }
        )
    return {"segments": segments, "language": "pt"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    args = ap.parse_args()

    settings = get_settings()
    job_dir = settings.tmp_dir / args.job
    if not job_dir.exists():
        print(f"job dir not found: {job_dir}", file=sys.stderr)
        return 1

    csvs = list(job_dir.glob("*chordino_simplechord.csv"))
    if not csvs:
        print(f"no Chordino CSV in {job_dir}", file=sys.stderr)
        return 1
    raw_segments = _parse_chord_csv(csvs[0])
    raw_chords = {"mode": "real", "warning": "", "segments": raw_segments}
    print(f"  raw segments (re-parsed): {len(raw_segments)}")

    normalized = normalize_chord_vocabulary(
        raw_chords, simplify_to_triads=settings.chords.simplify_to_triads
    )
    print(f"  after normalize (simplify_to_triads={settings.chords.simplify_to_triads}): "
          f"{len(normalized['segments'])}")

    beats = _load_json(job_dir / "beats.json")
    if settings.chords.refine_to_key and beats.get("key"):
        normalized = refine_chords_to_key(normalized, beats["key"])
        print(f"  after refine_to_key ({beats['key']}): {len(normalized['segments'])}")
    smoothed = smooth_to_beats(normalized, beats)
    print(f"  after smooth_to_beats: {len(smoothed['segments'])}")
    (job_dir / "chords.json").write_text(
        json.dumps(smoothed, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    existing_aligned = _load_json(job_dir / "aligned.json")
    aligned_asr = _reconstruct_aligned_asr(existing_aligned)
    sections_payload = _load_json(job_dir / "sections.json")
    sections = sections_payload.get("sections", [])

    new_aligned = align_chords_by_word_time(
        transcription=aligned_asr,
        chords=smoothed,
        sections=sections,
    )
    new_aligned.setdefault("warnings", existing_aligned.get("warnings", []))
    new_aligned.setdefault("source_file", existing_aligned.get("source_file", ""))
    new_aligned.setdefault("transcription_mode", existing_aligned.get("transcription_mode", "real"))
    new_aligned.setdefault("chord_mode", smoothed.get("mode", "real"))

    (job_dir / "aligned.json").write_text(
        json.dumps(new_aligned, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  rewrote {job_dir}/aligned.json with {len(new_aligned.get('lines', []))} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

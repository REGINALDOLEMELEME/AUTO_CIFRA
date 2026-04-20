from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .config import get_settings
from .jobs import JobRepo
from .pipeline import run as run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AUTO_CIFRA end-to-end pipeline (CLI). "
        "Runs separation -> ASR -> alignment -> chords -> structure -> DOCX."
    )
    parser.add_argument("--input", required=True, help="Path to input audio file.")
    parser.add_argument("--language", default="", help="Override language code.")
    parser.add_argument("--model-size", default="", help="Override ASR model size.")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    input_audio = Path(args.input)
    if not input_audio.exists():
        print(f"Input file not found: {input_audio}", file=sys.stderr)
        return 1

    settings = get_settings()
    if args.language:
        settings.asr.language = args.language
        settings.app.language = args.language
    if args.model_size:
        settings.asr.model = args.model_size

    dest = settings.input_dir / input_audio.name
    if dest.resolve() != input_audio.resolve():
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(input_audio.read_bytes())

    repo = JobRepo(settings.db_path)
    try:
        job = repo.create(input_audio.name)
        repo.advance(job.id, "queued")
        job = repo.get(job.id)
        if job is None:
            print("failed to create job row", file=sys.stderr)
            return 1
        aligned_path = run_pipeline(job=job, repo=repo, settings=settings)
        print(f"aligned JSON: {aligned_path}")
        return 0
    finally:
        repo.close()


if __name__ == "__main__":
    raise SystemExit(main())

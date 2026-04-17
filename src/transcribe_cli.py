from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .paths import ensure_directories
from .settings import load_settings
from .transcription import transcribe_audio, write_transcription_json


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe audio to lyric segments with timestamps."
    )
    parser.add_argument("--input", required=True, help="Path to input audio file.")
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional output JSON path. Defaults to data/tmp/<name>.transcription.json",
    )
    parser.add_argument(
        "--language",
        default="",
        help="Language code. Defaults to value from config.",
    )
    parser.add_argument(
        "--model-size",
        default="small",
        help="Whisper model size (tiny, base, small, medium, large-v3).",
    )
    parser.add_argument(
        "--use-vad",
        action="store_true",
        help="Enable VAD filter (usually disable for sung music).",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    settings = load_settings()
    paths = settings["paths"]
    ensure_directories(paths)

    input_audio = Path(args.input)
    if not input_audio.exists():
        print(f"Input file not found: {input_audio}", file=sys.stderr)
        return 1

    tmp_dir = Path(paths["tmp_dir"])
    language = args.language or settings["app"]["language"]

    output_json = (
        Path(args.output_json)
        if args.output_json
        else tmp_dir / f"{input_audio.stem}.transcription.json"
    )

    result = transcribe_audio(
        input_audio=input_audio,
        tmp_dir=tmp_dir,
        language=language,
        model_size=args.model_size,
        use_vad=args.use_vad,
    )
    write_transcription_json(result, output_json)
    print(f"Transcription saved to: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

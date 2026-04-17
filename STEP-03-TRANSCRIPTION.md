# STEP 3 - Lyrics Transcription Status

## Completed
- Added audio normalization helper using ffmpeg (`src/audio.py`).
- Added transcription pipeline with faster-whisper (`src/transcription.py`).
- Added CLI entrypoint for transcription (`src/transcribe_cli.py`).
- Updated README with Step 3 run instructions.

## Output Contract
Default output JSON: `data/tmp/<audio-name>.transcription.json`

JSON includes:
- source_file
- normalized_audio
- language
- language_probability
- duration
- segments[] with {start, end, text}

## Validation
- Syntax validation passed: `python -m compileall src app`

## Notes
- Runtime transcription requires dependencies installed from `requirements.txt`.
- Step 4 still depends on installing sonic-annotator + chordino.

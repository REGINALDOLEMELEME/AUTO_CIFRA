# AUTO_CIFRA - Step 1 Scope

## Goal
Upload a song audio file and automatically generate a Word document with lyrics and chord symbols (C, Dm, G, etc.) aligned to lyrics.

## Inputs
- Audio file: .mp3, .wav, .m4a, .flac
- Optional metadata: song title, artist, language

## Outputs
- .docx file with:
  - Title
  - Chord line above each lyric line
  - Lyric line text
  - Optional timestamps (debug mode)

## Functional Requirements
1. User uploads one audio file.
2. System transcribes lyrics from audio.
3. System extracts chord sequence from audio.
4. System aligns chords to lyric lines by timestamps.
5. System exports a Word document.

## Non-Functional Requirements
- 100% free software stack.
- Runs locally on Windows.
- No paid APIs.
- Basic error handling and logs.

## Proposed Free Stack
- Python 3.11+
- ffmpeg (audio conversion)
- faster-whisper (speech-to-text)
- demucs (optional source separation)
- sonic-annotator + chordino (chord detection)
- python-docx (Word generation)
- streamlit (web UI)

## Pipeline Summary
1. Normalize audio with ffmpeg.
2. (Optional) Separate vocals/instrumental with demucs.
3. Transcribe vocals with faster-whisper.
4. Detect chords on instrumental/full mix.
5. Align chords to lyric lines.
6. Generate .docx output.

## Acceptance Criteria (Step 1)
- Scope documented.
- Toolchain chosen (free-only).
- Next implementation step clearly defined.

## Approval Gate
Before Step 2, user approves this scope and confirms:
- Main lyric language for first version (PT-BR or EN)
- If source separation (demucs) should be ON by default
- Desired output style: compact or spaced

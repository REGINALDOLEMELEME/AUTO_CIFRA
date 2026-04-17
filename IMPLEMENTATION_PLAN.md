# AUTO_CIFRA - Implementation Plan With Approval Gates

## Step 1 - Scope and Constraints
- Define requirements and free-only stack.
- Output: STEP-01-SCOPE.md
- Gate: User approval required.

## Step 2 - Environment Setup
- Create Python project structure.
- Add requirements.txt and install script instructions.
- Validate ffmpeg and Python availability.
- Gate: User approval required.

## Step 3 - Lyrics Transcription Module
- Implement audio preprocessing.
- Implement Whisper transcription with timestamps.
- Save intermediate JSON.
- Gate: User approval required.

## Step 4 - Chord Detection Module
- Integrate sonic-annotator/chordino workflow.
- Parse chord timeline output.
- Save intermediate chord JSON.
- Gate: User approval required.

## Step 5 - Alignment Module
- Align chord timestamps to lyric lines.
- Produce chord+lyric structured representation.
- Gate: User approval required.

## Step 6 - DOCX Export Module
- Generate Word document with chord-over-lyric layout.
- Include title/artist metadata.
- Gate: User approval required.

## Step 7 - Streamlit UI
- Implement upload, process, and download flow.
- Add processing progress and error messages.
- Gate: User approval required.

## Step 8 - Validation and Tuning
- Run end-to-end on sample songs.
- Tune thresholds for better alignment.
- Document limitations and usage.
- Final gate: User approval required.

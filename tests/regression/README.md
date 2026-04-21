# Regression harness

Ground-truth-backed pipeline evaluation. Use this to catch quality regressions
on real songs before they ship.

## Fixture layout

```
tests/regression/
├── README.md                          (this file)
├── fixtures/
│   └── <slug>/
│       ├── audio.mp3                  or .wav / .m4a / .flac
│       ├── lyrics.txt                 ground-truth lyrics (plain text, UTF-8)
│       └── chords.txt                 ground-truth chord progression
│                                      (one chord per line, in order)
└── baseline.json                      auto-written by eval_regression.py
```

`<slug>` is anything — song title slug, ID, whatever. One folder per song.

## How to add a song

1. Drop the audio in `tests/regression/fixtures/<slug>/audio.<ext>`.
2. Copy the lyrics into `lyrics.txt`. Strip the chord-line metadata — keep
   only what is actually sung, one lyric line per actual phrase in the song.
3. Copy the chord progression (as it appears in the chord sheet) into
   `chords.txt`, one chord per line, in the order they appear.

That's it. The evaluator discovers fixtures automatically on next run.

## Running

```powershell
.\.venv\Scripts\python.exe scripts\eval_regression.py
```

Reports per-song **WER** (lyrics) and **chord-accuracy** (multiset overlap of
chord labels). Fails (non-zero exit) if either falls below the floor in
`tests/regression/baseline.json`. Update the baseline manually after a
deliberate improvement:

```powershell
.\.venv\Scripts\python.exe scripts\eval_regression.py --update-baseline
```

## What counts as a regression

- **Lyrics:** WER above the baseline + a small tolerance (default 3 pp).
- **Chords:** accuracy below the baseline − 5 pp.

Tolerances are deliberately asymmetric: lyrics are bounded above, chord
accuracy is bounded below.

## Notes

- This harness deliberately does NOT run in the default pytest invocation —
  each song runs the full pipeline (separation + Whisper + Chordino), which
  takes minutes per song. Invoke it manually or from a nightly job.
- The ground-truth files never contain audio fingerprints or secrets, so
  they can be checked in. Audio files are on the user's local machine only.
- For each new song added, regenerate the baseline with `--update-baseline`
  after confirming the per-song numbers are acceptable.

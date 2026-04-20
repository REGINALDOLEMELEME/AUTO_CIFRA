# Regression Set — AUTO_CIFRA

Benchmarks S-1 (WER), S-2 (chord F1), and S-3 (chord-to-word alignment) run against
the tracks stored here. Ground truth is self-labeled — you own the files, the bar
is self-measured.

## Layout

```
data/regression/
├── <name>.mp3                 # input audio (or .wav/.m4a/.flac/.ogg)
├── <name>.lyrics.txt          # ground-truth lyrics, one stanza line per line
├── <name>.chords.lab          # MIREX-style chord labels (start end chord)
├── <name>.alignment.json      # ground-truth chord-event times keyed to words
└── <name>.aligned.json        # produced by the pipeline (do not edit)
```

## Minimum set

Drop **3 PT-BR studio tracks** you know well. Suggested mix:

- 1 MPB (sparse instrumentation, long vowels)
- 1 sertanejo (fast vocal delivery, dense chord rhythm)
- 1 pop BR (loud mix, doubled vocals)

## Generating predicted JSON

1. `scripts/run_api_server.ps1`
2. Open `http://127.0.0.1:8000/` and upload each track.
3. Run the pipeline and, when `stage=ready_for_review`, copy
   `data/tmp/<job_id>/aligned.json` → `data/regression/<name>.aligned.json`
4. Run the benchmarks:
   - `python scripts/eval_wer.py --dir data/regression`
   - `python scripts/eval_chords.py --dir data/regression`
   - `python scripts/eval_alignment.py --dir data/regression`

## Targets

| Metric | Script | Target |
|---|---|---|
| WER | `eval_wer.py` | ≤ 0.06 |
| Chord F1 | `eval_chords.py` | ≥ 0.75 |
| Chord→word alignment | `eval_alignment.py` | ≥ 0.95 within ±150 ms |

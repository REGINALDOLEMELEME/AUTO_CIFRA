# AUTO_CIFRA

Professional-grade chord-sheet generator for **Brazilian Portuguese** studio recordings. 100% local, 100% free.

Upload an MP3, review the AI's work in your browser, export a polished DOCX with chords anchored above the correct word of every lyric.

---

## Stack — every component is free and runs locally

| Stage | Tool | License |
|---|---|---|
| Audio normalize | `ffmpeg` | LGPL/GPL |
| Vocal isolation | `demucs htdemucs_ft` | MIT |
| Lyrics ASR | `faster-whisper large-v3` (int8 CPU) | MIT |
| Word-level forced alignment | `whisperx` + `wav2vec2 PT-BR` | BSD-4 / Apache-2.0 |
| Chord detection | `sonic-annotator` + Chordino | GPL |
| Beat / BPM / key | `librosa` | ISC |
| Section labeling | librosa agglomerative + heuristics | ISC |
| Review UI | FastAPI + Jinja2 + HTMX | MIT / BSD / 0BSD |
| DOCX export | `python-docx` | MIT |
| Job queue | SQLite (stdlib) | public domain |

No API key, no cloud call, no paid service.

---

## Install

### 1. Python 3.11

```powershell
winget install -e --id Python.Python.3.11 --scope user --silent
```

Verify: `py -3.11 --version` → `Python 3.11.9` (or later 3.11.x).

### 2. Project env

```powershell
cd AUTO_CIFRA
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade "pip<26" "setuptools<81" wheel
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchaudio==2.5.1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Native tools

Install these and make sure they are on PATH:

- **ffmpeg** — https://www.gyan.dev/ffmpeg/builds/ (release build). Required.
- **sonic-annotator** + **NNLS-Chroma VAMP plugin (Chordino)** — optional. If absent, AUTO_CIFRA falls back to a pure-Python chord detector (lower F1 but functional). Drop `sonic-annotator.exe` into `tools/sonic-annotator/sonic-annotator-win64/` and the `.cat`/`.n3`/`.dll` plugin files into `tools/vamp-plugins/`.

### 4. Pre-download models (one-time, ~6 GB)

```powershell
.\.venv\Scripts\python.exe scripts/prefetch_models.py
```

After this step the pipeline runs fully offline (S-7).

---

## Run

```powershell
.\scripts\run_api_server.ps1
```

This runs uvicorn with the `--factory` flag (`app.api:create_app`) so settings are re-read on every cold boot.

Open http://127.0.0.1:8000/ in your browser.

Flow:

1. **Upload** — drag-drop an MP3. You get a `job_id`.
2. **Process** — kicks off the pipeline. Status polls automatically.
3. **Review** — when status = `ready_for_review`, open `/review/<job_id>`. Edit any word, change or remove any chord, set transpose or capo.
4. **Export DOCX** — downloads a polished chord sheet with chords above words in a borderless 2-row table (renders correctly in Word and LibreOffice, any font).

---

## CLI (power users)

```powershell
.\.venv\Scripts\python.exe -m src.transcribe_cli --input path\to\song.mp3
```

Writes the aligned JSON and returns its path.

---

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest tests/
```

Benchmarks on your regression set (see `data/regression/README.md`):

```powershell
.\.venv\Scripts\python.exe scripts/eval_wer.py
.\.venv\Scripts\python.exe scripts/eval_chords.py
.\.venv\Scripts\python.exe scripts/eval_alignment.py
.\.venv\Scripts\python.exe scripts/check_licenses.py
```

Targets: WER ≤ 6%, chord F1 ≥ 0.75, chord-to-word alignment ≥ 95% within ±150 ms.

---

## Graceful degradation

Any of these can be missing — the pipeline warns and proceeds:

| Missing | Effect |
|---|---|
| Demucs | ASR runs on the full mix (lower WER on loud choruses) |
| WhisperX | Word timestamps come from Whisper (±500 ms drift possible) |
| Chordino | Pure-Python chroma-template chord detector kicks in |

You'll see a yellow warning banner on the review page for each fallback.

---

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/upload` | multipart file upload → returns job |
| POST | `/process/{job_id}` | enqueue for pipeline |
| GET | `/jobs/{job_id}` | poll status |
| GET | `/review/{job_id}` | editable HTML page |
| POST | `/save/{job_id}` | persist edits, transpose, capo |
| POST | `/export/{job_id}` | render DOCX, stream file |
| GET | `/history` | recent jobs |
| GET | `/health` | liveness |

---

## Project layout

```
AUTO_CIFRA/
├── app/api/             FastAPI app, routes, schemas
├── src/                 pipeline modules, job DB, config
├── frontend/            Jinja2 templates, static assets
├── scripts/             setup, runner, evaluation
├── tests/               pytest suite (69 tests)
├── config/settings.yaml tunables
├── data/                input / tmp / output + regression set
├── models/              cached ML models (~6 GB once downloaded)
└── requirements.txt
```

# AUTO_CIFRA — Quick Start

Reopen the project, run the pipeline, upload a song, review, export. No code changes needed.

Project root: `C:\Users\Meu Computador\Downloads\PROJECT_CLAUDE\projeto_analysis_ia\AUTO_CIFRA`

---

## 1. Open a PowerShell at the project root

```powershell
cd "C:\Users\Meu Computador\Downloads\PROJECT_CLAUDE\projeto_analysis_ia\AUTO_CIFRA"
```

## 2. Check the basics (5 seconds)

```powershell
.\.venv\Scripts\python.exe --version     # should print: Python 3.11.9
where ffmpeg                             # should print: C:\ffmpeg\bin\ffmpeg.exe
```

If `ffmpeg` is missing, the pipeline will fail in the **Transcrição** phase. Re-add it to PATH or reinstall it.

## 3. Start the API server

```powershell
.\scripts\run_api_server.ps1
```

Expected output (keep the terminal open):

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

Leave that window running. Close it (or press `Ctrl+C`) when you're done for the day.

## 4. Open the dashboard

Open in your browser:

**http://127.0.0.1:8000/**

## 5. Run a test

1. Drag-drop your MP3 (or click the dashed zone → pick a file)
2. The pipeline auto-starts. You'll see 6 phases light up:
   - Isolamento de vocais (Demucs) — ~4 min
   - Transcrição (Whisper large-v3) — ~3–5 min
   - Alinhamento (WhisperX) — ~30 s
   - Acordes (Python detector) — ~30 s
   - Estrutura — ~10 s
   - Renderização — instant
3. Each bar smoothly fills 0→~98% during its phase, snaps to 100% when done.
4. When status turns green with **"pronto para revisar"**, click **Revisar**.
5. Edit any wrong word or chord. Set transpose/capo if needed.
6. Click **Exportar DOCX** — the file downloads.

**Want to abort?** Click the red **Cancelar** button (visible anytime the pipeline is running).

## 6. Stop the server

In the PowerShell where uvicorn is running, press **Ctrl+C**.

---

## Run the tests (optional sanity check)

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q
```

Should print `74 passed in ~1.5s`. If any test fails, the pipeline has a broken dependency — fix before running the UI.

## CLI mode (no browser)

```powershell
.\.venv\Scripts\python.exe -m src.transcribe_cli --input "C:\path\to\song.mp3"
```

Writes `data\tmp\<job_id>\aligned.json`. You still need the server to render the review page or export DOCX.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `.venv\Scripts\python.exe` not found | venv is gone — rebuild: `py -3.11 -m venv .venv` then `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` and reinstall torch from the CPU index (see README §2) |
| Port 8000 already in use | Another instance is running. Kill it: `taskkill /F /IM python.exe` (or change `--port` in `scripts\run_api_server.ps1`) |
| "chordino unavailable" warning on review page | Expected — we use the Python chord detector fallback. Install `sonic-annotator` + NNLS-Chroma VAMP plugin into `tools\` to silence the warning and get slightly better chord quality. |
| Review page shows `Internal Server Error` | Check the server terminal for a traceback. Usually a template issue or missing `aligned.json`. |
| Pipeline stuck at "separating" forever | First run downloads models (~5 GB). Check `models\huggingface\` is growing; otherwise kill and run `.\.venv\Scripts\python.exe scripts\prefetch_models.py` once, then retry. |
| Upload works but pipeline never starts | Worker crashed. Restart the server. Any job already in `queued` stage will be picked up automatically. |
| DOCX chord column mis-aligned in Word | Shouldn't happen — the export uses a borderless 2-row table, which is font-independent. If it happens, open the DOCX in LibreOffice to cross-check; file a bug with the job ID. |

## File locations

- **Uploaded audio:** `data\input\*.mp3`
- **Per-job intermediates:** `data\tmp\<job_id>\`
- **Exported DOCX:** `data\output\<job_id>\<name>.docx`
- **Job database:** `data\jobs.sqlite`
- **Model cache (~5 GB):** `models\`
- **Server logs:** only in the terminal window — redirect to a file if you want persistence

## Key URLs

- Dashboard: http://127.0.0.1:8000/
- Health check: http://127.0.0.1:8000/health
- API docs: http://127.0.0.1:8000/docs (FastAPI auto-generated)
- Job status: http://127.0.0.1:8000/jobs/{job_id}
- Review: http://127.0.0.1:8000/review/{job_id}
- History: http://127.0.0.1:8000/history

## Update from Git (when you come back later)

```powershell
cd "C:\Users\Meu Computador\Downloads\PROJECT_CLAUDE\projeto_analysis_ia\AUTO_CIFRA"
git pull
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest tests/ -q
```

If tests pass, start the server and go.

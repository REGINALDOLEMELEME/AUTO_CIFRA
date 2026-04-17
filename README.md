# AUTO_CIFRA

Free local system to upload a song and generate a Word chord sheet.

## Stack (free)
- Python
- ffmpeg
- faster-whisper
- demucs
- sonic-annotator + chordino
- python-docx
- streamlit

## Install
1. `powershell -ExecutionPolicy Bypass -File .\\scripts\\setup.ps1`
2. Install ffmpeg and ensure `ffmpeg` is in PATH.
3. Install sonic-annotator and chordino plugin for real chord detection.

## Web Test Console
File:
- `frontend/upload_test.html`

Run locally:
1. Start backend:
`powershell -ExecutionPolicy Bypass -File .\\scripts\\run_upload_server.ps1`
2. Start static page in another terminal:
`python -m http.server 5500`
3. Open:
`http://localhost:5500/frontend/upload_test.html`
4. Keep endpoint as:
`http://127.0.0.1:8000/upload`
5. Choose file and click `Run Full Pipeline`.

## Step 8 Hardening
- Backend endpoint `GET /history` returns recent `.docx` / `.json` outputs.
- Backend writes structured events to `data/tmp/server.log`.
- Frontend shows:
  - Full response JSON
  - Clickable result links
  - Recent output history list

Health check:
- `http://127.0.0.1:8000/health` should return `{"status":"ok"}`

## API Contracts
- Upload:
  - `POST /upload?filename=<name.ext>`
  - body: raw file bytes
- Process:
  - `POST /process?filename=<name.ext>&language=pt&model_size=small`
- Arrange:
  - `POST /arrange?filename=<name.ext>&language=pt&model_size=small`
- History:
  - `GET /history`

## Notes
- If dependencies are missing, endpoints can return `mode: mock`.
- Outputs are written under `data/tmp` and `data/output`.

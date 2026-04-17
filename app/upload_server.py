from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
from datetime import datetime
from urllib.parse import parse_qs, urlparse

from src.paths import ensure_directories
from src.settings import load_settings
from src.transcription import transcribe_audio, write_transcription_json
from src.chords import detect_chords, write_json
from src.alignment import align_chords_to_lyrics
from src.docx_export import export_transcription_docx, export_aligned_chord_docx


ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac"}

settings = load_settings()
paths = settings["paths"]
ensure_directories(paths)
INPUT_DIR = Path(paths["input_dir"])
TMP_DIR = Path(paths["tmp_dir"])
OUTPUT_DIR = Path(paths["output_dir"])
SERVER_LOG = TMP_DIR / "server.log"


def _mock_transcription(input_audio: Path, language: str, reason: str) -> dict:
    return {
        "source_file": input_audio.name,
        "normalized_audio": "",
        "language": language,
        "language_probability": 1.0,
        "duration": 0.0,
        "mode": "mock",
        "warning": reason,
        "segments": [
            {
                "start": 0.0,
                "end": 0.0,
                "text": (
                    "[MOCK TRANSCRIPTION] faster-whisper is unavailable in this environment. "
                    "Install dependency to get real lyrics."
                ),
            }
        ],
    }


class UploadHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        return

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()
        self.wfile.write(body)

    def _append_log(self, level: str, message: str, extra: dict | None = None) -> None:
        SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "level": level,
            "message": message,
            "path": self.path,
        }
        if extra:
            row["extra"] = extra
        with SERVER_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route = parsed.path.rstrip("/") or "/"
        if route == "/health":
            self._send_json(200, {"status": "ok"})
            return
        if route == "/history":
            self._handle_history()
            return
        self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            route_path = parsed.path
            if route_path.startswith("http://") or route_path.startswith("https://"):
                route_path = urlparse(route_path).path

            normalized_route = route_path.rstrip("/") or "/"

            if normalized_route == "/upload":
                self._handle_upload(parsed)
                return
            if normalized_route == "/process":
                self._handle_process(parsed)
                return
            if normalized_route == "/arrange":
                self._handle_arrange(parsed)
                return

            self._send_json(404, {"error": "Not found"})
        except Exception as exc:  # pragma: no cover
            self._append_log("error", "Unhandled POST exception", {"detail": str(exc)})
            self._send_json(500, {"error": "Internal server error", "detail": str(exc)})

    def _handle_upload(self, parsed) -> None:
        query = parse_qs(parsed.query)
        filename = query.get("filename", [""])[0].strip()
        if not filename:
            self._send_json(400, {"error": "Missing query parameter: filename"})
            return

        safe_name = Path(filename).name
        ext = Path(safe_name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            self._send_json(400, {"error": "Invalid file format"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            self._send_json(400, {"error": "Missing content"})
            return

        file_bytes = self.rfile.read(content_length)
        destination = INPUT_DIR / safe_name
        destination.write_bytes(file_bytes)
        self._append_log("info", "Upload completed", {"filename": safe_name, "bytes": len(file_bytes)})

        self._send_json(
            200,
            {
                "message": "Upload successful",
                "filename": safe_name,
                "saved_to": str(destination),
                "bytes": len(file_bytes),
            },
        )

    def _handle_process(self, parsed) -> None:
        query = parse_qs(parsed.query)
        filename = query.get("filename", [""])[0].strip()
        model_size = query.get("model_size", ["small"])[0].strip() or "small"
        language = query.get("language", [settings["app"]["language"]])[0].strip()
        use_vad = query.get("use_vad", ["0"])[0].strip().lower() in {"1", "true", "yes", "on"}

        if not filename:
            self._send_json(400, {"error": "Missing query parameter: filename"})
            return

        safe_name = Path(filename).name
        input_audio = INPUT_DIR / safe_name
        if not input_audio.exists():
            self._send_json(404, {"error": "Input file not found", "filename": safe_name})
            return

        transcription_json = TMP_DIR / f"{input_audio.stem}.transcription.json"
        docx_path = OUTPUT_DIR / f"{input_audio.stem}.lyrics.docx"

        result = self._get_transcription(
            input_audio=input_audio,
            transcription_json=transcription_json,
            language=language,
            model_size=model_size,
            use_vad=use_vad,
        )

        docx_status = "ok"
        docx_detail = ""
        try:
            export_transcription_docx(result, docx_path, title=input_audio.stem)
        except Exception as exc:
            docx_status = "failed"
            docx_detail = str(exc)

        self._send_json(
            200,
            {
                "message": "Processing completed",
                "filename": safe_name,
                "transcription_json": str(transcription_json),
                "docx_path": str(docx_path) if docx_status == "ok" else "",
                "docx_status": docx_status,
                "docx_detail": docx_detail,
                "segments": len(result.get("segments", [])),
                "mode": result.get("mode", "real"),
                "warning": result.get("warning", ""),
            },
        )
        self._append_log("info", "Process completed", {"filename": safe_name, "mode": result.get("mode", "real")})

    def _handle_arrange(self, parsed) -> None:
        query = parse_qs(parsed.query)
        filename = query.get("filename", [""])[0].strip()
        model_size = query.get("model_size", ["small"])[0].strip() or "small"
        language = query.get("language", [settings["app"]["language"]])[0].strip()
        use_vad = query.get("use_vad", ["0"])[0].strip().lower() in {"1", "true", "yes", "on"}

        if not filename:
            self._send_json(400, {"error": "Missing query parameter: filename"})
            return

        safe_name = Path(filename).name
        input_audio = INPUT_DIR / safe_name
        if not input_audio.exists():
            self._send_json(404, {"error": "Input file not found", "filename": safe_name})
            return

        transcription_json = TMP_DIR / f"{input_audio.stem}.transcription.json"
        chords_json = TMP_DIR / f"{input_audio.stem}.chords.json"
        arrangement_json = TMP_DIR / f"{input_audio.stem}.arrangement.json"
        chord_docx = OUTPUT_DIR / f"{input_audio.stem}.chords.docx"

        transcription = self._get_transcription(
            input_audio=input_audio,
            transcription_json=transcription_json,
            language=language,
            model_size=model_size,
            use_vad=use_vad,
        )
        chords = detect_chords(input_audio=input_audio, tmp_dir=TMP_DIR)
        write_json(chords, chords_json)
        arrangement = align_chords_to_lyrics(transcription=transcription, chords=chords)
        write_json(arrangement, arrangement_json)

        docx_status = "ok"
        docx_detail = ""
        try:
            export_aligned_chord_docx(arrangement, chord_docx, title=input_audio.stem)
        except Exception as exc:
            docx_status = "failed"
            docx_detail = str(exc)

        self._send_json(
            200,
            {
                "message": "Arrangement completed",
                "filename": safe_name,
                "transcription_json": str(transcription_json),
                "chords_json": str(chords_json),
                "arrangement_json": str(arrangement_json),
                "docx_path": str(chord_docx) if docx_status == "ok" else "",
                "docx_status": docx_status,
                "docx_detail": docx_detail,
                "transcription_mode": transcription.get("mode", "real"),
                "chord_mode": chords.get("mode", "real"),
                "lines": len(arrangement.get("lines", [])),
                "warnings": arrangement.get("warnings", []),
            },
        )
        self._append_log(
            "info",
            "Arrange completed",
            {
                "filename": safe_name,
                "transcription_mode": transcription.get("mode", "real"),
                "chord_mode": chords.get("mode", "real"),
                "lines": len(arrangement.get("lines", [])),
            },
        )

    def _get_transcription(
        self,
        input_audio: Path,
        transcription_json: Path,
        language: str,
        model_size: str,
        use_vad: bool = False,
    ) -> dict:
        try:
            result = transcribe_audio(
                input_audio=input_audio,
                tmp_dir=TMP_DIR,
                language=language,
                model_size=model_size,
                use_vad=use_vad,
            )
            write_transcription_json(result, transcription_json)
            return result
        except Exception as exc:
            reason = str(exc)
            result = _mock_transcription(input_audio=input_audio, language=language, reason=reason)
            write_transcription_json(result, transcription_json)
            self._append_log("warn", "Transcription fallback to mock", {"filename": input_audio.name, "detail": reason})
            return result

    def _handle_history(self) -> None:
        entries: list[dict] = []
        for directory, kind in ((OUTPUT_DIR, "output"), (TMP_DIR, "tmp")):
            if not directory.exists():
                continue
            for file_path in directory.glob("*"):
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in {".docx", ".json"}:
                    continue
                stat = file_path.stat()
                entries.append(
                    {
                        "kind": kind,
                        "name": file_path.name,
                        "path": str(file_path),
                        "size": stat.st_size,
                        "modified_ts": stat.st_mtime,
                    }
                )
        entries.sort(key=lambda x: x["modified_ts"], reverse=True)
        self._send_json(200, {"items": entries[:30], "log_path": str(SERVER_LOG)})


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), UploadHandler)
    print(f"Upload server running on http://{host}:{port}")
    print("Health: /health | Upload: POST /upload | Process: POST /process | Arrange: POST /arrange")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down upload server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run(host=args.host, port=args.port)

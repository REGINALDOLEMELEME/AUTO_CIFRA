"""Backward-compatible thin wrapper around the new typed settings module."""
from __future__ import annotations

from pathlib import Path

from .config import get_settings as _get_settings


def load_settings(config_path: Path | None = None) -> dict:
    """Return settings as a plain dict for legacy callers (upload_server, CLI)."""
    s = _get_settings(config_path)
    return {
        "app": s.app.model_dump(),
        "paths": {
            "input_dir": str(s.input_dir),
            "output_dir": str(s.output_dir),
            "tmp_dir": str(s.tmp_dir),
            "models_dir": str(s.models_dir),
            "db_path": str(s.db_path),
            "frontend_dir": str(s.frontend_dir),
        },
        "asr": s.asr.model_dump(),
        "separation": s.separation.model_dump(),
        "alignment": s.alignment.model_dump(),
        "chords": s.chords.model_dump(),
        "structure": s.structure.model_dump(),
        "docx": s.docx.model_dump(),
        "worker": s.worker.model_dump(),
    }

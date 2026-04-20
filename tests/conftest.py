from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterator

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))


@pytest.fixture
def tmp_project(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Fresh AUTO_CIFRA-like directory tree for tests that read settings."""
    base = Path(tempfile.mkdtemp(prefix="autocifra_"))
    try:
        (base / "data" / "input").mkdir(parents=True)
        (base / "data" / "tmp").mkdir(parents=True)
        (base / "data" / "output").mkdir(parents=True)
        (base / "models").mkdir(parents=True)
        (base / "config").mkdir(parents=True)
        (base / "frontend").mkdir(parents=True)
        (base / "config" / "settings.yaml").write_text(
            "app:\n  language: pt\n"
            "paths:\n  input_dir: data/input\n  output_dir: data/output\n"
            "  tmp_dir: data/tmp\n  models_dir: models\n"
            "  db_path: data/jobs.sqlite\n  frontend_dir: frontend\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("AUTO_CIFRA_ROOT", str(base))
        monkeypatch.chdir(base)
        from src.config import get_settings

        get_settings.cache_clear()
        yield base
        get_settings.cache_clear()
    finally:
        shutil.rmtree(base, ignore_errors=True)


@pytest.fixture
def sample_audio() -> Path:
    wav = Path(__file__).parent / "data" / "silence_4s.wav"
    assert wav.exists(), f"Missing test fixture: {wav}"
    return wav


@pytest.fixture
def sample_aligned() -> dict:
    return {
        "source_file": "song.mp3",
        "transcription_mode": "real",
        "chord_mode": "real",
        "warnings": [],
        "lines": [
            {
                "section": "Verso 1",
                "start": 0.0,
                "end": 3.0,
                "words": [
                    {"text": "Eu", "start": 0.0, "end": 0.25, "chord": "C"},
                    {"text": "te", "start": 0.30, "end": 0.50, "chord": None},
                    {"text": "amo", "start": 0.55, "end": 1.20, "chord": "G"},
                    {"text": "muito", "start": 1.25, "end": 2.00, "chord": None},
                ],
                "lyric_line": "Eu te amo muito",
                "chord_line": "C G",
            },
            {
                "section": None,
                "start": 3.5,
                "end": 6.0,
                "words": [
                    {"text": "Sim", "start": 3.5, "end": 3.9, "chord": "Am"},
                    {"text": "sempre", "start": 4.0, "end": 4.8, "chord": None},
                    {"text": "serei", "start": 5.0, "end": 5.9, "chord": "F"},
                ],
                "lyric_line": "Sim sempre serei",
                "chord_line": "Am F",
            },
        ],
    }

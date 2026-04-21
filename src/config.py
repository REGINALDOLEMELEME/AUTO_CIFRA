from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class _AsrCfg(BaseModel):
    model: str = "large-v3"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "pt"


class _SeparationCfg(BaseModel):
    enabled: bool = True
    model: str = "htdemucs_ft"


class _AlignmentCfg(BaseModel):
    enabled: bool = True
    # Empty = whisperx picks the right default wav2vec2 for the configured language.
    # Override only if you want to force a specific HF repo or torchaudio pipeline.
    model: str = ""


class _ChordsCfg(BaseModel):
    backend: str = "chordino"
    vocabulary: list[str] = Field(
        default_factory=lambda: ["maj", "min", "5", "7", "m7", "maj7", "m7b5", "dim", "sus4"]
    )
    # When True, reduce Chordino's extended voicings (Cmaj7, Am7, Dm7, ...) to
    # their triad (C, Am, Dm). Matches how Cifra-Club-style lead sheets notate
    # chords; guitarists typically play these positions identically regardless
    # of the 7th being sounded in the recording.
    simplify_to_triads: bool = False
    # When True, coerce bare major chords to the diatonic minor when they fall
    # on ii/iii/vi of the detected major key (or ii°/iv/v of a minor key).
    # Fixes Chordino's systematic major-bias on simplechord output — e.g.
    # `A` in C major becomes `Am`. Leaves 7ths, dim, and slash chords alone.
    refine_to_key: bool = False


class _StructureCfg(BaseModel):
    enabled: bool = True
    n_segments: int = 6
    confidence_threshold: float = 0.6


class _DocxCfg(BaseModel):
    style: str = "table"
    body_font: str = "Calibri"
    chord_font: str = "Calibri"
    body_size_pt: int = 11
    chord_size_pt: int = 11


class _WorkerCfg(BaseModel):
    stage_timeout_seconds: int = 900
    poll_interval_seconds: float = 1.0


class _AppCfg(BaseModel):
    language: str = "pt"


class Settings(BaseModel):
    project_root: Path
    input_dir: Path
    output_dir: Path
    tmp_dir: Path
    models_dir: Path
    db_path: Path
    frontend_dir: Path
    app: _AppCfg = Field(default_factory=_AppCfg)
    asr: _AsrCfg = Field(default_factory=_AsrCfg)
    separation: _SeparationCfg = Field(default_factory=_SeparationCfg)
    alignment: _AlignmentCfg = Field(default_factory=_AlignmentCfg)
    chords: _ChordsCfg = Field(default_factory=_ChordsCfg)
    structure: _StructureCfg = Field(default_factory=_StructureCfg)
    docx: _DocxCfg = Field(default_factory=_DocxCfg)
    worker: _WorkerCfg = Field(default_factory=_WorkerCfg)

    def ensure_dirs(self) -> None:
        for p in (self.input_dir, self.output_dir, self.tmp_dir, self.models_dir):
            p.mkdir(parents=True, exist_ok=True)


def _project_root() -> Path:
    override = os.environ.get("AUTO_CIFRA_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[1]


def _merge(defaults: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    if not override:
        return defaults
    out = dict(defaults)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


@lru_cache(maxsize=1)
def get_settings(config_path: Path | None = None) -> Settings:
    root = _project_root()
    cfg_path = config_path or (root / "config" / "settings.yaml")
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    paths_raw = raw.get("paths", {}) or {}
    input_dir = root / paths_raw.get("input_dir", "data/input")
    output_dir = root / paths_raw.get("output_dir", "data/output")
    tmp_dir = root / paths_raw.get("tmp_dir", "data/tmp")
    models_dir = root / paths_raw.get("models_dir", "models")
    db_path = root / paths_raw.get("db_path", "data/jobs.sqlite")
    frontend_dir = root / paths_raw.get("frontend_dir", "frontend")

    _apply_model_cache_env(models_dir)

    settings = Settings(
        project_root=root,
        input_dir=input_dir,
        output_dir=output_dir,
        tmp_dir=tmp_dir,
        models_dir=models_dir,
        db_path=db_path,
        frontend_dir=frontend_dir,
        app=_AppCfg(**(raw.get("app") or {})),
        asr=_AsrCfg(**(raw.get("asr") or {})),
        separation=_SeparationCfg(**(raw.get("separation") or {})),
        alignment=_AlignmentCfg(**(raw.get("alignment") or {})),
        chords=_ChordsCfg(**(raw.get("chords") or {})),
        structure=_StructureCfg(**(raw.get("structure") or {})),
        docx=_DocxCfg(**(raw.get("docx") or {})),
        worker=_WorkerCfg(**(raw.get("worker") or {})),
    )
    settings.ensure_dirs()
    return settings


def _apply_model_cache_env(models_dir: Path) -> None:
    """Pin every ML library's cache under <project>/models/ (ADR-010)."""
    models_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(models_dir / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(models_dir / "torch"))
    os.environ.setdefault("XDG_CACHE_HOME", str(models_dir / "cache"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

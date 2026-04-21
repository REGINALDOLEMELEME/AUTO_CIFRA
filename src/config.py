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
    # initial_prompt biases Whisper's vocabulary (max 224 tokens). Keep it
    # FREE OF PROPER NOUNS AND COUNTRY NAMES — anything specific here leaks
    # as a hallucination when the audio is ambiguous. Learned the hard way:
    #   - A franciscan prompt produced "Santa Clara, São Francisco" tails on
    #     secular songs.
    #   - A prompt containing "Brasil" produced "A CIDADE NO BRASIL" on the
    #     whole first verse of Legião Urbana's "Será".
    # A maximally-generic hint still helps language lock without biasing
    # toward specific strings.
    initial_prompt: str = "Letra de música cantada com clareza."
    # hotwords (faster-whisper ≥ 1.0) reinforces decoder bias for rare terms.
    # Kept EMPTY by default — we measured that a religious-only hotword list
    # hallucinated those same terms onto secular tracks. Enable per-job from
    # the review UI or from batch_process.py when you know the repertoire.
    hotwords: str = ""
    # Apply deterministic regex corrections from src/lexicon.py after ASR.
    lexicon_correction: bool = True
    # Tail-fallback: when the primary ASR pass on Demucs-separated vocals
    # leaves a trailing gap longer than tail_fallback_min_gap_s, re-run
    # Whisper on the ORIGINAL mix over that window. Recovers quiet-tail
    # lyrics that Demucs over-suppressed (observed on 'São Francisco da
    # Misericórdia' — final refrain was lost in the separated track).
    tail_fallback_enabled: bool = True
    tail_fallback_min_gap_s: float = 8.0
    # Always probe at least this many seconds from the end of the audio when
    # tail-fallback fires, even if Whisper claims the last word ends later.
    # Guards against word-end timestamp stretching into silence windows.
    tail_fallback_probe_last_s: float = 30.0
    # Head-fallback: (superseded by gap_fallback below — kept as a config
    # knob for backwards compatibility, no-op if gap_fallback is enabled.)
    head_fallback_enabled: bool = False
    head_fallback_min_gap_s: float = 8.0
    # Gap-fallback: detect any gap between consecutive primary-pass segments
    # longer than `gap_fallback_min_s` and re-probe that window on the
    # ORIGINAL mix with chunked transcribe_clip. Covers head gaps (filtered
    # intro hallucinations), mid-song gaps, and late-entry first verses.
    gap_fallback_enabled: bool = True
    gap_fallback_min_s: float = 8.0


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
    # When True, drop short out-of-key chord segments that are sandwiched
    # between in-key neighbours (detection noise — real borrowed chords are
    # either longer or appear in runs, so the filter leaves them alone).
    filter_out_of_key_flickers: bool = True
    # When True, run a second-opinion chord classifier (librosa chroma +
    # key-biased templates) over Chordino's segments and replace Chordino's
    # label when it is out-of-key AND the chroma's in-key candidate scores
    # higher by at least `chroma_reclassify_margin`.
    chroma_reclassify: bool = True
    chroma_reclassify_margin: float = 0.05
    # If False, the chroma classifier can override ANY Chordino label (not
    # only out-of-key ones). Turn this on cautiously — Chordino's HMM is
    # usually better at in-key chord disambiguation than raw templates.
    chroma_reclassify_only_out_of_key: bool = True


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


class _StemsCfg(BaseModel):
    # htdemucs_6s produces {drums, bass, vocals, other, guitar, piano} —
    # guitar gets its own stem instead of being mixed into `other`,
    # which noticeably improves guitar clarity when drums are removed.
    # Use "htdemucs_ft" for the faster (but 4-stem only) fine-tuned model.
    model: str = "htdemucs_6s"
    mp3_bitrate: int = 320
    cache_ttl_hours: int = 24
    max_duration_sec: int = 1200       # 20 min
    max_bytes: int = 62_914_560        # 60 MiB
    janitor_interval_sec: int = 6 * 3600


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
    stems: _StemsCfg = Field(default_factory=_StemsCfg)

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
        stems=_StemsCfg(**(raw.get("stems") or {})),
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

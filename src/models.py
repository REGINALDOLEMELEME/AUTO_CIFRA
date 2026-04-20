"""Lazy singletons for ML models (ADR-003). Each loader is wrapped so missing
optional deps return (None, error_message) instead of raising."""
from __future__ import annotations

import gc
import os
import threading
from typing import Any

_lock = threading.RLock()
_whisper: Any = None
_whisper_key: tuple[str, str, str] | None = None
_demucs: Any = None
_wav2vec: Any = None
_wav2vec_meta: Any = None
_wav2vec_lang: str | None = None


def _try_import(name: str):
    try:
        import importlib

        return importlib.import_module(name), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{name} import failed: {exc}"


def get_whisper(model_size: str, device: str = "cpu", compute_type: str = "int8"):
    global _whisper, _whisper_key
    with _lock:
        key = (model_size, device, compute_type)
        if _whisper is not None and _whisper_key == key:
            return _whisper, None
        mod, err = _try_import("faster_whisper")
        if err:
            return None, err
        try:
            _whisper = mod.WhisperModel(model_size, device=device, compute_type=compute_type)
            _whisper_key = key
            return _whisper, None
        except Exception as exc:  # noqa: BLE001
            return None, f"WhisperModel load failed: {exc}"


def release_whisper() -> None:
    global _whisper, _whisper_key
    with _lock:
        _whisper = None
        _whisper_key = None
        gc.collect()


def get_demucs(model_name: str = "htdemucs_ft"):
    global _demucs
    with _lock:
        if _demucs is not None:
            return _demucs, None
        mod, err = _try_import("demucs.pretrained")
        if err:
            return None, err
        try:
            _demucs = mod.get_model(model_name)
            _demucs.cpu().eval()
            return _demucs, None
        except Exception as exc:  # noqa: BLE001
            return None, f"Demucs load failed: {exc}"


def release_demucs() -> None:
    global _demucs
    with _lock:
        _demucs = None
        gc.collect()


def get_wav2vec_aligner(language: str = "pt", model_name: str | None = None, device: str = "cpu"):
    """Return (model, metadata, error) ready for whisperx.align."""
    global _wav2vec, _wav2vec_meta, _wav2vec_lang
    with _lock:
        if _wav2vec is not None and _wav2vec_lang == language:
            return _wav2vec, _wav2vec_meta, None
        whisperx, err = _try_import("whisperx")
        if err:
            return None, None, err
        kwargs: dict = {"language_code": language, "device": device}
        if model_name:
            kwargs["model_name"] = model_name
        try:
            model, metadata = whisperx.load_align_model(**kwargs)
            _wav2vec = model
            _wav2vec_meta = metadata
            _wav2vec_lang = language
            return model, metadata, None
        except Exception as exc:  # noqa: BLE001
            return None, None, f"wav2vec aligner load failed: {exc}"


def release_wav2vec() -> None:
    global _wav2vec, _wav2vec_meta, _wav2vec_lang
    with _lock:
        _wav2vec = None
        _wav2vec_meta = None
        _wav2vec_lang = None
        gc.collect()


def release_all() -> None:
    release_whisper()
    release_demucs()
    release_wav2vec()

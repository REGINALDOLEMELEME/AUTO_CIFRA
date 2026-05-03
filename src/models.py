"""Lazy singletons for ML models (ADR-003). Each loader is wrapped so missing
optional deps return (None, error_message) instead of raising."""
from __future__ import annotations

import gc
import os
from pathlib import Path
from shutil import copy2
import threading
from typing import Any

_lock = threading.RLock()
_whisper: Any = None
_whisper_key: tuple[str, str, str] | None = None
_demucs: Any = None
_wav2vec: Any = None
_wav2vec_meta: Any = None
_wav2vec_key: tuple[str, str, str] | None = None

_ALIGN_REQUIRED_META = (
    "config.json",
    "preprocessor_config.json",
    "vocab.json",
)
_ALIGN_OPTIONAL_META = (
    "special_tokens_map.json",
    "tokenizer_config.json",
)


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


def _resolve_align_model_name(whisperx_alignment: Any, language: str, model_name: str | None) -> str | None:
    if model_name:
        return model_name
    if language in whisperx_alignment.DEFAULT_ALIGN_MODELS_TORCH:
        return whisperx_alignment.DEFAULT_ALIGN_MODELS_TORCH[language]
    return whisperx_alignment.DEFAULT_ALIGN_MODELS_HF.get(language)


def _repo_cache_dirs(hf_home: Path, repo_id: str) -> list[Path]:
    repo_key = repo_id.replace("/", "--")
    return [
        hf_home / "hub" / f"models--{repo_key}",
        hf_home / f"models--{repo_key}",
    ]


def _iter_snapshot_dirs(hf_home: Path, repo_id: str) -> list[Path]:
    out: list[Path] = []
    for base in _repo_cache_dirs(hf_home, repo_id):
        snaps = base / "snapshots"
        if not snaps.exists():
            continue
        for item in snaps.iterdir():
            if item.is_dir():
                out.append(item)
    return out


def _best_snapshot_parts(hf_home: Path, repo_id: str) -> tuple[Path | None, str | None, Path | None]:
    meta_snapshot: Path | None = None
    weight_snapshot: Path | None = None
    weight_name: str | None = None
    for snap in _iter_snapshot_dirs(hf_home, repo_id):
        if meta_snapshot is None and all((snap / name).exists() for name in _ALIGN_REQUIRED_META):
            meta_snapshot = snap
        if (snap / "model.safetensors").exists():
            weight_snapshot = snap
            weight_name = "model.safetensors"
            break
        if weight_snapshot is None and (snap / "pytorch_model.bin").exists():
            weight_snapshot = snap
            weight_name = "pytorch_model.bin"
    return meta_snapshot, weight_name, weight_snapshot


def _safe_link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except Exception:  # noqa: BLE001
        copy2(src, dst)


def _prepare_local_align_dir(model_name: str, hf_home: Path | None) -> Path | None:
    direct = Path(model_name)
    if direct.exists():
        return direct
    if hf_home is None or "/" not in model_name:
        return None
    meta_snapshot, weight_name, weight_snapshot = _best_snapshot_parts(hf_home, model_name)
    if meta_snapshot is None or weight_snapshot is None or weight_name is None:
        return None
    if meta_snapshot == weight_snapshot and all((meta_snapshot / name).exists() for name in _ALIGN_REQUIRED_META):
        return meta_snapshot

    merged_root = hf_home / "alignment-cache" / model_name.replace("/", "--")
    merged_root.mkdir(parents=True, exist_ok=True)
    for name in _ALIGN_REQUIRED_META + _ALIGN_OPTIONAL_META:
        src = meta_snapshot / name
        if src.exists():
            copy2(src, merged_root / name)
    _safe_link_or_copy(weight_snapshot / weight_name, merged_root / weight_name)
    return merged_root


def _load_hf_align_model(
    source: str,
    language: str,
    device: str,
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> tuple[Any | None, Any | None, str | None]:
    transformers, err = _try_import("transformers")
    if err:
        return None, None, err
    try:
        processor = transformers.Wav2Vec2Processor.from_pretrained(
            source,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        align_model = transformers.Wav2Vec2ForCTC.from_pretrained(
            source,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        ).to(device)
        vocab = processor.tokenizer.get_vocab()
        metadata = {
            "language": language,
            "dictionary": {char.lower(): code for char, code in vocab.items()},
            "type": "huggingface",
        }
        return align_model, metadata, None
    except Exception as exc:  # noqa: BLE001
        return None, None, str(exc)


def _load_torchaudio_align_model(model_name: str, language: str, device: str) -> tuple[Any | None, Any | None, str | None]:
    torchaudio, err = _try_import("torchaudio")
    if err:
        return None, None, err
    try:
        bundle = torchaudio.pipelines.__dict__[model_name]
        model_dir = os.environ.get("TORCH_HOME")
        kwargs = {"model_dir": model_dir} if model_dir else {}
        align_model = bundle.get_model(dl_kwargs=kwargs).to(device)
        labels = bundle.get_labels()
        metadata = {
            "language": language,
            "dictionary": {c.lower(): i for i, c in enumerate(labels)},
            "type": "torchaudio",
        }
        return align_model, metadata, None
    except Exception as exc:  # noqa: BLE001
        return None, None, str(exc)


def get_wav2vec_aligner(language: str = "pt", model_name: str | None = None, device: str = "cpu"):
    """Return (model, metadata, error) ready for whisperx.align."""
    global _wav2vec, _wav2vec_meta, _wav2vec_key
    with _lock:
        cache_key = (language, model_name or "", device)
        if _wav2vec is not None and _wav2vec_key == cache_key:
            return _wav2vec, _wav2vec_meta, None

        whisperx_alignment, err = _try_import("whisperx.alignment")
        if err:
            return None, None, err

        requested = _resolve_align_model_name(whisperx_alignment, language, model_name)
        if not requested:
            return None, None, f"wav2vec aligner unavailable (no model configured for {language!r})"

        if requested in whisperx_alignment.torchaudio.pipelines.__all__:
            model, metadata, load_err = _load_torchaudio_align_model(requested, language, device)
            if model is None or load_err:
                return None, None, f"wav2vec aligner unavailable ({requested}): {load_err}"
            _wav2vec = model
            _wav2vec_meta = metadata
            _wav2vec_key = cache_key
            return model, metadata, None

        hf_home_raw = os.environ.get("HF_HOME")
        hf_home = Path(hf_home_raw) if hf_home_raw else None
        local_dir = _prepare_local_align_dir(requested, hf_home)
        if local_dir is not None:
            model, metadata, load_err = _load_hf_align_model(
                str(local_dir),
                language=language,
                device=device,
                local_files_only=True,
            )
            if model is not None and not load_err:
                _wav2vec = model
                _wav2vec_meta = metadata
                _wav2vec_key = cache_key
                return model, metadata, None

        model, metadata, load_err = _load_hf_align_model(
            requested,
            language=language,
            device=device,
            cache_dir=str(hf_home) if hf_home else None,
            local_files_only=False,
        )
        if model is None or load_err:
            return None, None, f"wav2vec aligner unavailable ({requested}): {load_err}"

        _wav2vec = model
        _wav2vec_meta = metadata
        _wav2vec_key = cache_key
        return model, metadata, None


def release_wav2vec() -> None:
    global _wav2vec, _wav2vec_meta, _wav2vec_key
    with _lock:
        _wav2vec = None
        _wav2vec_meta = None
        _wav2vec_key = None
        gc.collect()


def release_all() -> None:
    release_whisper()
    release_demucs()
    release_wav2vec()

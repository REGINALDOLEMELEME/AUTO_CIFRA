"""Demucs vocal isolation. Returns the path to vocals.wav (or the input path and a
warning if Demucs is unavailable — graceful degradation per AT-005)."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

from .models import get_demucs, release_demucs


def separate_vocals(
    input_audio: Path,
    output_dir: Path,
    model_name: str = "htdemucs_ft",
) -> Tuple[Path, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{input_audio.stem}.vocals.wav"
    if target.exists():
        return target, ""

    model, err = get_demucs(model_name)
    if err or model is None:
        return input_audio, f"Demucs unavailable: {err}. Using full mix for ASR."

    try:
        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio
        from demucs.apply import apply_model
    except Exception as exc:  # noqa: BLE001
        return input_audio, f"Demucs runtime deps missing: {exc}. Using full mix for ASR."

    try:
        waveform, sr = torchaudio.load(str(input_audio))
        if sr != model.samplerate:
            waveform = torchaudio.functional.resample(waveform, sr, model.samplerate)
            sr = model.samplerate
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2]

        wav = waveform.unsqueeze(0)
        with torch.no_grad():
            sources = apply_model(model, wav, device="cpu", progress=False, split=True, overlap=0.1)
        stems = {name: sources[0, i].cpu().numpy() for i, name in enumerate(model.sources)}
        vocals = stems.get("vocals")
        if vocals is None:
            return input_audio, "Demucs returned no 'vocals' stem. Using full mix."
        mono = np.mean(vocals, axis=0)
        sf.write(str(target), mono, sr, subtype="PCM_16")
        return target, ""
    except Exception as exc:  # noqa: BLE001
        return input_audio, f"Demucs run failed: {exc}. Using full mix for ASR."
    finally:
        release_demucs()

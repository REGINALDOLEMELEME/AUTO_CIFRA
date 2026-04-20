"""Beat/BPM/key detection via librosa. Returns graceful defaults if librosa is
missing."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def detect_beats_key(input_audio: Path) -> dict[str, Any]:
    try:
        import librosa
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        return {"bpm": 0.0, "beats": [], "key": "", "warning": f"librosa missing: {exc}"}

    try:
        y, sr = librosa.load(str(input_audio), sr=22050, mono=True)
        if y.size == 0:
            return {"bpm": 0.0, "beats": [], "key": "", "warning": "empty audio"}
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        best = ("", -1.0)
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        for i in range(12):
            maj = np.corrcoef(np.roll(major_profile, i), chroma_mean)[0, 1]
            minr = np.corrcoef(np.roll(minor_profile, i), chroma_mean)[0, 1]
            if maj > best[1]:
                best = (f"{names[i]}", float(maj))
            if minr > best[1]:
                best = (f"{names[i]}m", float(minr))

        return {
            "bpm": float(tempo),
            "beats": [round(float(t), 3) for t in beat_times],
            "key": best[0],
            "key_confidence": round(best[1], 3),
            "warning": "",
        }
    except Exception as exc:  # noqa: BLE001
        return {"bpm": 0.0, "beats": [], "key": "", "warning": f"beat/key detection failed: {exc}"}

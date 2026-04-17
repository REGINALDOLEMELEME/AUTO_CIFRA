from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import os
from typing import Any


def _mock_chords(reason: str) -> dict[str, Any]:
    return {
        "mode": "mock",
        "warning": reason,
        "segments": [
            {"start": 0.0, "end": 4.0, "chord": "C"},
            {"start": 4.0, "end": 8.0, "chord": "G"},
            {"start": 8.0, "end": 12.0, "chord": "Am"},
            {"start": 12.0, "end": 16.0, "chord": "F"},
        ],
    }


_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _build_chord_templates():
    import numpy as np

    major_intervals = [0, 4, 7]
    minor_intervals = [0, 3, 7]
    templates: dict[str, Any] = {}
    for root in range(12):
        major = np.zeros(12, dtype=float)
        minor = np.zeros(12, dtype=float)
        for interval in major_intervals:
            major[(root + interval) % 12] = 1.0
        for interval in minor_intervals:
            minor[(root + interval) % 12] = 1.0
        templates[_NOTE_NAMES[root]] = major
        templates[f"{_NOTE_NAMES[root]}m"] = minor
    return templates


def _python_chord_estimation(input_audio: Path) -> dict[str, Any]:
    try:
        import numpy as np
        import librosa
    except Exception as exc:
        return _mock_chords(f"python chord detector unavailable: {exc}")

    try:
        y, sr = librosa.load(str(input_audio), sr=22050, mono=True)
        if y.size == 0:
            return _mock_chords("audio decoded empty in python detector")

        hop_length = 512
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
        if chroma.shape[1] == 0:
            return _mock_chords("no chroma frames in python detector")

        templates = _build_chord_templates()
        chord_names = list(templates.keys())
        template_matrix = np.stack([templates[name] for name in chord_names], axis=0)
        template_norm = np.linalg.norm(template_matrix, axis=1, keepdims=True) + 1e-9
        template_matrix = template_matrix / template_norm

        frame_chords: list[str] = []
        for i in range(chroma.shape[1]):
            v = chroma[:, i].astype(float)
            v_norm = np.linalg.norm(v)
            if v_norm <= 1e-9:
                frame_chords.append("N")
                continue
            v = v / v_norm
            scores = template_matrix @ v
            idx = int(np.argmax(scores))
            frame_chords.append(chord_names[idx])

        times = librosa.frames_to_time(
            list(range(chroma.shape[1])),
            sr=sr,
            hop_length=hop_length,
        )

        segments: list[dict[str, float | str]] = []
        start_idx = 0
        for i in range(1, len(frame_chords) + 1):
            if i == len(frame_chords) or frame_chords[i] != frame_chords[start_idx]:
                chord = frame_chords[start_idx]
                if chord != "N":
                    start_t = float(times[start_idx])
                    end_t = float(times[min(i, len(times) - 1)])
                    if end_t - start_t >= 0.25:
                        segments.append(
                            {
                                "start": round(start_t, 3),
                                "end": round(end_t, 3),
                                "chord": chord,
                            }
                        )
                start_idx = i

        if not segments:
            return _mock_chords("python detector produced zero chord segments")

        return {"mode": "python", "warning": "", "segments": segments}
    except Exception as exc:
        return _mock_chords(f"python chord detection failed: {exc}")


def _parse_chord_csv(csv_path: Path) -> list[dict[str, float | str]]:
    items: list[dict[str, float | str]] = []
    with csv_path.open("r", encoding="utf-8", errors="ignore") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                start = float(row[1])
            except ValueError:
                continue
            chord = row[-1].strip()
            if not chord:
                continue
            duration = 2.0
            try:
                duration = float(row[2])
            except ValueError:
                duration = 2.0
            items.append(
                {
                    "start": round(start, 3),
                    "end": round(start + max(duration, 0.5), 3),
                    "chord": chord,
                }
            )
    return items


def detect_chords(input_audio: Path, tmp_dir: Path) -> dict[str, Any]:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[1]
    local_sonic = project_root / "tools" / "sonic-annotator" / "sonic-annotator-win64" / "sonic-annotator.exe"
    sonic_exe = str(local_sonic) if local_sonic.exists() else "sonic-annotator"

    env = os.environ.copy()
    local_vamp = project_root / "tools" / "vamp-plugins"
    if local_vamp.exists():
        env["VAMP_PATH"] = str(local_vamp)

    try:
        cmd = [
            sonic_exe,
            "-d",
            "vamp:nnls-chroma:chordino:chord",
            "-w",
            "csv",
            "--csv-force",
            "-o",
            str(tmp_dir),
            str(input_audio),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    except Exception as exc:
        python_result = _python_chord_estimation(input_audio)
        if python_result.get("mode") == "python":
            python_result["warning"] = f"chordino unavailable, using python detector: {exc}"
            return python_result
        return _mock_chords(f"sonic-annotator/chordino unavailable: {exc}")

    pattern = f"{input_audio.stem}*chord*.csv"
    csv_candidates = sorted(tmp_dir.glob(pattern))
    if not csv_candidates:
        return _mock_chords("Chord CSV not generated by sonic-annotator.")

    segments = _parse_chord_csv(csv_candidates[0])
    if not segments:
        python_result = _python_chord_estimation(input_audio)
        if python_result.get("mode") == "python":
            python_result["warning"] = "chordino returned empty output, using python detector"
            return python_result
        return _mock_chords("Chord CSV parsed with zero segments.")

    return {"mode": "real", "warning": "", "segments": segments}


def write_json(payload: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path

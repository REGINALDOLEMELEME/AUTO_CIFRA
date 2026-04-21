from __future__ import annotations

import csv
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


# -------- Vocabulary normalization (ADR-007) --------------------------------

_ROOT = re.compile(r"^([A-Ga-g])([#b]?)")


def _split_root(label: str) -> tuple[str, str] | None:
    m = _ROOT.match(label)
    if not m:
        return None
    root = m.group(1).upper() + (m.group(2) or "")
    tail = label[m.end():]
    return root, tail


def _canonicalize_quality(tail: str) -> tuple[str, bool]:
    """Return (quality_suffix, warn). Maps exotic Chordino tails to the
    allowed vocabulary {maj, min, 5, 7, m7, maj7, m7b5, dim, sus4}."""
    t = tail.strip().replace("Δ", "maj7").replace("°", "dim").replace("ø", "m7b5")
    # Order matters — longest match first.
    if re.search(r"maj7|M7", t):
        return "maj7", False
    if re.search(r"m7b5", t):
        return "m7b5", False
    if re.search(r"^min?7|^m7", t):
        return "m7", False
    if re.search(r"^min|^m(?![a-zA-Z])", t):
        return "m", False
    if re.search(r"^dim", t):
        return "dim", False
    if re.search(r"^aug|^\+", t):
        return "", True
    if re.search(r"^sus2", t):
        return "", True
    if re.search(r"^sus4?", t):
        return "sus4", False
    if re.search(r"^5($|[^0-9])", t):
        return "5", False
    if re.search(r"^7", t):
        return "7", False
    if re.search(r"^(9|11|13|add9|add11|add13)", t):
        return "7", True
    if t in {"", "maj", "M"}:
        return "", False
    return "", True


def normalize_chord_label(label: str) -> tuple[str, bool]:
    """Return (canonical_label, was_modified)."""
    raw = label.strip()
    if raw in {"", "N", "X", "-"}:
        return "", False
    if "/" in raw:
        head, bass = raw.split("/", 1)
        h, warn_h = normalize_chord_label(head)
        b, warn_b = normalize_chord_label(bass)
        if not h:
            return "", True
        if not b:
            return h, warn_h or True
        return f"{h}/{b}", warn_h or warn_b
    parts = _split_root(raw)
    if not parts:
        return "", True
    root, tail = parts
    qual, warn = _canonicalize_quality(tail)
    return f"{root}{qual}", warn


def simplify_to_triad(label: str) -> str:
    """Reduce a canonical chord label to its triad form.

    Keeps root, the minor marker, diminished, and slash-bass. Drops maj7 / m7 /
    7 / 6 / 5 / sus4 since those are voicing details that a guitarist reading a
    Cifra-Club-style chart would play the same way as the plain triad.
    """
    if not label:
        return ""
    if "/" in label:
        head, bass = label.split("/", 1)
        head_t = simplify_to_triad(head)
        return f"{head_t}/{bass}" if head_t else ""
    parts = _split_root(label)
    if not parts:
        return label
    root, tail = parts
    if tail.startswith("dim"):
        return f"{root}dim"
    if tail.startswith("m7b5"):
        return f"{root}m"
    if tail.startswith("m") and not tail.startswith("maj"):
        return f"{root}m"
    return root


# Scale-degree triad qualities for major / natural-minor keys. Used by
# refine_chords_to_key() to coerce Chordino's major-biased "simplechord"
# output to the diatonic minor when the root lands on ii/iii/vi (major key)
# or similar (minor key). Heuristic only — it does not try to model secondary
# dominants or borrowed chords.
_SHARP_ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
_MAJOR_SCALE_QUALITIES = ["", "m", "m", "", "", "m", "dim"]  # I ii iii IV V vi vii°
_MINOR_SCALE_QUALITIES = ["m", "dim", "", "m", "m", "", ""]  # i ii° III iv v VI VII


def _norm_root(root: str) -> str:
    r = (root[:1].upper() + root[1:]) if root else ""
    return _FLAT_TO_SHARP.get(r, r)


def _expected_quality_for_root(key: str, root: str) -> str | None:
    """Return '', 'm', or 'dim' for the root's diatonic triad in `key`.
    `key` is like 'C', 'Am', 'F#m'. Returns None if inputs can't be interpreted."""
    key = (key or "").strip()
    if not key:
        return None
    key_is_minor = key.endswith("m") and not key.endswith("dim")
    key_root = _norm_root(key[:-1] if key_is_minor else key)
    root_n = _norm_root(root)
    if key_root not in _SHARP_ROOTS or root_n not in _SHARP_ROOTS:
        return None
    degree = (_SHARP_ROOTS.index(root_n) - _SHARP_ROOTS.index(key_root)) % 12
    # Major scale degrees live at semitones {0,2,4,5,7,9,11}; minor at {0,2,3,5,7,8,10}
    major_degrees = [0, 2, 4, 5, 7, 9, 11]
    minor_degrees = [0, 2, 3, 5, 7, 8, 10]
    degrees = minor_degrees if key_is_minor else major_degrees
    if degree not in degrees:
        return None
    scale_index = degrees.index(degree)
    qualities = _MINOR_SCALE_QUALITIES if key_is_minor else _MAJOR_SCALE_QUALITIES
    return qualities[scale_index]


def refine_chords_to_key(
    chords: dict[str, Any], key: str
) -> dict[str, Any]:
    """Coerce bare major triads to their diatonic quality when they land on an
    out-of-quality scale degree. Leaves already-qualified chords alone (7ths,
    dim, slash chords, etc.) and only touches the head of slash labels."""
    if not key:
        return chords
    segs = chords.get("segments") or []
    refined: list[dict[str, Any]] = []
    for seg in segs:
        label = str(seg.get("chord") or "").strip()
        if not label:
            refined.append(seg)
            continue
        head, sep, bass = label.partition("/")
        parts = _split_root(head)
        if not parts:
            refined.append(seg)
            continue
        root, tail = parts
        if tail:
            refined.append(seg)
            continue
        expected = _expected_quality_for_root(key, root)
        if expected is None or expected == "":
            refined.append(seg)
            continue
        new_head = f"{root}{expected}"
        new_label = f"{new_head}{sep}{bass}" if sep else new_head
        refined.append({**seg, "chord": new_label})
    out = dict(chords)
    out["segments"] = refined
    return out


def normalize_chord_vocabulary(
    chords: dict[str, Any],
    simplify_to_triads: bool = False,
) -> dict[str, Any]:
    warnings_accum: list[str] = list(chords.get("warning", "") and [chords["warning"]] or [])
    seen_exotic: set[str] = set()
    new_segments: list[dict[str, Any]] = []
    for seg in chords.get("segments", []) or []:
        raw_label = str(seg.get("chord") or "").strip()
        canon, warn = normalize_chord_label(raw_label)
        if not canon:
            continue
        if warn and raw_label not in seen_exotic:
            seen_exotic.add(raw_label)
            warnings_accum.append(f"chord simplified: {raw_label!r} -> {canon!r}")
        if simplify_to_triads:
            canon = simplify_to_triad(canon)
            if not canon:
                continue
        new_segments.append({**seg, "chord": canon})
    out = {**chords, "segments": new_segments}
    out["warning"] = " | ".join([w for w in warnings_accum if w])
    return out


# -------- Beat smoothing ----------------------------------------------------


def _is_in_key(label: str, key: str) -> bool:
    """True when `label`'s root is diatonic to `key`. Slash basses and
    extensions are ignored — we only check the root."""
    if not label or not key:
        return True
    head = label.split("/", 1)[0]
    parts = _split_root(head)
    if not parts:
        return True
    root, _ = parts
    return _expected_quality_for_root(key, root) is not None


def filter_out_of_key_flickers(
    chords: dict[str, Any],
    key: str,
    max_flicker_s: float = 1.6,
) -> dict[str, Any]:
    """Remove short out-of-key chord segments that are sandwiched between
    in-key neighbours.

    Chordino's simplechord output routinely emits 0.5–1.5 s bursts of an
    out-of-key chord inside an otherwise diatonic progression (e.g. a stray
    F or Dm inside a C–G–Am–F verse). Those are almost always detection
    noise — quantising the harmonic grid smooths them out without touching
    genuinely non-diatonic passages (those are either longer or flanked by
    another non-diatonic chord)."""
    if not key:
        return chords
    segs = list(chords.get("segments") or [])
    if len(segs) < 3:
        return chords
    kept: list[dict[str, Any]] = []
    i = 0
    while i < len(segs):
        cur = segs[i]
        if i == 0 or i == len(segs) - 1:
            kept.append(cur)
            i += 1
            continue
        prev_kept = kept[-1] if kept else segs[i - 1]
        nxt = segs[i + 1]
        cur_label = str(cur.get("chord") or "")
        prev_label = str(prev_kept.get("chord") or "")
        next_label = str(nxt.get("chord") or "")
        dur = float(cur.get("end", 0.0)) - float(cur.get("start", 0.0))
        # Drop cur iff: short, out-of-key, and BOTH neighbours are in-key.
        if (
            dur <= max_flicker_s
            and not _is_in_key(cur_label, key)
            and _is_in_key(prev_label, key)
            and _is_in_key(next_label, key)
        ):
            # Absorb cur's time into the previous kept segment so there's no
            # gap in the chord timeline.
            if kept:
                kept[-1] = {**kept[-1], "end": float(cur.get("end", kept[-1]["end"]))}
            i += 1
            continue
        kept.append(cur)
        i += 1
    # Collapse consecutive runs of the same chord the filter may have created.
    collapsed: list[dict[str, Any]] = []
    for seg in kept:
        if collapsed and collapsed[-1]["chord"] == seg.get("chord"):
            collapsed[-1] = {**collapsed[-1], "end": float(seg.get("end", collapsed[-1]["end"]))}
        else:
            collapsed.append(dict(seg))
    return {**chords, "segments": collapsed}


def smooth_to_beats(chords: dict[str, Any], beats: dict[str, Any]) -> dict[str, Any]:
    """Drop chord flickers shorter than one beat and collapse consecutive runs."""
    beat_times = beats.get("beats") or []
    if not chords.get("segments") or not beat_times or len(beat_times) < 2:
        return chords
    beat_interval = max(
        (beat_times[i + 1] - beat_times[i]) for i in range(len(beat_times) - 1)
    )
    min_duration = max(0.25, beat_interval * 0.75)

    segs = sorted(chords["segments"], key=lambda s: float(s.get("start", 0.0)))
    kept: list[dict[str, Any]] = []
    for seg in segs:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        dur = end - start
        if dur < min_duration:
            continue
        if kept and kept[-1]["chord"] == seg.get("chord"):
            kept[-1]["end"] = max(float(kept[-1]["end"]), end)
        else:
            kept.append({**seg, "start": round(start, 3), "end": round(end, 3)})
    return {**chords, "segments": kept}


# -------- Detection (existing Chordino + Python fallback) -------------------


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

# Chroma templates used by the Python chord detector.
# Each template is a 12-bin pitch-class vector + a score weight applied post-
# cosine to bias the detector toward simpler chords (matches typical chord-sheet
# notation where 7th extensions are often omitted unless prominent).
_QUALITIES: list[tuple[str, tuple[int, ...], float]] = [
    ("",      (0, 4, 7),      1.00),  # major triad
    ("m",     (0, 3, 7),      1.00),  # minor triad
    ("5",     (0, 7),         0.95),  # power chord
    ("7",     (0, 4, 7, 10),  0.85),  # dominant 7th
    ("m7",    (0, 3, 7, 10),  0.85),  # minor 7th
    ("maj7",  (0, 4, 7, 11),  0.85),  # major 7th
    ("dim",   (0, 3, 6),      0.85),  # diminished triad
]

# Diatonic chord qualities per scale degree, used to bias scores toward
# chords that "fit" the detected key (major / natural minor).
_MAJOR_DEGREES = {0: "", 2: "m", 4: "m", 5: "", 7: "", 9: "m", 11: "dim"}
_MINOR_DEGREES = {0: "m", 2: "dim", 3: "", 5: "m", 7: "m", 8: "", 10: ""}


def _build_chord_templates():
    import numpy as np

    templates: dict[str, Any] = {}
    weights: dict[str, float] = {}
    for root in range(12):
        for suffix, intervals, weight in _QUALITIES:
            vec = np.zeros(12, dtype=float)
            for interval in intervals:
                vec[(root + interval) % 12] = 1.0
            name = f"{_NOTE_NAMES[root]}{suffix}"
            templates[name] = vec
            weights[name] = weight
    return templates, weights


def _key_bias(chord_name: str, key: str) -> float:
    """Return a small log-prob boost (0 .. +0.15) for chords that belong to
    the detected key. Non-key chords get 0."""
    if not key:
        return 0.0
    import re as _re

    m = _re.match(r"^([A-G][#b]?)(m?)$", key)
    if not m:
        return 0.0
    key_root = m.group(1)
    key_is_minor = bool(m.group(2))
    sharp_to_flat = {"C#": "C#", "D#": "D#", "F#": "F#", "G#": "G#", "A#": "A#",
                     "Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
    key_root = sharp_to_flat.get(key_root, key_root)
    if key_root not in _NOTE_NAMES:
        return 0.0
    key_idx = _NOTE_NAMES.index(key_root)
    degrees = _MINOR_DEGREES if key_is_minor else _MAJOR_DEGREES

    # split chord_name into root + suffix
    cm = _re.match(r"^([A-G][#b]?)(.*)$", chord_name)
    if not cm:
        return 0.0
    croot = sharp_to_flat.get(cm.group(1), cm.group(1))
    if croot not in _NOTE_NAMES:
        return 0.0
    croot_idx = _NOTE_NAMES.index(croot)
    csuffix = cm.group(2)
    base_suffix = csuffix
    if csuffix in {"7", "m7", "maj7"}:
        base_suffix = {"7": "", "m7": "m", "maj7": ""}[csuffix]
    elif csuffix == "5":
        base_suffix = ""

    diff = (croot_idx - key_idx) % 12
    return 0.12 if degrees.get(diff) == base_suffix else 0.0


def _viterbi_decode(emission_log: "np.ndarray", self_bonus: float = 0.35) -> list[int]:
    """Tiny Viterbi with a uniform transition cost and a self-loop bonus
    that strongly discourages flickering between chords.
    emission_log: (T, K) log-probabilities. Returns list of K indices."""
    import numpy as np

    T, K = emission_log.shape
    if T == 0:
        return []
    dp = np.full((T, K), -np.inf, dtype=np.float64)
    bp = np.zeros((T, K), dtype=np.int32)
    dp[0] = emission_log[0]
    for t in range(1, T):
        # Best predecessor for each state
        prev = dp[t - 1]
        best_prev = prev.max()
        best_prev_idx = int(prev.argmax())
        # Transition score: self-loop gets a bonus; all others get best_prev
        for k in range(K):
            stay_score = prev[k] + self_bonus
            if stay_score >= best_prev:
                dp[t, k] = stay_score + emission_log[t, k]
                bp[t, k] = k
            else:
                dp[t, k] = best_prev + emission_log[t, k]
                bp[t, k] = best_prev_idx
    # Back-trace
    path = [int(dp[-1].argmax())]
    for t in range(T - 1, 0, -1):
        path.append(int(bp[t, path[-1]]))
    path.reverse()
    return path


def _python_chord_estimation(input_audio: Path, key: str = "") -> dict[str, Any]:
    """Beat-synchronous, multi-template, Viterbi-smoothed chord detector.
    Works in pure Python (librosa + numpy), no VAMP hosts required."""
    try:
        import librosa
        import numpy as np
    except Exception as exc:
        return _mock_chords(f"python chord detector unavailable: {exc}")

    try:
        y, sr = librosa.load(str(input_audio), sr=22050, mono=True)
        if y.size == 0:
            return _mock_chords("audio decoded empty in python detector")

        # High-resolution chroma, then beat-sync so one frame = one beat.
        hop_length = 512
        chroma = librosa.feature.chroma_cqt(
            y=y, sr=sr, hop_length=hop_length, bins_per_octave=36
        )
        # Median-filter chroma slightly to remove transients.
        chroma = np.clip(chroma, 0, 1)
        if chroma.shape[1] < 4:
            return _mock_chords("too few chroma frames")

        try:
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
            if len(beat_frames) >= 4:
                chroma_beats = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
                beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
            else:
                chroma_beats = chroma
                beat_times = librosa.frames_to_time(
                    np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length
                )
        except Exception:
            chroma_beats = chroma
            beat_times = librosa.frames_to_time(
                np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length
            )

        templates, weights = _build_chord_templates()
        chord_names = list(templates.keys())
        template_matrix = np.stack([templates[n] for n in chord_names], axis=0)
        template_matrix /= np.linalg.norm(template_matrix, axis=1, keepdims=True) + 1e-9
        template_weights = np.array([weights[n] for n in chord_names], dtype=np.float64)
        T = chroma_beats.shape[1]
        K = len(chord_names)

        # Cosine-similarity emission; append a fixed "No chord" score.
        emission = np.zeros((T, K + 1), dtype=np.float64)
        silence_threshold = 0.05
        for t in range(T):
            v = chroma_beats[:, t].astype(np.float64)
            n = np.linalg.norm(v)
            if n < silence_threshold:
                emission[t, K] = 1.0
                continue
            v /= n + 1e-9
            emission[t, :K] = (template_matrix @ v) * template_weights
            emission[t, K] = 0.25  # baseline noise floor for "No chord"

        # Key bias
        if key:
            bias = np.array([_key_bias(n, key) for n in chord_names] + [0.0])
            emission += bias[None, :]

        # log-probabilities (add small floor to avoid log(0))
        emission_log = np.log(np.clip(emission, 1e-4, None))

        # Viterbi
        names_with_n = chord_names + ["N"]
        path = _viterbi_decode(emission_log, self_bonus=0.35)
        frame_chords = [names_with_n[i] for i in path]

        # Convert frame indices to time intervals.
        segments: list[dict[str, float | str]] = []
        end_time = float(len(y) / sr)
        for i in range(T):
            start_t = float(beat_times[i]) if i < len(beat_times) else 0.0
            if i + 1 < T and i + 1 < len(beat_times):
                end_t = float(beat_times[i + 1])
            else:
                end_t = end_time
            chord = frame_chords[i]
            if chord == "N":
                continue
            if end_t - start_t < 0.1:
                continue
            if segments and segments[-1]["chord"] == chord:
                segments[-1]["end"] = round(end_t, 3)
            else:
                segments.append(
                    {"start": round(start_t, 3), "end": round(end_t, 3), "chord": chord}
                )

        if not segments:
            return _mock_chords("python detector produced zero chord segments")
        return {"mode": "python", "warning": "", "segments": segments, "key": key}
    except Exception as exc:
        return _mock_chords(f"python chord detection failed: {exc}")


def _parse_chord_csv(csv_path: Path) -> list[dict[str, float | str]]:
    """Parse sonic-annotator CSV output for Chordino simplechord.

    With `--csv-basedir --csv-fill-ends --csv-end-times`, sonic-annotator 1.7
    emits rows of `start,end,"chord"` (3 columns, no filename prefix).
    Older sonic-annotator 1.6 emitted `filename,start,duration,"chord"` (4
    columns). This parser handles both by detecting a trailing numeric in
    column 1.
    """
    items: list[dict[str, float | str]] = []
    with csv_path.open("r", encoding="utf-8", errors="ignore") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                float(row[0])
                offset = 0
            except ValueError:
                offset = 1
            try:
                start = float(row[offset])
                second = float(row[offset + 1])
            except (ValueError, IndexError):
                continue
            chord = row[-1].strip().strip('"')
            if not chord:
                continue
            if second > start:
                end = second
            elif second >= 0.05:
                end = start + second
            else:
                continue
            items.append(
                {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "chord": chord,
                }
            )
    return items


def detect_chords(input_audio: Path, tmp_dir: Path, key: str = "") -> dict[str, Any]:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[1]
    local_sonic = (
        project_root / "tools" / "sonic-annotator" / "sonic-annotator-win64" / "sonic-annotator.exe"
    )
    sonic_exe = str(local_sonic) if local_sonic.exists() else "sonic-annotator"

    env = os.environ.copy()
    local_vamp = project_root / "tools" / "vamp-plugins"
    if local_vamp.exists():
        env["VAMP_PATH"] = str(local_vamp)

    try:
        cmd = [
            sonic_exe,
            "-d",
            "vamp:nnls-chroma:chordino:simplechord",
            "-w",
            "csv",
            "--csv-force",
            "--csv-fill-ends",
            "--csv-end-times",
            "--csv-basedir",
            str(tmp_dir),
            str(input_audio),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    except Exception as exc:
        python_result = _python_chord_estimation(input_audio, key=key)
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
        python_result = _python_chord_estimation(input_audio, key=key)
        if python_result.get("mode") == "python":
            python_result["warning"] = "chordino empty output, using python detector"
            return python_result
        return _mock_chords("Chord CSV parsed with zero segments.")

    return {"mode": "real", "warning": "", "segments": segments}


def write_json(payload: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_path

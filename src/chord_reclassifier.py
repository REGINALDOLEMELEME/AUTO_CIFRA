"""Second-opinion chord classifier using chroma template matching.

Chordino (our primary detector via sonic-annotator) has a trained HMM that
biases toward certain voicings and sometimes produces out-of-key roots
(e.g. `F Dm` where the audio is clearly `C G` in a C-major song). This
module cross-validates each Chordino segment against librosa's chroma
features using a simple template bank of 24 triads, with a configurable
key-bias penalty.

We deliberately keep this narrow: re-label, don't re-segment. Chordino's
timing is generally accurate — what we're correcting is label choice when
the harmonic content contradicts it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .chords import _expected_quality_for_root, _split_root


# Pitch-class index: C=0, C#=1, D=2, ..., B=11
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PITCH_INDEX = {p: i for i, p in enumerate(PITCH_CLASSES)}
# Enharmonic aliases — normalize flats to sharps for template matching.
_FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}


def _normalize_root(root: str) -> str | None:
    """Return a sharp-based root name in {C, C#, D, ..., B}, or None.
    Uppercases only the first letter so the flat indicator "b" is preserved.
    """
    if not root:
        return None
    r = root.strip().replace("♭", "b").replace("♯", "#")
    if not r:
        return None
    r = r[0].upper() + r[1:]
    if len(r) >= 2 and r[1] == "b":
        r = _FLAT_TO_SHARP.get(r[:2], r)
    if r not in PITCH_INDEX:
        r = r[:1]
    return r if r in PITCH_INDEX else None


def _template(root_idx: int, quality: str) -> list[float]:
    """12-d binary template for a triad. Equal-weight on root/3rd/5th."""
    vec = [0.0] * 12
    if quality == "m":
        intervals = (0, 3, 7)
    else:  # major
        intervals = (0, 4, 7)
    for iv in intervals:
        vec[(root_idx + iv) % 12] = 1.0 / 3.0
    return vec


def _score(chroma: list[float], template: list[float]) -> float:
    """Cosine-like similarity between a normalized chroma frame and a
    template. Both are non-negative; we normalize chroma to sum=1 first.
    """
    total = sum(chroma) or 1e-9
    c_norm = [c / total for c in chroma]
    return sum(c * t for c, t in zip(c_norm, template))


def classify_chroma(
    chroma_vec: list[float],
    key: str = "",
    out_of_key_penalty: float = 0.6,
) -> tuple[str, float]:
    """Pick the best triad label for a 12-d chroma vector. Diatonic chords
    of `key` (if provided) score at full weight; out-of-key roots are
    multiplied by `out_of_key_penalty` so only a much stronger audio match
    can pick them."""
    best_label = ""
    best_score = -1.0
    for i, root in enumerate(PITCH_CLASSES):
        for qual in ("", "m"):
            tmpl = _template(i, qual)
            score = _score(chroma_vec, tmpl)
            if key:
                expected = _expected_quality_for_root(key, root)
                if expected is None:
                    score *= out_of_key_penalty
            if score > best_score:
                best_score = score
                best_label = f"{root}{qual}"
    return best_label, best_score


def _score_label(chroma_vec: list[float], label: str) -> float:
    """Score Chordino's label against the chroma window (no key bias)."""
    head = label.split("/", 1)[0]
    parts = _split_root(head)
    if not parts:
        return 0.0
    root = _normalize_root(parts[0])
    if not root:
        return 0.0
    tail = parts[1]
    qual = "m" if tail.startswith("m") and not tail.startswith("maj") else ""
    tmpl = _template(PITCH_INDEX[root], qual)
    return _score(chroma_vec, tmpl)


def reclassify_with_chroma(
    chords: dict[str, Any],
    audio_path: Path,
    key: str,
    margin: float = 0.05,
    only_out_of_key: bool = True,
) -> dict[str, Any]:
    """Second-opinion pass over Chordino segments using librosa chroma.

    For each segment, average the audio's chroma over its time window. If
    the chroma's top pick scores higher than Chordino's label by at least
    `margin` AND (when `only_out_of_key` is set) Chordino's label is
    out-of-key, replace Chordino's label with the chroma pick.

    Narrow by design: we are correcting *labels*, not timings.
    """
    if not chords.get("segments"):
        return chords
    try:
        import librosa
        import numpy as np
    except Exception as exc:
        return {
            **chords,
            "warning": " | ".join(
                filter(None, [chords.get("warning", ""), f"chroma_reclassify skipped: {exc}"])
            ),
        }

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    # chroma is shape (12, n_frames). Default hop_length=512 → ~43 fps at 22.05 kHz.
    hop = 512
    fps = sr / hop

    new_segments: list[dict[str, Any]] = []
    replaced = 0
    for seg in chords["segments"]:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        if end <= start:
            new_segments.append(seg)
            continue
        s_f = max(0, int(start * fps))
        e_f = min(chroma.shape[1], int(end * fps) + 1)
        if e_f <= s_f:
            new_segments.append(seg)
            continue
        window = chroma[:, s_f:e_f].mean(axis=1)
        vec = [float(x) for x in window]

        chordino_label = str(seg.get("chord") or "").strip()
        chroma_label, chroma_score = classify_chroma(vec, key=key)
        chordino_score = _score_label(vec, chordino_label)

        # Gate the replacement.
        chordino_in_key = True
        if chordino_label:
            head = chordino_label.split("/", 1)[0]
            parts = _split_root(head)
            if parts:
                root = _normalize_root(parts[0])
                chordino_in_key = (
                    key == ""
                    or (root is not None and _expected_quality_for_root(key, root) is not None)
                )

        should_replace = (
            chroma_label
            and chroma_label != chordino_label
            and chroma_score - chordino_score > margin
        )
        if only_out_of_key:
            should_replace = should_replace and (not chordino_in_key)

        if should_replace:
            new_segments.append({**seg, "chord": chroma_label})
            replaced += 1
        else:
            new_segments.append(seg)

    out = {**chords, "segments": new_segments}
    if replaced:
        w = f"chroma_reclassify: replaced {replaced} segment(s) with in-key candidates"
        out["warning"] = " | ".join(filter(None, [out.get("warning", ""), w]))
    return out

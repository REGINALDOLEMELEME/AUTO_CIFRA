from __future__ import annotations

import re

_SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
_SHARP_TO_INDEX = {n: i for i, n in enumerate(_SHARP_NAMES)}
_FLAT_TO_INDEX = {n: i for i, n in enumerate(_FLAT_NAMES)}

_ROOT_PATTERN = re.compile(r"^([A-Ga-g])([#b]?)")


def _parse_root(label: str) -> tuple[int, str]:
    m = _ROOT_PATTERN.match(label)
    if not m:
        raise ValueError(f"Cannot parse chord root: {label!r}")
    root_letter = m.group(1).upper()
    accidental = m.group(2) or ""
    root_text = f"{root_letter}{accidental}"
    if accidental == "#":
        idx = _SHARP_TO_INDEX.get(root_text)
    elif accidental == "b":
        idx = _FLAT_TO_INDEX.get(root_text)
    else:
        idx = _SHARP_TO_INDEX.get(root_text)
    if idx is None:
        raise ValueError(f"Unknown root: {root_text!r}")
    return idx, label[len(root_text):]


def shift_chord(label: str, semitones: int, prefer_flats: bool = True) -> str:
    if not label or not label.strip():
        return label
    label = label.strip()
    if label in {"N", "X", "-"}:
        return label
    # Slash chord: transpose both sides, keep the slash.
    if "/" in label:
        head, bass = label.split("/", 1)
        return f"{shift_chord(head, semitones, prefer_flats)}/{shift_chord(bass, semitones, prefer_flats)}"
    try:
        idx, quality = _parse_root(label)
    except ValueError:
        return label
    new_idx = (idx + semitones) % 12
    names = _FLAT_NAMES if prefer_flats else _SHARP_NAMES
    return f"{names[new_idx]}{quality}"


def effective_semitones(transpose_semitones: int, capo_fret: int) -> int:
    """Capo raises pitch of the played chord; the written chord drops by capo_fret."""
    return transpose_semitones - capo_fret

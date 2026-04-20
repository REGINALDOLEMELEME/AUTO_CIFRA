from __future__ import annotations

import pytest

from src.chords import normalize_chord_label, normalize_chord_vocabulary


@pytest.mark.parametrize(
    "raw,expected,warned",
    [
        ("C", ("C", False), False),
        ("Cmaj", ("C", False), False),
        ("CM", ("C", False), False),
        ("Cm", ("Cm", False), False),
        ("Cmin", ("Cm", False), False),
        ("C7", ("C7", False), False),
        ("Cm7", ("Cm7", False), False),
        ("Cmin7", ("Cm7", False), False),
        ("Cmaj7", ("Cmaj7", False), False),
        ("CM7", ("Cmaj7", False), False),
        ("CΔ", ("Cmaj7", False), False),
        ("Cm7b5", ("Cm7b5", False), False),
        ("Cø", ("Cm7b5", False), False),
        ("Cdim", ("Cdim", False), False),
        ("C°", ("Cdim", False), False),
        ("Csus", ("Csus4", False), False),
        ("Csus4", ("Csus4", False), False),
        ("C5", ("C5", False), False),
        ("D#5", ("D#5", False), False),
        ("Csus2", ("C", True), True),
        ("Caug", ("C", True), True),
        ("C+", ("C", True), True),
        ("C9", ("C7", True), True),
        ("Cadd9", ("C7", True), True),
        ("C11", ("C7", True), True),
        ("C13", ("C7", True), True),
        ("", ("", False), False),
        ("N", ("", False), False),
        ("C/E", ("C/E", False), False),
        ("G/B", ("G/B", False), False),
        ("Cmaj7/E", ("Cmaj7/E", False), False),
    ],
)
def test_normalize_chord_label(raw, expected, warned):
    out, warn = normalize_chord_label(raw)
    assert (out, warn) == expected


def test_normalize_vocabulary_emits_warning():
    payload = {
        "mode": "real",
        "warning": "",
        "segments": [
            {"start": 0.0, "end": 2.0, "chord": "C"},
            {"start": 2.0, "end": 4.0, "chord": "Csus2"},
            {"start": 4.0, "end": 6.0, "chord": "Cadd9"},
            {"start": 6.0, "end": 8.0, "chord": "Cadd9"},
            {"start": 8.0, "end": 10.0, "chord": "N"},
        ],
    }
    out = normalize_chord_vocabulary(payload)
    assert [s["chord"] for s in out["segments"]] == ["C", "C", "C7", "C7"]
    assert "Csus2" in out["warning"]
    assert "Cadd9" in out["warning"]

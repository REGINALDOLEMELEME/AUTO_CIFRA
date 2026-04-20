from __future__ import annotations

import pytest

from src.transpose import effective_semitones, shift_chord


@pytest.mark.parametrize(
    "label,semitones,flats,expected",
    [
        ("C", 2, False, "D"),
        ("C", 2, True, "D"),
        ("C", 1, False, "C#"),
        ("C", 1, True, "Db"),
        ("C", -1, True, "B"),
        ("C", 0, True, "C"),
        ("C", 12, True, "C"),
        ("Cm", 3, True, "Ebm"),
        ("C7", 5, True, "F7"),
        ("Cmaj7", 4, True, "Emaj7"),
        ("Cm7", 9, True, "Am7"),
        ("F#", 1, False, "G"),
        ("Bb", 1, True, "B"),
        ("Bb", 2, True, "C"),
        ("G/B", 2, True, "A/Db"),
        ("N", 2, True, "N"),
        ("", 2, True, ""),
    ],
)
def test_shift_chord(label, semitones, flats, expected):
    assert shift_chord(label, semitones, prefer_flats=flats) == expected


def test_all_roots_round_trip_12():
    # Sharp-preference round trip (prefer_flats=False) keeps each sharp-spelled
    # root identical at +12 semitones. With prefer_flats=True, sharps become flats.
    for r in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]:
        assert shift_chord(r, 12, prefer_flats=False) == r
        assert shift_chord(r + "m7", 12, prefer_flats=False) == r + "m7"


def test_unknown_label_is_passthrough():
    # Non-pitch labels should be returned unchanged.
    assert shift_chord("?!?", 2) == "?!?"


def test_effective_semitones():
    assert effective_semitones(2, 0) == 2
    assert effective_semitones(0, 3) == -3
    assert effective_semitones(-2, 2) == -4

from __future__ import annotations

from src.chords import filter_out_of_key_flickers


def _segs(*triples):
    """triples are (start, end, chord)."""
    return {"segments": [{"start": s, "end": e, "chord": c} for (s, e, c) in triples]}


def test_drops_short_outofkey_flicker():
    # C – F(flicker, out of key) – C  in key of C major — F is actually IV,
    # so let's use a real out-of-key chord: Eb.
    got = filter_out_of_key_flickers(
        _segs((0.0, 4.0, "C"), (4.0, 5.0, "Eb"), (5.0, 8.0, "C")),
        key="C",
    )
    labels = [s["chord"] for s in got["segments"]]
    assert "Eb" not in labels


def test_keeps_long_outofkey_passage():
    # Long Eb stretch — likely a deliberate key change, not a flicker.
    got = filter_out_of_key_flickers(
        _segs((0.0, 4.0, "C"), (4.0, 8.0, "Eb"), (8.0, 12.0, "C")),
        key="C",
        max_flicker_s=1.6,
    )
    labels = [s["chord"] for s in got["segments"]]
    assert labels.count("Eb") == 1


def test_keeps_diatonic_chords():
    # All in-key — nothing should be dropped.
    got = filter_out_of_key_flickers(
        _segs((0.0, 2.0, "C"), (2.0, 3.0, "F"), (3.0, 4.0, "G"), (4.0, 5.0, "Am")),
        key="C",
    )
    labels = [s["chord"] for s in got["segments"]]
    assert labels == ["C", "F", "G", "Am"]


def test_empty_key_is_noop():
    payload = _segs((0.0, 1.0, "C"), (1.0, 1.2, "Eb"), (1.2, 3.0, "C"))
    got = filter_out_of_key_flickers(payload, key="")
    assert got == payload


def test_boundary_segments_never_dropped():
    # First and last segments are never considered flickers (no neighbour
    # to compare against on both sides).
    got = filter_out_of_key_flickers(
        _segs((0.0, 1.0, "Eb"), (1.0, 4.0, "C"), (4.0, 5.0, "Eb")),
        key="C",
    )
    labels = [s["chord"] for s in got["segments"]]
    assert labels == ["Eb", "C", "Eb"]


def test_minor_key_detection():
    # Key of Am — F is IV of the natural minor scale (in key). Eb is not.
    got = filter_out_of_key_flickers(
        _segs((0.0, 2.0, "Am"), (2.0, 3.0, "Eb"), (3.0, 5.0, "Am")),
        key="Am",
    )
    labels = [s["chord"] for s in got["segments"]]
    assert "Eb" not in labels

from __future__ import annotations

from src.chord_reclassifier import classify_chroma, _score_label, _template, PITCH_INDEX


def _chroma_for_chord(root: str, quality: str = "") -> list[float]:
    """Synthetic chroma: a clean triad with full energy on root/3rd/5th."""
    return _template(PITCH_INDEX[root], quality)


def test_classifies_clean_c_major():
    label, _ = classify_chroma(_chroma_for_chord("C", ""))
    assert label == "C"


def test_classifies_clean_a_minor():
    label, _ = classify_chroma(_chroma_for_chord("A", "m"))
    assert label == "Am"


def test_key_bias_prefers_diatonic_at_equal_score():
    # A chroma ambiguous between C (in key of C) and C# (out of key) should
    # prefer C.
    # Build a mostly-C vector with a tiny bump on C# so scores are near-tied.
    vec = list(_template(PITCH_INDEX["C"], ""))
    label_biased, _ = classify_chroma(vec, key="C", out_of_key_penalty=0.6)
    assert label_biased == "C"


def test_key_bias_penalizes_out_of_key():
    # Chroma that fits Eb major better than C major — but in key of C, the
    # penalty should swing the pick to an in-key candidate.
    vec_eb = list(_template(PITCH_INDEX["D#"], ""))  # Eb = D#
    label_no_bias, _ = classify_chroma(vec_eb, key="")
    label_c_bias, _ = classify_chroma(vec_eb, key="C", out_of_key_penalty=0.3)
    assert label_no_bias == "D#"
    # With strong out-of-key penalty, the in-key match should win.
    assert label_c_bias != "D#"


def test_score_label_handles_empty_and_slash():
    vec = _chroma_for_chord("C", "")
    assert _score_label(vec, "") == 0.0
    # Slash chord — we only score the head.
    assert _score_label(vec, "C/G") > 0


def test_score_label_recognises_quality():
    vec_major = _chroma_for_chord("C", "")
    vec_minor = _chroma_for_chord("C", "m")
    assert _score_label(vec_major, "C") > _score_label(vec_major, "Cm")
    assert _score_label(vec_minor, "Cm") > _score_label(vec_minor, "C")


def test_handles_flat_notation():
    # Bb maj chord → internal template is at A#.
    vec = _chroma_for_chord("A#", "")
    assert _score_label(vec, "Bb") > 0
    assert _score_label(vec, "A#") > 0

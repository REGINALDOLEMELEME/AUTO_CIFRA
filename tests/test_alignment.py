from __future__ import annotations

from src.alignment import align_chords_by_word_time


def test_simple_word_time_alignment():
    transcription = {
        "source_file": "x.mp3",
        "segments": [
            {
                "start": 0.0,
                "end": 3.0,
                "text": "Eu te amo",
                "words": [
                    {"word": "Eu", "start": 0.00, "end": 0.30},
                    {"word": "te", "start": 0.35, "end": 0.55},
                    {"word": "amo", "start": 0.60, "end": 1.10},
                ],
            }
        ],
    }
    chords = {
        "segments": [
            {"start": 0.0, "end": 0.30, "chord": "C"},
            {"start": 0.30, "end": 0.60, "chord": "G"},
            {"start": 0.60, "end": 1.20, "chord": "Am"},
        ]
    }
    aligned = align_chords_by_word_time(transcription, chords)
    assert len(aligned["lines"]) == 1
    line = aligned["lines"][0]
    assert line["words"][0]["chord"] == "C"
    assert line["words"][1]["chord"] == "G"
    assert line["words"][2]["chord"] == "Am"


def test_multiple_chord_changes_within_one_word_attach_first():
    transcription = {
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "solo",
                "words": [{"word": "solo", "start": 0.0, "end": 2.0}],
            }
        ]
    }
    chords = {
        "segments": [
            {"start": 0.0, "end": 0.5, "chord": "C"},
            {"start": 0.5, "end": 1.5, "chord": "G"},
            {"start": 1.5, "end": 2.0, "chord": "Am"},
        ]
    }
    aligned = align_chords_by_word_time(transcription, chords)
    # The word should be decorated with its starting chord, "C".
    assert aligned["lines"][0]["words"][0]["chord"] == "C"


def test_no_chord_duplicates_across_words():
    transcription = {
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "a b c",
                "words": [
                    {"word": "a", "start": 0.0, "end": 0.3},
                    {"word": "b", "start": 0.3, "end": 0.6},
                    {"word": "c", "start": 0.6, "end": 1.0},
                ],
            }
        ]
    }
    chords = {"segments": [{"start": 0.0, "end": 1.0, "chord": "C"}]}
    aligned = align_chords_by_word_time(transcription, chords)
    words = aligned["lines"][0]["words"]
    assert words[0]["chord"] == "C"
    assert words[1]["chord"] is None
    assert words[2]["chord"] is None


def test_section_label_is_emitted_once_per_section():
    transcription = {
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "line one",
                "words": [{"word": "line", "start": 0.0, "end": 0.4}, {"word": "one", "start": 0.4, "end": 1.0}],
            },
            {
                "start": 1.5,
                "end": 2.5,
                "text": "line two",
                "words": [{"word": "line", "start": 1.5, "end": 1.9}, {"word": "two", "start": 1.9, "end": 2.5}],
            },
        ]
    }
    sections = [{"start": 0.0, "end": 3.0, "label": "Verso 1", "confidence": 0.8}]
    aligned = align_chords_by_word_time(transcription, {"segments": []}, sections)
    assert aligned["lines"][0]["section"] == "Verso 1"
    assert aligned["lines"][1]["section"] is None


def test_empty_inputs_produce_empty_output():
    aligned = align_chords_by_word_time(
        {"source_file": "x.mp3", "segments": []}, {"segments": []}
    )
    assert aligned["lines"] == []

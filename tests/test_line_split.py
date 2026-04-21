from __future__ import annotations

from src.docx_export import _split_line_on_punctuation


def _mk(text: str) -> list[dict]:
    return [{"text": t} for t in text.split()]


def test_short_line_passes_through():
    w = _mk("três palavras curtas")
    chunks = _split_line_on_punctuation(w, max_words=12)
    assert len(chunks) == 1
    assert chunks[0] == w


def test_splits_on_commas():
    # "Tire suas mãos de mim, eu não pertenço a você, não é me dominando..."
    words = _mk(
        "Tire suas mãos de mim, eu não pertenço a você, "
        "não é me dominando assim, que você vai me entender, eu"
    )
    chunks = _split_line_on_punctuation(words, max_words=10)
    # No chunk should exceed 10 words.
    assert all(len(c) <= 10 for c in chunks)
    # Each chunk except the last should end on a comma (greedy breaks at
    # the LAST punctuation within the window — this maximises phrase size).
    for c in chunks[:-1]:
        assert c[-1]["text"].endswith(","), f"chunk didn't end on comma: {[w['text'] for w in c]}"


def test_splits_on_period():
    words = _mk("um dois três quatro cinco. seis sete oito nove dez.")
    chunks = _split_line_on_punctuation(words, max_words=6)
    # Split at the period, not a hard cut.
    assert chunks[0][-1]["text"] == "cinco."


def test_hard_split_when_no_punctuation():
    # Long line with no punctuation — must still be split to stay readable.
    words = _mk(" ".join(["palavra"] * 30))
    chunks = _split_line_on_punctuation(words, max_words=10)
    assert len(chunks) == 3
    assert all(len(c) <= 10 for c in chunks)


def test_preserves_word_order():
    words = _mk("a, b, c, d, e, f, g, h")
    chunks = _split_line_on_punctuation(words, max_words=3)
    flat = [w for chunk in chunks for w in chunk]
    assert flat == words


def test_edge_single_word_line():
    words = _mk("misericórdia")
    assert _split_line_on_punctuation(words, max_words=12) == [words]


def test_no_empty_chunks():
    words = _mk("dois, quatro, seis, oito, dez")
    chunks = _split_line_on_punctuation(words, max_words=3)
    assert all(c for c in chunks)

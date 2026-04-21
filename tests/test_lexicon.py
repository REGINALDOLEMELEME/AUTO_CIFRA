from __future__ import annotations

from src.lexicon import apply_lexicon


def _segments(*pairs):
    """Build a fake transcription dict with (text, [words]) tuples."""
    return {
        "segments": [
            {
                "text": text,
                "words": [{"word": w} for w in words],
            }
            for text, words in pairs
        ]
    }


def test_castro_to_casto():
    t = _segments(("Obediente Castro e associada", ["Castro"]))
    apply_lexicon(t)
    assert t["segments"][0]["text"] == "Obediente casto e associada"
    assert t["segments"][0]["words"][0]["word"] == "casto"


def test_experiencia_variants_to_experienciar():
    cases = [
        ("Experiência a sua misericórdia", "experienciar sua misericórdia"),
        ("Experiencia sua misericórdia", "experienciar sua misericórdia"),
        ("experiência sua misericórdia", "experienciar sua misericórdia"),
    ]
    for src, expected in cases:
        t = _segments((src, []))
        apply_lexicon(t)
        assert t["segments"][0]["text"] == expected, (src, t["segments"][0]["text"])


def test_tenha_protecao_to_tem_a_protecao():
    t = _segments(("Santa Clara tenha proteção e de Maria", []))
    apply_lexicon(t)
    assert t["segments"][0]["text"] == "Santa Clara tem a proteção e de Maria"


def test_antonio_em_sao_to_antonio_e_sao():
    t = _segments(("Santo Antônio em São Francisco", ["em"]))
    apply_lexicon(t)
    assert t["segments"][0]["text"] == "Santo Antônio e São Francisco"


def test_idempotent_on_clean_text():
    t = _segments(
        ("Na juventude, o Senhor encontrou", []),
        ("experienciar sua misericórdia", []),
    )
    apply_lexicon(t)
    assert t["segments"][0]["text"] == "Na juventude, o Senhor encontrou"
    assert t["segments"][1]["text"] == "experienciar sua misericórdia"


def test_marker_set():
    t = _segments(("anything", []))
    apply_lexicon(t)
    assert t.get("lexicon_applied") is True


def test_sos_franciscana_becomes_sois_franciscana():
    # Contextual: only rewrite "Sós" → "Sois" when followed by "franciscana".
    t = _segments(("sós franciscana da misericórdia", ["sós", "franciscana"]))
    apply_lexicon(t)
    assert t["segments"][0]["text"] == "Sois Franciscana da misericórdia"


def test_sos_alone_is_not_rewritten():
    # Real PT-BR word "sós" (= alone, plural) must survive outside the refrain.
    t = _segments(("nós ficamos sós na sala", []))
    apply_lexicon(t)
    assert t["segments"][0]["text"] == "nós ficamos sós na sala"


def test_vigar_becomes_ficar_at_both_levels():
    # Must rewrite BOTH segment.text and the per-word tokens so the DOCX
    # table cells show "Ficar" (not "Vigar"). Earlier multi-word rule only
    # hit the segment text and left the word tokens stale.
    t = _segments(("Vigar pra quê se é sem querer", ["Vigar", "pra", "quê"]))
    apply_lexicon(t)
    assert t["segments"][0]["text"].startswith("Ficar")
    assert t["segments"][0]["words"][0]["word"] == "Ficar"


def test_handles_missing_words_key():
    t = {"segments": [{"text": "Obediente Castro"}]}
    apply_lexicon(t)
    assert t["segments"][0]["text"] == "Obediente casto"

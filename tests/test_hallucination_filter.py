from __future__ import annotations

from src.transcription import _looks_like_hallucinated_credit


def test_detects_legenda_credit():
    assert _looks_like_hallucinated_credit("Legenda Adriana Zanotto")
    assert _looks_like_hallucinated_credit("Legendas por João Silva")
    assert _looks_like_hallucinated_credit("Legendado por Maria")


def test_detects_youtube_tail():
    assert _looks_like_hallucinated_credit("Obrigado por assistir!")
    assert _looks_like_hallucinated_credit("Curta e se inscreva no canal")


def test_detects_lone_tchau():
    # Whisper emits "Tchau!" on quiet intro bars on rock tracks.
    assert _looks_like_hallucinated_credit("Tchau!")
    assert _looks_like_hallucinated_credit("Tchau.")
    assert _looks_like_hallucinated_credit("tchau")
    # But "tchau" inside a real lyric must survive.
    assert not _looks_like_hallucinated_credit("Então tchau, até mais")


def test_detects_cidade_no_brasil():
    # Repeatable PT Whisper hallucination on low-energy intros; observed on
    # São Francisco and Legião Urbana's "Será".
    assert _looks_like_hallucinated_credit("A CIDADE NO BRASIL")
    assert _looks_like_hallucinated_credit("a cidade no brasil.")
    # But lyrics that mention cities should survive:
    assert not _looks_like_hallucinated_credit("A cidade no Brasil inteiro escuta essa canção")


def test_detects_musica_hallucinations():
    # Whisper emits "Música religiosa", "Música de fundo" etc. during silent
    # stretches. Observed on São Francisco tail after silence post-last-lyric.
    assert _looks_like_hallucinated_credit("Música religiosa.")
    assert _looks_like_hallucinated_credit("Música de fundo")
    assert _looks_like_hallucinated_credit("Música instrumental")
    assert _looks_like_hallucinated_credit("Música eletrônica")
    assert _looks_like_hallucinated_credit("Música clássica")
    assert _looks_like_hallucinated_credit("Música.")
    assert _looks_like_hallucinated_credit("música")
    assert _looks_like_hallucinated_credit("Som ambiente")


def test_musica_in_lyric_context_not_filtered():
    # But "A música do coração" is a legitimate lyric — leading "música"
    # alone (or followed by a noun matching our hallucination list) triggers;
    # other contexts do not.
    assert not _looks_like_hallucinated_credit("A música do coração")
    assert not _looks_like_hallucinated_credit("Tocamos nossa música juntos")


def test_detects_subtitulos_variants():
    assert _looks_like_hallucinated_credit("Subtítulos: equipe X")
    assert _looks_like_hallucinated_credit("Tradução e legenda: Y")


def test_accepts_real_lyrics():
    # Must NOT filter anything that is actually sung vocabulary
    assert not _looks_like_hallucinated_credit(
        "És Franciscana da misericórdia, chamada a ser entre os mais pobres"
    )
    assert not _looks_like_hallucinated_credit(
        "Na juventude, o Senhor encontrou, por Ele tudo deixou"
    )
    assert not _looks_like_hallucinated_credit("sagrado coração de Jesus")


def test_empty_text_is_not_credit():
    assert not _looks_like_hallucinated_credit("")
    assert not _looks_like_hallucinated_credit("   ")


def test_lone_amen_filtered():
    # Whisper emits "Amém." when fed a silent buffer — drop it as an artefact.
    assert _looks_like_hallucinated_credit("Amém.")
    assert _looks_like_hallucinated_credit("amém")

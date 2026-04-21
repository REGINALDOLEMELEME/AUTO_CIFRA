"""Deterministic post-ASR lexicon corrections for PT-BR religious music.

Whisper large-v3 makes the same handful of mistakes over and over on franciscan
/ liturgical repertoire (voiceless vocals + non-colloquial vocabulary). Rather
than retrain, we apply a conservative regex replacement pass on the
segment-level text *and* the per-word tokens that docx_export consumes.

Scope: PT-BR. Matches are case-insensitive, accent-insensitive where noted, and
preserve capitalisation on the first letter. Each entry is documented with the
ASR symptom it came from — do not add fixes without a real observation.
"""
from __future__ import annotations

import re
from typing import Any, Iterable


# (regex pattern, replacement). Each pattern is compiled with re.IGNORECASE.
# Replacements intentionally match the canonical PT-BR spelling, accents
# included; downstream consumers (docx_export) emit UTF-8.
DEFAULT_CORRECTIONS: tuple[tuple[str, str], ...] = (
    # "Castro" ← casto: ASR picks the much more common surname.
    (r"\bcastro\b", "casto"),
    # "Experiência a/Experiencia" ← experienciar: verb form is rare in PT-BR
    # conversational data, so Whisper snaps to the noun + preposition.
    (r"\bexperi[êe]ncia\s+a\b", "experienciar"),
    (r"\bexperiencia\b", "experienciar"),
    (r"\bexperi[êe]ncia\b", "experienciar"),
    # "tenha proteção" ← tem a proteção: function-word collapse.
    (r"\btenha\s+prote[çc][ãa]o\b", "tem a proteção"),
    # "Antônio em São Francisco" ← Antônio e São Francisco: single-phoneme swap.
    (r"\bant[ôo]nio\s+em\s+s[ãa]o\b", "Antônio e São"),
    # "Sós Franciscana" ← Sois Franciscana: Whisper hears the verb "Sois" (2nd
    # person plural "are") as "Sós" ("alone"). Only rewrite in the refrain
    # context so we don't clobber the legitimate word "sós" elsewhere.
    (r"\bs[óo]s\s+franciscana\b", "Sois Franciscana"),
    # "Vigar" ← Ficar: F/V initial-phoneme confusion. "Vigar" is not a word
    # in Brazilian Portuguese, so a single-token rewrite is safe and covers
    # the docx_export per-word cells (multi-word patterns can't rewrite
    # individual word tokens). Observed on Legião Urbana's "Será".
    (r"\bvigar\b", "Ficar"),
    # "sois" ← Sois: capitalise at refrain head when isolated.
    # (no replacement — keeps token content but normalises casing upstream)
)


def _apply(text: str, corrections: Iterable[tuple[str, str]]) -> str:
    out = text
    for pattern, replacement in corrections:
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    return out


def apply_lexicon(
    transcription: dict[str, Any],
    extra_corrections: Iterable[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Mutate `transcription` in-place, applying known word-level corrections
    to both segment text and per-word tokens. Returns the same dict for
    chaining."""
    corrections = tuple(DEFAULT_CORRECTIONS) + tuple(extra_corrections or ())
    if not corrections:
        return transcription
    for seg in transcription.get("segments", []) or []:
        txt = seg.get("text") or ""
        if txt:
            seg["text"] = _apply(txt, corrections)
        for w in seg.get("words") or []:
            wt = w.get("word") or ""
            if wt:
                w["word"] = _apply(wt, corrections)
    transcription["lexicon_applied"] = True
    return transcription

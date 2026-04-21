from __future__ import annotations

from dataclasses import dataclass, asdict, field
import json
from pathlib import Path
from typing import Any

import re

from .audio import normalize_audio
from .models import get_whisper


# Whisper PT-BR is partially trained on TV subtitle corpora and consistently
# emits subtitle-credit strings during silent / near-silent intervals. These
# are not present in the actual audio — they are memorised corpus artefacts.
# Drop any segment whose text matches (case-insensitive, accent-insensitive).
_SUBTITLE_HALLUCINATION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, flags=re.IGNORECASE)
    for p in (
        r"^\s*legenda[sd]?\b",           # "Legenda Adriana Zanotto", "Legendas por..."
        r"^\s*legendado\b",
        r"^\s*subt[íi]tulos?\b",
        r"^\s*tradu[çc][ãa]o\s+(e\s+)?legenda",
        r"^\s*revis[ãa]o\b.*\blegenda",
        r"^\s*transcri[çc][ãa]o\b.*\b(por|de)\b",
        r"^\s*obrigad[oa]\s+por\s+assistir",     # YouTube-tail hallucination
        r"curta\s+(e\s+)?(se\s+)?inscreva",
        r"^\s*www\.",
        r"^\s*am[eé]m\.?$",                      # single-word fallback when fed silence
        # "Música religiosa", "Música de fundo", "Música instrumental" —
        # standard Whisper emissions for unrecognised music during silent
        # intervals. Observed on the São Francisco tail (v4 run).
        r"^\s*m[úu]sica\s+(religios[ao]|de\s+fundo|instrumental|eletr[ôo]nica|ambiente|cl[áa]ssica|cl[áa]ssico)",
        r"^\s*m[úu]sica\s*\.?\s*$",              # lone "Música."
        r"^\s*som\s+ambiente",
        # "A CIDADE NO BRASIL" — specific PT Whisper hallucination pattern
        # observed on multiple songs during low-energy intros/outros. The
        # phrase is not a real lyric; match only when it's the entire
        # segment text so legitimate uses (e.g. a song that actually contains
        # this line) aren't filtered.
        r"^\s*a\s+cidade\s+no\s+brasil\.?\s*$",
        # Whisper sometimes ends a transcription with "Tchau!" or "Tchau."
        # when fed the quiet tail of an intro bar. Only filter when it's
        # the ENTIRE segment (so "tchau" inside a real lyric survives).
        r"^\s*tchau[!\.]?\s*$",
    )
)


def _looks_like_hallucinated_credit(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return any(p.search(t) for p in _SUBTITLE_HALLUCINATION_PATTERNS)


@dataclass
class LyricSegment:
    start: float
    end: float
    text: str
    words: list[dict[str, Any]] = field(default_factory=list)


def _collect_segments(whisper_segments, *, avg_logprob_floor: float = -1.0,
                      no_speech_cap: float = 0.6) -> list[LyricSegment]:
    """Shared post-processing: apply hallucination filters and return clean
    LyricSegment objects. Used by both the primary transcription pass and the
    tail-fallback pass so both benefit from the same quality gates."""
    items: list[LyricSegment] = []
    for segment in whisper_segments:
        text = (segment.text or "").strip()
        if not text:
            continue
        avg_logprob = float(getattr(segment, "avg_logprob", 0.0) or 0.0)
        no_speech_prob = float(getattr(segment, "no_speech_prob", 0.0) or 0.0)
        compression_ratio = float(getattr(segment, "compression_ratio", 1.0) or 1.0)
        if avg_logprob < avg_logprob_floor and no_speech_prob > no_speech_cap:
            continue
        if compression_ratio > 2.4:
            continue
        if _looks_like_hallucinated_credit(text):
            continue
        words: list[dict[str, Any]] = []
        for w in (getattr(segment, "words", None) or []):
            wt = (getattr(w, "word", "") or "").strip()
            if not wt:
                continue
            w_start = float(getattr(w, "start", segment.start) or segment.start)
            w_end = float(getattr(w, "end", w_start) or w_start)
            words.append(
                {"start": round(w_start, 3), "end": round(w_end, 3), "word": wt}
            )
        items.append(
            LyricSegment(
                start=round(float(segment.start), 3),
                end=round(float(segment.end), 3),
                text=text,
                words=words,
            )
        )
    return items


def transcribe_audio(
    input_audio: Path,
    tmp_dir: Path,
    language: str = "pt",
    model_size: str = "large-v3",
    use_vad: bool = False,
    compute_type: str = "int8",
    device: str = "cpu",
    initial_prompt: str | None = None,
    hotwords: str | None = None,
) -> dict[str, Any]:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    normalized_audio = tmp_dir / f"{input_audio.stem}.normalized.wav"
    normalize_audio(input_audio=input_audio, output_audio=normalized_audio)

    model, err = get_whisper(model_size=model_size, device=device, compute_type=compute_type)
    if model is None or err:
        raise RuntimeError(err or "faster-whisper is not available")

    # Deterministic + hallucination-resistant settings:
    #   - temperature=0.0 + single-temp list force greedy decoding (no sampling
    #     fallback pass that would reintroduce randomness).
    #   - condition_on_previous_text=False prevents a single bad early segment
    #     from contaminating downstream segmentation (saw 16 vs 7 segments on
    #     byte-identical re-runs when this was enabled).
    #   - no_speech_threshold=0.5 (was 0.7): still trims silent Demucs pre-roll
    #     but preserves quiet tails — the final refrain was being dropped on
    #     "São Francisco da Misericórdia".
    #   - log_prob_threshold / compression_ratio_threshold: drop garbage and
    #     repetitive hallucinated segments at the token level.
    #   - hallucination_silence_threshold: faster-whisper ≥ 1.0 specific —
    #     skip segments whose timings sit inside a silence window.
    #   - initial_prompt / hotwords: bias vocabulary towards the song's lexicon
    #     (religious / franciscan terms) so we stop getting "Castro" for casto,
    #     "Experiência" for experienciar, etc.
    kwargs: dict[str, Any] = dict(
        language=language,
        vad_filter=use_vad,
        word_timestamps=True,
        beam_size=5,
        temperature=[0.0],
        condition_on_previous_text=False,
        no_speech_threshold=0.5,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,
    )
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt
    # hotwords and hallucination_silence_threshold were added in newer
    # faster-whisper releases; feature-detect instead of hard-requiring them.
    import inspect

    transcribe_params = inspect.signature(model.transcribe).parameters
    if hotwords and "hotwords" in transcribe_params:
        kwargs["hotwords"] = hotwords
    if "hallucination_silence_threshold" in transcribe_params:
        kwargs["hallucination_silence_threshold"] = 2.0

    segments, info = model.transcribe(str(normalized_audio), **kwargs)
    items = _collect_segments(segments)

    return {
        "source_file": input_audio.name,
        "normalized_audio": str(normalized_audio),
        "language": getattr(info, "language", language),
        "language_probability": float(getattr(info, "language_probability", 0.0)),
        "duration": float(getattr(info, "duration", 0.0)),
        "segments": [asdict(item) for item in items],
        "mode": "real",
        "model_size": model_size,
    }


def transcribe_clip(
    input_audio: Path,
    tmp_dir: Path,
    start_s: float,
    end_s: float,
    language: str = "pt",
    model_size: str = "large-v3",
    compute_type: str = "int8",
    device: str = "cpu",
    initial_prompt: str | None = None,
    hotwords: str | None = None,
    chunk_window_s: float = 6.0,
    chunk_overlap_s: float = 1.0,
) -> list[dict[str, Any]]:
    """Transcribe only the [start_s, end_s] window of `input_audio`.

    Primary use: tail-fallback and head-fallback on the **original mix**
    (not the Demucs-separated vocals). Demucs's Wiener filter over-suppresses
    quiet tails, and the separated track can make Whisper hallucinate long
    single segments (e.g. a 30-second "A CIDADE NO BRASIL" covering the
    whole intro of Legião Urbana's "Será"). Re-running Whisper on the mix
    recovers the real lyrics.

    Returns a list of segment dicts with absolute timestamps.

    **Chunking**: when the window exceeds `chunk_window_s` and chunking is
    enabled (chunk_window_s > 0), we split the window into overlapping
    chunks and transcribe each independently. This defeats Whisper's
    tendency to short-circuit a long window to just the recognized song
    title (saw exactly this on Legião Urbana: single-window emitted only
    "Tire Suas Mãos de Mim"; 5s chunks pulled the full verse).

    Thresholds are slightly looser than the primary pass because the
    fallback usually runs on quieter or harder-to-transcribe windows; the
    subtitle/music hallucination filter still runs so corpus artefacts
    ("Música", "Legenda...") don't leak through.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    normalized_audio = tmp_dir / f"{input_audio.stem}.tailfallback.wav"
    normalize_audio(input_audio=input_audio, output_audio=normalized_audio)

    model, err = get_whisper(model_size=model_size, device=device, compute_type=compute_type)
    if model is None or err:
        raise RuntimeError(err or "faster-whisper is not available")

    kwargs: dict[str, Any] = dict(
        language=language,
        vad_filter=False,
        word_timestamps=True,
        beam_size=5,
        temperature=[0.0],
        condition_on_previous_text=False,
        no_speech_threshold=0.3,
        log_prob_threshold=-1.5,
        compression_ratio_threshold=2.4,
    )
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt

    import inspect

    params = inspect.signature(model.transcribe).parameters
    if hotwords and "hotwords" in params:
        kwargs["hotwords"] = hotwords
    span = max(0.0, float(end_s) - float(start_s))
    if (
        chunk_window_s
        and chunk_window_s > 0
        and span > chunk_window_s
        and "clip_timestamps" in params
    ):
        # Chunked mode: break [start, end] into overlapping windows and
        # transcribe each separately. Measured +1200% lyric recovery on
        # Legião Urbana's "Será" intro (1 segment → 4+ segments of real
        # verse text) vs a single long clip.
        all_items: list[LyricSegment] = []
        step = max(1.0, chunk_window_s - chunk_overlap_s)
        t = float(start_s)
        while t < float(end_s):
            chunk_end = min(t + chunk_window_s, float(end_s))
            chunk_kwargs = dict(kwargs, clip_timestamps=[t, chunk_end])
            # condition_on_previous_text improves continuity within a chunk
            # without risking contamination across far-apart regions since
            # chunks are short.
            chunk_kwargs["condition_on_previous_text"] = True
            segs, _ = model.transcribe(str(normalized_audio), **chunk_kwargs)
            all_items.extend(
                _collect_segments(segs, avg_logprob_floor=-1.2, no_speech_cap=0.7)
            )
            t += step
        # De-dup overlapping emissions by (start, text). Sort by start.
        seen: set[tuple[float, str]] = set()
        unique: list[LyricSegment] = []
        for it in sorted(all_items, key=lambda x: x.start):
            key = (round(it.start, 1), it.text.strip().lower())
            if key in seen:
                continue
            seen.add(key)
            unique.append(it)
        return [asdict(item) for item in unique]

    # Single-window mode (short clips don't need chunking).
    if "clip_timestamps" in params:
        kwargs["clip_timestamps"] = [float(start_s), float(end_s)]
    else:
        # Older faster-whisper: fall back to a re-encoded slice via ffmpeg.
        import subprocess

        sliced = tmp_dir / f"{input_audio.stem}.tailfallback.slice.wav"
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(normalized_audio),
                "-ss", f"{start_s:.3f}",
                "-to", f"{end_s:.3f}",
                "-c", "copy",
                str(sliced),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        normalized_audio = sliced

    segments, _ = model.transcribe(str(normalized_audio), **kwargs)
    items = _collect_segments(segments, avg_logprob_floor=-1.2, no_speech_cap=0.7)
    return [asdict(item) for item in items]


def write_transcription_json(result: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path

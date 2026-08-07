"""Verbatimize Chirp-3 transcripts with CrisperWhisper 2.0.

CrisperWhisper 2.0 exposes a dedicated verbatimize task: given audio and a
trusted clean ("intended") transcript, the model reproduces that transcript
word-for-word and inserts only the disfluencies and vocal events actually
present in the audio. That is exactly the Chirp-3 upgrade path -- unlike
re-transcribing, it keeps Chirp's rare-word and proper-noun recall, and
unlike forced alignment it can add words the reference never contained.

The one hard constraint is length: verbatimize takes a single decoder prompt
with no longform strategy, and the model warns above 30 s of audio. Chirp-3
transcripts carry per-word offsets, so the session is split into sub-30 s
windows whose audio slice and intended text are cut at the same points, each
window is verbatimized independently, and the results are concatenated.

Window audio boundaries fall at the midpoint of the silence between windows
so every sample is covered exactly once, except where Chirp left a long gap
(a failed chunk); there the slice hugs the transcribed words instead, so a
window never drags in minutes of untranscribed audio the intended text does
not account for.

Speaker labels come from Chirp: each output word is aligned back to the
window's input words, matched words inherit that speaker, and inserted
disfluencies inherit the speaker of the word they follow.
"""

from __future__ import annotations

import difflib
import logging
from pathlib import Path
from typing import Any

import numpy as np
from crisperwhisper.audio import SAMPLE_RATE, load_audio
from crisperwhisper.forced_align import default_normalize

logger = logging.getLogger(__name__)

# Verbatimize is a single-prompt task; the model warns past 30 s. Leave
# headroom so the padded audio slice also stays under the threshold.
MAX_WINDOW_SECONDS = 26.0
# A pause this long is a natural boundary -- splitting there keeps window
# edges inside silence, where a cut cannot truncate a word.
SPLIT_GAP_SECONDS = 2.0
MIN_WINDOW_SECONDS = 3.0
# Words per window. Output is longer than input (that is the point), so this
# stays well inside the token budget.
MAX_WINDOW_WORDS = 90
SLICE_PAD_SECONDS = 0.25

# A window whose output falls outside these ratios of the input word count is
# treated as a failed decode and the original Chirp words are kept instead.
MIN_OUTPUT_RATIO = 0.5
MAX_OUTPUT_RATIO = 3.0

ORIGIN_CHIRP = "chirp"
ORIGIN_INSERTED = "inserted"
ORIGIN_FALLBACK = "chirp-fallback"


def build_windows(
    words: list[dict],
    duration: float,
    *,
    max_window: float = MAX_WINDOW_SECONDS,
    max_words: int = MAX_WINDOW_WORDS,
    split_gap: float = SPLIT_GAP_SECONDS,
    min_window: float = MIN_WINDOW_SECONDS,
    pad: float = SLICE_PAD_SECONDS,
) -> list[dict]:
    """Split Chirp words into sub-30 s windows with matching audio slices.

    Returns a list of
        {"words": [...], "text": str,
         "word_start": float, "word_end": float,
         "audio_start": float, "audio_end": float}
    covering every input word exactly once, in order.
    """
    if not words:
        return []

    groups: list[list[dict]] = [[words[0]]]
    for previous, word in zip(words, words[1:]):
        current = groups[-1]
        span = word["end"] - current[0]["start"]
        gap = word["start"] - previous["end"]
        span_full = len(current) >= max_words or span > max_window
        at_pause = gap >= split_gap and (previous["end"] - current[0]["start"]) >= min_window
        if span_full or at_pause:
            groups.append([word])
        else:
            current.append(word)

    windows: list[dict] = []
    for index, group in enumerate(groups):
        word_start, word_end = group[0]["start"], group[-1]["end"]
        previous_end = groups[index - 1][-1]["end"] if index else 0.0
        next_start = groups[index + 1][0]["start"] if index + 1 < len(groups) else duration

        # Midpoint of the surrounding silence, but never reach further than
        # `pad` beyond the words themselves (guards Chirp's long gaps).
        audio_start = max(word_start - pad, (previous_end + word_start) / 2.0)
        audio_end = min(word_end + pad, (word_end + next_start) / 2.0)
        audio_start = max(0.0, min(audio_start, word_start))
        audio_end = min(duration, max(audio_end, word_end))

        windows.append(
            {
                "words": group,
                "text": " ".join(w["word"].strip() for w in group),
                "word_start": word_start,
                "word_end": word_end,
                "audio_start": audio_start,
                "audio_end": audio_end,
            }
        )
    return windows


def attach_speakers(
    chirp_words: list[dict], output_words: list[dict]
) -> tuple[list[dict], int]:
    """Carry Chirp speaker labels onto verbatimized words.

    Aligns the two word sequences with difflib. Matched and substituted words
    take the speaker of their Chirp counterpart and are tagged
    ``origin="chirp"``; words with no counterpart are the newly recovered
    disfluencies, tagged ``origin="inserted"`` and given the speaker of the
    nearest labelled neighbour.

    Returns (words, dropped) where `dropped` counts Chirp words the model
    failed to reproduce -- verbatimize is supposed to be content-preserving,
    so a nonzero count is a quality signal worth reporting.
    """
    reference = [default_normalize(w["word"]) for w in chirp_words]
    hypothesis = [default_normalize(w["word"]) for w in output_words]

    speakers: list[str | None] = [None] * len(output_words)
    origins = [ORIGIN_INSERTED] * len(output_words)
    dropped = 0

    matcher = difflib.SequenceMatcher(a=reference, b=hypothesis, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for di, dj in zip(range(i1, i2), range(j1, j2)):
                speakers[dj] = chirp_words[di]["speaker"]
                origins[dj] = ORIGIN_CHIRP
        elif tag == "replace":
            # A substitution occupies the same slot as the words it replaced,
            # so take the speaker positionally and clamp at the span end.
            for offset, dj in enumerate(range(j1, j2)):
                speakers[dj] = chirp_words[min(i1 + offset, i2 - 1)]["speaker"]
                origins[dj] = ORIGIN_CHIRP
        elif tag == "delete":
            dropped += i2 - i1
        # "insert" leaves the disfluency unlabelled; filled in below.

    default = chirp_words[0]["speaker"] if chirp_words else "SPEAKER_UNKNOWN"
    last = None
    for index, speaker in enumerate(speakers):
        if speaker is None:
            speakers[index] = last
        else:
            last = speaker
    following = default
    for index in range(len(speakers) - 1, -1, -1):
        if speakers[index] is None:
            speakers[index] = following
        else:
            following = speakers[index]

    for word, speaker, origin in zip(output_words, speakers, origins):
        word["speaker"] = speaker
        word["origin"] = origin
    return output_words, dropped


def _fallback_words(window: dict) -> list[dict]:
    return [
        {
            "word": w["word"],
            "start": w["start"],
            "end": w["end"],
            "speaker": w["speaker"],
            "origin": ORIGIN_FALLBACK,
        }
        for w in window["words"]
    ]


def verbatimize_window(
    model,
    audio: np.ndarray,
    window: dict,
    *,
    language: str = "en",
    max_new_tokens: int = 448,
) -> tuple[list[dict], bool]:
    """Verbatimize one window. Returns (words, ok); ok=False means fallback."""
    start_sample = int(window["audio_start"] * SAMPLE_RATE)
    end_sample = int(window["audio_end"] * SAMPLE_RATE)
    clip = audio[start_sample:end_sample]
    if clip.size == 0:
        return _fallback_words(window), False

    result = model.verbatimize(
        clip,
        window["text"],
        language=language,
        sr=SAMPLE_RATE,
        word_timestamps=True,
        max_new_tokens=max_new_tokens,
    )
    produced = result.words or []
    expected = len(window["words"])
    if not produced or not (
        MIN_OUTPUT_RATIO * expected <= len(produced) <= MAX_OUTPUT_RATIO * expected
    ):
        logger.warning(
            "Window %.1f-%.1fs produced %d words for %d input words; "
            "keeping the Chirp text for this window",
            window["audio_start"], window["audio_end"], len(produced), expected,
        )
        return _fallback_words(window), False

    offset = window["audio_start"]
    limit = window["audio_end"]
    words = []
    for word in produced:
        start = offset + float(word.start or 0.0)
        end = offset + float(word.end or 0.0)
        words.append(
            {
                "word": word.word,
                "start": round(min(max(start, offset), limit), 3),
                "end": round(min(max(end, start), limit), 3),
            }
        )
    return words, True


def verbatimize_session(
    model,
    audio_path: str | Path,
    chirp_words: list[dict],
    duration: float,
    *,
    language: str = "en",
    max_window: float = MAX_WINDOW_SECONDS,
    max_new_tokens: int = 448,
) -> dict[str, Any]:
    """Verbatimize a whole session against its Chirp-3 transcript.

    Returns:
        {"words": [{"word", "start", "end", "speaker", "origin"}, ...],
         "text": str, "duration": float, "stats": {...}}
    """
    audio_path = Path(audio_path)
    audio = load_audio(str(audio_path))
    duration = duration or len(audio) / SAMPLE_RATE

    windows = build_windows(chirp_words, duration, max_window=max_window)
    logger.info(
        "%s: %d Chirp words -> %d window(s) over %.1fs",
        audio_path.name, len(chirp_words), len(windows), duration,
    )

    words: list[dict] = []
    failed = 0
    dropped_total = 0
    for index, window in enumerate(windows, start=1):
        try:
            produced, ok = verbatimize_window(
                model, audio, window,
                language=language, max_new_tokens=max_new_tokens,
            )
        except Exception:
            logger.exception(
                "Verbatimize failed on window %d (%.1f-%.1fs); keeping Chirp text",
                index, window["audio_start"], window["audio_end"],
            )
            produced, ok = _fallback_words(window), False

        if ok:
            produced, dropped = attach_speakers(window["words"], produced)
            dropped_total += dropped
        else:
            failed += 1
        words.extend(produced)

        if index % 25 == 0 or index == len(windows):
            logger.info("  window %d/%d", index, len(windows))

    inserted = sum(1 for w in words if w["origin"] == ORIGIN_INSERTED)
    stats = {
        "windows": len(windows),
        "windows_fallback": failed,
        "chirp_words": len(chirp_words),
        "verbatim_words": len(words),
        "inserted_words": inserted,
        "dropped_chirp_words": dropped_total,
        "insertion_rate": round(inserted / len(words), 4) if words else 0.0,
    }
    logger.info("%s: %s", audio_path.name, stats)

    return {
        "words": words,
        "text": " ".join(w["word"].strip() for w in words),
        "duration": duration,
        "stats": stats,
    }

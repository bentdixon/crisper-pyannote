"""Reader for Chirp-3 word-level transcripts (final_gcp_transcripts/*.json).

The AMPSCZ bucket stores each Chirp-3 session transcript as a JSON document
with a flat word list and a metadata block:

    {"words": [{"startOffset": 1.6, "endOffset": 2.48,
                "word": "Right.", "speakerLabel": "0"}, ...],
     "metadata": {"site": ..., "subject": ..., "interview": ...,
                  "total_audio_length": 1213.056, "num_chunks": 2,
                  "overlap_sec": 80, "overlap_regions": [...]}}

Words are normalized to the pipeline's plain-dict convention
({"word", "start", "end", "speaker"}) so the rest of the pipeline does not
need to know about the Chirp field names.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SPEAKER_PREFIX = "SPEAKER_"


def format_speaker(label: Any) -> str:
    """Render a Chirp speakerLabel as a pipeline speaker id.

    Chirp emits bare integer strings ("0", "1", "2"); the pipeline uses
    pyannote-style SPEAKER_00 labels everywhere else.
    """
    text = str(label).strip()
    if text.isdigit():
        return f"{SPEAKER_PREFIX}{int(text):02d}"
    return text or f"{SPEAKER_PREFIX}UNKNOWN"


def load_transcript(path: str | Path) -> dict[str, Any]:
    """Load one Chirp-3 transcript.

    Returns:
        {"words": [{"word", "start", "end", "speaker"}, ...],
         "metadata": {...},
         "duration": float | None}

    Words are sorted by start time. Entries with missing or non-monotonic
    offsets are repaired rather than dropped: a word that ends before it
    starts is given a zero-length span, which the windower tolerates.
    """
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    metadata = raw.get("metadata") or {}

    words: list[dict[str, Any]] = []
    for entry in raw.get("words") or []:
        text = (entry.get("word") or "").strip()
        if not text:
            continue
        start = float(entry.get("startOffset") or 0.0)
        end = float(entry.get("endOffset") or start)
        words.append(
            {
                "word": text,
                "start": start,
                "end": max(start, end),
                "speaker": format_speaker(entry.get("speakerLabel")),
            }
        )
    words.sort(key=lambda w: (w["start"], w["end"]))

    duration = metadata.get("total_audio_length")
    duration = float(duration) if duration else None
    if duration is None and words:
        duration = words[-1]["end"]

    logger.info(
        "Loaded %s: %d Chirp words, %d speaker(s), duration %s",
        path.name, len(words), len({w["speaker"] for w in words}),
        f"{duration:.1f}s" if duration else "unknown",
    )
    return {"words": words, "metadata": metadata, "duration": duration}

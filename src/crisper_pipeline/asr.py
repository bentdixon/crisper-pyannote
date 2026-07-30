"""CrisperWhisper 2.0 ASR: verbatim transcription with word-level timestamps."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from crisperwhisper import CrisperWhisperModel

logger = logging.getLogger(__name__)


def load_model(
    model_name: str = "large",
    *,
    backend: str = "ct2",
    draft_model: str | None = "turbo",
    compute_type: str = "float16",
    device: str = "auto",
    device_index: int = 0,
) -> CrisperWhisperModel:
    """Load CrisperWhisper 2.0 with the CTranslate2 backend.

    The default configuration pairs the "large" model with a "turbo" draft
    model so speculative decoding is available at transcribe time.
    """
    logger.info(
        "Loading CrisperWhisper model=%s backend=%s draft=%s compute_type=%s",
        model_name, backend, draft_model, compute_type,
    )
    return CrisperWhisperModel(
        model_name,
        backend=backend,
        draft_model=draft_model,
        compute_type=compute_type,
        device=device,
        device_index=device_index,
    )


def transcribe(
    model: CrisperWhisperModel,
    audio_path: str | Path,
    *,
    language: str = "en",
    speculative_decoding: bool = True,
) -> dict[str, Any]:
    """Transcribe a wav file in verbatim mode with word-level timestamps.

    Returns a plain dict:
        {
            "text": str,
            "language": str,
            "duration": float,
            "processing_time": float,
            "words": [{"word": str, "start": float, "end": float}, ...],
        }
    """
    audio_path = Path(audio_path)
    logger.info("Transcribing %s", audio_path)
    result = model.transcribe(
        str(audio_path),
        language=language,
        mode="verbatim",
        word_timestamps=True,
        speculative_decoding=speculative_decoding,
    )
    words = [
        {"word": w.word, "start": float(w.start), "end": float(w.end)}
        for w in (result.words or [])
    ]
    return {
        "text": result.text,
        "language": result.language,
        "duration": float(result.duration),
        "processing_time": float(result.processing_time),
        "words": words,
    }

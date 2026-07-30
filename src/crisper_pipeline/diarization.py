"""Speaker diarization with pyannote/speaker-diarization-community-1."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from pyannote.audio import Pipeline

logger = logging.getLogger(__name__)

DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"


def load_pipeline(token: str | None = None, device: str | None = None) -> Pipeline:
    """Load the community-1 diarization pipeline and move it to GPU if available.

    Authentication: pass a HuggingFace token explicitly, or rely on the
    HF_TOKEN environment variable / cached `huggingface-cli login` credentials.
    The model is gated, so its terms must be accepted on huggingface.co first.
    """
    token = token or os.environ.get("HF_TOKEN")
    logger.info("Loading diarization pipeline %s", DIARIZATION_MODEL)
    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=token)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(torch.device(device))
    logger.info("Diarization pipeline on %s", device)
    return pipeline


def diarize(
    pipeline: Pipeline,
    audio_path: str | Path,
    *,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[dict]:
    """Diarize a wav file and return exclusive speaker segments.

    Uses the pipeline's exclusive speaker diarization (non-overlapping
    segments), which the pyannoteAI merge tutorial recommends for
    reconciliation with ASR timestamps.

    Returns a list sorted by start time:
        [{"start": float, "end": float, "speaker": str}, ...]
    """
    audio_path = Path(audio_path)
    kwargs: dict = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    logger.info("Diarizing %s %s", audio_path, kwargs or "")
    output = pipeline(str(audio_path), **kwargs)
    annotation = output.exclusive_speaker_diarization

    segments = [
        {"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)}
        for turn, _track, speaker in annotation.itertracks(yield_label=True)
    ]
    segments.sort(key=lambda s: s["start"])
    logger.info(
        "Found %d segments across %d speakers",
        len(segments), len({s["speaker"] for s in segments}),
    )
    return segments

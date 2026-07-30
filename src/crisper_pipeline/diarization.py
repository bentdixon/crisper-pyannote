"""Speaker diarization with pyannote/speaker-diarization-community-1."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import soundfile as sf
import torch

from .cuda_preload import preload_torchcodec_libs

preload_torchcodec_libs()

from pyannote.audio import Pipeline  # noqa: E402

logger = logging.getLogger(__name__)

DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"


def load_pipeline(
    token: str | None = None,
    device: str | None = None,
    model: str | None = None,
) -> Pipeline:
    """Load a diarization pipeline and move it to GPU if available.

    model may be a HuggingFace pipeline id or a local config.yaml path (for
    example one produced by finetune/optimize_pipeline.py); it defaults to
    stock community-1.

    Authentication: pass a HuggingFace token explicitly, or rely on the
    HF_TOKEN environment variable / cached `hf auth login` credentials.
    The model is gated, so its terms must be accepted on huggingface.co first.
    """
    token = token or os.environ.get("HF_TOKEN")
    model = model or DIARIZATION_MODEL
    logger.info("Loading diarization pipeline %s", model)
    pipeline = Pipeline.from_pretrained(model, token=token)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(torch.device(device))
    logger.info("Diarization pipeline on %s", device)
    return pipeline


def load_audio(audio_path: str | Path) -> dict:
    """Load a wav file into the in-memory format pyannote pipelines accept.

    Bypasses pyannote's built-in torchcodec decoding, which requires a
    torchcodec build matching the installed CUDA/FFmpeg stack. The pipeline
    downmixes and resamples as needed.
    """
    data, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)  # (channel, time)
    return {"waveform": waveform, "sample_rate": sample_rate}


def diarize(
    pipeline: Pipeline,
    audio_path: str | Path,
    *,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    exclusive: bool = True,
) -> list[dict]:
    """Diarize a wav file and return speaker segments.

    With exclusive=True (default), uses the pipeline's exclusive speaker
    diarization (non-overlapping segments), which the pyannoteAI merge
    tutorial recommends for reconciliation with ASR timestamps. With
    exclusive=False, returns the raw diarization, where segments from
    different speakers may overlap.

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

    logger.info(
        "Diarizing %s (%s) %s",
        audio_path, "exclusive" if exclusive else "overlapping", kwargs or "",
    )
    output = pipeline(load_audio(audio_path), **kwargs)
    annotation = (
        output.exclusive_speaker_diarization if exclusive
        else output.speaker_diarization
    )

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

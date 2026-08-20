"""CrisperWhisper 2.0 ASR: verbatim transcription with word-level timestamps.

Longform audio is transcribed as a sequence of sub-30 s windows rather than
through the model's own longform strategy. Measured on a 1094 s interview
against two independent references (the other team's VAD-segment pipeline:
1752 words; Chirp-3: ~1730):

    longform_strategy="continuation"   651 words   <- the model default
    ... with stride 20                 802
    ... with stride 15                1203
    strategy="chunked_lcs"            1637        (no word timestamps)
    windowed short-form               (this)

Each 30 s continuation chunk emitted only ~16 words regardless of stride --
dense speech in 30 s is 60-90 -- so chunks were ending early and shrinking the
stride merely packed in more of them. The same audio transcribed in isolation
as 25 s clips came back complete, so the loss is in longform stitching, not in
decoding. "chunked_lcs" and "token_lcs" recover the content but raise
NotImplementedError with word_timestamps=True, which every downstream stage
here needs, so windowing is done here instead.
"""

from __future__ import annotations

import math
import time

import logging
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from crisperwhisper import CrisperWhisperModel

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
MAX_WINDOW_SECONDS = 25.0
WINDOW_PAD_SECONDS = 0.2
VAD_THRESHOLD = 0.3
SHORT_AUDIO_SECONDS = 30.0


def _load_mono(audio_path: str | Path) -> tuple[np.ndarray, int]:
    """Read a wav as mono float32 at the model's sample rate."""
    data, rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if rate != SAMPLE_RATE:
        # Linear resample: adequate for 16 kHz speech and avoids pulling in a
        # resampling dependency just for the rare non-16k file.
        target_len = int(round(len(mono) * SAMPLE_RATE / rate))
        mono = np.interp(
            np.linspace(0.0, len(mono), target_len, endpoint=False),
            np.arange(len(mono)),
            mono,
        ).astype("float32")
        rate = SAMPLE_RATE
    return mono, rate


def speech_windows(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    max_window: float = MAX_WINDOW_SECONDS,
    threshold: float = VAD_THRESHOLD,
) -> list[tuple[float, float]]:
    """Speech spans merged into windows of at most max_window seconds.

    Cuts fall in silence between VAD spans wherever possible; a single speech
    span longer than the cap is split hard, since the encoder cannot see more
    than 30 s at once either way.
    """
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad",
        force_reload=False, onnx=False, verbose=False,
    )
    get_speech_timestamps = utils[0]
    spans = get_speech_timestamps(
        torch.from_numpy(audio), model, sampling_rate=sample_rate, threshold=threshold
    )
    if not spans:
        return []

    windows: list[tuple[float, float]] = []
    start = end = None
    for span in spans:
        span_start = span["start"] / sample_rate
        span_end = span["end"] / sample_rate
        while span_end - span_start > max_window:
            windows.append((span_start, span_start + max_window))
            span_start += max_window
        if start is None:
            start, end = span_start, span_end
        elif span_end - start <= max_window:
            end = span_end
        else:
            windows.append((start, end))
            start, end = span_start, span_end
    if start is not None:
        windows.append((start, end))
    return windows


# The other team's segmentation constants, ported with their values so the
# comparison is of architecture rather than of tuning. Their cap splits an
# over-long segment into equal pieces regardless of where the silence is;
# speech_windows seeks a pause instead, and that difference is deliberate --
# reproducing their order should not quietly import our cut-point choice.
SEGMENT_PAD_SECONDS = 0.3
SEGMENT_MAX_GAP = 1.0


def merge_segments(
    segments: list[dict],
    max_segment: float = MAX_WINDOW_SECONDS,
    max_gap: float = SEGMENT_MAX_GAP,
) -> list[dict]:
    """Merge consecutive same-speaker diarization segments, then cap the length.

    Port of `_merge_and_cap` in the other team's transcription_core, on our
    dict convention rather than their (start, end, speaker) tuples. Merging
    matters because raw diarization emits many short adjacent turns for one
    speaker, and transcribing each in isolation denies the model the context
    that makes a one-word answer decodable.
    """
    if not segments:
        return []
    ordered = sorted(segments, key=lambda s: s["start"])
    merged: list[dict] = [dict(ordered[0])]
    for segment in ordered[1:]:
        current = merged[-1]
        gap = segment["start"] - current["end"]
        span = segment["end"] - current["start"]
        if (
            segment["speaker"] == current["speaker"]
            and span <= max_segment
            and gap <= max_gap
        ):
            current["end"] = segment["end"]
        else:
            merged.append(dict(segment))

    capped: list[dict] = []
    for segment in merged:
        span = segment["end"] - segment["start"]
        if span <= max_segment:
            capped.append(segment)
            continue
        pieces = math.ceil(span / max_segment)
        length = span / pieces
        for index in range(pieces):
            start = segment["start"] + index * length
            capped.append({
                "start": start,
                "end": min(start + length, segment["end"]),
                "speaker": segment["speaker"],
            })
    return capped


def transcribe_segments(
    model: CrisperWhisperModel,
    audio_path: str | Path,
    segments: list[dict],
    *,
    language: str = "en",
    speculative_decoding: bool = False,
    pad: float = SEGMENT_PAD_SECONDS,
) -> dict[str, Any]:
    """Transcribe each diarization segment, taking its speaker from the segment.

    The other team's order: diarize the whole file, then transcribe segment by
    segment. Two consequences distinguish it from transcribe_windowed. Speech
    the diarizer did not mark is never transcribed -- the same exposure that
    silero windowing has, but pyannote's segmentation catches brief turns that
    silero's 250 ms minimum discards. And every word arrives already attributed,
    so merge.assign_speakers is not involved and no word can land in the
    UNKNOWN bucket.
    """
    audio, rate = _load_mono(audio_path)
    duration = len(audio) / rate
    segments = merge_segments(segments)
    if not segments:
        logger.warning("No diarization segments for %s", audio_path)
        return {
            "text": "", "language": language, "duration": duration,
            "processing_time": 0.0, "words": [],
        }

    logger.info(
        "Transcribing %s in %d segment(s) (%.0f s)",
        Path(audio_path).name, len(segments), duration,
    )
    words: list[dict] = []
    pieces: list[str] = []
    started = time.time()
    for index, segment in enumerate(segments, start=1):
        lo = max(0.0, segment["start"] - pad)
        hi = min(duration, segment["end"] + pad)
        clip = audio[int(lo * rate):int(hi * rate)]
        if clip.size == 0:
            continue
        try:
            result = model.transcribe(
                clip, sr=rate, language=language, mode="verbatim",
                word_timestamps=True, speculative_decoding=speculative_decoding,
            )
        except Exception:
            logger.exception("  segment %d/%d failed; skipping", index, len(segments))
            continue
        if result.text:
            pieces.append(result.text.strip())
        for word in result.words or []:
            start = lo + float(word.start)
            end = lo + float(word.end)
            words.append({
                "word": word.word,
                "start": max(start, 0.0),
                "end": max(end, start),
                "speaker": segment["speaker"],
            })
        if index % 25 == 0 or index == len(segments):
            logger.info("  segment %d/%d", index, len(segments))

    words.sort(key=lambda w: w["start"])
    return {
        "text": " ".join(pieces),
        "language": language,
        "duration": duration,
        "processing_time": time.time() - started,
        "words": words,
    }


def transcribe_windowed(
    model: CrisperWhisperModel,
    audio_path: str | Path,
    *,
    language: str = "en",
    speculative_decoding: bool = False,
    max_window: float = MAX_WINDOW_SECONDS,
) -> dict[str, Any]:
    """Transcribe by short-form windows, offsetting each window's timestamps.

    Every window is under the 30 s encoder limit, so the model's longform
    stitching -- which drops most of the transcript on this corpus -- is never
    engaged.
    """
    audio, rate = _load_mono(audio_path)
    duration = len(audio) / rate
    windows = speech_windows(audio, rate, max_window=max_window)
    if not windows:
        logger.warning("No speech detected in %s", audio_path)
        return {
            "text": "", "language": language, "duration": duration,
            "processing_time": 0.0, "words": [],
        }

    logger.info(
        "Transcribing %s in %d window(s) (%.0f s)", Path(audio_path).name, len(windows), duration
    )
    words: list[dict] = []
    pieces: list[str] = []
    started = time.time()
    for index, (start, end) in enumerate(windows, start=1):
        lo = max(0.0, start - WINDOW_PAD_SECONDS)
        hi = min(duration, end + WINDOW_PAD_SECONDS)
        clip = audio[int(lo * rate):int(hi * rate)]
        if clip.size == 0:
            continue
        try:
            result = model.transcribe(
                clip, sr=rate, language=language, mode="verbatim",
                word_timestamps=True, speculative_decoding=speculative_decoding,
            )
        except Exception:
            logger.exception("  window %d/%d failed; skipping", index, len(windows))
            continue
        if result.text:
            pieces.append(result.text.strip())
        for word in result.words or []:
            # Clamp into the unpadded window so the pad cannot let a word from
            # the neighbouring window be emitted twice.
            word_start = lo + float(word.start)
            word_end = lo + float(word.end)
            if word_end < start or word_start > end:
                continue
            words.append({
                "word": word.word,
                "start": max(word_start, 0.0),
                "end": max(word_end, word_start),
            })
        if index % 25 == 0 or index == len(windows):
            logger.info("  window %d/%d", index, len(windows))

    words.sort(key=lambda w: w["start"])
    return {
        "text": " ".join(pieces),
        "language": language,
        "duration": duration,
        "processing_time": time.time() - started,
        "words": words,
    }


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
    longform: str = "windowed",
    segments: list[dict] | None = None,
) -> dict[str, Any]:
    """Transcribe a wav file in verbatim mode with word-level timestamps.

    longform selects how audio over 30 s is handled: "windowed" (default)
    transcribes sub-30 s speech windows and offsets their timestamps;
    "continuation" uses the model's own longform strategy, which drops most of
    the transcript on this corpus (see the module docstring).

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

    if longform == "diarization":
        if not segments:
            raise ValueError(
                "longform='diarization' needs the diarization segments; run "
                "diarization before ASR in this mode"
            )
        return transcribe_segments(
            model, audio_path, segments, language=language,
            speculative_decoding=speculative_decoding,
        )

    if longform == "windowed":
        with sf.SoundFile(str(audio_path)) as handle:
            duration = len(handle) / handle.samplerate
        if duration > SHORT_AUDIO_SECONDS:
            return transcribe_windowed(
                model, audio_path, language=language,
                speculative_decoding=speculative_decoding,
            )

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

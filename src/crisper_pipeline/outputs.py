"""Writers for word-level JSON and human-readable transcripts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm."""
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    logger.info("Wrote %s", path)


def _render_turns(turns: list[dict]) -> str:
    lines = [
        f"[{format_timestamp(t['start'])} - {format_timestamp(t['end'])}] "
        f"{t['speaker']}: {t['text']}"
        for t in turns
    ]
    return "\n".join(lines) + "\n"


def write_outputs(
    output_dir: str | Path,
    audio_path: str | Path,
    transcript: dict,
    diarization_segments: list[dict],
    turns: list[dict],
    metadata: dict,
) -> Path:
    """Write all pipeline outputs for one audio file.

    Each run gets its own directory named <audio stem>_<run timestamp>, so
    transcribing the same file twice produces two separate outputs. Layout
    (under <output_dir>/<audio stem>_<timestamp>/):
        metadata.json          run timestamp plus transcription and
                               diarization settings
        transcript.json        full word-level transcript with speakers
        transcript.txt         human-readable, speaker-attributed transcript
        diarization.json       raw exclusive diarization segments
        speakers/<SPK>.json    word-level JSON per participant
        speakers/<SPK>.txt     human-readable transcript per participant
    """
    audio_path = Path(audio_path)
    stamp = metadata["run_timestamp_compact"]
    session_dir = Path(output_dir) / f"{audio_path.stem}_{stamp}"
    suffix = 1
    while session_dir.exists():
        suffix += 1
        session_dir = Path(output_dir) / f"{audio_path.stem}_{stamp}-{suffix}"
    speakers_dir = session_dir / "speakers"
    speakers_dir.mkdir(parents=True)

    words = transcript["words"]

    _write_json(session_dir / "metadata.json", metadata)

    _write_json(
        session_dir / "transcript.json",
        {
            "audio": audio_path.name,
            "language": transcript["language"],
            "duration": transcript["duration"],
            "text": transcript["text"],
            "words": words,
        },
    )
    _write_json(session_dir / "diarization.json", diarization_segments)
    (session_dir / "transcript.txt").write_text(_render_turns(turns))
    logger.info("Wrote %s", session_dir / "transcript.txt")

    speakers = sorted({w["speaker"] for w in words})
    for speaker in speakers:
        speaker_words = [w for w in words if w["speaker"] == speaker]
        speaker_turns = [t for t in turns if t["speaker"] == speaker]
        _write_json(
            speakers_dir / f"{speaker}.json",
            {
                "audio": audio_path.name,
                "speaker": speaker,
                "num_words": len(speaker_words),
                "words": speaker_words,
            },
        )
        (speakers_dir / f"{speaker}.txt").write_text(_render_turns(speaker_turns))
        logger.info("Wrote per-speaker outputs for %s", speaker)

    return session_dir

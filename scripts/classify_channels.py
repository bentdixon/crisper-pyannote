"""Classify every cohort visit by the diarization path its audio would take.

The mono-versus-stereo comparison needs to know which sessions the other team's
pipeline actually diarized from channel dominance rather than from pyannote,
because that is the subset where their speaker labels never came from a model at
all. The sweep logged that decision per file, but those logs are gone, so it is
recomputed here.

The decision is not reimplemented: `has_real_channel_separation` and
`get_vad_ranges` are imported from baseline/transcription_core.py and run with
their own threshold. A container reporting two channels says nothing about
whether the channels carry different speakers -- one of our files measured 0.03
dB average separation against a 3.0 dB threshold -- so the VAD pass is what
distinguishes real stereo from a stereo wrapper.

Writes {visit: {channels, avg_db_separation, method}} where method is one of
channel_dominance | pyannote_fallback_no_real_stereo_separation | pyannote,
matching the strings transcription_core reports.

Usage:
    uv run python scripts/classify_channels.py --cohort /path/to/cohort \
        --output outputs/channels.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baseline"))

import transcription_core as core  # noqa: E402

logger = logging.getLogger("classify_channels")


def channel_count(path: Path) -> int:
    """Channel count from the wav header, without decoding the audio."""
    with wave.open(str(path)) as handle:
        return handle.getnchannels()


def separation_db(left, right, left_ranges, right_ranges) -> float | None:
    """The number has_real_channel_separation thresholds, for the record.

    That function prints its average and returns a bool; the comparison needs
    the value itself so borderline files are visible rather than assumed
    stable, so the same arithmetic is repeated here over the same windows.
    """
    ranges = sorted(left_ranges + right_ranges, key=lambda r: r[0])
    diffs = []
    for start, end in ranges:
        start_ms, end_ms = int(start * 1000), int(end * 1000)
        left_db = left[start_ms:end_ms].dBFS
        right_db = right[start_ms:end_ms].dBFS
        if left_db == float("-inf") or right_db == float("-inf"):
            continue
        diffs.append(abs(left_db - right_db))
    return sum(diffs) / len(diffs) if diffs else None


def classify(audio_path: Path, work_dir: Path) -> dict:
    channels = channel_count(audio_path)
    if channels == 1:
        return {"channels": 1, "avg_db_separation": None, "method": "pyannote"}

    split = core.load_audio_and_check_channels(str(audio_path))
    if len(split) != 2:
        return {"channels": channels, "avg_db_separation": None, "method": "pyannote"}

    left, right = split[0], split[1]
    left_ranges = core.get_vad_ranges(left, str(work_dir / "_vad_left.wav"))
    right_ranges = core.get_vad_ranges(right, str(work_dir / "_vad_right.wav"))
    real = core.has_real_channel_separation(left_ranges, right_ranges, left, right)
    return {
        "channels": 2,
        "avg_db_separation": separation_db(left, right, left_ranges, right_ranges),
        "method": (
            "channel_dominance" if real
            else "pyannote_fallback_no_real_stereo_separation"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    cohort = Path(args.cohort)
    visits = sorted(
        p for p in cohort.glob("Pronet*/*/*")
        if (p / "human").is_dir() and any((p / "human").glob("*.txt"))
        and any((p / "audio").glob("*.wav"))
    )
    if args.limit:
        visits = visits[: args.limit]
    logger.info("Classifying %d visit(s)", len(visits))

    out: dict[str, dict] = {}
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        for index, visit in enumerate(visits, start=1):
            relative = visit.relative_to(cohort).as_posix()
            audio = sorted((visit / "audio").glob("*.wav"))[0]
            try:
                out[relative] = classify(audio, work_dir)
            except Exception as error:
                # Counted, not swallowed: a file that fails to classify would
                # otherwise silently land in the mono condition by absence.
                failures += 1
                logger.error("  %s: %s: %s", relative, type(error).__name__, error)
                out[relative] = {
                    "channels": None, "avg_db_separation": None, "method": "error",
                }
            if index % 10 == 0 or index == len(visits):
                logger.info("  %d/%d", index, len(visits))

    counts: dict[str, int] = {}
    for entry in out.values():
        counts[entry["method"]] = counts.get(entry["method"], 0) + 1
    logger.info("methods: %s", counts)
    if failures:
        logger.error("%d visit(s) failed to classify", failures)

    borderline = sorted(
        (e["avg_db_separation"], v) for v, e in out.items()
        if e.get("avg_db_separation") is not None
    )
    if borderline:
        logger.info(
            "channel separation: min %.2f dB, max %.2f dB, threshold %.1f dB",
            borderline[0][0], borderline[-1][0], core.MIN_AVG_DB_SEPARATION,
        )
        near = [
            (round(db, 2), v) for db, v in borderline
            if abs(db - core.MIN_AVG_DB_SEPARATION) < 1.0
        ]
        if near:
            # Within 1 dB of the cut, so the original sweep could have decided
            # these differently. Named rather than counted.
            logger.warning("within 1 dB of the threshold: %s", near)

    Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    logger.info("wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

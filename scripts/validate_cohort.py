"""Check that a staged cohort really has usable audio, Chirp and human data.

A session counts as "complete" during fetching only because a file exists in
each of the three folders. That is not the same as the file being usable, so
this opens every one and checks it:

    audio   RIFF/WAVE header parses; sample rate, channels and duration
            recovered from the header rather than trusted
    chirp   JSON parses, has a non-empty word list, offsets run forwards,
            and its declared duration agrees with the wav
    human   parses into timestamped turns, turn times run forwards and stay
            inside the audio, and it carries a plausible amount of text

Reports per-session problems and a summary, so a session that is complete on
paper but unusable in practice shows up before it reaches a training or
scoring run. Prints counts and durations only, never transcript text.

Usage:
    uv run python scripts/validate_cohort.py --root /path/to/Chirp3_PSYCHS_NDA4
"""

from __future__ import annotations

import argparse
import json
import logging
import struct
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "finetune"))

from prepare_data import load_timestamped_text  # noqa: E402

logger = logging.getLogger("validate_cohort")

# A clinical interview shorter than this is almost certainly a stub.
MIN_DURATION_S = 60.0
MIN_HUMAN_WORDS = 50
# Chirp's declared length and the wav header should agree closely.
DURATION_TOLERANCE_S = 5.0


def wav_info(path: Path) -> dict:
    """Sample rate, channels and duration from the wav header alone."""
    with path.open("rb") as handle:
        header = handle.read(12)
        if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise ValueError("not a RIFF/WAVE file")
        fmt = None
        data_bytes = None
        while True:
            chunk = handle.read(8)
            if len(chunk) < 8:
                break
            chunk_id, size = struct.unpack("<4sI", chunk)
            if chunk_id == b"fmt ":
                body = handle.read(size)
                _, channels, rate, byte_rate, _, bits = struct.unpack("<HHIIHH", body[:16])
                fmt = (channels, rate, bits, byte_rate)
            elif chunk_id == b"data":
                data_bytes = size
                break
            else:
                handle.seek(size, 1)
    if not fmt:
        raise ValueError("no fmt chunk")
    channels, rate, bits, byte_rate = fmt
    if data_bytes is None:
        data_bytes = path.stat().st_size - 44
    return {
        "channels": channels,
        "sample_rate": rate,
        "bits": bits,
        "duration": data_bytes / byte_rate if byte_rate else 0.0,
    }


def check_session(visit: Path) -> tuple[dict, list[str]]:
    """Validate one <SITE>/<SUBJECT>/<visit>/ directory."""
    problems: list[str] = []
    facts: dict = {"visit": str(visit)}

    def only(folder: str, suffix: str) -> Path | None:
        found = sorted((visit / folder).glob(f"*{suffix}"))
        if not found:
            problems.append(f"missing {folder}")
            return None
        if len(found) > 1:
            problems.append(f"{len(found)} files in {folder}")
        return found[0]

    audio = only("audio", ".wav")
    chirp_path = only("chirp", ".json")
    human_path = only("human", ".txt")

    duration = None
    if audio:
        try:
            info = wav_info(audio)
            duration = info["duration"]
            facts.update(info)
            if duration < MIN_DURATION_S:
                problems.append(f"audio only {duration:.0f}s")
        except Exception as error:
            problems.append(f"audio unreadable: {error}")

    if chirp_path:
        try:
            document = json.loads(chirp_path.read_text())
            words = document.get("words") or []
            facts["chirp_words"] = len(words)
            if not words:
                problems.append("chirp has no words")
            else:
                starts = [float(w.get("startOffset") or 0.0) for w in words]
                if any(b < a for a, b in zip(starts, starts[1:])):
                    problems.append("chirp offsets not monotonic")
                declared = document.get("metadata", {}).get("total_audio_length")
                if declared and duration and abs(float(declared) - duration) > DURATION_TOLERANCE_S:
                    problems.append(
                        f"chirp duration {float(declared):.0f}s vs wav {duration:.0f}s"
                    )
                if duration:
                    facts["chirp_wpm"] = round(len(words) / (duration / 60.0), 1)
        except Exception as error:
            problems.append(f"chirp unreadable: {error}")

    if human_path:
        try:
            turns = load_timestamped_text(human_path, duration or 0.0)
            words = sum(len(t["text"].split()) for t in turns)
            facts["human_turns"] = len(turns)
            facts["human_words"] = words
            facts["human_speakers"] = len({t["speaker"] for t in turns})
            if words < MIN_HUMAN_WORDS:
                problems.append(f"human transcript only {words} words")
            starts = [t["start"] for t in turns]
            if any(b < a for a, b in zip(starts, starts[1:])):
                problems.append("human turn times not monotonic")
            if duration and starts and starts[-1] > duration + DURATION_TOLERANCE_S:
                problems.append(f"human turn at {starts[-1]:.0f}s past audio end {duration:.0f}s")
        except Exception as error:
            problems.append(f"human unreadable: {error}")

    return facts, problems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="staged cohort root")
    parser.add_argument("--output", default=None, help="write full results JSON here")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    root = Path(args.root)

    visits = sorted(
        path for path in root.glob("Pronet*/*/*")
        if path.is_dir() and (path / "audio").is_dir()
    )
    logger.info("Validating %d visit(s) under %s", len(visits), root)

    results = []
    flagged = []
    for visit in visits:
        facts, problems = check_session(visit)
        facts["problems"] = problems
        results.append(facts)
        if problems:
            flagged.append((visit.relative_to(root), problems))

    print(f"\n  visits checked : {len(results)}")
    print(f"  clean          : {len(results) - len(flagged)}")
    print(f"  with problems  : {len(flagged)}")
    if results:
        hours = sum(r.get("duration", 0.0) for r in results) / 3600
        print(f"  audio          : {hours:.1f} h")
        print(f"  participants   : {len({Path(r['visit']).parent.name for r in results})}")
        print(f"  sites          : {len({Path(r['visit']).parent.parent.name for r in results})}")
        rates = Counter()
        for r in results:
            for key in ("sample_rate", "channels"):
                if key in r:
                    rates[f"{key}={r[key]}"] += 1
        print(f"  formats        : {dict(rates)}")
    for visit, problems in flagged:
        print(f"    {visit}: {'; '.join(problems)}")

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
        logger.info("Wrote %s", args.output)
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())

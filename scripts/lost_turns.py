"""How often a whole speaker turn goes missing, and whether overlap explains it.

sWER cannot answer this. It pools every word a speaker said into one stream and
scores that stream as a block, so a lost four-word turn is four deletions among
five thousand words -- present in the number, invisible in practice. It also
cannot say why a turn vanished, and because it compares concatenated text it
says nothing about whether the exchange survived as an exchange. This measure is
turn-shaped instead.

For every turn in the human transcript, over the same visits, the same adapters
and the same coverage window as the WER scorers:

  lost entirely   the system produced no word at all inside that turn's time
                  span. The speech was never transcribed -- an ASR failure, and
                  what happens on CM02493/day0001 at 18:48, where an interviewer
                  question present in both the human transcript and Chirp-3 is
                  absent from ours in every window framing tried.
  lost to the speaker
                  words are there, but none of them carry the predicted speaker
                  matched to this transcript speaker. The words survived and the
                  attribution did not -- a diarization failure. This count
                  includes the turns lost entirely, since those lost both.

Reported whole, then split three ways:

  by turn length      short turns are the ones that vanish;
  by the previous turn's length
                      turns tile the timeline here (load_timestamped_text
                      synthesizes each end from the next start), so there are no
                      literal gaps to measure. A previous turn under about 1.5 s
                      means the speakers were trading rapidly, which is the best
                      available proxy for talking over each other without
                      listening to the audio;
  by speaker role     losing interviewer questions costs more than losing a
                      backchannel, because a downstream model reads the
                      questions as context.

That makes the overlap claim testable. Flat across the previous-length buckets
means overlap is not the driver and the 18:48 case was ordinary error; a sharp
climb under 1.5 s means it is. Confirming that two people really were speaking
at once needs pyannote's overlap detection on a sample; this is the screen that
decides whether that is worth running.

CPU only.

Usage:
    uv run python scripts/lost_turns.py --cohort /path/to/cohort \\
        --system chirp3 --system ours=outputs/ours \\
        --output lost_turns.json --csv lost_turns.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "finetune"))

import systems as registry  # noqa: E402
from coverage import clip_words, covered_turns  # noqa: E402
from prepare_data import load_timestamped_text  # noqa: E402

from evaluate_systems import (  # noqa: E402
    ADAPTERS,
    align_streams,
    as_words,
    predicted_streams,
    reference_streams,
)

logger = logging.getLogger("lost_turns")

# Upper edges, in seconds. The last bucket is everything above the last edge.
LENGTH_BUCKETS = [1.0, 2.0, 5.0]
PREVIOUS_BUCKETS = [1.5, 3.0, 8.0]

# Roles the human transcripts name directly. Everything else (S1/S2 files, where
# the transcriber numbered the speakers rather than naming them) is reported
# under its own label, because guessing which number is the interviewer would
# put a fabricated distinction into the table.
ROLE_LABELS = {"INTERVIEWER", "PARTICIPANT"}


def bucket(value: float, edges: list[float]) -> str:
    for edge in edges:
        if value < edge:
            return f"<{edge:g}s"
    return f"{edges[-1]:g}s+"


def turn_rows(turns: list[dict], words: list[dict]) -> list[dict]:
    """One row per human turn: was it transcribed, and to the right speaker?"""
    reference = reference_streams(turns)
    hypothesis = predicted_streams(words)
    if not reference or not hypothesis:
        return []
    mapping = align_streams(reference, hypothesis)

    timed = sorted(
        (
            (float(w["start"]), max(float(w["end"]), float(w["start"])),
             (w.get("speaker") or "UNKNOWN"))
            for w in words
            if w.get("start") is not None and w.get("end") is not None
        )
    )

    rows = []
    previous_length = None
    for turn in turns:
        start = float(turn["start"])
        end = float(turn["end"]) if turn["end"] is not None else start
        end = max(end, start)
        matched = mapping.get(turn["speaker"])

        inside = [w for w in timed if w[1] > start and w[0] < end]
        speaker_words = [w for w in inside if matched is not None and w[2] == matched]
        speaker = str(turn["speaker"]).upper()

        rows.append({
            "speaker": speaker if speaker in ROLE_LABELS else f"unnamed ({speaker})",
            "length": round(end - start, 3),
            "length_bucket": bucket(end - start, LENGTH_BUCKETS),
            "previous_bucket": (
                bucket(previous_length, PREVIOUS_BUCKETS)
                if previous_length is not None else "first turn"
            ),
            "ref_words": len(str(turn["text"]).split()),
            "hyp_words": len(inside),
            "lost_entirely": not inside,
            "lost_to_speaker": not speaker_words,
        })
        previous_length = end - start
    return rows


def summarize(rows: list[dict]) -> dict:
    """Whole-corpus rates plus the three splits, counts kept alongside rates."""
    def rates(subset: list[dict]) -> dict:
        total = len(subset)
        entirely = sum(1 for r in subset if r["lost_entirely"])
        speaker = sum(1 for r in subset if r["lost_to_speaker"])
        return {
            "turns": total,
            "lost_entirely": entirely,
            "lost_to_speaker": speaker,
            "lost_entirely_rate": round(entirely / total, 4) if total else None,
            "lost_to_speaker_rate": round(speaker / total, 4) if total else None,
        }

    entry = rates(rows)
    for field, order in (
        ("length_bucket", [bucket(0, LENGTH_BUCKETS)]),
        ("previous_bucket", ["first turn"]),
        ("speaker", []),
    ):
        keys = sorted({r[field] for r in rows})
        entry[f"by_{field}"] = {
            key: rates([r for r in rows if r[field] == key]) for key in keys
        }
    return entry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument(
        "--system", action="append", required=True, metavar="NAME[=DIR]",
        help=f"one of {sorted(ADAPTERS)}; all but chirp3 need an output directory",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--csv", default=None, help="per-turn rows, one per system")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    args = build_parser().parse_args(argv)
    cohort = Path(args.cohort)

    requested = []
    for spec in args.system:
        name, _, directory = spec.partition("=")
        if name not in ADAPTERS:
            logger.error("Unknown system %s; known: %s", name, sorted(ADAPTERS))
            return 1
        requested.append((name, ADAPTERS[name][0], Path(directory) if directory else None))

    visits = sorted(
        p for p in cohort.glob("Pronet*/*/*")
        if (p / "human").is_dir() and any((p / "human").glob("*.txt"))
        and any((p / "audio").glob("*.wav"))
    )
    if args.limit:
        visits = visits[: args.limit]
    logger.info("Scoring %d system(s) over %d visit(s)", len(requested), len(visits))

    rows: dict[str, list[dict]] = {name: [] for name, _, _ in requested}
    per_visit: dict[str, dict] = {}
    adapter_errors: dict[str, Counter] = {name: Counter() for name, _, _ in requested}

    for index, visit in enumerate(visits, start=1):
        relative = visit.relative_to(cohort)
        human = sorted((visit / "human").glob("*.txt"))[0]
        try:
            turns = load_timestamped_text(human, 0.0)
        except Exception:
            continue
        turns, window_start, window_end = covered_turns(turns)
        if not turns:
            continue

        for name, adapter, root in requested:
            try:
                words = as_words(adapter(visit, root, relative))
            except Exception as error:
                adapter_errors[name][f"{type(error).__name__}: {error}"] += 1
                words = None
            if not words:
                continue
            words = clip_words(words, window_start, window_end)
            if not words:
                continue
            visit_rows = turn_rows(turns, words)
            if not visit_rows:
                continue
            for row in visit_rows:
                row["visit"] = relative.as_posix()
            rows[name].extend(visit_rows)
            per_visit.setdefault(relative.as_posix(), {})[name] = summarize(visit_rows)
        if index % 25 == 0 or index == len(visits):
            logger.info("  %d/%d visits", index, len(visits))

    for name, errors in adapter_errors.items():
        if errors:
            logger.error(
                "%s: adapter raised on %d visit(s): %s",
                name, sum(errors.values()), dict(errors.most_common(3)),
            )
        if not rows[name]:
            logger.error("%s: scored 0 turns -- treat as a failure, not an absence", name)

    aggregate = {name: summarize(items) for name, items in rows.items() if items}

    table = []
    for name, entry in sorted(
        aggregate.items(), key=lambda kv: kv[1]["lost_entirely_rate"]
    ):
        table.append((
            registry.label_of(name),
            [
                ("turns", str(entry["turns"])),
                ("never transcribed", f"{entry['lost_entirely_rate']:.4f}"),
                ("wrong speaker or missing", f"{entry['lost_to_speaker_rate']:.4f}"),
            ],
        ))
    print()
    print(registry.report(table))

    for name, entry in aggregate.items():
        print()
        print(f"{registry.label_of(name)} -- turns never transcribed")
        for field in ("by_length_bucket", "by_previous_bucket", "by_speaker"):
            print(f"  {field[3:].replace('_', ' ')}")
            for key, stats in entry[field].items():
                print(
                    f"    {key:<20} {stats['lost_entirely_rate']:.4f}"
                    f"  ({stats['lost_entirely']} of {stats['turns']})"
                )

    if args.output:
        Path(args.output).write_text(
            json.dumps(
                {
                    "aggregate": {registry.label_of(k): v for k, v in aggregate.items()},
                    "per_visit": per_visit,
                },
                indent=2,
            )
        )
        logger.info("Wrote %s", args.output)

    if args.csv:
        fields = [
            "visit", "system", "speaker", "length", "length_bucket",
            "previous_bucket", "ref_words", "hyp_words", "lost_entirely",
            "lost_to_speaker",
        ]
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for name, items in rows.items():
                for row in items:
                    writer.writerow({**row, "system": registry.label_of(name)})
        logger.info("Wrote %s", args.csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

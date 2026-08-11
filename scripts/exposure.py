"""Per-transcript exposure: how many known PII locations each system leaves open.

The leak rate in score_redaction.py answers a narrow question -- of the gold
spans whose surface form the transcriber happened to leave intact, how many
survive verbatim. That is only 310 spans in 35 files, because five of the ten
sites scrubbed identifiers to {redacted} at transcription time and nothing can
be searched for. It is the wrong denominator for the question that matters,
which is whether a transcript is safe to hand out.

This uses everything available instead. A *PII location* is any reference
position where either

  - the human transcriber marked a span, surface form or not, or
  - any system in the run emitted a redaction placeholder,

projected into the human transcript's own token coordinates so every system is
judged against the same list. A system is **exposed** at a location when it
emitted no placeholder overlapping it. That covers all 269 files rather than
35, and it counts a scrubbed {redacted} span as a location, which the verbatim
search never could.

The surface form is recovered where possible -- from the human transcript when
the transcriber left it, otherwise from an unredacted system's own words at
that position, since a pipeline that redacts nothing transcribes the identifier
in the clear. That makes the exposure report readable without needing the audio.

Two honest limits. Locations only exist where somebody found something, so a
name that every system missed and no transcriber marked is invisible here --
exposure is a lower bound on risk, never an all-clear. And a location proposed
by one system's false positive counts against every system that skipped it, so
the absolute count is pessimistic; the per-transcript comparison between
systems is the reliable read.

Usage:
    uv run python scripts/exposure.py --cohort /path/to/cohort \
        --system chirp3 --system ours_redacted=outputs/ours \
        --reference-words ours=outputs/ours \
        --output outputs/exposure.json --csv outputs/private/exposure.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "finetune"))

import redaction  # noqa: E402
import systems as registry  # noqa: E402
from coverage import clip_words, covered_turns  # noqa: E402
from prepare_data import load_timestamped_text  # noqa: E402

from score_redaction import REDACTION_ADAPTERS  # noqa: E402

logger = logging.getLogger("exposure")

PAD = 2


def project_back(blocks, start: int, end: int, pad: int = PAD) -> tuple[int, int]:
    """Map a hypothesis token range onto the reference.

    The mirror of redaction.project. Needed because placeholder spans live in
    each system's own coordinates and the union has to be built in one shared
    frame -- the human transcript's, which is the only frame all systems share.
    """
    low, high = None, None
    for i1, i2, j1, j2 in blocks:
        if j2 <= start or j1 >= end:
            continue
        if i2 - i1 == j2 - j1:
            lo = i1 + max(start - j1, 0)
            hi = i1 + min(end - j1, j2 - j1)
        else:
            lo, hi = i1, i2
        low = lo if low is None else min(low, lo)
        high = hi if high is None else max(high, hi)
    if low is None:
        return 0, 0
    return max(low - pad, 0), high + pad


def merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Overlapping candidate locations collapse into one."""
    out: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument(
        "--system", action="append", required=True, metavar="NAME[=DIR]",
        help="systems to judge; only those that emit placeholders are scored",
    )
    parser.add_argument(
        "--reference-words", action="append", default=[], metavar="NAME=DIR",
        help=(
            "an unredacted system used to recover the surface form at a "
            "location; not itself scored"
        ),
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--csv", default=None,
        help="per-transcript rows. Contains identifiers in the clear.",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser


def resolve(specs, table):
    out = []
    for spec in specs:
        name, _, directory = spec.partition("=")
        if name not in table:
            raise SystemExit(f"unknown system {name}")
        out.append((name, table[name][0], Path(directory) if directory else None))
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    cohort = Path(args.cohort)

    systems = resolve(args.system, REDACTION_ADAPTERS)
    sources = resolve(args.reference_words, REDACTION_ADAPTERS)

    visits = sorted(
        p for p in cohort.glob("Pronet*/*/*")
        if (p / "human").is_dir() and any((p / "human").glob("*.txt"))
        and any((p / "audio").glob("*.wav"))
    )
    if args.limit:
        visits = visits[: args.limit]

    per_visit: dict[str, dict] = {}
    totals: dict[str, Counter] = defaultdict(Counter)
    clean_transcripts: Counter = Counter()
    scored_transcripts: Counter = Counter()
    rows: list[dict] = []

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

        ref_tokens, gold = redaction.human_tokens(human)

        # Each system's placeholders, in reference coordinates.
        placed: dict[str, list[tuple[int, int]]] = {}
        available: dict[str, list[str]] = {}
        for name, adapter, root in systems + sources:
            try:
                words = adapter(visit, root, relative)
            except Exception:
                words = None
            if not words:
                continue
            words = clip_words(words, window_start, window_end)
            if not words:
                continue
            hyp_tokens, spans = redaction.system_tokens(words)
            blocks = redaction.index_map(ref_tokens, hyp_tokens)
            available[name] = hyp_tokens
            if any(name == n for n, _, _ in systems):
                placed[name] = [
                    project_back(blocks, s["start"], s["end"]) for s in spans
                ]

        if not placed:
            continue

        gold_ranges = [(g["start"], g["end"]) for g in gold]

        for name in placed:
            # Leave-one-out: a system is judged against the human's spans plus
            # what the OTHER systems found, never its own detections. Including
            # them lets a system propose locations it satisfies by construction,
            # which hands the win to whichever redactor is most aggressive --
            # with its own detections in, Chirp-3 (1996 spans) scored 34%
            # exposed against Gemma's 52% (1449 spans), reversing the ranking
            # every other measure gives.
            locations = merge_ranges(
                gold_ranges
                + [
                    r for other, spans in placed.items() if other != name
                    for r in spans if r != (0, 0)
                ]
            )
            if not locations:
                # Nothing known to protect here; not evidence of safety, so the
                # transcript is skipped rather than scored as perfect.
                continue

            exposed = []
            for start, end in locations:
                covered = any(
                    s < end and e > start for s, e in placed[name]
                )
                if covered:
                    continue
                surface = " ".join(ref_tokens[start:end]).strip()
                if not surface or surface == "redacted":
                    for other, tokens in available.items():
                        if other in placed:
                            continue
                        surface = surface or "?"
                        break
                exposed.append((start, end, surface))

            # The same question over human-marked spans only. Those are
            # verified PII, so this is the lower bound; the leave-one-out union
            # above includes other systems' false positives and is the upper
            # bound. Quoting one without the other overstates the certainty.
            gold_merged = merge_ranges(gold_ranges)
            gold_exposed = sum(
                1 for start, end in gold_merged
                if not any(s2 < end and e2 > start for s2, e2 in placed[name])
            )
            totals[name]["gold_locations"] += len(gold_merged)
            totals[name]["gold_exposed"] += gold_exposed

            totals[name]["locations"] += len(locations)
            totals[name]["exposed"] += len(exposed)
            scored_transcripts[name] += 1
            if not exposed:
                clean_transcripts[name] += 1

            per_visit.setdefault(relative.as_posix(), {})[name] = {
                "locations": len(locations),
                "exposed": len(exposed),
                "exposed_rate": len(exposed) / len(locations),
                "gold_locations": len(gold_merged),
                "gold_exposed": gold_exposed,
            }
            if args.csv:
                rows.append({
                    "site": relative.parts[0],
                    "subject": relative.parts[1],
                    "session": relative.parts[2],
                    "system": registry.label_of(name),
                    "pii_locations": len(locations),
                    "exposed": len(exposed),
                    "exposed_rate": round(len(exposed) / len(locations), 4),
                    "exposed_text": " | ".join(
                        s for _, _, s in exposed[:40] if s and s != "?"
                    ),
                    "human_transcript_path": str(human),
                })
        if index % 25 == 0 or index == len(visits):
            logger.info("  %d/%d visits", index, len(visits))

    aggregate = {}
    for name, counter in totals.items():
        scored = scored_transcripts[name] or 1
        aggregate[registry.label_of(name)] = {
            "transcripts": scored_transcripts[name],
            "pii_locations": counter["locations"],
            "exposed": counter["exposed"],
            "exposed_rate": counter["exposed"] / (counter["locations"] or 1),
            "clean_transcripts": clean_transcripts[name],
            "clean_transcript_rate": clean_transcripts[name] / scored,
            "gold_locations": counter["gold_locations"],
            "gold_exposed": counter["gold_exposed"],
            "gold_exposed_rate": (
                counter["gold_exposed"] / counter["gold_locations"]
                if counter["gold_locations"] else None
            ),
        }

    width = max((len(n) for n in aggregate), default=10)
    print(
        f"\n{'system':{width}}  transcripts   gold-exposed   union-exposed  "
        f"fully clean"
    )
    for name, s in sorted(aggregate.items(), key=lambda kv: kv[1]["exposed_rate"]):
        gold = (
            f"{s['gold_exposed']}/{s['gold_locations']} "
            f"{s['gold_exposed_rate'] * 100:.0f}%"
            if s["gold_exposed_rate"] is not None else "-"
        )
        print(
            f"{name:{width}}  {s['transcripts']:11d}  {gold:>14}  "
            f"{s['exposed']}/{s['pii_locations']} {s['exposed_rate'] * 100:.0f}%".ljust(16)
            + f"  {s['clean_transcripts']:3d} of {s['transcripts']} "
            f"({s['clean_transcript_rate'] * 100:.0f}%)"
        )

    if args.output:
        Path(args.output).write_text(json.dumps(
            {"aggregate": aggregate, "per_visit": per_visit}, indent=2,
        ) + "\n")
        logger.info("wrote %s", args.output)
    if args.csv and rows:
        rows.sort(key=lambda r: (-r["exposed"], r["subject"]))
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        logger.warning(
            "wrote %s: %d row(s). Contains identifiers in the clear.",
            args.csv, len(rows),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Choose a validation set of transcripts for a redaction experiment.

The turn-rewrite pilot ran on four transcripts and 59 gold spans, which is far
too few to separate a real improvement from four lucky files: its whole leak
advantage came from ten spans in one visit. This picks a larger set on the two
properties that decide what can be measured.

  leak-testable   the visit must carry at least one gold span whose surface
                  form the transcriber left intact. Five of the ten sites
                  scrubbed theirs to {redacted}, and a scrubbed span can be
                  scored for detection but never leak-tested -- so a validation
                  set drawn at random would answer the wrong question.
  length spread   sampled evenly across the word-count range rather than
                  randomly, because cost scales with length and the failure
                  modes (long monologues, ragged batches) live at the ends.

Subjects are capped so one talkative participant cannot carry the result, and
visits already spent on the pilot can be excluded so the validation set is
genuinely held out.

Usage:
    uv run python scripts/select_validation.py --cohort /path/to/cohort \
        --outputs outputs/ours --count 24 --output /tmp/validation.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "finetune"))

import redaction  # noqa: E402
from coverage import covered_turns  # noqa: E402
from prepare_data import load_timestamped_text  # noqa: E402

logger = logging.getLogger("select_validation")


def survey(cohort: Path, outputs: Path) -> list[dict]:
    """Every visit that can be used, with what makes it usable."""
    rows: list[dict] = []
    for visit in sorted(cohort.glob("Pronet*/*/*")):
        human = sorted((visit / "human").glob("*.txt"))
        transcript = outputs / visit.relative_to(cohort) / "transcript.json"
        if not human or not transcript.exists():
            continue
        # Scored over the covered span, like every other measurement here, so
        # the span counts match what the scorer will actually see.
        try:
            turns = load_timestamped_text(human[0], 0.0)
        except Exception:
            continue
        turns, _, _ = covered_turns(turns)
        if not turns:
            continue
        _, spans = redaction.tokens_from_texts([t["text"] for t in turns])
        testable = [s for s in spans if not s["scrubbed"] and s["surface"].strip()]
        if not testable:
            continue
        payload = json.loads(transcript.read_text())
        words = payload["words"] if isinstance(payload, dict) else payload
        rows.append({
            "visit": visit.relative_to(cohort).as_posix(),
            "subject": visit.parent.name,
            "site": visit.parent.parent.name,
            "words": len(words),
            "gold": len(spans),
            "testable": len(testable),
        })
    return rows


def choose(rows: list[dict], count: int, per_subject: int) -> list[dict]:
    """Evenly spaced over word count, capped per subject."""
    ordered = sorted(rows, key=lambda r: r["words"])
    chosen: list[dict] = []
    seen: dict[str, int] = {}
    if not ordered:
        return chosen

    # Walk the length-ordered list at a stride that would land `count` visits,
    # skipping any that would break the per-subject cap and continuing from
    # there, so the spread survives the cap instead of collapsing to the head.
    stride = max(len(ordered) / count, 1.0)
    position = 0.0
    used: set[int] = set()
    while len(chosen) < count and len(used) < len(ordered):
        index = min(int(position), len(ordered) - 1)
        # Find the next unused candidate at or after this position that fits.
        for offset in range(len(ordered)):
            candidate = (index + offset) % len(ordered)
            if candidate in used:
                continue
            row = ordered[candidate]
            if seen.get(row["subject"], 0) >= per_subject:
                used.add(candidate)
                continue
            used.add(candidate)
            seen[row["subject"]] = seen.get(row["subject"], 0) + 1
            chosen.append(row)
            break
        position += stride
    return sorted(chosen, key=lambda r: r["visit"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--outputs", required=True, help="tree of transcript.json files")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument(
        "--per-subject", type=int, default=2,
        help="cap on visits from one participant; this study is longitudinal",
    )
    parser.add_argument(
        "--exclude", default=None, metavar="FILE",
        help="visit list to hold out, e.g. the pilot's four",
    )
    parser.add_argument("--output", default=None, help="write the visit list here")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    rows = survey(Path(args.cohort), Path(args.outputs))
    logger.info("%d visit(s) carry at least one leak-testable gold span", len(rows))

    if args.exclude:
        excluded = {
            line.strip() for line in Path(args.exclude).read_text().splitlines()
            if line.strip()
        }
        before = len(rows)
        rows = [r for r in rows if r["visit"] not in excluded]
        logger.info("Excluded %d visit(s)", before - len(rows))

    chosen = choose(rows, args.count, args.per_subject)
    if len(chosen) < args.count:
        logger.warning(
            "Only %d visit(s) available under the per-subject cap of %d",
            len(chosen), args.per_subject,
        )

    print()
    print(f"{'visit':46s} {'words':>6s} {'gold':>5s} {'testable':>9s}")
    for row in chosen:
        print(f"{row['visit']:46s} {row['words']:6d} {row['gold']:5d} {row['testable']:9d}")
    sites = sorted({r["site"] for r in chosen})
    print()
    print(
        f"{len(chosen)} visit(s), {sum(r['words'] for r in chosen)} words, "
        f"{sum(r['gold'] for r in chosen)} gold spans, "
        f"{sum(r['testable'] for r in chosen)} leak-testable, "
        f"{len(sites)} site(s): {', '.join(s.replace('Pronet', '') for s in sites)}"
    )

    if args.output:
        Path(args.output).write_text(
            "".join(f"{row['visit']}\n" for row in chosen)
        )
        logger.info("wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

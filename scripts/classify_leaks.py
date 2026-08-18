"""Summarise the leaks CSV by kind, emitting counts only -- never identifiers.

A leak is any gold span whose surface form survives in a system's output, but
the transcribers' braces mark more than identifiers: a bare month, a statement
about religion, a hobby. Judging which leaks matter needs a human read of the
CSV, and that file is PII in the clear and stays on the cluster. This produces
the part that can safely travel: how many leaks of each shape each system left,
so a chart can separate "single word, could be a name" from "a bare month".

The classes are deliberately crude and named for what they are -- a shape, not
a verdict. Nothing here decides that a span is safe; it decides that a span
needs a person to look at it.

Usage:
    uv run python scripts/classify_leaks.py --leaks-csv outputs/private/leaks.csv \
        --output outputs/leak_kinds.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import systems as registry  # noqa: E402

logger = logging.getLogger("classify_leaks")

MONTHS = (
    r"(january|february|march|april|may|june|july|august|september|october"
    r"|november|december)"
)

# Order matters: it is the order the chart stacks them in, least identifying
# last, so the eye lands on the part that still needs review.
KINDS = ["single word", "two or three words", "longer phrase", "month or date", "number"]


def kind(text: str) -> str:
    body = text.strip().lower()
    if re.fullmatch(r"[\d\s\-/.]+", body):
        return "number"
    if re.search(MONTHS, body) or re.search(r"\b\d{1,2}(st|nd|rd|th)\b", body):
        return "month or date"
    words = body.split()
    if len(words) == 1:
        return "single word"
    if len(words) <= 3:
        return "two or three words"
    return "longer phrase"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaks-csv", required=True)
    parser.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    counts: dict[str, Counter] = defaultdict(Counter)
    for row in csv.DictReader(open(args.leaks_csv)):
        if row.get("leaked") != "1":
            continue
        counts[row["system"]][kind(row.get("identifier", ""))] += 1

    payload = {
        system: {"kinds": dict(kinds), "total": sum(kinds.values())}
        for system, kinds in counts.items()
    }
    for system, entry in sorted(payload.items(), key=lambda kv: -kv[1]["total"]):
        parts = ", ".join(f"{n} {k}" for k, n in sorted(entry["kinds"].items()))
        print(f"{registry.key_of(system):26s} {entry['total']:3d} leaked  ({parts})")

    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2))
        logger.info("wrote %s -- counts only, no identifiers", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

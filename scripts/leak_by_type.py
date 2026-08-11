"""Leak rate broken down by the kind of identifier that leaked.

The gold convention in these transcripts is a bare curly brace -- {isaiah} --
carrying no label, so the type has to come from somewhere else. Two sources, in
order of trust:

  1. Any system's own label for that span. A span one system leaves in the clear
     is often caught by another, and the catching system names it PERSON_NAME,
     LOCATION, DATE and so on. Labels are pooled across every system and every
     visit the same surface form appears in, so a name identified once is typed
     everywhere.
  2. A lexical fallback for spans nothing ever caught: month names, ordinals and
     bare numbers are dates. Everything still unresolved is reported as
     unclassified rather than guessed into a category -- an unclassified bucket
     that stays visible is worth more than a tidy chart built on assumptions.

Usage:
    uv run python scripts/leak_by_type.py --cohort /path/to/cohort \
        --system chirp3 --system ours_redacted=outputs/ours \
        --output outputs/leak_by_type.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
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

logger = logging.getLogger("leak_by_type")

MONTHS = {
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}
ORDINAL = re.compile(r"^\d{1,2}(st|nd|rd|th)$|^(first|second|third|fourth|fifth|"
                     r"sixth|seventh|eighth|ninth|tenth|eleventh|twelfth)$")

CATEGORY_ORDER = ["name", "location", "date", "age", "gender", "unclassified"]


def lexical_type(surface: str) -> str | None:
    tokens = [redaction.normalize_token(t) for t in surface.split()]
    tokens = [t for t in tokens if t]
    if not tokens:
        return None
    if any(t in MONTHS for t in tokens) or any(ORDINAL.match(t) for t in tokens):
        return "date"
    if all(t.isdigit() for t in tokens):
        return "date"
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--system", action="append", required=True, metavar="NAME[=DIR]")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    cohort = Path(args.cohort)

    chosen = []
    for spec in args.system:
        name, _, directory = spec.partition("=")
        if name not in REDACTION_ADAPTERS:
            logger.error("Unknown system %s", name)
            return 1
        chosen.append((name, REDACTION_ADAPTERS[name][0], Path(directory) if directory else None))

    visits = sorted(
        p for p in cohort.glob("Pronet*/*/*")
        if (p / "human").is_dir() and any((p / "human").glob("*.txt"))
        and any((p / "audio").glob("*.wav"))
    )
    if args.limit:
        visits = visits[: args.limit]

    # (visit, span index) -> {"surface", "leaked": {system: bool}}
    spans: dict[tuple[str, int], dict] = {}
    # Placeholders each system emitted, so the chart can drop systems that
    # redact nothing: they leak everything by construction and comparing
    # against them says nothing about a redactor.
    redactions: Counter = Counter()
    # surface form -> Counter of labels any system gave it, anywhere
    labels_for: dict[str, Counter] = defaultdict(Counter)

    for index, visit in enumerate(visits, start=1):
        relative = visit.relative_to(cohort).as_posix()
        human = sorted((visit / "human").glob("*.txt"))[0]
        try:
            turns = load_timestamped_text(human, 0.0)
        except Exception:
            continue
        turns, window_start, window_end = covered_turns(turns)
        if not turns:
            continue

        for name, adapter, root in chosen:
            try:
                words = adapter(visit, root, relative and Path(relative))
            except Exception:
                words = None
            if not words:
                continue
            words = clip_words(words, window_start, window_end)
            if not words:
                continue
            result = redaction.score_visit(turns, words)
            redactions[name] += result["predicted_spans"]
            for order, span in enumerate(result["spans"]):
                if not span["testable"]:
                    continue
                key = (relative, order)
                entry = spans.setdefault(
                    key, {"surface": span["surface"], "leaked": {}}
                )
                entry["leaked"][name] = span["leaked"]
                for label in span["labels"]:
                    labels_for[span["surface"].lower()][label] += 1
        if index % 25 == 0 or index == len(visits):
            logger.info("  %d/%d visits", index, len(visits))

    def categorize(surface: str) -> str:
        counter = labels_for.get(surface.lower())
        if counter:
            label = counter.most_common(1)[0][0]
            return redaction.CATEGORY.get(label, "unclassified")
        return lexical_type(surface) or "unclassified"

    totals: dict[str, Counter] = defaultdict(Counter)
    leaked: dict[str, Counter] = defaultdict(Counter)
    for entry in spans.values():
        category = categorize(entry["surface"])
        for name, was_leaked in entry["leaked"].items():
            totals[name][category] += 1
            if was_leaked:
                leaked[name][category] += 1

    typed = Counter(categorize(e["surface"]) for e in spans.values())
    logger.info("gold spans by inferred type: %s", dict(typed))
    logger.info(
        "typed from a system label: %d of %d distinct surface forms",
        len(labels_for), len({e["surface"].lower() for e in spans.values()}),
    )

    aggregate = {}
    for name in totals:
        aggregate[registry.label_of(name)] = {
            "redactions": redactions[name],
            "by_type": {
                category: {
                    "spans": totals[name][category],
                    "leaked": leaked[name][category],
                    "leak_rate": (
                        leaked[name][category] / totals[name][category]
                        if totals[name][category] else None
                    ),
                }
                for category in CATEGORY_ORDER if totals[name][category]
            },
            "spans": sum(totals[name].values()),
            "leaked": sum(leaked[name].values()),
        }

    width = max((len(n) for n in aggregate), default=10)
    categories = [c for c in CATEGORY_ORDER if any(
        c in v["by_type"] for v in aggregate.values()
    )]
    print(f"\n{'system':{width}}  " + "  ".join(f"{c:>14}" for c in categories))
    for name, stats in aggregate.items():
        cells = []
        for category in categories:
            entry = stats["by_type"].get(category)
            cells.append(
                f"{entry['leaked']}/{entry['spans']} {entry['leak_rate'] * 100:4.0f}%".rjust(14)
                if entry else "".rjust(14)
            )
        print(f"{name:{width}}  " + "  ".join(cells))

    if args.output:
        Path(args.output).write_text(json.dumps(
            {"aggregate": aggregate, "spans_by_type": dict(typed)}, indent=2,
        ) + "\n")
        logger.info("wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

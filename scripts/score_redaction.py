"""Score each system's PII redaction against the human transcripts' gold spans.

Follows DIALOG-DeID section 2.3: span-level precision, recall and micro-F1.
Category-agnostic, because the gold convention in this corpus (curly braces)
carries no label -- the paper's per-category table needs labelled gold, which we
do not have. What is reported per category is the *distribution* of what each
system chose to redact, which is still informative about where a redactor is
over-eager.

Alongside the paper's metric, two numbers this corpus makes possible:

  leak rate       gold spans whose surface form appears verbatim in the output.
                  The privacy question stated directly, with no alignment step
                  to distrust.
  over/under      false positives and false negatives as rates over gold spans.
                  A redactor can reach a respectable F1 by being aggressive in
                  one place and blind in another; splitting the error tells you
                  which failure you are buying.

Usage:
    uv run python scripts/score_redaction.py --cohort /path/to/cohort \
        --system chirp3 --system ours=outputs/ours \
        --output outputs/redaction.json
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

import redaction  # noqa: E402
import systems as registry  # noqa: E402
from coverage import clip_words, covered_turns  # noqa: E402
from evaluate_systems import ADAPTERS, make_file_adapter, make_run_adapter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "finetune"))
from prepare_data import load_timestamped_text  # noqa: E402

logger = logging.getLogger("score_redaction")

# Systems that carry redacted output. chirp3 redacts natively; the *_redacted
# entries read what redact_llm.py wrote next to each transcript.
REDACTION_ADAPTERS = dict(ADAPTERS)
REDACTION_ADAPTERS.update({
    "ours_redacted": (make_run_adapter("transcript_redacted.json"), "transcript_redacted.json"),
    "baseline_redacted": (make_file_adapter("_words_redacted.json"), "_words_redacted.json"),
    # redact_llm.py --mode turn, written alongside the chunk-mode output with
    # its own suffix so the two protocols can be scored against each other on
    # the same transcripts rather than one replacing the other.
    "ours_redacted_turn": (
        make_run_adapter("transcript_redacted_turn.json"), "transcript_redacted_turn.json",
    ),
    "baseline_redacted_turn": (
        make_file_adapter("_words_redacted_turn.json"), "_words_redacted_turn.json",
    ),
    # Same two protocols after the possessive rule was added to both prompts.
    # Written to their own suffixes so the before and after can be scored on the
    # same transcripts in one run rather than one overwriting the other.
    "ours_redacted_poss": (
        make_run_adapter("transcript_redacted_poss.json"), "transcript_redacted_poss.json",
    ),
    "ours_redacted_turn_poss": (
        make_run_adapter("transcript_redacted_turn_poss.json"),
        "transcript_redacted_turn_poss.json",
    ),
})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument(
        "--system", action="append", required=True, metavar="NAME[=DIR]",
        help=f"one of {sorted(REDACTION_ADAPTERS)}",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--subset", default=None, metavar="FILE",
        help="file of cohort-relative visit paths, one per line; score only these",
    )
    parser.add_argument(
        "--leaks-csv", default=None, metavar="FILE",
        help=(
            "write one row per gold span that leaked, for inspection. WARNING: "
            "this file contains the identifiers in the clear, by construction -- "
            "it is the one output here that is not de-identified"
        ),
    )
    parser.add_argument(
        "--all-spans", action="store_true",
        help="put every gold span in the CSV, not only the leaked ones",
    )
    parser.add_argument(
        "--gold-only", action="store_true",
        help=(
            "score only visits whose human transcript has at least one gold "
            "span; 127 of 269 have none, and whether that means 'no PII' or "
            "'not annotated' is unknown"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    cohort = Path(args.cohort)

    systems = []
    for spec in args.system:
        name, _, directory = spec.partition("=")
        if name not in REDACTION_ADAPTERS:
            logger.error("Unknown system %s; known: %s", name, sorted(REDACTION_ADAPTERS))
            return 1
        systems.append((name, REDACTION_ADAPTERS[name][0], Path(directory) if directory else None))

    visits = sorted(
        p for p in cohort.glob("Pronet*/*/*")
        if (p / "human").is_dir() and any((p / "human").glob("*.txt"))
        and any((p / "audio").glob("*.wav"))
    )
    if args.subset:
        wanted = {
            line.strip() for line in Path(args.subset).read_text().splitlines()
            if line.strip()
        }
        visits = [p for p in visits if p.relative_to(cohort).as_posix() in wanted]
        # Scoring zero visits would otherwise look like a completed run with no
        # PII in it, which is the failure this whole module is built to avoid.
        if not visits:
            logger.error("--subset %s matched none of the cohort's visits", args.subset)
            return 1
        if len(visits) != len(wanted):
            logger.warning(
                "--subset listed %d visit(s), %d matched the cohort",
                len(wanted), len(visits),
            )
    if args.limit:
        visits = visits[: args.limit]

    per_visit: dict[str, dict] = {}
    leak_rows: list[dict] = []
    totals: dict[str, Counter] = {name: Counter() for name, _, _ in systems}
    categories: dict[str, Counter] = {name: Counter() for name, _, _ in systems}
    errors: dict[str, Counter] = {name: Counter() for name, _, _ in systems}
    scored = Counter()
    gold_visits = 0

    for index, visit in enumerate(visits, start=1):
        relative = visit.relative_to(cohort)
        human = sorted((visit / "human").glob("*.txt"))[0]
        # The gold spans all sit inside the covered span by construction, but
        # the hypothesis must be clipped to it or every placeholder the system
        # emitted after the transcript stopped counts as a false positive.
        try:
            turns = load_timestamped_text(human, 0.0)
        except Exception:
            continue
        turns, window_start, window_end = covered_turns(turns)
        if not turns:
            continue
        gold_tokens, gold_spans = redaction.tokens_from_texts(
            [t['text'] for t in turns]
        )
        if gold_spans:
            gold_visits += 1
        elif args.gold_only:
            continue

        for name, adapter, root in systems:
            try:
                words = adapter(visit, root, relative)
            except Exception as error:
                errors[name][f"{type(error).__name__}: {error}"] += 1
                continue
            if not words:
                continue
            words = clip_words(words, window_start, window_end)
            if not words:
                continue
            result = redaction.score_visit(turns, words)
            scored[name] += 1
            for key in (
                "gold_spans", "predicted_spans", "true_positives",
                "false_positives", "false_negatives", "leak_testable", "leaked",
                "leaked_fuzzy", "identifiers_testable", "identifiers_readable",
            ):
                totals[name][key] += result[key]
            categories[name].update(result["categories"])
            per_visit.setdefault(relative.as_posix(), {})[name] = {
                k: v for k, v in result.items() if k not in ("categories", "spans")
            }

            if args.leaks_csv:
                site, subject, session = relative.parts[:3]
                for order, span in enumerate(result["spans"], start=1):
                    # readable_somewhere without leaked is an occurrence this
                    # system did redact, whose name survives elsewhere. Worth
                    # reading, so it is written, and flagged rather than
                    # counted as this occurrence leaking.
                    if not (span["leaked"] or span.get("leaked_fuzzy")
                            or span.get("readable_somewhere") or args.all_spans):
                        continue
                    leak_rows.append({
                        "site": site,
                        "subject": subject,
                        "session": session,
                        "system": registry.label_of(name),
                        "span": order,
                        "identifier": span["surface"],
                        "leaked": int(span["leaked"]),
                        "leaked_fuzzy": int(span.get("leaked_fuzzy", False)),
                        "similarity": span.get("similarity", ""),
                        "nearest_text": span.get("nearest_text", ""),
                        "readable_elsewhere": int(span.get("readable_somewhere", False)),
                        "redacted": int(span["redacted"]),
                        "leak_testable": int(span["testable"]),
                        "labels": " ".join(span["labels"]),
                        "human_context": span["human_context"],
                        "system_context": span["system_context"],
                    })
        if index % 25 == 0 or index == len(visits):
            logger.info("  %d/%d visits", index, len(visits))

    logger.info("%d of %d visit(s) carry at least one gold span", gold_visits, len(visits))
    for name, counter in errors.items():
        if counter:
            logger.error("%s: adapter raised on %d visit(s): %s",
                         name, sum(counter.values()), dict(counter.most_common(3)))

    aggregate = {}
    for name, counter in totals.items():
        if not scored[name]:
            logger.error("%s: scored 0 visits -- a failure, not an absence", name)
            continue
        tp, fp, fn = (
            counter["true_positives"], counter["false_positives"], counter["false_negatives"],
        )
        gold = counter["gold_spans"] or 1
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        aggregate[name] = {
            "visits": scored[name],
            "gold_spans": counter["gold_spans"],
            "predicted_spans": counter["predicted_spans"],
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1": (
                2 * precision * recall / (precision + recall)
                if precision and recall else 0.0
            ),
            # Signed, over gold spans: negative is under-redaction, positive is
            # over-redaction, zero is a redactor whose errors cancel in count
            # (which is not the same as a redactor that is right).
            "under_rate": fn / gold,
            "over_rate": fp / gold,
            "net_rate": (fp - fn) / gold,
            "leak_testable": counter["leak_testable"],
            "leaked": counter["leaked"],
            "leak_rate": (
                counter["leaked"] / counter["leak_testable"]
                if counter["leak_testable"] else None
            ),
            # The same test allowing a near-miss spelling, at
            # redaction.FUZZY_THRESHOLD. Reported beside the exact figure
            # rather than replacing it, because the threshold is a judgement
            # and the exact count is not.
            "leaked_fuzzy": counter["leaked_fuzzy"],
            "leak_rate_fuzzy": (
                counter["leaked_fuzzy"] / counter["leak_testable"]
                if counter["leak_testable"] else None
            ),
            # The other privacy question: not "did this occurrence survive" but
            # "can this person still be identified anywhere in the transcript".
            # One entry per distinct identifier per visit, so a name marked
            # seventeen times cannot contribute seventeen failures.
            "identifiers_testable": counter["identifiers_testable"],
            "identifiers_readable": counter["identifiers_readable"],
            "identifier_leak_rate": (
                counter["identifiers_readable"] / counter["identifiers_testable"]
                if counter["identifiers_testable"] else None
            ),
            "categories": dict(categories[name]),
        }

    rows = []
    for name, stats in sorted(aggregate.items(), key=lambda kv: -(kv[1]["f1"] or 0)):
        leak = (
            f"{stats['leak_rate'] * 100:.1f}%" if stats["leak_rate"] is not None else "-"
        )
        named = (
            f"{stats['identifier_leak_rate'] * 100:.1f}%"
            if stats["identifier_leak_rate"] is not None else "-"
        )
        rows.append((
            registry.label_of(name),
            [
                ("visits", str(stats["visits"])),
                ("gold", str(stats["gold_spans"])),
                ("redacted", str(stats["predicted_spans"])),
                ("TP", str(stats["true_positives"])),
                ("FP", str(stats["false_positives"])),
                ("FN", str(stats["false_negatives"])),
                ("recall", f"{(stats['recall'] or 0) * 100:.1f}%"),
                ("precision", f"{(stats['precision'] or 0) * 100:.1f}%"),
                ("F1", f"{stats['f1'] * 100:.1f}%"),
                ("under", f"{stats['under_rate'] * 100:.1f}%"),
                ("over", f"{stats['over_rate'] * 100:.1f}%"),
                ("leak", leak),
                ("leak (near-miss)", (
                    f"{stats['leak_rate_fuzzy'] * 100:.1f}%"
                    if stats["leak_rate_fuzzy"] is not None else "-"
                )),
                ("names readable", named),
            ],
        ))
    print()
    print(registry.report(rows))

    if args.leaks_csv:
        # Sorted so a reviewer reads one participant's sessions together and the
        # same span appears under each system in turn.
        leak_rows.sort(key=lambda r: (r["subject"], r["session"], r["span"], r["system"]))
        fields = [
            "site", "subject", "session", "system", "span", "identifier",
            "leaked", "leaked_fuzzy", "similarity", "nearest_text",
            "readable_elsewhere", "redacted", "leak_testable", "labels",
            "human_context", "system_context",
        ]
        with open(args.leaks_csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(leak_rows)
        subjects = len({r["subject"] for r in leak_rows})
        logger.warning(
            "wrote %s: %d row(s) across %d participant(s). This file contains "
            "identifiers in the clear -- keep it beside the audio, not with the "
            "de-identified outputs",
            args.leaks_csv, len(leak_rows), subjects,
        )

    if args.output:
        Path(args.output).write_text(
            json.dumps(
                {
                    "aggregate": {registry.label_of(k): v for k, v in aggregate.items()},
                    "per_visit": {
                        visit: {registry.label_of(k): v for k, v in entry.items()}
                        for visit, entry in per_visit.items()
                    },
                },
                indent=2,
            ) + "\n"
        )
        logger.info("wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

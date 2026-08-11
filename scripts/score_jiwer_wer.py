"""Score every system with the NVIDIA-transcript WER script's method.

`scripts/nvidia_wer.py` is that script, vendored unmodified; this one imports
its `preprocess_transcript` and `calculate_wer` so the numbers come from its
code rather than a reimplementation. Only the plumbing is ours: the same visit
list, the same system adapters and the same coverage window as
`evaluate_systems.py` and `score_partner_wer.py`, so all three metrics describe
identical inputs and can be compared visit by visit.

What this metric does that ours does not:

  - Everything in square brackets is deleted before scoring. That removes the
    human transcripts' [inaudible]/[crosstalk] markup, but it also removes
    every CrisperWhisper filled pause, because CW2 writes them as [UM]/[UH]
    while Chirp-3 writes "um"/"uh" as plain words. The deletion is therefore
    asymmetric across systems, and `filler_symmetric` below quantifies it by
    dropping filled pauses on both sides for every system alike.
  - All punctuation is stripped and everything is lowercased, so this is a
    normalized WER with no separate raw tier.
  - jiwer `process_words`, i.e. true Levenshtein alignment. The partner team's
    metric uses difflib and is an upper bound; this one is the standard number,
    which is why it lands lower than theirs on the same audio.
  - Braces are not stripped, so the transcribers' PII spans leave their surface
    text in the reference ({isaiah} -> "isaiah"). 876 spans corpus-wide, a
    rounding error, and left alone rather than deviating from their code.

Two departures from the vendored script, both forced by this corpus:

  - Reference lines are parsed with prepare_data's TIMESTAMPED_LINE rather than
    its `^S\\d+:\\s+HH:MM:SS.mmm` pattern. 56k of 69k lines here use
    "INTERVIEWER:"/"PARTICIPANT:", which that pattern does not match, and the
    speaker tag and timestamp digits would survive into the reference as words
    -- the defect already found and fixed once in score_partner_wer.py.
  - Scoring is restricted to the span the human transcript covers. These
    transcripts stop early on a third of the cohort (median 96% of the audio,
    97 visits under 80%), and the systems transcribe to the end of the file, so
    an unrestricted comparison charges the untranscribed remainder as
    insertions. See coverage.py.

Usage:
    uv run python scripts/score_jiwer_wer.py --cohort /path/to/cohort \\
        --system chirp3 --system ours=outputs/ours \\
        --output jiwer_wer.json --csv jiwer_wer.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "finetune"))

import systems as registry  # noqa: E402
from coverage import clip_words, covered_turns  # noqa: E402
from nvidia_wer import calculate_wer, preprocess_transcript  # noqa: E402
from partner_compare import FILLER_RE  # noqa: E402
from prepare_data import TIMESTAMPED_LINE, load_timestamped_text  # noqa: E402

from evaluate_systems import ADAPTERS  # noqa: E402

logger = logging.getLogger("score_jiwer_wer")

COUNTS = ("hits", "substitutions", "deletions", "insertions", "total_words")


def reference_prose(turns: list[dict]) -> str:
    """Turn text as one string, tags and timestamps already gone."""
    return re.sub(r"\s+", " ", " ".join(t["text"] for t in turns)).strip()


def strip_fillers(text: str) -> str:
    """Drop filled pauses from already-preprocessed text.

    Applied to both sides of every system, so the bracket deletion in
    `preprocess_transcript` -- which silently removes CW2's [UM] but keeps
    Chirp's "um" -- stops being a per-system advantage.
    """
    return " ".join(t for t in text.split() if not FILLER_RE.match(t))


def score_pair(reference_text: str, hypothesis_text: str) -> dict:
    """Their WER, plus the filler-symmetric variant, for one visit."""
    metrics, _ = calculate_wer(reference_text, hypothesis_text)
    row = {k: metrics[k] for k in COUNTS}
    row["wer"] = metrics["wer"]
    row["mer"] = metrics["mer"]
    row["wil"] = metrics["wil"]
    row["hyp_words"] = row["hits"] + row["substitutions"] + row["insertions"]

    ref_nofill = strip_fillers(preprocess_transcript(reference_text))
    hyp_nofill = strip_fillers(preprocess_transcript(hypothesis_text))
    symmetric, _ = calculate_wer(ref_nofill, hyp_nofill)
    row["wer_filler_symmetric"] = symmetric["wer"]
    row["total_words_filler_symmetric"] = symmetric["total_words"]
    row["errors_filler_symmetric"] = (
        symmetric["substitutions"] + symmetric["deletions"] + symmetric["insertions"]
    )
    return row


def aggregate_rows(rows: list[dict]) -> dict:
    """Per-visit means and medians, plus corpus-pooled (micro) rates.

    Both are reported because they answer different questions and diverge
    sharply here: a tail of visits with truncated references pushes the mean
    far above the median, and the pooled rate weights a 100-minute session
    above a 5-minute one, which is the number a corpus-level claim needs.
    """
    entry: dict[str, float | int] = {"visits": len(rows)}
    for key in ("wer", "wer_filler_symmetric", "mer", "wil"):
        values = [row[key] for row in rows]
        entry[key] = round(statistics.fmean(values), 4)
        entry[f"{key}_median"] = round(statistics.median(values), 4)
    totals = {k: sum(row[k] for row in rows) for k in COUNTS}
    entry.update(totals)
    reference_words = max(totals["total_words"], 1)
    errors = totals["substitutions"] + totals["deletions"] + totals["insertions"]
    entry["wer_micro"] = round(errors / reference_words, 4)
    for key in ("substitutions", "deletions", "insertions"):
        entry[f"{key}_rate"] = round(totals[key] / reference_words, 4)
    entry["hyp_words"] = sum(row["hyp_words"] for row in rows)
    entry["word_ratio"] = round(entry["hyp_words"] / reference_words, 4)
    entry["wer_micro_filler_symmetric"] = round(
        sum(row["errors_filler_symmetric"] for row in rows)
        / max(sum(row["total_words_filler_symmetric"] for row in rows), 1), 4,
    )
    return entry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument(
        "--system", action="append", required=True, metavar="NAME[=DIR]",
        help=f"one of {sorted(ADAPTERS)}; all but chirp3 need an output directory",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--csv", default=None, help="per-visit rows, one per system")
    parser.add_argument(
        "--max-minutes", type=float, default=None,
        help="hard cap on the scored window, on top of the per-transcript one",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    cohort = Path(args.cohort)
    cap = args.max_minutes * 60 if args.max_minutes else None

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

    per_visit: dict[str, dict] = {}
    scores: dict[str, list[dict]] = {name: [] for name, _, _ in requested}
    adapter_errors: dict[str, Counter] = {name: Counter() for name, _, _ in requested}
    missing: Counter = Counter()

    for index, visit in enumerate(visits, start=1):
        relative = visit.relative_to(cohort)
        human = sorted((visit / "human").glob("*.txt"))[0]
        try:
            turns = load_timestamped_text(human, 0.0)
        except Exception:
            continue
        turns, window_start, window_end = covered_turns(turns, cap)
        if not turns:
            continue
        reference_text = reference_prose(turns)
        if not preprocess_transcript(reference_text):
            continue

        for name, adapter, root in requested:
            try:
                words = adapter(visit, root, relative)
            except Exception as error:
                adapter_errors[name][f"{type(error).__name__}: {error}"] += 1
                words = None
            if not words:
                if words is None:
                    missing[name] += 1
                continue
            words = clip_words(words, window_start, window_end)
            if not words:
                continue
            hypothesis_text = " ".join(str(w.get("word", "")).strip() for w in words).strip()
            if not preprocess_transcript(hypothesis_text):
                continue
            row = score_pair(reference_text, hypothesis_text)
            row["visit"] = relative.as_posix()
            scores[name].append(row)
            per_visit.setdefault(relative.as_posix(), {})[name] = {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in row.items() if k != "visit"
            }
        if index % 25 == 0 or index == len(visits):
            logger.info("  %d/%d visits", index, len(visits))

    for name, errors in adapter_errors.items():
        if errors:
            logger.error(
                "%s: adapter raised on %d visit(s): %s",
                name, sum(errors.values()), dict(errors.most_common(3)),
            )
        if not scores[name]:
            logger.error("%s: scored 0 visits -- treat as a failure, not an absence", name)

    aggregate = {name: aggregate_rows(rows) for name, rows in scores.items() if rows}

    table = []
    for name, entry in sorted(aggregate.items(), key=lambda kv: kv[1]["wer_micro"]):
        table.append((
            registry.label_of(name),
            [
                ("visits", str(entry["visits"])),
                ("WER pooled", f"{entry['wer_micro']:.4f}"),
                ("WER mean", f"{entry['wer']:.4f}"),
                ("WER median", f"{entry['wer_median']:.4f}"),
                ("sub", f"{entry['substitutions_rate']:.4f}"),
                ("del", f"{entry['deletions_rate']:.4f}"),
                ("ins", f"{entry['insertions_rate']:.4f}"),
                ("WER no fillers", f"{entry['wer_micro_filler_symmetric']:.4f}"),
                ("word ratio", f"{entry['word_ratio']:.3f}"),
            ],
        ))
    print()
    print(registry.report(table))

    if args.output:
        payload = {
            "metric": "nvidia_wer.py (jiwer, brackets and punctuation stripped)",
            "window": "human-transcript coverage span"
                      + (f", capped at {args.max_minutes} min" if cap else ""),
            "aggregate": {registry.label_of(k): v for k, v in aggregate.items()},
            "per_visit": {
                visit: {registry.label_of(k): v for k, v in entry.items()}
                for visit, entry in per_visit.items()
            },
            "missing": dict(missing),
            "adapter_errors": {k: dict(v) for k, v in adapter_errors.items() if v},
        }
        Path(args.output).write_text(json.dumps(payload, indent=2))
        logger.info("wrote %s", args.output)

    if args.csv:
        fields = ["visit", "system", "wer", "wer_filler_symmetric", "mer", "wil",
                  "hits", "substitutions", "deletions", "insertions",
                  "total_words", "hyp_words"]
        with Path(args.csv).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for name, rows in scores.items():
                for row in rows:
                    writer.writerow({**row, "system": registry.label_of(name)})
        logger.info("wrote %s", args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

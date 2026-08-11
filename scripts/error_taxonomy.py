"""Classify the edits behind each system's WER, so the number can be explained.

WER on this corpus runs 84-88%, which reads as catastrophic and is not. The
report already splits it into substitutions, deletions and insertions, and shows
insertions carrying about half the total -- but "insertions" is not an
explanation, and the obvious guess is wrong: `evaluate_systems.normalize`
removes filled pauses from *both* sides before scoring, so um and uh cannot be
responsible for any of it.

This walks the same jiwer alignment over the same normalized text and puts every
edit in a category, so the total reconstructs the reported WER exactly and the
question "what is the machine saying that the human transcript does not" gets a
concrete answer.

Categories, chosen to separate transcription failure from convention mismatch:

  repetition        an inserted word the speaker actually said twice -- the
                    hypothesis repeats its own neighbour. Verbatim ASR keeps
                    stutters and restarts; semi-verbatim transcribers delete
                    them silently.
  backchannel       yeah, okay, right, mm -- acknowledgement tokens a
                    transcriber routinely drops as noise.
  discourse marker  like, so, just, well, you know, I mean.
  near-miss         a substitution whose two sides are one or two characters
                    apart, or differ only by inflection or number formatting:
                    the machine heard the word, and spelled it differently.
  different word    a substitution that is neither -- a genuine mishearing.
  deletion          reference words the machine did not produce.
  other insertion   everything else the machine added.

Usage:
    uv run python scripts/error_taxonomy.py --cohort /path/to/cohort \
        --system chirp3 --system ours=outputs/ours --output outputs/taxonomy.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "finetune"))

import jiwer  # noqa: E402
import systems as registry  # noqa: E402
from prepare_data import load_timestamped_text  # noqa: E402

from evaluate_systems import ADAPTERS, normalize  # noqa: E402

logger = logging.getLogger("error_taxonomy")

BACKCHANNEL = {
    "yeah", "yes", "yep", "okay", "ok", "right", "sure", "alright", "true",
    "exactly", "gotcha", "wow", "oh", "no", "nope", "cool", "good", "great",
}
DISCOURSE = {
    "like", "so", "just", "well", "actually", "basically", "literally",
    "anyway", "anyways", "know", "mean", "guess", "think", "kinda", "sorta",
}

# Digits spelled out, so "9th" against "ninth" is a formatting difference
# rather than a mishearing. Only the range that actually occurs in dates and
# ages here.
NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand", "first", "second", "third", "fourth", "fifth", "sixth",
    "seventh", "eighth", "ninth", "tenth",
}

CATEGORIES = [
    "repetition",
    "backchannel",
    "discourse marker",
    "other insertion",
    "near-miss",
    "different word",
    "deletion",
]


def edit_distance(a: str, b: str) -> int:
    """Character-level Levenshtein, small strings only."""
    if a == b:
        return 0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb),
            ))
        previous = current
    return previous[-1]


def is_near_miss(reference: str, hypothesis: str) -> bool:
    """Same word, written differently, rather than a different word heard."""
    if reference in NUMBER_WORDS and hypothesis in NUMBER_WORDS:
        return True
    if reference.isdigit() or hypothesis.isdigit():
        return True
    shorter, longer = sorted((reference, hypothesis), key=len)
    # Inflection: one is the other plus a short suffix.
    if len(shorter) >= 4 and longer.startswith(shorter) and len(longer) - len(shorter) <= 3:
        return True
    threshold = 1 if min(len(reference), len(hypothesis)) <= 4 else 2
    return edit_distance(reference, hypothesis) <= threshold


def classify_insertion(tokens: list[str], index: int) -> str:
    word = tokens[index]
    neighbours = {
        tokens[i] for i in (index - 2, index - 1, index + 1, index + 2)
        if 0 <= i < len(tokens) and i != index
    }
    if word in neighbours:
        return "repetition"
    if word in BACKCHANNEL:
        return "backchannel"
    if word in DISCOURSE:
        return "discourse marker"
    return "other insertion"


def classify_visit(reference_text: str, hypothesis_text: str) -> tuple[Counter, int]:
    reference = reference_text.split()
    hypothesis = hypothesis_text.split()
    if not reference or not hypothesis:
        return Counter(), 0

    output = jiwer.process_words(reference_text, hypothesis_text)
    counts: Counter = Counter()
    for chunk in output.alignments[0]:
        if chunk.type == "equal":
            continue
        if chunk.type == "insert":
            for index in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                counts[classify_insertion(hypothesis, index)] += 1
        elif chunk.type == "delete":
            counts["deletion"] += chunk.ref_end_idx - chunk.ref_start_idx
        elif chunk.type == "substitute":
            pairs = zip(
                range(chunk.ref_start_idx, chunk.ref_end_idx),
                range(chunk.hyp_start_idx, chunk.hyp_end_idx),
            )
            for ref_index, hyp_index in pairs:
                counts[
                    "near-miss" if is_near_miss(reference[ref_index], hypothesis[hyp_index])
                    else "different word"
                ] += 1
    return counts, len(reference)


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
        if name not in ADAPTERS:
            logger.error("Unknown system %s", name)
            return 1
        chosen.append((name, ADAPTERS[name][0], Path(directory) if directory else None))

    visits = sorted(
        p for p in cohort.glob("Pronet*/*/*")
        if (p / "human").is_dir() and any((p / "human").glob("*.txt"))
        and any((p / "audio").glob("*.wav"))
    )
    if args.limit:
        visits = visits[: args.limit]

    totals: dict[str, Counter] = {name: Counter() for name, _, _ in chosen}
    reference_words: Counter = Counter()
    scored: Counter = Counter()

    for index, visit in enumerate(visits, start=1):
        relative = visit.relative_to(cohort)
        human = sorted((visit / "human").glob("*.txt"))[0]
        try:
            turns = load_timestamped_text(human, 0.0)
        except Exception:
            continue
        reference_text = normalize(" ".join(t["text"] for t in turns))
        if not reference_text:
            continue

        for name, adapter, root in chosen:
            try:
                words = adapter(visit, root, relative)
            except Exception:
                words = None
            if not words:
                continue
            hypothesis_text = normalize(" ".join(str(w.get("word", "")) for w in words))
            counts, length = classify_visit(reference_text, hypothesis_text)
            if not length:
                continue
            totals[name].update(counts)
            reference_words[name] += length
            scored[name] += 1
        if index % 25 == 0 or index == len(visits):
            logger.info("  %d/%d visits", index, len(visits))

    aggregate = {}
    for name, counter in totals.items():
        if not scored[name]:
            logger.error("%s: scored 0 visits -- a failure, not an absence", name)
            continue
        length = reference_words[name] or 1
        rates = {c: counter[c] / length for c in CATEGORIES}
        aggregate[name] = {
            "visits": scored[name],
            "reference_words": reference_words[name],
            "counts": {c: counter[c] for c in CATEGORIES},
            "rates": rates,
            # Reconstructed WER. It must match the reported figure; if it does
            # not, the taxonomy is describing a different alignment than the
            # one the headline number came from.
            "wer_reconstructed": sum(rates.values()),
        }

    rows = []
    for name, stats in aggregate.items():
        rows.append((
            registry.label_of(name),
            [("WER", f"{stats['wer_reconstructed'] * 100:.1f}%")]
            + [(c, f"{stats['rates'][c] * 100:.1f}%") for c in CATEGORIES],
        ))
    print()
    print(registry.report(rows))

    if args.output:
        Path(args.output).write_text(
            json.dumps(
                {registry.label_of(k): v for k, v in aggregate.items()}, indent=2,
            ) + "\n"
        )
        logger.info("wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

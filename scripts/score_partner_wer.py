"""Score every system with the partner team's WER implementation.

`scripts/partner_compare.py` is their compareFiles.py, vendored unmodified;
this script imports its `tokenize`, `replace_fillers` and `analyze` so the
numbers below come from their code, not a reimplementation of it. Only the
plumbing is ours: the same visit list and the same system adapters that
`evaluate_systems.py` uses, so the two metrics are computed over identical
inputs and can be compared visit by visit.

Their metric differs from ours in three ways worth stating, because they move
the number:

  - difflib SequenceMatcher, not Levenshtein. A "replace" block costs
    max(ref_len, hyp_len) rather than an aligned edit count, and difflib
    optimises for the longest matching subsequence rather than the minimum
    edit distance, so this is an upper bound on WER, not the standard one.
  - Three tiers: raw (case and punctuation intact), normalized (both
    stripped), and filler-normalized (every filled pause on both sides
    collapsed to a single [filler] token). Their tier 3 keeps fillers as
    one comparable token; ours deletes them outright.
  - Bracketed and braced markup is stripped but the token it prefixes is
    kept, so CrisperWhisper's [UM] and the human transcripts' [inaudible]
    both vanish before tier 1 even runs.

Reference text is the human transcript file with only the "S1 HH:MM:SS.mmm"
line prefixes removed -- their script expects prose, and leaving the metadata
in would score every system against tokens no ASR can emit. Nothing else is
touched: the transcriber's brackets reach their tokenizer intact, which is
what would happen if they ran compareFiles.py on the file themselves. (This
is deliberately not `load_timestamped_text`, whose normalizer unwraps
"[psychs?]" to "psychs" for forced alignment; their tokenizer deletes the
bracket instead, and the reference must follow their rule, not ours.)
Hypothesis text is the system's words joined in time order.

Usage:
    uv run python scripts/score_partner_wer.py --cohort /path/to/cohort \
        --system chirp3 --system ours=outputs/ours \
        --output partner_wer.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from partner_compare import analyze, replace_fillers, tokenize  # noqa: E402

from evaluate_systems import ADAPTERS  # noqa: E402

logger = logging.getLogger("score_partner_wer")

TIERS = ("raw", "normalized", "filler_normalized")

# "S1 00:00:01.451 " at the head of a turn; continuation lines carry no prefix.
TURN_PREFIX = re.compile(r"^\s*S\d+\s+\d{1,2}:\d{2}:\d{2}(?:\.\d+)?\s*")


def reference_prose(path: Path) -> str:
    """The human transcript as prose: speaker tags and timestamps removed."""
    lines = [TURN_PREFIX.sub("", line).strip() for line in path.read_text().splitlines()]
    return re.sub(r"\s+", " ", " ".join(line for line in lines if line)).strip()


def partner_wer(reference_text: str, hypothesis_text: str) -> dict[str, float]:
    """The three numbers their compareFiles.py prints, from their functions."""
    ref_raw = tokenize(reference_text)
    hyp_raw = tokenize(hypothesis_text)
    ref_norm = tokenize(reference_text, normalize=True)
    hyp_norm = tokenize(hypothesis_text, normalize=True)
    return {
        "raw": analyze(ref_raw, hyp_raw),
        "normalized": analyze(ref_norm, hyp_norm),
        "filler_normalized": analyze(replace_fillers(ref_norm), replace_fillers(hyp_norm)),
        "ref_words": len(ref_norm),
        "hyp_words": len(hyp_norm),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument(
        "--system", action="append", required=True, metavar="NAME[=DIR]",
        help=f"one of {sorted(ADAPTERS)}; all but chirp3 need an output directory",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    cohort = Path(args.cohort)

    systems = []
    for spec in args.system:
        name, _, directory = spec.partition("=")
        if name not in ADAPTERS:
            logger.error("Unknown system %s; known: %s", name, sorted(ADAPTERS))
            return 1
        systems.append((name, ADAPTERS[name][0], Path(directory) if directory else None))

    visits = sorted(
        p for p in cohort.glob("Pronet*/*/*")
        if (p / "human").is_dir() and any((p / "human").glob("*.txt"))
        and any((p / "audio").glob("*.wav"))
    )
    if args.limit:
        visits = visits[: args.limit]
    logger.info("Scoring %d system(s) over %d visit(s)", len(systems), len(visits))

    per_visit: dict[str, dict] = {}
    scores: dict[str, list[dict]] = {name: [] for name, _, _ in systems}
    adapter_errors: dict[str, Counter] = {name: Counter() for name, _, _ in systems}
    missing: Counter = Counter()

    for index, visit in enumerate(visits, start=1):
        relative = visit.relative_to(cohort)
        human = sorted((visit / "human").glob("*.txt"))[0]
        reference_text = reference_prose(human)
        if not reference_text:
            continue

        for name, adapter, root in systems:
            try:
                words = adapter(visit, root, relative)
            except Exception as error:
                adapter_errors[name][f"{type(error).__name__}: {error}"] += 1
                words = None
            if not words:
                if words is None:
                    missing[name] += 1
                continue
            hypothesis_text = " ".join(str(w.get("word", "")).strip() for w in words).strip()
            if not hypothesis_text:
                continue
            result = partner_wer(reference_text, hypothesis_text)
            scores[name].append(result)
            per_visit.setdefault(relative.as_posix(), {})[name] = {
                k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.items()
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

    aggregate: dict[str, dict] = {}
    for name, rows in scores.items():
        if not rows:
            continue
        entry: dict[str, float | int] = {"visits": len(rows)}
        for tier in TIERS:
            values = [row[tier] for row in rows]
            entry[tier] = round(statistics.fmean(values), 4)
            entry[f"{tier}_median"] = round(statistics.median(values), 4)
        entry["ref_words"] = sum(row["ref_words"] for row in rows)
        entry["hyp_words"] = sum(row["hyp_words"] for row in rows)
        entry["word_ratio"] = round(entry["hyp_words"] / max(entry["ref_words"], 1), 4)
        aggregate[name] = entry

    width = max((len(n) for n in aggregate), default=6)
    print(f"\n{'system':{width}}  visits  {'raw':>8}  {'normalized':>11}  {'filler-norm':>12}  {'ratio':>6}")
    print("-" * (width + 56))
    for name, entry in sorted(aggregate.items(), key=lambda kv: kv[1]["filler_normalized"]):
        print(
            f"{name:{width}}  {entry['visits']:6d}  {entry['raw']:7.2f}%  "
            f"{entry['normalized']:10.2f}%  {entry['filler_normalized']:11.2f}%  "
            f"{entry['word_ratio']:6.3f}"
        )

    if args.output:
        payload = {
            "metric": "partner compareFiles.py (difflib SequenceMatcher, percent)",
            "aggregate": aggregate,
            "per_visit": per_visit,
            "missing": dict(missing),
            "adapter_errors": {k: dict(v) for k, v in adapter_errors.items() if v},
        }
        Path(args.output).write_text(json.dumps(payload, indent=2))
        logger.info("wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

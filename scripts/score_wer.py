"""Score transcript variants against the human transcripts by WER.

Answers one question: does a change to the transcription actually move it
closer to what a human wrote? Any number of variants can be scored over the
same visits, so "baseline" and "baseline + LLM corrections" (or Chirp-3, or
verbatimize output) sit in one table.

Because the human transcripts are semi-verbatim and the ASR is fully
verbatim, absolute WER is inflated by disfluencies the human never wrote
down -- the meaningful signal is the *difference* between variants over the
same reference, not the absolute level. Text is normalized identically for
every variant: transcriber markup stripped, vocal-event tokens like [UM]
removed, lowercased, punctuation dropped.

Only aggregate error rates are printed, never transcript text.

Usage:
    uv run python scripts/score_wer.py --cohort /path/to/cohort \
        --variant baseline=baseline/outputs:_transcript.txt \
        --variant corrected=baseline/outputs:_transcript_corrected.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "finetune"))

import jiwer  # noqa: E402
from prepare_data import load_timestamped_text, normalize_text  # noqa: E402

logger = logging.getLogger("score_wer")

TURN_LINE = re.compile(r"^\[[\d.?]+ - [\d.?]+\] \S+: (?P<text>.*)$")
# CrisperWhisper marks vocal events in brackets ([UM], [laughter]); human
# transcribers never write these, so they are noise for a WER comparison.
EVENT_TOKEN = re.compile(r"\[[^\]]*\]")


def normalize_for_wer(text: str) -> str:
    text = EVENT_TOKEN.sub(" ", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def transcript_body(path: Path) -> str:
    """Strip "[start - end] SPEAKER:" prefixes, keeping only spoken text."""
    parts = []
    for line in path.read_text().splitlines():
        match = TURN_LINE.match(line.strip())
        parts.append(match.group("text") if match else line)
    return " ".join(parts)


def human_body(path: Path) -> str:
    turns = load_timestamped_text(path, 0.0)
    return " ".join(normalize_text(t["text"]) for t in turns)


def parse_variant(spec: str) -> tuple[str, Path, str]:
    name, _, rest = spec.partition("=")
    root, _, suffix = rest.partition(":")
    return name, Path(root), (suffix or "_transcript.txt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, help="cohort root holding human/ transcripts")
    parser.add_argument(
        "--variant", action="append", required=True, metavar="NAME=DIR[:SUFFIX]",
        help="a transcript tree to score (repeatable)",
    )
    parser.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    cohort = Path(args.cohort)
    variants = [parse_variant(v) for v in args.variant]

    # Visits scoreable by every variant, so the comparison is like for like.
    per_variant_visits = []
    for _, root, suffix in variants:
        found = {}
        for path in root.rglob(f"*{suffix}"):
            found[path.parent.relative_to(root).as_posix()] = path
        per_variant_visits.append(found)
    shared = set.intersection(*(set(v) for v in per_variant_visits)) if per_variant_visits else set()

    usable = []
    for visit in sorted(shared):
        human = sorted((cohort / visit / "human").glob("*.txt"))
        if human:
            usable.append((visit, human[0]))
    logger.info("Scoring %d visit(s) across %d variant(s)", len(usable), len(variants))
    if not usable:
        logger.error("No visits have both a human transcript and every variant")
        return 1

    results: dict[str, dict] = {}
    per_visit: dict[str, dict] = {}

    for (name, _, _), found in zip(variants, per_variant_visits):
        references, hypotheses = [], []
        for visit, human_path in usable:
            reference = normalize_for_wer(human_body(human_path))
            hypothesis = normalize_for_wer(transcript_body(found[visit]))
            if not reference or not hypothesis:
                continue
            references.append(reference)
            hypotheses.append(hypothesis)
            measure = jiwer.process_words(reference, hypothesis)
            per_visit.setdefault(visit, {})[name] = round(measure.wer, 4)

        measures = jiwer.process_words(references, hypotheses)
        results[name] = {
            "visits": len(references),
            "wer": measures.wer,
            "mer": measures.mer,
            "substitutions": measures.substitutions,
            "deletions": measures.deletions,
            "insertions": measures.insertions,
            "hits": measures.hits,
        }

    baseline_name = variants[0][0]
    baseline_wer = results[baseline_name]["wer"]
    print(f"\n  {'variant':22s} {'visits':>6} {'WER':>9} {'vs first':>10} {'sub':>7} {'del':>7} {'ins':>7}")
    for name, stats in results.items():
        delta = stats["wer"] - baseline_wer
        arrow = "" if name == baseline_name else f"{delta:+.4f}"
        print(
            f"  {name:22s} {stats['visits']:6d} {stats['wer']:9.4f} {arrow:>10} "
            f"{stats['substitutions']:7d} {stats['deletions']:7d} {stats['insertions']:7d}"
        )

    if len(variants) > 1:
        moved = {n: 0 for n, _, _ in variants[1:]}
        worsened = dict(moved)
        for visit, scores in per_visit.items():
            for name in moved:
                if name in scores and baseline_name in scores:
                    if scores[name] < scores[baseline_name]:
                        moved[name] += 1
                    elif scores[name] > scores[baseline_name]:
                        worsened[name] += 1
        print()
        for name in moved:
            print(f"  {name}: better on {moved[name]} visit(s), worse on {worsened[name]}, "
                  f"unchanged on {len(per_visit) - moved[name] - worsened[name]}")

    if args.output:
        Path(args.output).write_text(
            json.dumps({"aggregate": results, "per_visit": per_visit}, indent=2) + "\n"
        )
        logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

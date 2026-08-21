"""Reassign short speaker runs to their dominant neighbour. No model.

This is the comparator the Gemma corrector has to beat. The repo has already
published one LLM post-processor that moved every metric by exactly zero, and
without a trivial baseline a second null result would be uninterpretable: it
would not distinguish "the model cannot do this" from "this class of error
cannot be fixed by relabelling at all".

The rule: any run of MAX_RUN words or fewer is given the speaker of the longer
adjacent run. A run bounded by two runs of the same speaker is the clearest
case -- one or two words attributed to the other party in the middle of an
uninterrupted stretch of speech. Runs at the very start or end of the
transcript have only one neighbour and take it.

The rule is deliberately blind to the words. It cannot tell a genuine
one-word answer ("no") from a diarization slip, and roughly six of every ten
one-word runs in our output are already correct, so it will move some words
the wrong way. That is the honest baseline: whatever a model contributes over
this is its contribution to reading the conversation rather than the timeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from speaker_rewrite import (  # noqa: E402
    apply_changes,
    destination_for,
    load_words,
    speaker_runs,
    write_corrected,
)

logger = logging.getLogger("correct_speakers_rule")

MAX_RUN = 3
SOURCE = "rule"


def propose(words: list[dict], max_run: int = MAX_RUN) -> tuple[dict[int, str], Counter]:
    """Speaker changes by index, and a count of why each run was or was not moved.

    Neighbours are read from the *original* run layout, not from a partially
    rewritten one: applying changes as we go would let one reassignment merge
    two runs and change what the next short run sees, making the result depend
    on iteration order.
    """
    runs = speaker_runs(words)
    changes: dict[int, str] = {}
    reasons: Counter = Counter()
    for position, (start, end) in enumerate(runs):
        if end - start > max_run:
            reasons["kept: run too long"] += 1
            continue
        before = runs[position - 1] if position > 0 else None
        after = runs[position + 1] if position + 1 < len(runs) else None
        candidates = [r for r in (before, after) if r is not None]
        if not candidates:
            reasons["kept: no neighbour"] += 1
            continue
        # Longer neighbour wins; a tie goes to the preceding run, because a
        # short interjection is more often absorbed from the speech it
        # interrupts than from the speech that follows it.
        longest = max(candidates, key=lambda r: (r[1] - r[0], r is before))
        target = words[longest[0]].get("speaker")
        if target is None or target == words[start].get("speaker"):
            reasons["kept: neighbour has the same speaker"] += 1
            continue
        for index in range(start, end):
            changes[index] = target
        reasons["moved"] += 1
    return changes, reasons


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", required=True, help="tree of transcripts to correct")
    parser.add_argument("--pattern", default="transcript.json")
    parser.add_argument("--suffix", default="_speakers_rule.json")
    parser.add_argument("--max-run", type=int, default=MAX_RUN)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", action="append", default=None, help="substring filter, repeatable")
    parser.add_argument("--redo", action="store_true", help="rewrite files that already exist")
    parser.add_argument("--report", default=None, help="JSON summary over the whole tree")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    args = build_parser().parse_args(argv)

    root = Path(args.outputs)
    files = sorted(p for p in root.rglob(args.pattern) if not p.name.endswith(args.suffix))
    if args.only:
        files = [f for f in files if any(token in str(f) for token in args.only)]
    if not args.redo:
        files = [f for f in files if not destination_for(f, args.suffix).exists()]
    if args.limit:
        files = files[: args.limit]
    logger.info("%d transcripts under %s", len(files), root)

    totals: Counter = Counter()
    per_file: dict[str, dict] = {}
    for position, path in enumerate(files, start=1):
        payload, words = load_words(path)
        if not words:
            logger.warning("%s: no words", path)
            continue
        changes, reasons = propose(words, args.max_run)
        corrected = apply_changes(words, changes, SOURCE)
        moved = sum(1 for old, new in zip(words, corrected)
                    if old.get("speaker") != new.get("speaker"))
        report = {
            "source": SOURCE,
            "max_run": args.max_run,
            "words": len(words),
            "runs_moved": reasons["moved"],
            "words_moved": moved,
            "reasons": dict(reasons),
        }
        write_corrected(path, payload, words, corrected, report, args.suffix)
        per_file[str(path.relative_to(root))] = report
        totals.update(reasons)
        totals["words"] += len(words)
        totals["words_moved"] += moved
        if position % 50 == 0:
            logger.info("%d/%d", position, len(files))

    logger.info(
        "moved %d of %d words (%.2f%%) across %d runs",
        totals["words_moved"], totals["words"],
        totals["words_moved"] / max(totals["words"], 1) * 100, totals["moved"],
    )
    for reason, count in sorted(totals.items()):
        if reason.startswith("kept") or reason == "moved":
            logger.info("    %-40s %d", reason, count)

    if args.report:
        Path(args.report).write_text(json.dumps(
            {"totals": dict(totals), "per_file": per_file}, indent=1))
        logger.info("wrote %s", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

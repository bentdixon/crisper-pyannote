"""Score word timestamps against the human transcripts' turn start times.

The human transcriptions carry one trusted time per turn (the moment that
speaker starts talking). That is coarse, but it is the only independent
ground truth available -- Chirp's offsets and CrisperWhisper's
cross-attention timings are both machine estimates and agreeing with each
other proves nothing.

Method: concatenate the human turn texts into one reference token stream,
remembering which token opens each turn, and align that stream to the
candidate transcript's words with difflib. Wherever a turn's opening token
matches a candidate word, the absolute difference between the candidate
word's start and the human turn time is one error sample. Turns whose
opening token does not match are skipped rather than guessed at.

Compare any number of candidate transcripts (Chirp, verbatimize, and
verbatimize --realign) over the same sessions to see which timing source is
closest to the human annotation. Only aggregate error statistics are
printed, never transcript text.

Usage:
    uv run python scripts/score_timestamps.py --human-dir testdata/human \
        --chirp-dir testdata/chirp \
        --candidate windowed=testdata/out2 --candidate realigned=testdata/out3
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "finetune"))

import difflib

from crisperwhisper.forced_align import default_normalize
from prepare_data import load_timestamped_text  # noqa: E402

from crisper_pipeline import chirp
from crisper_pipeline.verbatimize_cli import normalize_stem

logger = logging.getLogger("score_timestamps")

RUN_SUFFIX = re.compile(r"_\d{8}-\d{6}(-\d+)?$")


def turn_anchors(turns: list[dict]) -> tuple[list[str], dict[int, float]]:
    """Flatten turns into tokens plus {token index: human turn start}."""
    tokens: list[str] = []
    anchors: dict[int, float] = {}
    for turn in turns:
        words = turn["text"].split()
        if not words:
            continue
        anchors[len(tokens)] = turn["start"]
        tokens.extend(words)
    return tokens, anchors


def score(candidate_words: list[dict], turns: list[dict]) -> list[float]:
    """Absolute error in seconds at each matched turn boundary."""
    tokens, anchors = turn_anchors(turns)
    matcher = difflib.SequenceMatcher(
        a=[default_normalize(t) for t in tokens],
        b=[default_normalize(w["word"]) for w in candidate_words],
        autojunk=False,
    )
    errors: list[float] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            continue
        for offset, index in enumerate(range(i1, i2)):
            if index in anchors:
                errors.append(abs(candidate_words[j1 + offset]["start"] - anchors[index]))
    return errors


def load_candidate(directory: Path) -> dict[str, list[dict]]:
    """Map session key -> word list for a verbatimize-session output dir."""
    found: dict[str, list[dict]] = {}
    for path in sorted(directory.glob("*/transcript.json")):
        key = normalize_stem(RUN_SUFFIX.sub("", path.parent.name))
        found[key] = json.loads(path.read_text())["words"]
    return found


def summarize(errors: list[float]) -> dict[str, float]:
    ordered = sorted(errors)
    return {
        "turns": len(ordered),
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p90": ordered[int(0.9 * (len(ordered) - 1))],
        "within_0.1s": sum(1 for e in ordered if e <= 0.1) / len(ordered),
        "within_0.5s": sum(1 for e in ordered if e <= 0.5) / len(ordered),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-dir", required=True, help="directory of human *_REDACTED.txt")
    parser.add_argument("--chirp-dir", required=True, help="Chirp transcripts, scored as a baseline")
    parser.add_argument(
        "--candidate", action="append", default=[], metavar="NAME=DIR",
        help="a verbatimize-session output directory to score (repeatable)",
    )
    parser.add_argument("--output", default=None, help="write results JSON here")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    human = {normalize_stem(p.stem): p for p in sorted(Path(args.human_dir).glob("*.txt"))}
    chirp_paths = {normalize_stem(p.stem): p for p in sorted(Path(args.chirp_dir).glob("*.json"))}
    candidates = {}
    for entry in args.candidate:
        name, _, directory = entry.partition("=")
        candidates[name] = load_candidate(Path(directory))
    logger.info(
        "%d human transcript(s), candidates: %s",
        len(human), ", ".join(candidates) or "none",
    )

    errors: dict[str, list[float]] = {name: [] for name in ["chirp", *candidates]}
    for key, human_path in sorted(human.items()):
        if key not in chirp_paths:
            logger.warning("No Chirp transcript for %s; skipping", human_path.name)
            continue
        source = chirp.load_transcript(chirp_paths[key])
        turns = load_timestamped_text(human_path, source["duration"] or 0.0)

        per_session = {"chirp": score(source["words"], turns)}
        for name, words_by_key in candidates.items():
            if key in words_by_key:
                per_session[name] = score(words_by_key[key], turns)

        logger.info(
            "%s | %d human turns | matched %s",
            human_path.stem[:34], len(turns),
            " ".join(f"{n}={len(e)}" for n, e in per_session.items()),
        )
        for name, values in per_session.items():
            errors[name].extend(values)

    print(f"\n  {'source':12s} {'turns':>6} {'median':>8} {'mean':>8} {'p90':>8} "
          f"{'<=0.1s':>8} {'<=0.5s':>8}")
    results = {}
    for name, values in errors.items():
        if not values:
            continue
        stats = summarize(values)
        results[name] = stats
        print(
            f"  {name:12s} {stats['turns']:6d} {stats['median']:7.3f}s {stats['mean']:7.3f}s "
            f"{stats['p90']:7.3f}s {stats['within_0.1s']:7.1%} {stats['within_0.5s']:7.1%}"
        )

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
        logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

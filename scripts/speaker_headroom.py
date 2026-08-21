"""How much a speaker-relabelling post-processor could possibly win.

A corrector that only rewrites `word["speaker"]` cannot invent words, so its
ceiling is fixed by the words already in the transcript. This measures that
ceiling directly rather than guessing at it, by relabelling every word to the
reference speaker who was talking at that instant and rescoring:

    baseline    the system as it stands
    oracle      every word given its true speaker -- the ceiling for any
                relabeller, and a number no real corrector can reach
    short-run   only words in runs of MAX_RUN or fewer given their true
                speaker -- the ceiling for a corrector aimed at the small words
                between utterances, which is the class short turns fall into

The gap between `short-run` and `oracle` is how much of the problem lives
outside short runs; the gap between `baseline` and `short-run` is the whole
prize on offer. Run this before spending GPU time on a model: the repo has one
published LLM post-processor that moved every metric by exactly zero, and it
was built without ever asking what the best possible outcome would have been.

The oracle labels words with reference speaker names, so the Hungarian
assignment in `score_visit` becomes an identity mapping. That is intended --
an oracle is allowed to know the answer -- but it means oracle sWER is not
comparable to a real system's sWER for any purpose except bounding it.
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "finetune"))

import systems as registry  # noqa: E402
from coverage import clip_words, covered_turns  # noqa: E402
from evaluate_systems import ADAPTERS, as_words, score_visit  # noqa: E402
from prepare_data import load_timestamped_text  # noqa: E402

logger = logging.getLogger("speaker_headroom")

# A run of this many words or fewer is the class a corrector for "small words
# between utterances" would act on. Chosen from the measured error profile:
# one-word runs are wrong 40% of the time and four-word-plus runs 5%, with the
# crossover inside this range.
MAX_RUN = 3

# Metrics that can move when only speaker labels change. WER is deliberately
# absent -- it ignores speakers, so it is an invariant rather than a result,
# and it is checked as one below.
MOVABLE = ["swer", "der", "der_confusion", "qtp_f1"]


def speaker_at(turns: list[dict], starts: list[float], moment: float) -> str | None:
    """Reference speaker talking at `moment`, or None outside every turn."""
    index = bisect.bisect_right(starts, moment) - 1
    if index < 0:
        return None
    turn = turns[index]
    return turn["speaker"] if moment <= turn["end"] else None


def speaker_runs(words: list[dict]) -> list[tuple[int, int]]:
    """Half-open index ranges of consecutive words sharing a speaker label."""
    runs: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(words) + 1):
        if index == len(words) or words[index].get("speaker") != words[start].get("speaker"):
            runs.append((start, index))
            start = index
    return runs


def relabel(words: list[dict], turns: list[dict], max_run: int | None) -> tuple[list[dict], int]:
    """Copy of `words` with true speakers filled in, and how many changed.

    `max_run` of None relabels everything; an integer restricts the rewrite to
    runs of at most that length. Words the reference places in no turn keep
    their original label -- an oracle should not be credited for guessing in a
    region the transcript never covered.
    """
    starts = [t["start"] for t in turns]
    out = [dict(word) for word in words]
    changed = 0
    for start, end in speaker_runs(out):
        if max_run is not None and end - start > max_run:
            continue
        for index in range(start, end):
            word = out[index]
            middle = (float(word["start"]) + float(word["end"])) / 2
            truth = speaker_at(turns, starts, middle)
            if truth is None or truth == word.get("speaker"):
                continue
            word["speaker"] = truth
            changed += 1
    return out, changed


def word_accuracy(words: list[dict], turns: list[dict], mapping: dict[str, str]) -> tuple[int, int]:
    """Words whose mapped speaker matches the reference, and words scored."""
    starts = [t["start"] for t in turns]
    correct = scored = 0
    for word in words:
        truth = speaker_at(turns, starts, (float(word["start"]) + float(word["end"])) / 2)
        if truth is None:
            continue
        scored += 1
        if mapping.get(word.get("speaker") or "UNKNOWN") == truth:
            correct += 1
    return correct, scored


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument(
        "--system", action="append", required=True, metavar="NAME[=DIR]",
        help=f"one of {sorted(ADAPTERS)}; all but chirp3 need an output directory",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-run", type=int, default=MAX_RUN)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    args = build_parser().parse_args(argv)
    cohort = Path(args.cohort)

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
    logger.info("%d visits, %d systems, max run %d", len(visits), len(requested), args.max_run)

    import soundfile as sf

    rows: dict[str, list[dict]] = {name: [] for name, _, _ in requested}
    for position, visit in enumerate(visits, start=1):
        human = sorted((visit / "human").glob("*.txt"))
        audio = sorted((visit / "audio").glob("*.wav"))
        try:
            duration = sf.info(str(audio[0])).duration
        except Exception as error:  # noqa: BLE001 - a bad wav must not stop the sweep
            logger.warning("%s: unreadable audio (%s)", visit.name, error)
            continue
        turns, window_start, window_end = covered_turns(load_timestamped_text(human[0], duration))
        if not turns or window_start is None:
            continue
        relative = visit.relative_to(cohort)

        for name, adapter, root in requested:
            try:
                words = as_words(adapter(visit, root, relative))
            except Exception as error:  # noqa: BLE001 - counted, never silent
                logger.warning("%s/%s: adapter failed (%s)", name, visit.name, error)
                continue
            if not words:
                continue
            words = clip_words(words, window_start, window_end)
            if not words:
                continue

            entry = {"visit": relative.as_posix(), "words": len(words)}
            for variant, cap in (("baseline", 0), ("short_run", args.max_run), ("oracle", None)):
                if variant == "baseline":
                    candidate, changed = words, 0
                else:
                    candidate, changed = relabel(words, turns, cap)
                scored = score_visit(turns, candidate)
                if scored is None:
                    entry = None
                    break
                entry[f"{variant}_changed"] = changed
                for key in MOVABLE + ["wer"]:
                    entry[f"{variant}_{key}"] = scored[key]
            if entry:
                rows[name].append(entry)

        if position % 25 == 0:
            logger.info("%d/%d visits", position, len(visits))

    aggregate: dict[str, dict] = {}
    for name, entries in rows.items():
        if not entries:
            logger.error("%s scored zero visits", registry.label_of(name))
            continue
        block: dict = {"visits": len(entries)}
        total_words = sum(e["words"] for e in entries)
        for variant in ("baseline", "short_run", "oracle"):
            changed = sum(e[f"{variant}_changed"] for e in entries)
            block[f"{variant}_changed_words"] = changed
            block[f"{variant}_changed_share"] = round(changed / max(total_words, 1), 4)
            for key in MOVABLE + ["wer"]:
                values = [e[f"{variant}_{key}"] for e in entries if e.get(f"{variant}_{key}") is not None]
                block[f"{variant}_{key}"] = round(sum(values) / len(values), 4) if values else None
        aggregate[registry.label_of(name)] = block

        # WER ignores speakers, so relabelling cannot move it. A difference
        # here means the relabeller corrupted the word list, not that the
        # oracle helped -- report it as the failure it is rather than letting a
        # plausible table hide it.
        for variant in ("short_run", "oracle"):
            if block[f"{variant}_wer"] != block["baseline_wer"]:
                logger.error(
                    "%s: %s WER moved %.4f -> %.4f; relabelling must not touch words",
                    registry.label_of(name), variant, block["baseline_wer"],
                    block[f"{variant}_wer"],
                )

    for label, block in aggregate.items():
        print(f"\n{label}")
        print(f"    visits {block['visits']}")
        for variant in ("baseline", "short_run", "oracle"):
            print(
                f"    {variant:10s} "
                + "  ".join(f"{k} {block[f'{variant}_{k}']}" for k in MOVABLE)
                + f"   words relabelled {block[f'{variant}_changed_words']}"
                f" ({block[f'{variant}_changed_share'] * 100:.2f}%)"
            )

    if args.output:
        payload = {"max_run": args.max_run, "aggregate": aggregate, "per_visit": rows}
        Path(args.output).write_text(json.dumps(payload, indent=1))
        logger.info("wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

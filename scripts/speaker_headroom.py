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
import difflib
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "finetune"))

import systems as registry  # noqa: E402
from coverage import clip_words, covered_turns  # noqa: E402
from evaluate_systems import (  # noqa: E402
    ADAPTERS,
    align_streams,
    as_words,
    predicted_streams,
    reference_streams,
    score_visit,
)
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

# Same tokenizer as the lost-turns content measure: bracketed markup dropped so
# CrisperWhisper's [UM] does not have to match a typed "um", braces dropped so
# the PII convention does not break an alignment.
TOKEN = re.compile(r"[a-z0-9']+")


def content_tokens(text: str) -> list[str]:
    text = re.sub(r"\[[^\]]*\]", " ", (text or "").lower())
    return TOKEN.findall(text.replace("{", " ").replace("}", " "))


def speaker_at(turns: list[dict], starts: list[float], moment: float) -> str | None:
    """Reference speaker talking at `moment`, or None outside every turn."""
    index = bisect.bisect_right(starts, moment) - 1
    if index < 0:
        return None
    turn = turns[index]
    return turn["speaker"] if moment <= turn["end"] else None


def truth_by_content(words: list[dict], turns: list[dict]) -> dict[int, str]:
    """Speaker per hypothesis word, taken from the transcript text it matches.

    The obvious oracle -- give each word the speaker the reference has talking
    at that instant -- does not work here, and the failure is instructive. Human
    turn marks are accurate to about a third of a second, so a timing anchor
    misfiles the words either side of every boundary, and there are tens of
    thousands of boundaries. Measured on six visits it scored *worse* than the
    system it was meant to bound: sWER 0.246 against a baseline 0.144. The
    diarizer's boundaries are sharper than the annotation's, so a timing oracle
    is not an upper bound on anything.

    Aligning the two token streams instead makes the anchor the words
    themselves, which is what `lost_turns.py` had to do for the same reason.
    Words that match nothing in the reference -- insertions, and speech the
    typist never transcribed -- get no entry and keep whatever label they had.
    """
    ref_tokens: list[str] = []
    ref_speaker: list[str] = []
    for turn in turns:
        for token in content_tokens(turn["text"]):
            ref_tokens.append(token)
            ref_speaker.append(turn["speaker"])

    hyp_tokens: list[str] = []
    owner: list[int] = []
    for index, word in enumerate(words):
        for token in content_tokens(word["word"]):
            hyp_tokens.append(token)
            owner.append(index)

    truth: dict[int, str] = {}
    matcher = difflib.SequenceMatcher(None, ref_tokens, hyp_tokens, autojunk=False)
    for i, j, size in matcher.get_matching_blocks():
        for offset in range(size):
            # First match wins: a word contributing two tokens that straddle a
            # turn boundary belongs to the turn it starts in.
            truth.setdefault(owner[j + offset], ref_speaker[i + offset])
    return truth


def speaker_runs(words: list[dict]) -> list[tuple[int, int]]:
    """Half-open index ranges of consecutive words sharing a speaker label."""
    runs: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(words) + 1):
        if index == len(words) or words[index].get("speaker") != words[start].get("speaker"):
            runs.append((start, index))
            start = index
    return runs


def relabel(
    words: list[dict], truth: dict[int, str], max_run: int | None,
    mapping: dict[str, str] | None = None,
) -> tuple[list[dict], int, int]:
    """Copy of `words` with true speakers filled in, changed count, anchored count.

    `max_run` of None relabels every anchored word; an integer restricts the
    rewrite to runs of at most that length. Words with no entry in `truth`
    match nothing in the transcript and keep their original label -- an oracle
    should not be credited for guessing where there is no answer.

    "Changed" counts words whose speaker actually moved, compared through
    `mapping` (hypothesis label -> reference label). Without it every word
    would count as changed, because the pipeline writes SPEAKER_00 where the
    transcript writes INTERVIEWER, and a rename is not a correction.
    """
    mapping = mapping or {}
    # Every word is first renamed into the reference's label space. Without
    # this the output carries INTERVIEWER on the words the oracle touched and
    # SPEAKER_00 on the rest, which is four streams where there are two: sWER
    # improved but DER nearly doubled, because the two halves of one speaker
    # could not be matched to each other. A label that has no counterpart
    # (UNKNOWN, or a third speaker the assignment did not use) keeps its own
    # name and is charged as its own stream, which is correct.
    out = [dict(word) for word in words]
    for word in out:
        label = word.get("speaker") or "UNKNOWN"
        word["speaker"] = mapping.get(label, label)
    changed = anchored = 0
    for start, end in speaker_runs(out):
        if max_run is not None and end - start > max_run:
            continue
        for index in range(start, end):
            if index not in truth:
                continue
            anchored += 1
            if out[index].get("speaker") != truth[index]:
                changed += 1
            out[index]["speaker"] = truth[index]
    return out, changed, anchored


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

            # Hypothesis labels are SPEAKER_00/01, reference labels are
            # INTERVIEWER/PARTICIPANT or S1/S2, so "changed" is only meaningful
            # through the same assignment the scorer itself uses.
            assignment = align_streams(reference_streams(turns), predicted_streams(words))
            mapping = {hyp: ref for ref, hyp in assignment.items() if hyp is not None}
            truth = truth_by_content(words, turns)

            entry = {"visit": relative.as_posix(), "words": len(words)}
            for variant, cap in (("baseline", 0), ("short_run", args.max_run), ("oracle", None)):
                if variant == "baseline":
                    candidate, changed, anchored = words, 0, 0
                else:
                    candidate, changed, anchored = relabel(words, truth, cap, mapping)
                scored = score_visit(turns, candidate)
                if scored is None:
                    entry = None
                    break
                entry[f"{variant}_changed"] = changed
                entry[f"{variant}_anchored"] = anchored
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
            block[f"{variant}_anchored_words"] = sum(e[f"{variant}_anchored"] for e in entries)
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

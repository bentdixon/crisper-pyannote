"""Run the LLM review on finished transcripts and apply what it suggests.

The original pipeline deliberately stops at suggestions: it writes
{prefix}_llm_suggestion.json and never touches the transcript. That is the
right default for production, but it also means nobody can tell whether the
suggestions would actually help. This script closes that loop -- it produces
the suggestions with their exact prompt and their exact validation, then
writes a *separate* corrected copy so the original and corrected transcripts
can be scored against the human transcript side by side.

Nothing here overwrites a pipeline output. For each visit it adds:
    {prefix}_llm_suggestion.json    the raw (validated) suggestions
    {prefix}_transcript_corrected.txt
    {prefix}_words_corrected.json
    {prefix}_llm_applied.json       what was actually changed, and what was
                                    suggested but could not be applied

Their three categories map onto edits like this:
    word_corrections    replace whole-word occurrences of "original" with
                        "suggested" in both the text and the word list
    speaker_flags       flip the flagged turn to the other role
    role_mapping_check  if looks_correct is false, swap INTERVIEWER and
                        PARTICIPANT across the whole transcript

Usage:
    uv run python baseline/apply_llm_corrections.py --outputs baseline/outputs
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

logger = logging.getLogger("apply_llm_corrections")

TURN_LINE = re.compile(r"^\[(?P<start>[\d.?]+) - (?P<end>[\d.?]+)\] (?P<speaker>\S+): (?P<text>.*)$")
ROLES = ("INTERVIEWER", "PARTICIPANT")


def other_role(role: str) -> str:
    if role == ROLES[0]:
        return ROLES[1]
    if role == ROLES[1]:
        return ROLES[0]
    return role


def apply_word_corrections(text: str, words: list[dict], corrections: list[dict]) -> tuple[str, list[dict], list[dict]]:
    """Substitute suggested words in the transcript text and the word list.

    Only whole-word, case-sensitive matches are replaced, mirroring the
    prompt's requirement that "original" be copied verbatim. A correction
    that matches nothing is reported rather than silently dropped.
    """
    applied: list[dict] = []
    for correction in corrections:
        original = (correction.get("original") or "").strip()
        suggested = (correction.get("suggested") or "").strip()
        if not original or not suggested:
            continue

        pattern = re.compile(rf"(?<!\w){re.escape(original)}(?!\w)")
        text, replacements = pattern.subn(suggested, text)

        word_hits = 0
        if len(original.split()) == 1:
            for word in words:
                if word["word"] == original:
                    word["word"] = suggested
                    word_hits += 1

        applied.append({
            **correction,
            "text_replacements": replacements,
            "word_replacements": word_hits,
            "applied": replacements > 0,
        })
    return text, words, applied


def apply_speaker_flags(text: str, words: list[dict], flags: list[dict]) -> tuple[str, list[dict], list[dict]]:
    """Flip the role on turns the reviewer flagged as mis-attributed."""
    lines = text.split("\n")
    applied: list[dict] = []

    for flag in flags:
        target = (flag.get("turn_text") or "").strip()
        if not target:
            continue
        hit = False
        for index, line in enumerate(lines):
            match = TURN_LINE.match(line)
            if not match or match.group("text").strip() != target:
                continue
            flipped = other_role(match.group("speaker"))
            lines[index] = (
                f"[{match.group('start')} - {match.group('end')}] {flipped}: {match.group('text')}"
            )
            start, end = match.group("start"), match.group("end")
            for word in words:
                if (
                    word["start"] is not None
                    and start != "?" and end != "?"
                    and float(start) - 1e-6 <= word["start"] <= float(end) + 1e-6
                    and word["speaker"] == match.group("speaker")
                ):
                    word["speaker"] = flipped
            hit = True
            break
        applied.append({**flag, "applied": hit})

    return "\n".join(lines), words, applied


def apply_role_swap(text: str, words: list[dict]) -> tuple[str, list[dict]]:
    """Swap the two role labels across the whole transcript."""
    placeholder = "\x00ROLE\x00"
    text = text.replace(ROLES[0], placeholder).replace(ROLES[1], ROLES[0]).replace(placeholder, ROLES[1])
    for word in words:
        word["speaker"] = other_role(word["speaker"])
    return text, words


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", required=True, help="run_baseline.py output tree")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--shard", default=None, metavar="I/N",
        help=(
            "review only shard I of N (1-based) so several workers can share "
            "the tree across GPUs; the review is the slowest stage per file"
        ),
    )
    parser.add_argument(
        "--skip-done", action="store_true",
        help="skip transcripts that already have a corrected copy (resume)",
    )
    parser.add_argument("--model", default=None, help="review model id (default: Qwen2.5-7B-Instruct)")
    parser.add_argument("--device", default=None, help="e.g. cuda:1, to avoid a GPU busy with transcription")
    parser.add_argument(
        "--skip", default="", help="categories not to apply (words,speakers,roles)"
    )
    parser.add_argument(
        "--reuse-suggestions", action="store_true",
        help="reuse existing *_llm_suggestion.json instead of re-running the model",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    outputs = Path(args.outputs)
    transcripts = sorted(outputs.rglob("*_transcript.txt"))
    transcripts = [t for t in transcripts if not t.name.endswith("_transcript_corrected.txt")]
    if args.skip_done:
        transcripts = [
            t for t in transcripts
            if not (t.parent / f"{t.name[: -len('_transcript.txt')]}_transcript_corrected.txt").exists()
        ]
    if args.shard:
        index_str, _, count_str = args.shard.partition("/")
        shard_index, shard_count = int(index_str), int(count_str)
        transcripts = transcripts[shard_index - 1 :: shard_count]
        logger.info("Shard %d/%d: %d transcript(s)", shard_index, shard_count, len(transcripts))
    if args.limit:
        transcripts = transcripts[: args.limit]
    if not transcripts:
        logger.error("No *_transcript.txt found under %s", outputs)
        return 1
    logger.info("Reviewing %d transcript(s)", len(transcripts))

    model = tokenizer = None
    if not args.reuse_suggestions:
        from llm_review import LLM_MODEL_ID, load_llm

        logger.info("Loading %s", args.model or LLM_MODEL_ID)
        model, tokenizer = load_llm(args.model or LLM_MODEL_ID, device=args.device)

    totals = {"suggested_words": 0, "applied_words": 0, "suggested_flags": 0,
              "applied_flags": 0, "role_swaps": 0, "visits": 0}

    for index, transcript_path in enumerate(transcripts, start=1):
        prefix = transcript_path.name[: -len("_transcript.txt")]
        directory = transcript_path.parent
        words_path = directory / f"{prefix}_words.json"
        suggestion_path = directory / f"{prefix}_llm_suggestion.json"
        logger.info("[%d/%d] %s", index, len(transcripts), prefix[:48])

        text = transcript_path.read_text()
        words = json.loads(words_path.read_text()) if words_path.exists() else []

        if args.reuse_suggestions:
            if not suggestion_path.exists():
                logger.warning("  no suggestions on disk; skipping")
                continue
            suggestions = json.loads(suggestion_path.read_text())
        else:
            from llm_review import run_llm_verification

            suggestions = run_llm_verification(model, tokenizer, text)
            suggestion_path.write_text(json.dumps(suggestions, indent=2) + "\n")

        corrections = suggestions.get("word_corrections") or []
        flags = suggestions.get("speaker_flags") or []
        role_check = suggestions.get("role_mapping_check") or {}

        applied_words: list[dict] = []
        applied_flags: list[dict] = []
        swapped = False

        if "words" not in skip:
            text, words, applied_words = apply_word_corrections(text, words, corrections)
        if "speakers" not in skip:
            text, words, applied_flags = apply_speaker_flags(text, words, flags)
        if "roles" not in skip and role_check.get("looks_correct") is False:
            text, words = apply_role_swap(text, words)
            swapped = True

        (directory / f"{prefix}_transcript_corrected.txt").write_text(text)
        (directory / f"{prefix}_words_corrected.json").write_text(json.dumps(words, indent=2) + "\n")
        (directory / f"{prefix}_llm_applied.json").write_text(json.dumps({
            "word_corrections": applied_words,
            "speaker_flags": applied_flags,
            "role_swapped": swapped,
            "role_mapping_check": role_check,
        }, indent=2) + "\n")

        landed = sum(1 for c in applied_words if c["applied"])
        flags_landed = sum(1 for f in applied_flags if f["applied"])
        totals["visits"] += 1
        totals["suggested_words"] += len(corrections)
        totals["applied_words"] += landed
        totals["suggested_flags"] += len(flags)
        totals["applied_flags"] += flags_landed
        totals["role_swaps"] += int(swapped)
        logger.info(
            "  words %d/%d applied | flags %d/%d applied | role swapped: %s",
            landed, len(corrections), flags_landed, len(flags), swapped,
        )

    print("\n  " + json.dumps(totals, indent=2).replace("\n", "\n  "))
    (outputs / "llm_correction_totals.json").write_text(json.dumps(totals, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

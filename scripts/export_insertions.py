"""Export every insertion block with audio timestamps, for listening to.

The categories in error_taxonomy.py are heuristics, not measurements:
"repetition" is a word repeated within two positions, "backchannel" and
"discourse marker" are two hand-written word lists, and "other insertion" is
whatever matched none of them. This writes the underlying events out so the
classification can be checked against the audio rather than taken on trust.

One row per contiguous insertion block, because that is the unit a reviewer
listens to -- a 4000-word block is one event, not 4000. Each row carries the
audio time span from the system's own word timestamps, the surrounding text on
both sides, the nearest human turn and its timestamp, and absolute paths to the
audio, the human transcript and the system transcript.

Token indices are tracked back to source words so a normalized token can be
turned into a timestamp. `evaluate_systems.normalize` is applied per word and
the concatenation is asserted to equal normalizing the joined string, so this
reads exactly the alignment the scores came from; a visit where that assertion
fails is reported and skipped rather than silently mistimed.

Usage:
    uv run python scripts/export_insertions.py --cohort /path/to/cohort \
        --system ours=outputs/ours --output insertions.csv --min-words 5
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "finetune"))

import jiwer  # noqa: E402
import systems as registry  # noqa: E402
from coverage import clip_words, covered_turns  # noqa: E402
from prepare_data import load_timestamped_text  # noqa: E402

from error_taxonomy import classify_insertion  # noqa: E402
from evaluate_systems import ADAPTERS, normalize  # noqa: E402

logger = logging.getLogger("export_insertions")

CONTEXT_WORDS = 12
PREVIEW_WORDS = 40


def tokenize_tracked(items: list[str]) -> tuple[list[str], list[int]]:
    """Normalized tokens plus, for each, the index of the item it came from."""
    tokens: list[str] = []
    owners: list[int] = []
    for index, item in enumerate(items):
        for token in normalize(item).split():
            tokens.append(token)
            owners.append(index)
    return tokens, owners


def clock(seconds: float | None) -> str:
    if seconds is None:
        return ""
    seconds = max(float(seconds), 0.0)
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def preview(words: list[str], limit: int = PREVIEW_WORDS) -> str:
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit]) + f" ... [+{len(words) - limit} more words]"


def nearest_turn(turns: list[dict], when: float | None) -> dict | None:
    if when is None or not turns:
        return None
    for turn in turns:
        end = turn["end"] if turn["end"] is not None else turn["start"]
        if turn["start"] <= when <= end:
            return turn
    return min(turns, key=lambda t: abs(t["start"] - when))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--system", action="append", required=True, metavar="NAME[=DIR]")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--min-words", type=int, default=1,
        help="skip insertion blocks shorter than this (default 1, keep all)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    cohort = Path(args.cohort).resolve()

    chosen = []
    for spec in args.system:
        name, _, directory = spec.partition("=")
        if name not in ADAPTERS:
            logger.error("Unknown system %s", name)
            return 1
        chosen.append((name, ADAPTERS[name][0], Path(directory).resolve() if directory else None))

    visits = sorted(
        p for p in cohort.glob("Pronet*/*/*")
        if (p / "human").is_dir() and any((p / "human").glob("*.txt"))
        and any((p / "audio").glob("*.wav"))
    )
    if args.limit:
        visits = visits[: args.limit]

    rows: list[dict] = []
    mismatched = 0

    for index, visit in enumerate(visits, start=1):
        relative = visit.relative_to(cohort)
        human = sorted((visit / "human").glob("*.txt"))[0]
        audio = sorted((visit / "audio").glob("*.wav"))[0]
        try:
            turns = load_timestamped_text(human, 0.0)
        except Exception:
            continue
        turns, window_start, window_end = covered_turns(turns)
        if not turns:
            continue

        ref_tokens, ref_owner = tokenize_tracked([t["text"] for t in turns])
        if " ".join(ref_tokens) != normalize(" ".join(t["text"] for t in turns)):
            mismatched += 1
            continue

        for name, adapter, root in chosen:
            try:
                words = adapter(visit, root, relative)
            except Exception:
                words = None
            if not words:
                continue
            words = clip_words(words, window_start, window_end)
            if not words:
                continue
            raw = [str(w.get("word", "")) for w in words]
            hyp_tokens, hyp_owner = tokenize_tracked(raw)
            if not hyp_tokens or not ref_tokens:
                continue
            if " ".join(hyp_tokens) != normalize(" ".join(raw)):
                mismatched += 1
                continue

            source = adapter_path(visit, root, relative, name)
            output = jiwer.process_words(" ".join(ref_tokens), " ".join(hyp_tokens))
            for chunk in output.alignments[0]:
                if chunk.type != "insert":
                    continue
                length = chunk.hyp_end_idx - chunk.hyp_start_idx
                if length < args.min_words:
                    continue

                categories: dict[str, int] = {}
                for position in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                    label = classify_insertion(hyp_tokens, position)
                    categories[label] = categories.get(label, 0) + 1
                dominant = max(categories.items(), key=lambda kv: kv[1])[0]

                first_word = words[hyp_owner[chunk.hyp_start_idx]]
                last_word = words[hyp_owner[chunk.hyp_end_idx - 1]]
                start = first_word.get("start")
                end = last_word.get("end")

                turn = nearest_turn(turns, start)
                ref_at = chunk.ref_start_idx
                ref_context = " ".join(
                    ref_tokens[max(ref_at - CONTEXT_WORDS, 0):ref_at + CONTEXT_WORDS]
                )
                before = " ".join(
                    hyp_tokens[max(chunk.hyp_start_idx - CONTEXT_WORDS, 0):chunk.hyp_start_idx]
                )
                after = " ".join(
                    hyp_tokens[chunk.hyp_end_idx:chunk.hyp_end_idx + CONTEXT_WORDS]
                )

                rows.append({
                    "site": relative.parts[0],
                    "subject": relative.parts[1],
                    "session": relative.parts[2],
                    "system": registry.label_of(name),
                    "category": dominant,
                    "category_counts": " ".join(
                        f"{k}={v}" for k, v in sorted(categories.items())
                    ),
                    "words_inserted": length,
                    "audio_start_s": f"{start:.2f}" if start is not None else "",
                    "audio_end_s": f"{end:.2f}" if end is not None else "",
                    "audio_start_hms": clock(start),
                    "audio_end_hms": clock(end),
                    "inserted_text": preview(
                        hyp_tokens[chunk.hyp_start_idx:chunk.hyp_end_idx]
                    ),
                    "system_context_before": before,
                    "system_context_after": after,
                    "human_text_at_this_point": ref_context,
                    "human_turn_time": clock(turn["start"]) if turn else "",
                    "human_turn_speaker": turn["speaker"] if turn else "",
                    "human_turn_text": preview(turn["text"].split(), 25) if turn else "",
                    "audio_path": str(audio),
                    "human_transcript_path": str(human),
                    "system_transcript_path": str(source) if source else "",
                })
        if index % 25 == 0 or index == len(visits):
            logger.info("  %d/%d visits", index, len(visits))

    if mismatched:
        logger.warning(
            "%d visit/system pair(s) skipped: per-word normalization did not "
            "reproduce the pooled text, so timestamps could not be trusted",
            mismatched,
        )

    rows.sort(key=lambda r: -r["words_inserted"])
    fields = list(rows[0]) if rows else []
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    total = sum(r["words_inserted"] for r in rows)
    logger.info(
        "wrote %s: %d block(s), %d inserted word(s), largest %d",
        args.output, len(rows), total, rows[0]["words_inserted"] if rows else 0,
    )
    return 0


def adapter_path(visit: Path, root: Path | None, relative: Path, name: str) -> Path | None:
    """The file the adapter actually read, so a row can be traced to its source."""
    if name == "chirp3":
        found = sorted((visit / "chirp").glob("*.json"))
        return found[0] if found else None
    if root is None:
        return None
    for pattern in ("transcript.json", "*_words.json", "*_words_corrected.json"):
        found = sorted((root / relative).glob(pattern))
        if found:
            return found[-1]
    return None


if __name__ == "__main__":
    raise SystemExit(main())

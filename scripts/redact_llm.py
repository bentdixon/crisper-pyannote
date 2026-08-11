"""Redact PII from a pipeline's word-level transcript with Gemma 4, in chunks.

Our pipelines transcribe verbatim and redact nothing, so every name, date and
place a participant says survives in the clear. Chirp-3 redacts natively; this
gives the other systems a comparable capability so the comparison is redactor
against redactor rather than redactor against nothing.

Chunked deliberately. Sessions run to 75 minutes and 20k words, and the
previous LLM pass in this repo (Qwen2.5-7B reviewing whole transcripts) silently
exceeded its 32k context on at least one visit and produced a degraded review
nobody could distinguish from a good one. Here the transcript is cut into
overlapping windows of a few hundred words, each redacted independently, and
the windows are stitched back by word index. A window that fails, times out or
returns unusable JSON falls back to its original words and is counted -- a
failed window must never look like a window with no PII in it.

The model returns the indices of words to redact, not rewritten text: asking a
31B model to reproduce 400 words verbatim invites paraphrase, and the whole
point is that only the identifying tokens change. Indices are validated against
the window before anything is applied.

Usage:
    uv run python scripts/redact_llm.py --outputs outputs/ours \
        --cohort /path/to/cohort --shard 1/4 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

logger = logging.getLogger("redact_llm")

MODEL_ID = "google/gemma-4-31b-it"

# Window size in words. Small enough that the model attends to every token and
# that one bad window costs little, large enough to carry the context that makes
# "May" a date rather than a name.
WINDOW_WORDS = 400
OVERLAP_WORDS = 40

# The label set Chirp-3 emits, so both systems' output lands in the same space
# and score_redaction can compare categories without a translation table.
LABELS = ["PERSON_NAME", "DATE", "LOCATION", "AGE", "GENDER", "US_STATE"]

PROMPT = """You are de-identifying a transcript of a clinical research interview.

Below is a numbered list of words. Identify every word that is part of \
personally identifying information, under this label set:

PERSON_NAME  first names, surnames, nicknames of any real person
DATE         specific dates: "May 9th", "March", "the 14th". NOT relative time
             like "yesterday", "last week", "two months ago"
LOCATION     specific places: cities, neighbourhoods, streets, named buildings,
             schools, employers. NOT generic places like "the store", "my room"
US_STATE     US state names
AGE          a person's stated age in years
GENDER       an explicit gender term used to identify a specific person

Rules:
- Mark ONLY the words that are themselves identifying. Do not mark surrounding
  words like "in", "on", "my", "called".
- Do not mark the interviewer's generic questions, clinical terms, or common
  words that merely resemble names.
- If a name is already written as [PERSON_NAME] or similar, ignore it.
- Relative time expressions are NOT dates.

Return ONLY a JSON object of this exact shape, no prose, no markdown fence:
{{"redactions": [{{"index": <word number>, "label": "<LABEL>"}}]}}
If there is no identifying information, return {{"redactions": []}}.

Words:
{numbered}
"""


def load_model(model_id: str = MODEL_ID, device: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=device or "cuda:0",
    )
    model.eval()
    return model, tokenizer


def build_windows(words: list[dict]) -> list[tuple[int, int]]:
    """Overlapping [start, end) word-index windows covering the transcript.

    The overlap exists so a span sitting on a boundary is seen whole by at least
    one window; stitching takes the union of redactions, so seeing a span twice
    is harmless while seeing half of it is not.
    """
    if not words:
        return []
    step = max(WINDOW_WORDS - OVERLAP_WORDS, 1)
    windows = []
    start = 0
    while start < len(words):
        windows.append((start, min(start + WINDOW_WORDS, len(words))))
        if start + WINDOW_WORDS >= len(words):
            break
        start += step
    return windows


def parse_response(text: str) -> list[dict] | None:
    """Pull the JSON object out of a model response, or None if unusable."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    entries = payload.get("redactions")
    if not isinstance(entries, list):
        return None
    clean = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            index = int(entry.get("index"))
        except (TypeError, ValueError):
            continue
        label = str(entry.get("label", "")).strip().upper()
        if label not in LABELS:
            continue
        clean.append({"index": index, "label": label})
    return clean


def redact_window(model, tokenizer, words: list[dict], offset: int,
                  max_new_tokens: int = 900) -> tuple[dict[int, str], str | None]:
    """Redactions for one window as {absolute word index: label}.

    Returns the reason string when the window could not be used, so the caller
    can count failures instead of treating them as clean windows.
    """
    import torch

    numbered = "\n".join(
        f"{i} {str(w.get('word', '')).strip()}" for i, w in enumerate(words)
    )
    prompt = PROMPT.format(numbered=numbered)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    reply = tokenizer.decode(
        generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    )

    entries = parse_response(reply)
    if entries is None:
        return {}, "unparseable JSON"

    out: dict[int, str] = {}
    dropped = 0
    for entry in entries:
        index = entry["index"]
        # Out-of-range indices are the model hallucinating positions; applying
        # them would redact unrelated words elsewhere in the session.
        if not 0 <= index < len(words):
            dropped += 1
            continue
        out[offset + index] = entry["label"]
    if dropped:
        logger.debug("  dropped %d out-of-range index/indices", dropped)
    return out, None


def redact_words(model, tokenizer, words: list[dict]) -> tuple[list[dict], dict]:
    """Apply windowed redaction, returning new words and a per-file report."""
    windows = build_windows(words)
    redactions: dict[int, str] = {}
    failures = 0
    for start, end in windows:
        try:
            found, reason = redact_window(model, tokenizer, words[start:end], start)
        except Exception as error:
            found, reason = {}, f"{type(error).__name__}: {error}"
        if reason:
            failures += 1
            logger.warning("  window %d-%d unusable (%s); words kept", start, end, reason)
            continue
        redactions.update(found)

    out = []
    for index, word in enumerate(words):
        copy = dict(word)
        label = redactions.get(index)
        if label:
            # Punctuation is preserved so turn text still reads correctly and
            # the sentence structure a reviewer relies on survives.
            trailing = re.search(r"[.,!?;:]+$", str(word.get("word", "")))
            copy["word"] = f"[{label}]" + (trailing.group(0) if trailing else "")
            copy["redacted_from"] = word.get("word")
            copy["redaction_label"] = label
        out.append(copy)

    return out, {
        "windows": len(windows),
        "window_failures": failures,
        "redacted_words": len(redactions),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", required=True, help="tree of transcript.json files")
    parser.add_argument("--pattern", default="transcript.json")
    parser.add_argument("--suffix", default="transcript_redacted.json")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard", default=None, metavar="I/N")
    parser.add_argument(
        "--redo", action="store_true", help="re-redact files that already have output",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    root = Path(args.outputs)
    files = sorted(root.rglob(args.pattern))
    if not files:
        logger.error("No %s under %s", args.pattern, root)
        return 1
    if not args.redo:
        files = [f for f in files if not (f.parent / args.suffix).exists()]
    if args.shard:
        index_str, _, count_str = args.shard.partition("/")
        files = files[int(index_str) - 1 :: int(count_str)]
        logger.info("Shard %s: %d file(s)", args.shard, len(files))
    if args.limit:
        files = files[: args.limit]
    if not files:
        logger.info("Nothing to do")
        return 0

    model, tokenizer = load_model(args.model, args.device)

    total_failures = 0
    for index, path in enumerate(files, start=1):
        started = time.perf_counter()
        payload = json.loads(path.read_text())
        words = payload["words"] if isinstance(payload, dict) else payload
        redacted, report = redact_words(model, tokenizer, words)
        total_failures += report["window_failures"]

        destination = path.parent / args.suffix
        body = dict(payload) if isinstance(payload, dict) else {}
        body["words"] = redacted
        body["redaction"] = {"model": args.model, **report}
        destination.write_text(json.dumps(body, indent=2))
        logger.info(
            "[%d/%d] %s: %d word(s) redacted over %d window(s), %d failed, %.0fs",
            index, len(files), path.parent.name, report["redacted_words"],
            report["windows"], report["window_failures"], time.perf_counter() - started,
        )

    if total_failures:
        logger.warning("%d window(s) fell back to their original words", total_failures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Redact PII from a pipeline's word-level transcript with Gemma 4, in chunks.

Our pipelines transcribe verbatim and redact nothing, so every name, date and
place a participant says survives in the clear. Chirp-3 redacts natively; this
gives the other systems a comparable capability so the comparison is redactor
against redactor rather than redactor against nothing.

Chunked deliberately. Sessions run to 75 minutes and 20k words, and the
previous LLM pass in this repo (Qwen2.5-7B reviewing whole transcripts) silently
exceeded its 32k context on at least one visit and produced a degraded review
nobody could distinguish from a good one. Here the transcript is cut into
chunks of at most 5000 characters, always ending on a sentence boundary, and
each is redacted independently. A chunk that fails, times out or returns
unusable JSON falls back to its original words and is counted -- a failed chunk
must never look like a chunk with no PII in it.

Addressing is by sentence number plus the quoted text, not character offsets
and not word indices:

  - Character offsets require the model to count characters, which it cannot do
    reliably, and an off-by-a-few silently redacts the wrong span.
  - Word indices require it to track a running count over hundreds of tokens,
    and a wrong index is undetectable -- it names a real word, just not the
    intended one.
  - A sentence number plus the exact words is checkable. The quote is searched
    for inside the sentence it was attributed to; if it is not there, the model
    invented it and the span is dropped and counted. Sentences are short enough
    that the number is easy to get right, and the quote pins the span within it.

Mapping back to word indices is then deterministic, which is what the pipeline
needs since redaction replaces individual word tokens and must preserve their
timestamps.

Usage:
    uv run python scripts/redact_llm.py --outputs outputs/ours \
        --shard 1/3 --device cuda:0
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

# Chunks are measured in characters and cut on sentence boundaries: a chunk
# grows until the next sentence would take it past MAX_CHUNK_CHARS, so no
# sentence is ever split across two chunks and the model never sees half a
# clause. One sentence longer than the limit becomes its own oversized chunk
# rather than being cut mid-sentence.
MAX_CHUNK_CHARS = 5000
OVERLAP_SENTENCES = 1

# Sentences end at terminal punctuation on a word. CrisperWhisper and Chirp both
# attach punctuation to the word token, so this needs no separate tokenizer, and
# a run of words with no terminal punctuation at all is capped so one unpunctuated
# passage cannot swallow a whole transcript.
SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*$")
MAX_SENTENCE_WORDS = 120

# The label set Chirp-3 emits, so both systems' output lands in the same space
# and score_redaction can compare categories without a translation table.
LABELS = ["PERSON_NAME", "DATE", "LOCATION", "AGE", "GENDER", "US_STATE"]

PROMPT = """You are de-identifying a transcript of a clinical research interview.

Below are numbered sentences. Identify every span of personally identifying \
information, under this label set:

PERSON_NAME  first names, surnames, nicknames of any real person
DATE         specific dates: "May 9th", "March", "the 14th". NOT relative time
             like "yesterday", "last week", "two months ago"
LOCATION     specific places: cities, neighbourhoods, streets, named buildings,
             schools, employers. NOT generic places like "the store", "my room"
US_STATE     US state names
AGE          a person's stated age in years
GENDER       an explicit gender term used to identify a specific person

Rules:
- Quote ONLY the identifying words themselves, exactly as they appear in the
  sentence. Do not include surrounding words like "in", "on", "my", "called".
- Do not mark the interviewer's generic questions, clinical terms, or common
  words that merely resemble names.
- If a span is already written as [PERSON_NAME] or similar, ignore it.
- Relative time expressions are NOT dates.

Return ONLY a JSON object of this exact shape, no prose, no markdown fence:
{{"redactions": [{{"sentence": <sentence number>, "text": "<exact words>", \
"label": "<LABEL>"}}]}}
The "text" must appear verbatim in that sentence.
If there is no identifying information, return {{"redactions": []}}.

Sentences:
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


def split_sentences(words: list[dict]) -> list[tuple[int, int]]:
    """Word-index ranges, one per sentence."""
    sentences: list[tuple[int, int]] = []
    start = 0
    for index, word in enumerate(words):
        text = str(word.get("word", "")).strip()
        long_enough = index - start + 1 >= MAX_SENTENCE_WORDS
        if (text and SENTENCE_END.search(text)) or long_enough:
            sentences.append((start, index + 1))
            start = index + 1
    if start < len(words):
        sentences.append((start, len(words)))
    return sentences


def build_chunks(sentences: list[tuple[int, int]], words: list[dict]) -> list[list[int]]:
    """Group sentence indices into chunks of at most MAX_CHUNK_CHARS.

    Returns lists of sentence indices rather than word offsets, because the
    model is addressed in sentences and the mapping back has to agree with what
    it was shown. Each chunk repeats the previous chunk's last sentence, so a
    span in the first sentence of a chunk has been seen once with its left
    context; redactions are unioned, so seeing a span twice costs nothing.
    """
    lengths = [
        sum(len(str(words[i].get("word", "")).strip()) + 1 for i in range(a, b))
        for a, b in sentences
    ]
    chunks: list[list[int]] = []
    current: list[int] = []
    size = 0
    for index, length in enumerate(lengths):
        if current and size + length > MAX_CHUNK_CHARS:
            chunks.append(current)
            current = current[-OVERLAP_SENTENCES:] if OVERLAP_SENTENCES else []
            size = sum(lengths[i] for i in current)
        current.append(index)
        size += length
    if current:
        chunks.append(current)
    return chunks


def sentence_text(words: list[dict], span: tuple[int, int]) -> str:
    return " ".join(str(words[i].get("word", "")).strip() for i in range(*span))


def locate(words: list[dict], span: tuple[int, int], needle: str) -> list[int]:
    """Word indices inside one sentence whose text matches `needle`.

    The model quotes the identifying words back, so this is a search rather than
    a coordinate lookup: a quote that cannot be found is a hallucination and is
    dropped, which a bare index could never reveal. Matching is on normalized
    tokens so punctuation attached to a word does not defeat it.
    """
    target = [normalize(t) for t in needle.split() if normalize(t)]
    if not target:
        return []
    tokens = [normalize(str(words[i].get("word", ""))) for i in range(*span)]
    for offset in range(len(tokens) - len(target) + 1):
        if tokens[offset:offset + len(target)] == target:
            return list(range(span[0] + offset, span[0] + offset + len(target)))
    return []


def normalize(token: str) -> str:
    return re.sub(r"[^a-z0-9']", "", token.lower())


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
            sentence = int(entry.get("sentence"))
        except (TypeError, ValueError):
            continue
        text = str(entry.get("text", "")).strip()
        label = str(entry.get("label", "")).strip().upper()
        if not text or label not in LABELS:
            continue
        clean.append({"sentence": sentence, "text": text, "label": label})
    return clean


def redact_chunk(model, tokenizer, words: list[dict], sentences: list[tuple[int, int]],
                 chunk: list[int], max_new_tokens: int = 900,
                 ) -> tuple[dict[int, str], str | None, int]:
    """Redactions for one chunk as {absolute word index: label}.

    Returns the failure reason when the chunk could not be used -- so the caller
    can count failures rather than treat them as clean chunks -- and the number
    of quoted spans that could not be found in the sentence they were attributed
    to, which is the model inventing text.
    """
    import torch

    # Numbered by position within the chunk, so the model never has to reason
    # about absolute offsets in a 20,000-word transcript.
    numbered = "\n".join(
        f"{local} {sentence_text(words, sentences[index])}"
        for local, index in enumerate(chunk)
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
        return {}, "unparseable JSON", 0

    out: dict[int, str] = {}
    unmatched = 0
    for entry in entries:
        local = entry["sentence"]
        # A sentence number outside the chunk is the model addressing something
        # it was never shown; applying it would redact unrelated speech.
        if not 0 <= local < len(chunk):
            unmatched += 1
            continue
        span = sentences[chunk[local]]
        found = locate(words, span, entry["text"])
        if not found:
            unmatched += 1
            continue
        for index in found:
            out[index] = entry["label"]
    return out, None, unmatched


def redact_words(model, tokenizer, words: list[dict]) -> tuple[list[dict], dict]:
    """Apply chunked redaction, returning new words and a per-file report."""
    sentences = split_sentences(words)
    chunks = build_chunks(sentences, words)
    redactions: dict[int, str] = {}
    failures = 0
    unmatched = 0
    for number, chunk in enumerate(chunks, start=1):
        try:
            found, reason, missed = redact_chunk(
                model, tokenizer, words, sentences, chunk,
            )
        except Exception as error:
            found, reason, missed = {}, f"{type(error).__name__}: {error}", 0
        unmatched += missed
        if reason:
            failures += 1
            logger.warning(
                "  chunk %d/%d unusable (%s); its words are kept unredacted",
                number, len(chunks), reason,
            )
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
        "sentences": len(sentences),
        "chunks": len(chunks),
        "chunk_failures": failures,
        # Spans the model quoted that do not occur in the sentence it named.
        # Counted rather than silently dropped: a rising number means the model
        # is inventing text, which no index-based protocol could have detected.
        "unmatched_quotes": unmatched,
        "redacted_words": len(redactions),
    }


def destination_for(path: Path, suffix: str) -> Path:
    """Source stem plus suffix, so each tree keeps its own naming."""
    return path.with_name(path.stem + suffix)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", required=True, help="tree of transcript.json files")
    parser.add_argument("--pattern", default="transcript.json")
    parser.add_argument(
        "--suffix", default="_redacted.json",
        help=(
            "appended to the source stem, so transcript.json becomes "
            "transcript_redacted.json and X_words.json becomes "
            "X_words_redacted.json -- the names the scorer's adapters glob for"
        ),
    )
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
        files = [f for f in files if not destination_for(f, args.suffix).exists()]
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
        total_failures += report["chunk_failures"]

        destination = destination_for(path, args.suffix)
        body = dict(payload) if isinstance(payload, dict) else {}
        body["words"] = redacted
        body["redaction"] = {"model": args.model, **report}
        destination.write_text(json.dumps(body, indent=2))
        logger.info(
            "[%d/%d] %s: %d word(s) redacted over %d chunk(s) / %d sentence(s), "
            "%d chunk(s) failed, %d unmatched quote(s), %.0fs",
            index, len(files), path.parent.name, report["redacted_words"],
            report["chunks"], report["sentences"], report["chunk_failures"],
            report["unmatched_quotes"], time.perf_counter() - started,
        )

    if total_failures:
        logger.warning("%d chunk(s) fell back to their original words", total_failures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

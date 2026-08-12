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

A second protocol is available with `--mode turn`: one speaker turn per call,
the model rewriting the turn verbatim with PII replaced inline by [LABEL]. It
trades the checkability of the quote protocol for a much easier task -- copy
this text and change these words -- at the cost of one call per turn and an
output the model can silently paraphrase. Nothing is trusted: the rewrite is
difflib-aligned back to the original words, only placeholder-for-word
substitutions are honoured, and a turn whose text the model altered beyond
MIN_VERBATIM_RATIO is discarded unredacted and counted. Both modes write the
same file shape, so score_redaction.py scores them identically.

Usage:
    uv run python scripts/redact_llm.py --outputs outputs/ours \
        --shard 1/3 --device cuda:0
    uv run python scripts/redact_llm.py --outputs outputs/ours \
        --mode turn --suffix _redacted_turn.json --device cuda:0
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


# --- turn mode -------------------------------------------------------------
#
# One speaker turn per call. Turns longer than MAX_TURN_CHARS are cut on
# sentence boundaries first: the model has to reproduce every word it is shown,
# so a five-minute monologue in one call is both slow and where verbatim
# reproduction breaks down.
MAX_TURN_CHARS = 1200

# Share of the turn's words that must survive the rewrite unchanged for it to be
# used at all. A model asked to copy text will occasionally summarise or
# "clean up" instead, and that rewrite must never reach the transcript: the
# turn is dropped unredacted and counted as a failure. 0.85 tolerates the
# genuine edits (the placeholders themselves, some punctuation drift) while
# catching paraphrase.
MIN_VERBATIM_RATIO = 0.85

# The largest replace block whose words may all be attributed to a placeholder.
# Inside a block the alignment cannot say which input word each output token
# came from, so a long mixed block is dropped rather than guessed at -- the
# alternative is redacting a clause because it happened to sit beside a name.
MAX_AMBIGUOUS_BLOCK = 8

TURN_PROMPT = """You are de-identifying a transcript of a clinical research interview.

Below is one speaker turn. Return the SAME text, word for word, with every span \
of personally identifying information replaced by a label in square brackets.

Label set:
[PERSON_NAME]  first names, surnames, nicknames of any real person
[DATE]         specific dates: "May 9th", "March", "the 14th". NOT relative time
               like "yesterday", "last week", "two months ago"
[LOCATION]     specific places: cities, neighbourhoods, streets, named buildings,
               schools, employers. NOT generic places like "the store", "my room"
[US_STATE]     US state names
[AGE]          a person's stated age in years
[GENDER]       an explicit gender term used to identify a specific person

Rules:
- Reproduce every other word exactly as written, including disfluencies, \
repetitions and punctuation. Do not correct, shorten or rephrase anything.
- Replace ONLY the identifying words themselves. Keep the surrounding words: \
"I saw Maria on Tuesday" becomes "I saw [PERSON_NAME] on Tuesday".
- A multi-word identifier becomes a single label: "San Francisco" becomes \
"[LOCATION]", not "[LOCATION] [LOCATION]".
- Leave existing labels like [PERSON_NAME] exactly as they are.
- Do not mark clinical terms, the interviewer's generic questions, or common \
words that merely resemble names.
- Relative time expressions are NOT dates.

Return ONLY the rewritten text. No preamble, no explanation, no quotation marks \
around it, no markdown fence.

Turn:
{turn}
"""

# Every call after the first repeats only this, because the instructions above
# are already in the conversation.
TURN_FOLLOWUP = """Turn:
{turn}
"""

# Worked examples, sent as real conversation turns rather than quoted inside the
# instructions: the format is then demonstrated by the assistant's own reply, so
# a demonstration cannot contradict the "no quotation marks, no preamble" rule
# the way a quoted example does, and the model has seen the exact shape of a
# valid response before it writes one.
#
# Each pair is chosen to settle several decisions at once, because every example
# is re-encoded on every call -- there is no prefix cache across generate()
# calls, and turn mode makes one call per turn. Two dense examples cost less and
# teach more than six thin ones.
#
# Between them these cover: disfluencies and stuttered repetitions kept; city
# and state labelled separately; a month as DATE but "last week" left alone; a
# street as LOCATION but "the store" left alone; a spelled-out age; an existing
# [PERSON_NAME] passed through untouched; a turn with nothing to redact returned
# character-for-character, which is the case a model is most tempted to "help"
# with. The text is synthetic, written to look like this corpus's verbatim ASR
# output rather than clean prose.
EXAMPLES: list[tuple[str, str]] = [
    (
        "Yeah, so, um, I I moved back to Boston, Massachusetts in March, and my "
        "sister Rachel, she's twenty three, she was still living at the old place "
        "on Commonwealth Ave. I saw her again last week at the store.",
        "Yeah, so, um, I I moved back to [LOCATION], [US_STATE] in [DATE], and my "
        "sister [PERSON_NAME], she's [AGE], she was still living at the old place "
        "on [LOCATION]. I saw her again last week at the store.",
    ),
    (
        "[UM] I don't know, it just felt like, like everyone was watching me? And "
        "then [PERSON_NAME] said I should maybe talk to somebody about it.",
        "[UM] I don't know, it just felt like, like everyone was watching me? And "
        "then [PERSON_NAME] said I should maybe talk to somebody about it.",
    ),
]


def build_messages(turn: str) -> list[dict]:
    """Instructions, the worked examples as turns, then the real turn."""
    messages: list[dict] = []
    for index, (given, wanted) in enumerate(EXAMPLES):
        template = TURN_PROMPT if index == 0 else TURN_FOLLOWUP
        messages.append({"role": "user", "content": template.format(turn=given)})
        messages.append({"role": "assistant", "content": wanted})
    template = TURN_PROMPT if not EXAMPLES else TURN_FOLLOWUP
    messages.append({"role": "user", "content": template.format(turn=turn)})
    return messages


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


def group_turns(words: list[dict]) -> list[tuple[int, int]]:
    """Word-index ranges, one per speaker turn, split when over MAX_TURN_CHARS.

    Consecutive words sharing a speaker label are one turn; a tree whose words
    carry no speaker (nothing here, but the adapters allow it) degenerates to a
    single run, which the length cut then breaks up on sentence boundaries.
    """
    if not words:
        return []
    runs: list[tuple[int, int]] = []
    start = 0
    previous = words[0].get("speaker")
    for index in range(1, len(words)):
        speaker = words[index].get("speaker")
        if speaker != previous:
            runs.append((start, index))
            start, previous = index, speaker
    runs.append((start, len(words)))

    spans: list[tuple[int, int]] = []
    for begin, finish in runs:
        length = sum(
            len(str(words[i].get("word", "")).strip()) + 1 for i in range(begin, finish)
        )
        if length <= MAX_TURN_CHARS:
            spans.append((begin, finish))
            continue
        # Cut on sentence boundaries inside the turn, greedily, so a piece is
        # still a whole number of sentences and reads as language.
        piece_start, size = begin, 0
        for a, b in split_sentences(words[begin:finish]):
            a, b = a + begin, b + begin
            piece = sum(
                len(str(words[i].get("word", "")).strip()) + 1 for i in range(a, b)
            )
            if size and size + piece > MAX_TURN_CHARS:
                spans.append((piece_start, a))
                piece_start, size = a, 0
            size += piece
        if piece_start < finish:
            spans.append((piece_start, finish))
    return spans


PLACEHOLDER_TOKEN = re.compile(r"^\[([A-Z_]+)\][.,!?;:]*$")


def align_rewrite(words: list[dict], span: tuple[int, int], reply: str,
                  ) -> tuple[dict[int, str] | None, int]:
    """Map a rewritten turn back onto word indices, or None if it is not verbatim.

    The model returns prose, so the only way to know which words it replaced is
    to align its output against the input and look at what changed. Equal blocks
    are words it copied; a replace block whose output side is placeholders is a
    redaction and its input words take that label. Anything else -- a block with
    no placeholder in it, a placeholder inserted where no word stood, a long
    mixed block -- is the model editing text it was told to copy, and is counted
    rather than applied.
    """
    import difflib

    source = [str(words[i].get("word", "")).strip() for i in range(*span)]
    output = reply.split()
    if not source or not output:
        return None, 0

    def key(token: str) -> str:
        found = PLACEHOLDER_TOKEN.match(token)
        # Placeholders are given a key no real word can collide with, so the
        # matcher can never pair "[DATE]" with the word it replaced.
        return f"\x00{found.group(1)}" if found else normalize(token)

    # Both sides go through `key`, so a placeholder the transcript already
    # carried matches itself and is left alone rather than being counted as a
    # fresh redaction of the word "[PERSON_NAME]".
    left = [key(t) for t in source]
    right = [key(t) for t in output]
    matcher = difflib.SequenceMatcher(None, left, right, autojunk=False)

    out: dict[int, str] = {}
    unmatched = 0
    # Words the rewrite accounts for: copied verbatim, or replaced by a
    # placeholder. Anything else is the model changing text it was told to keep,
    # and the ratio below decides whether there is too much of it. Redacted
    # words must count as accounted -- they are the one edit that was asked for,
    # and charging them to the ratio would reject exactly the turns that carry
    # the most PII.
    accounted = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            accounted += i2 - i1
            continue
        found = [PLACEHOLDER_TOKEN.match(output[j]) for j in range(j1, j2)]
        placeholders = [m.group(1) for m in found if m]
        labels = [label for label in placeholders if label in LABELS]
        # One placeholder per replaced word means the labels can be assigned
        # positionally: "Boston, Massachusetts" -> "[LOCATION], [US_STATE]"
        # otherwise collapses to two LOCATIONs. Any other shape is ambiguous
        # inside the block and takes the first label for the whole of it.
        positional = (
            len(labels) == len(placeholders) == i2 - i1 and len(labels) > 1
        )
        if not labels:
            continue
        if i2 == i1 or i2 - i1 > MAX_AMBIGUOUS_BLOCK:
            unmatched += 1
            continue
        for offset in range(i1, i2):
            # A standalone punctuation token carries no identity and must not
            # become a placeholder: it lands inside the block whenever the model
            # writes "[LOCATION]," for a word plus its comma.
            if left[offset]:
                out[span[0] + offset] = labels[offset - i1] if positional else labels[0]
        accounted += i2 - i1

    if accounted < MIN_VERBATIM_RATIO * len(source):
        return None, 0
    return out, unmatched


def redact_turn(model, tokenizer, words: list[dict], span: tuple[int, int],
                ) -> tuple[dict[int, str], str | None, int]:
    """Redactions for one turn, as {absolute word index: label}."""
    import torch

    turn = sentence_text(words, span)
    messages = build_messages(turn)
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    # The reply is the turn again, so the budget scales with the turn itself --
    # not with the prompt, which now carries the instructions and both worked
    # examples and would make the budget grow with material the model is not
    # asked to reproduce. A truncated reply looks exactly like a paraphrase to
    # the aligner and would throw the turn away, hence the generous multiple.
    budget = int(len(tokenizer(turn)["input_ids"]) * 1.5) + 64
    with torch.no_grad():
        generated = model.generate(
            **inputs, max_new_tokens=budget, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    reply = tokenizer.decode(
        generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    ).strip()
    fence = re.search(r"```(?:\w+)?\s*(.*?)```", reply, re.S)
    if fence:
        reply = fence.group(1).strip()
    if not reply:
        return {}, "empty reply", 0

    found, unmatched = align_rewrite(words, span, reply)
    if found is None:
        return {}, "reply is not a verbatim copy of the turn", 0
    return found, None, unmatched


def redact_words_by_turn(model, tokenizer, words: list[dict]) -> tuple[dict[int, str], dict]:
    """One call per speaker turn. Returns redactions and the per-file report."""
    turns = group_turns(words)
    redactions: dict[int, str] = {}
    failures = 0
    unmatched = 0
    for number, span in enumerate(turns, start=1):
        try:
            found, reason, missed = redact_turn(model, tokenizer, words, span)
        except Exception as error:
            found, reason, missed = {}, f"{type(error).__name__}: {error}", 0
        unmatched += missed
        if reason:
            failures += 1
            logger.warning(
                "  turn %d/%d unusable (%s); its words are kept unredacted",
                number, len(turns), reason,
            )
            continue
        redactions.update(found)
    return redactions, {
        "mode": "turn",
        "turns": len(turns),
        # Same key names as chunk mode so the report, the scorer and every
        # downstream table read both without a translation table.
        "chunks": len(turns),
        "chunk_failures": failures,
        "unmatched_quotes": unmatched,
        "redacted_words": len(redactions),
    }


def apply_labels(words: list[dict], redactions: dict[int, str]) -> list[dict]:
    """Replace each labelled word with its placeholder, keeping the timestamps."""
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
    return out


def redact_words(model, tokenizer, words: list[dict], mode: str = "chunk",
                 ) -> tuple[list[dict], dict]:
    """Apply redaction in the requested mode, returning words and a report."""
    if mode == "turn":
        redactions, report = redact_words_by_turn(model, tokenizer, words)
        return apply_labels(words, redactions), report

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

    return apply_labels(words, redactions), {
        "mode": "chunk",
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
    parser.add_argument(
        "--mode", choices=("chunk", "turn"), default="chunk",
        help=(
            "chunk: numbered sentences, model quotes the spans back (default). "
            "turn: one speaker turn per call, model rewrites it verbatim with "
            "[LABEL] inline. Give turn mode its own --suffix so both survive"
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
        redacted, report = redact_words(model, tokenizer, words, args.mode)
        total_failures += report["chunk_failures"]

        destination = destination_for(path, args.suffix)
        body = dict(payload) if isinstance(payload, dict) else {}
        body["words"] = redacted
        body["redaction"] = {"model": args.model, **report}
        destination.write_text(json.dumps(body, indent=2))
        logger.info(
            "[%d/%d] %s: %d word(s) redacted over %d %s, %d failed, "
            "%d unmatched span(s), %.0fs",
            index, len(files), path.parent.name, report["redacted_words"],
            report["chunks"], "turn(s)" if args.mode == "turn" else "chunk(s)",
            report["chunk_failures"], report["unmatched_quotes"],
            time.perf_counter() - started,
        )

    if total_failures:
        logger.warning(
            "%d %s fell back to their original words",
            total_failures, "turn(s)" if args.mode == "turn" else "chunk(s)",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

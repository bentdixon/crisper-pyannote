"""Correct absorbed short utterances with Gemma 4, one whole transcript per call.

Our diarization fails in one specific place: short utterances. A one-word
speaker run in our output carries the right speaker 60% of the time against 95%
for a run of four words or more, and the lost-turns measure sees the same defect
from the other side -- turns under two seconds vanish into a neighbour's turn.
A participant's "no" lands inside the interviewer's question, and on a
symptom-checklist interview that is missing data rather than a cosmetic slip.

Those words are already in the transcript, wearing the wrong label, so a model
that reads the conversation can in principle put them back. The ceiling is
known and small: short runs are 3.1% of words, so perfect repair of every one
moves word-level speaker accuracy about 1.2 points. `speaker_headroom.py`
measures it, and `correct_speakers_rule.py` is the no-model comparator this has
to beat -- both exist because the last LLM post-processor in this repo moved
every metric by exactly zero and there was no way to tell an incapable model
from an unfixable problem.

Whole transcripts, unlike redact_llm's chunking. The judgement needed here is
"who was talking either side of this word", which a chunk boundary destroys,
and the unit of work is small: the model returns a handful of corrections, not
a rewritten transcript. A median session is around 4,400 words and fits in one
call. Transcripts over MAX_CONTEXT_TOKENS fall back to overlapping windows of
turns and every fallback is logged -- sessions run to 123 minutes, and a long
file must not silently take a different code path the way the Qwen review's did
when it overran its context unnoticed.

No batching and no prefix cache, deliberately. Both earn their keep in turn
mode, where a several-hundred-token instruction block sits in front of a turn
averaging twenty tokens. Here the instructions are a rounding error against a
whole transcript, and there is one call per file.

Addressing follows the chunk protocol exactly: a turn number local to this
request plus the exact words. A turn number the model was never shown, or a
quote that is not in the turn it was attributed to, is the model inventing
something -- dropped and counted. That check is the whole difference from the
Qwen attempt, which matched a paraphrasable `turn_text` against rendered lines
by string equality and applied 1 flag of 163.

Usage:
    uv run python scripts/correct_speakers_llm.py --outputs outputs/ours \
        --device cuda:0 --limit 8
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from redact_llm import MODEL_ID, load_model, locate  # noqa: E402
from speaker_rewrite import (  # noqa: E402
    apply_changes,
    destination_for,
    load_words,
    speaker_runs,
    write_corrected,
)

logger = logging.getLogger("correct_speakers_llm")

SOURCE = "gemma"

# A correction moving more than this many words is not the error this targets.
# Short runs are wrong 40% of the time and long ones 5%, so a model proposing to
# move a whole clause has either misread the task or hallucinated; either way
# applying it risks trading a small error for a large one.
MAX_MOVE_WORDS = 6

# The context guard. Gemma 4 takes far more than this, but a transcript sharing
# a window with its own reply is the failure mode that made the Qwen review
# useless, so the budget is deliberately conservative and the overflow path is
# exercised rather than theoretical: our longest sessions are around 20,000
# words and will window.
MAX_CONTEXT_TOKENS = 24000
OVERLAP_TURNS = 4

MAX_NEW_TOKENS = 1200

ACTIONS = ("reassign", "split")


def clock(seconds: float) -> str:
    """Wall-clock position of a turn, for the model to reason about pauses."""
    hours, rest = divmod(max(float(seconds), 0.0), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:04.1f}"


PROMPT = """You are correcting the speaker labels on a transcript of a clinical \
research interview.

Below are numbered speaker turns. Each line is:

    <number>  <speaker>  [<start time> +<silence before this turn>]  <words>

The words and their order are correct. Only the speaker labels can be wrong, \
and they go wrong in one specific way: a short utterance by one person is \
absorbed into the turn of the person speaking around it. A participant's "no" \
ends up inside the interviewer's question; a listener's "mm hmm" ends up inside \
the speaker's answer.

Find those, and only those.

In scope:
- One or two words at the start or end of a turn that answer or interrupt it.
- A short answer sitting inside a longer turn by the other speaker, which \
should stand alone as its own utterance.
- Backchannels and short answers: yeah, mm hmm, okay, right, no, yes, uh huh.

Out of scope. Do not:
- Change, add, remove or reword any word.
- Correct transcription errors, spelling, punctuation or disfluencies.
- Re-split a long turn because it covers two topics.
- Move more than six words in one correction. A longer span is not this error.

A long silence before a turn makes it more likely to be a real separate \
utterance. A turn that follows its neighbour with almost no gap is more likely \
to contain speech absorbed from that neighbour.

Return ONLY a JSON object of this exact shape, no prose, no markdown fence:
{{"corrections": [{{"turn": <turn number>, "text": "<exact words>", \
"speaker": "<speaker label>", "action": "reassign"}}]}}

- "turn" is the number of the line the words are on now.
- "text" must appear verbatim in that turn. A quote that does not appear there \
is discarded.
- "speaker" must be one of the speaker labels shown above, and must not be the \
label that turn already has.
- "action" is "reassign" when the words belong with the turn beside this one, \
or "split" when they sit inside this turn and should stand alone.

If every speaker label is already right, return {{"corrections": []}}.

Turns:
{numbered}
"""

FOLLOWUP = """Turns:
{numbered}
"""

# Worked examples as real conversation turns rather than quoted inside the
# instructions, for the reason redact_llm gives: the assistant's own reply
# demonstrates the format, so an example cannot contradict the "no prose, no
# fence" rule the way a quoted one can.
#
# Two examples, covering the two actions and the two ways a correction is
# addressed. The first is a backchannel at the end of a turn, answered by the
# neighbour, so it reassigns to an adjacent speaker. The second is a one-word
# answer buried mid-question, which has no adjacent turn to join and must split.
# Both replies also demonstrate leaving every other turn alone, which is the
# case a model is most tempted to improve on. The text is synthetic, written to
# look like this corpus's verbatim output rather than clean prose.
EXAMPLES: list[tuple[str, str]] = [
    (
        "0  SPEAKER_00  [00:03:11.4 +1.2s]  And over the past month have you "
        "felt that way more days than not yeah\n"
        "1  SPEAKER_01  [00:03:16.0 +0.3s]  I guess so, [UM] most days\n"
        "2  SPEAKER_00  [00:03:19.8 +0.5s]  Okay. And has that changed at all",
        '{"corrections": [{"turn": 0, "text": "yeah", "speaker": "SPEAKER_01", '
        '"action": "reassign"}]}',
    ),
    (
        "0  SPEAKER_01  [00:08:40.2 +2.1s]  It's been alright I think\n"
        "1  SPEAKER_00  [00:08:44.1 +0.9s]  Okay and have you had any trouble "
        "sleeping no okay and what about your appetite\n"
        "2  SPEAKER_01  [00:08:51.7 +0.4s]  That's been fine",
        '{"corrections": [{"turn": 1, "text": "no", "speaker": "SPEAKER_01", '
        '"action": "split"}]}',
    ),
]


def build_messages(numbered: str) -> list[dict]:
    """Instructions, the worked examples as turns, then the real transcript."""
    messages: list[dict] = []
    for index, (given, wanted) in enumerate(EXAMPLES):
        template = PROMPT if index == 0 else FOLLOWUP
        messages.append({"role": "user", "content": template.format(numbered=given)})
        messages.append({"role": "assistant", "content": wanted})
    template = PROMPT if not EXAMPLES else FOLLOWUP
    messages.append({"role": "user", "content": template.format(numbered=numbered)})
    return messages


def turn_line(words: list[dict], span: tuple[int, int], number: int, pause: float | None) -> str:
    """One rendered turn: request-local number, speaker, clock, pause, words.

    The pause is the silence before this turn, and it is the only signal in the
    line that is not recoverable from the text. Absorbed speech follows its
    neighbour with almost no gap; a real separate utterance usually does not.
    """
    speaker = words[span[0]].get("speaker") or "UNKNOWN"
    gap = "  ----" if pause is None else f" +{max(pause, 0.0):.1f}s"
    text = " ".join(str(words[i].get("word", "")).strip() for i in range(*span))
    return f"{number}  {speaker}  [{clock(words[span[0]]['start'])}{gap}]  {text}"


def render_window(words: list[dict], runs: list[tuple[int, int]], window: list[int]) -> str:
    """The numbered turns for one request.

    Numbered by position within the window, never by absolute turn index: the
    model is not asked to reason about an offset into a transcript it can only
    partly see. `interpret_window` maps the number back the same way, so the two
    cannot drift apart.
    """
    lines = []
    for number, run_index in enumerate(window):
        span = runs[run_index]
        previous = runs[run_index - 1] if run_index > 0 else None
        pause = None if previous is None else (
            float(words[span[0]]["start"]) - float(words[previous[1] - 1]["end"])
        )
        lines.append(turn_line(words, span, number, pause))
    return "\n".join(lines)


def window_labels(words: list[dict], runs: list[tuple[int, int]], window: list[int]) -> set[str]:
    """Speaker labels the model was actually shown in this request."""
    return {words[runs[i][0]].get("speaker") or "UNKNOWN" for i in window}


def build_windows(words: list[dict], runs: list[tuple[int, int]], measure,
                  budget: int = MAX_CONTEXT_TOKENS,
                  overlap: int = OVERLAP_TURNS) -> list[list[int]]:
    """Run indices grouped into requests, whole transcript first if it fits.

    `measure` maps rendered text to a token count, injected so the windowing can
    be tested without loading a tokenizer. Each window after the first re-seeds
    the previous window's last `overlap` turns, the way build_chunks re-seeds
    sentences: a turn at a window edge is otherwise judged with no left context,
    which is precisely the context this task needs. Corrections are unioned, so
    seeing a turn twice costs nothing.
    """
    if not runs:
        return []
    lengths = [measure(turn_line(words, span, 0, 0.0)) for span in runs]
    if sum(lengths) <= budget:
        return [list(range(len(runs)))]

    windows: list[list[int]] = []
    current: list[int] = []
    size = 0
    for index, length in enumerate(lengths):
        if current and size + length > budget:
            windows.append(current)
            current = current[-overlap:] if overlap else []
            size = sum(lengths[i] for i in current)
        current.append(index)
        size += length
    if current:
        windows.append(current)
    return windows


def parse_corrections(text: str) -> tuple[list[dict], int] | None:
    """Usable corrections and a count of malformed ones, or None if unparseable.

    Tolerant in the same two ways parse_response is -- a markdown fence the
    prompt forbade, and prose either side of the object -- because both are
    recoverable formatting slips rather than bad judgement.

    The malformed count is returned rather than dropped. An entry with no turn
    number, no quote or an invented action is the model misunderstanding the
    contract, and it must not be indistinguishable from a transcript with
    nothing to correct: that is the shape of every bug this repo has had to
    find twice.
    """
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
    entries = payload.get("corrections")
    if not isinstance(entries, list):
        return None

    clean: list[dict] = []
    malformed = 0
    for entry in entries:
        if not isinstance(entry, dict):
            malformed += 1
            continue
        try:
            turn = int(entry.get("turn"))
        except (TypeError, ValueError):
            malformed += 1
            continue
        quote = str(entry.get("text", "")).strip()
        speaker = str(entry.get("speaker", "")).strip()
        action = str(entry.get("action", "")).strip().lower()
        if not quote or not speaker or action not in ACTIONS:
            malformed += 1
            continue
        clean.append({"turn": turn, "text": quote, "speaker": speaker, "action": action})
    return clean, malformed


def interpret_window(words: list[dict], runs: list[tuple[int, int]], window: list[int],
                     entries: list[dict], claimed: dict[int, str],
                     ) -> tuple[dict[int, str], Counter]:
    """Validated speaker changes for one request, and why each entry was refused.

    `reassign` and `split` are the same edit at word level -- give these word
    indices this speaker -- because the turn structure is recomputed from the
    labels downstream, so a different label inside a run splits it into three by
    construction. The field is still required and still validated: it makes the
    model state its intent, it lets the report separate the two failure modes,
    and an action naming the turn's own speaker is a no-op the model should not
    be allowed to report as a correction.
    """
    labels = window_labels(words, runs, window)
    changes: dict[int, str] = {}
    counters: Counter = Counter()

    for entry in entries:
        local = entry["turn"]
        # A turn number outside the window is the model addressing something it
        # was never shown; applying it would relabel unrelated speech.
        if not 0 <= local < len(window):
            counters["unmatched"] += 1
            continue
        span = runs[window[local]]
        found = locate(words, span, entry["text"])
        # A quote that is not in the turn it was attributed to is invented. This
        # is the check the Qwen review lacked, and the reason its speaker flags
        # could never be located.
        if not found:
            counters["unmatched"] += 1
            continue
        if len(found) > MAX_MOVE_WORDS:
            counters["rejected_long"] += 1
            continue
        target = entry["speaker"]
        current = words[span[0]].get("speaker")
        # Either a label the model was never shown, or the label the turn
        # already has. Both are the model failing to name a usable target, and
        # neither can produce a correction.
        if target not in labels or target == current:
            counters["rejected_speaker"] += 1
            continue

        already = [i for i in found if i in claimed]
        if already:
            # Overlapping windows see the same turn twice, so the identical
            # correction arriving again is expected and harmless. Anything else
            # -- a different target, or only part of the span already spoken for
            # -- is two answers for one word, and neither is trustworthy. The
            # first is kept because something has to be.
            same = len(already) == len(found) and {claimed[i] for i in already} == {target}
            counters["duplicate" if same else "rejected_conflict"] += 1
            continue

        for index in found:
            changes[index] = target
            claimed[index] = target
        counters["applied"] += 1
        counters[f"applied_{entry['action']}"] += 1
    return changes, counters


def correct_words(words: list[dict], ask, measure,
                  budget: int = MAX_CONTEXT_TOKENS) -> tuple[list[dict], dict]:
    """Corrected word list and a report, given a way to ask the model.

    `ask` maps the numbered turns to a raw reply and `measure` maps text to a
    token count. Both are injected so the protocol -- windowing, addressing,
    validation, application -- is testable without a GPU, which is the only part
    that can be wrong in a way that produces plausible output.
    """
    runs = speaker_runs(words)
    windows = build_windows(words, runs, measure, budget)
    counters: Counter = Counter()
    changes: dict[int, str] = {}
    claimed: dict[int, str] = {}

    for window in windows:
        numbered = render_window(words, runs, window)
        try:
            reply = ask(numbered)
        except Exception as error:  # noqa: BLE001 - a failed window must be counted
            logger.warning("window failed: %s", error)
            counters["chunk_failures"] += 1
            continue
        parsed = parse_corrections(reply)
        # An unparseable reply keeps the window's original labels and is
        # counted. A failed window must never look like a window with nothing
        # to correct in it.
        if parsed is None:
            counters["chunk_failures"] += 1
            continue
        entries, malformed = parsed
        counters["corrections_proposed"] += len(entries) + malformed
        counters["malformed"] += malformed
        found, window_counters = interpret_window(words, runs, window, entries, claimed)
        changes.update(found)
        counters.update(window_counters)

    corrected = apply_changes(words, changes, SOURCE)
    moved = sum(1 for old, new in zip(words, corrected)
                if old.get("speaker") != new.get("speaker"))
    report = {
        "source": SOURCE,
        "turns": len(runs),
        "windows": len(windows),
        # How many extra requests the context guard forced. Zero is the normal
        # path: one call for the whole transcript.
        "window_fallbacks": max(len(windows) - 1, 0),
        "corrections_proposed": counters["corrections_proposed"],
        "corrections_applied": counters["applied"],
        "applied_reassign": counters["applied_reassign"],
        "applied_split": counters["applied_split"],
        "words_moved": moved,
        "unmatched": counters["unmatched"],
        "malformed": counters["malformed"],
        "rejected_speaker": counters["rejected_speaker"],
        "rejected_long": counters["rejected_long"],
        "rejected_conflict": counters["rejected_conflict"],
        "duplicate": counters["duplicate"],
        "chunk_failures": counters["chunk_failures"],
    }
    return corrected, report


def make_ask(model, tokenizer, max_new_tokens: int = MAX_NEW_TOKENS):
    """Bind a loaded model into the single-call interface `correct_words` wants."""
    import torch

    def ask(numbered: str) -> str:
        text = tokenizer.apply_chat_template(
            build_messages(numbered), tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(
            generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )
    return ask


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", required=True, help="tree of transcript.json files")
    parser.add_argument("--pattern", default="transcript.json")
    parser.add_argument(
        "--suffix", default="_speakers.json",
        help=(
            "appended to the source stem, so transcript.json becomes "
            "transcript_speakers.json -- the name the scorer's adapter globs for"
        ),
    )
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument(
        "--max-context-tokens", type=int, default=MAX_CONTEXT_TOKENS,
        help="transcripts rendering longer than this are split into overlapping windows",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--only", action="append", default=None, metavar="SUBSTRING",
        help=(
            "keep only files whose path contains this (repeatable). For pilots "
            "on named visits, where --limit would just take the first N"
        ),
    )
    parser.add_argument("--shard", default=None, metavar="I/N")
    parser.add_argument(
        "--redo", action="store_true", help="re-correct files that already have output",
    )
    parser.add_argument("--report", default=None, help="JSON summary over the whole tree")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    args = build_parser().parse_args(argv)

    root = Path(args.outputs)
    files = sorted(p for p in root.rglob(args.pattern) if not p.name.endswith(args.suffix))
    if not files:
        logger.error("No %s under %s", args.pattern, root)
        return 1
    if args.only:
        files = [f for f in files if any(part in str(f) for part in args.only)]
        logger.info("Filtered to %d file(s) by --only", len(files))
    if not args.redo:
        files = [f for f in files if not destination_for(f, args.suffix).exists()]
    if args.shard:
        index_str, _, count_str = args.shard.partition("/")
        files = files[int(index_str) - 1:: int(count_str)]
        logger.info("Shard %s: %d file(s)", args.shard, len(files))
    if args.limit:
        files = files[: args.limit]
    if not files:
        logger.info("Nothing to do")
        return 0

    model, tokenizer = load_model(args.model, args.device)
    ask = make_ask(model, tokenizer, args.max_new_tokens)

    def measure(text: str) -> int:
        return len(tokenizer(text)["input_ids"])

    totals: Counter = Counter()
    per_file: dict[str, dict] = {}
    for index, path in enumerate(files, start=1):
        started = time.perf_counter()
        payload, words = load_words(path)
        if not words:
            logger.warning("%s: no words", path)
            continue
        corrected, report = correct_words(words, ask, measure, args.max_context_tokens)
        report["model"] = args.model
        write_corrected(path, payload, words, corrected, report, args.suffix)
        per_file[str(path.relative_to(root))] = report
        for key, value in report.items():
            if isinstance(value, int):
                totals[key] += value
        logger.info(
            "[%d/%d] %s: %d proposed, %d applied, %d word(s) moved over %d "
            "window(s), %d unmatched, %d failed, %.0fs",
            index, len(files), path.parent.name, report["corrections_proposed"],
            report["corrections_applied"], report["words_moved"], report["windows"],
            report["unmatched"], report["chunk_failures"],
            time.perf_counter() - started,
        )

    logger.info(
        "moved %d word(s); proposed %d, applied %d, unmatched %d, "
        "long %d, bad speaker %d, conflict %d, duplicate %d, failed window %d",
        totals["words_moved"], totals["corrections_proposed"],
        totals["corrections_applied"], totals["unmatched"], totals["rejected_long"],
        totals["rejected_speaker"], totals["rejected_conflict"], totals["duplicate"],
        totals["chunk_failures"],
    )
    if totals["window_fallbacks"]:
        logger.warning(
            "%d transcript window(s) beyond the first: those files exceeded the "
            "context budget and were judged without their full context",
            totals["window_fallbacks"],
        )

    if args.report:
        Path(args.report).write_text(json.dumps(
            {"totals": dict(totals), "per_file": per_file}, indent=1))
        logger.info("wrote %s", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

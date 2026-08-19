"""Compare PII redaction between the human transcripts and a system's output.

The human transcripts carry the gold standard already: transcribers wrap
identifying material in curly braces -- {isaiah}, {san francisco}, {May 9th} --
and 514 of the 887 spans corpus-wide were scrubbed to {redacted} outright, so
the span marks a location even where the surface form is gone. (This is also
why the partner team's compareFiles.py strips \\{[^}]*\\} before scoring WER.)
Chirp-3 redacts natively into square-bracket labels: PERSON_NAME, DATE,
LOCATION, AGE, GENDER, US_STATE, DATE_OF_BIRTH, 4803 tokens corpus-wide. Our
own pipelines redact nothing, which is what redact_llm.py exists to fix.

Matching is done by aligning token sequences, not by timestamps. A gold span
and a system placeholder never share surface text -- "isaiah" against
"[PERSON_NAME]" -- so they land in a difflib replace block, and the block is
exactly the correspondence needed: the system tokens opposite a gold span are
the system's treatment of that span. Timestamps would work too, but human turn
ends are synthesized from the next turn's start (see prepare_data), so a
within-turn position would have to be interpolated and would carry seconds of
error in dense speech.

Two numbers come out of this, and they answer different questions:

  span P/R/F1   the paper's metric (DIALOG-DeID 2.3): did the system mark PII
                where the humans marked PII. Category-agnostic here, because
                the gold braces carry no label.
  leak rate     of the gold spans whose surface form survives in the human
                transcript, how many still read verbatim in the system's
                output at that point. The number a privacy reviewer asks for.
  identifier
  leak rate     of the distinct identifiers in a transcript, how many can
                still be read somewhere in the system's output.

The two leak numbers are deliberately separate, and conflating them is a
mistake this file made. The first version searched the whole transcript for
the surface form and then charged the result to every occurrence of it: on
SI00132/day0223_session004 a first name is marked seventeen times, Chirp-3
redacted eleven of them, and all seventeen were recorded as leaked -- with a
per-occurrence context showing the placeholder that had correctly replaced it.
That penalised the systems that redact most occurrences hardest: half of the
Gemma arms' recorded leaks were occurrences they had themselves redacted,
against 5.6% of verbatimize's. Occurrences are now tested where they stand,
and "is this name readable anywhere" is counted once per name.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path

# Exactly the PII labels, never a general uppercase-in-brackets pattern.
# CrisperWhisper writes its filled pauses and vocal events the same way --
# [UM], [UH], [LAUGHTER] -- so a broad pattern reads a verbatim transcript as
# one enormous redaction: it scored our unredacted output at 30,487 "redactions"
# over 269 visits, all of them fillers, and produced an entirely plausible
# table. The label set is closed and must stay in sync with redact_llm.LABELS.
PII_LABELS = [
    "PERSON_NAME", "NAME", "DATE", "DATE_OF_BIRTH", "LOCATION", "US_STATE",
    "ADDRESS", "AGE", "GENDER", "PHONE_NUMBER", "EMAIL", "ORGANIZATION",
]
PLACEHOLDER = re.compile(r"\[(" + "|".join(PII_LABELS) + r")\]")
GOLD_SPAN = re.compile(r"\{([^}]*)\}")

# "{redacted}" marks a span whose surface form the transcriber already removed,
# so it can be matched positionally but never leak-tested.
SCRUBBED = {"redacted"}

# Category mapping onto the paper's three reported categories, so per-category
# figures stay comparable with its Table 5. AGE and GENDER have no gold
# counterpart in the brace convention and are reported separately.
CATEGORY = {
    "PERSON_NAME": "name",
    "NAME": "name",
    "ADDRESS": "location",
    "PHONE_NUMBER": "contact",
    "EMAIL": "contact",
    "ORGANIZATION": "organization",
    "DATE": "date",
    "DATE_OF_BIRTH": "date",
    "LOCATION": "location",
    "US_STATE": "location",
    "AGE": "age",
    "GENDER": "gender",
}

TURN_PREFIX = re.compile(r"^(\S+)\s+((?:\d+:)?\d{1,2}:\d{2}(?:\.\d+)?)\s+(.+)$")


def normalize_token(token: str) -> str:
    return re.sub(r"[^a-z0-9']", "", token.lower())


def human_tokens(path: Path) -> tuple[list[str], list[dict]]:
    """Tokenize a whole human transcript file."""
    lines = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        match = TURN_PREFIX.match(line)
        lines.append(match.group(3) if match else line)
    return tokens_from_texts(lines)


def tokens_from_texts(texts: list[str]) -> tuple[list[str], list[dict]]:
    """Tokens and gold span records from already-parsed turn texts.

    Callers that clip the hypothesis to a coverage window must build the
    reference from the same window, or difflib aligns a whole-session reference
    against a partial hypothesis and every gold span past the window is charged
    as unprotected. Each span record carries its token index range, so a span
    can be located in the aligned system sequence.
    """
    tokens: list[str] = []
    spans: list[dict] = []
    for text in texts:
        # Walk the text so a span's token indices are known as it is consumed.
        position = 0
        for found in GOLD_SPAN.finditer(text):
            tokens.extend(text[position:found.start()].split())
            inner = found.group(1).split()
            start_index = len(tokens)
            tokens.extend(inner)
            surface = " ".join(inner)
            spans.append({
                "start": start_index,
                "end": len(tokens),
                "surface": surface,
                "scrubbed": all(normalize_token(t) in SCRUBBED for t in inner) or not inner,
            })
            position = found.end()
        tokens.extend(text[position:].split())
    return tokens, spans


def system_tokens(words: list[dict]) -> tuple[list[str], list[dict]]:
    """Tokenize a system word list, returning tokens and placeholder spans.

    Consecutive placeholder tokens of the same label collapse into one span:
    Chirp emits "[DATE]. [DATE]." for the single gold span "{May 9th}", and
    counting those as two redactions would report over-redaction that is really
    tokenization.
    """
    tokens = [str(w.get("word", "")).strip() for w in words]
    tokens = [t for t in tokens if t]

    spans: list[dict] = []
    index = 0
    while index < len(tokens):
        found = PLACEHOLDER.search(tokens[index])
        if not found:
            index += 1
            continue
        label = found.group(0).strip("[]")
        start = index
        index += 1
        while index < len(tokens):
            nxt = PLACEHOLDER.search(tokens[index])
            if not nxt or nxt.group(0).strip("[]") != label:
                break
            index += 1
        spans.append({
            "start": start, "end": index, "label": label,
            "category": CATEGORY.get(label, "other"),
        })
    return tokens, spans


def index_map(reference: list[str], hypothesis: list[str]) -> list[tuple[int, int, int, int]]:
    """difflib opcodes over normalized tokens, as (i1, i2, j1, j2) blocks."""
    ref = [normalize_token(t) for t in reference]
    hyp = [normalize_token(t) for t in hypothesis]
    matcher = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    return [(i1, i2, j1, j2) for _, i1, i2, j1, j2 in matcher.get_opcodes()]


def project(blocks, start: int, end: int, pad: int = 2) -> tuple[int, int]:
    """Map a reference token range onto the hypothesis, widened by `pad`.

    The padding matters: a redactor may drop the trailing punctuation token or
    emit one placeholder where the reference had two words, so the corresponding
    hypothesis region is rarely the same length. Two tokens is enough to absorb
    that without reaching into neighbouring speech.
    """
    low, high = None, None
    for i1, i2, j1, j2 in blocks:
        if i2 <= start or i1 >= end:
            continue
        # Offset within an equal-length block maps one-to-one; otherwise the
        # whole hypothesis block is the candidate region.
        if i2 - i1 == j2 - j1:
            lo = j1 + max(start - i1, 0)
            hi = j1 + min(end - i1, i2 - i1)
        else:
            lo, hi = j1, j2
        low = lo if low is None else min(low, lo)
        high = hi if high is None else max(high, hi)
    if low is None:
        return 0, 0
    return max(low - pad, 0), high + pad


def context(tokens: list[str], start: int, end: int, width: int = 7) -> str:
    """Surrounding words, for a human reading a leak report."""
    return " ".join(tokens[max(start - width, 0):min(end + width, len(tokens))])


def score_visit(human: Path | list[dict], words: list[dict]) -> dict:
    """Redaction metrics for one visit, with per-span detail for inspection.

    `human` is either the transcript path or the already-windowed turn list.
    Pass turns whenever the words have been clipped to a coverage window: a
    whole-file reference against a clipped hypothesis charges every gold span
    past the window as unprotected, which under a 30-minute cap moved exposure
    by 3-4 points.
    """
    if isinstance(human, Path):
        ref_tokens, gold = human_tokens(human)
    else:
        ref_tokens, gold = tokens_from_texts([t["text"] for t in human])
    hyp_tokens, predicted = system_tokens(words)
    blocks = index_map(ref_tokens, hyp_tokens)

    # A gold span is found if any placeholder overlaps its projected region.
    matched_predictions: set[int] = set()
    true_positives = 0
    details: list[dict] = []
    joined = " ".join(normalize_token(t) for t in hyp_tokens)

    for span in gold:
        low, high = project(blocks, span["start"], span["end"])
        hit = [
            k for k, p in enumerate(predicted)
            if p["start"] < high and p["end"] > low
        ]
        if hit:
            true_positives += 1
            matched_predictions.update(hit)

        # A span the transcriber already scrubbed to {redacted} has no surface
        # form left, so it can be matched positionally but never leak-tested.
        testable = not span["scrubbed"] and bool(span["surface"].strip())
        needle = " ".join(normalize_token(t) for t in span["surface"].split())
        pattern = rf"(?:^| ){re.escape(needle)}(?: |$)" if needle else None

        # Tested where the occurrence stands: the system tokens this occurrence
        # projects onto, and nothing else. Two occurrences close enough to fall
        # in one alignment block share a region and are therefore both charged
        # if the name survives inside it -- that is a genuine ambiguity about
        # which of them was redacted, not something a wider or narrower window
        # would resolve.
        region = " ".join(normalize_token(t) for t in hyp_tokens[low:high])
        leaked = bool(testable and pattern and re.search(pattern, region))
        # The same surface form anywhere in the output. Aggregated per distinct
        # identifier below; never charged to each occupation of it.
        elsewhere = bool(testable and pattern and re.search(pattern, joined))
        details.append({
            "surface": span["surface"],
            "testable": testable,
            "leaked": leaked,
            "readable_somewhere": elsewhere,
            "redacted": bool(hit),
            "labels": sorted({predicted[k]["label"] for k in hit}),
            "human_context": context(ref_tokens, span["start"], span["end"]),
            "system_context": context(hyp_tokens, low, high) if hyp_tokens else "",
        })

    false_negatives = len(gold) - true_positives
    false_positives = len(predicted) - len(matched_predictions)
    testable_spans = [d for d in details if d["testable"]]
    leaked = sum(1 for d in testable_spans if d["leaked"])

    # One entry per distinct identifier, so a name marked seventeen times
    # counts once toward "can this person still be identified".
    identifiers: dict[str, bool] = {}
    for detail in testable_spans:
        key = " ".join(normalize_token(t) for t in detail["surface"].split())
        identifiers[key] = identifiers.get(key, False) or detail["readable_somewhere"]

    categories: dict[str, int] = {}
    for span in predicted:
        categories[span["category"]] = categories.get(span["category"], 0) + 1

    precision = true_positives / len(predicted) if predicted else None
    recall = true_positives / len(gold) if gold else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall else (0.0 if gold or predicted else None)
    )
    return {
        "gold_spans": len(gold),
        "predicted_spans": len(predicted),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "leak_testable": len(testable_spans),
        "leaked": leaked,
        "identifiers_testable": len(identifiers),
        "identifiers_readable": sum(1 for v in identifiers.values() if v),
        "categories": categories,
        "spans": details,
    }

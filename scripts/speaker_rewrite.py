"""Shared pieces for anything that rewrites speaker labels on a finished transcript.

Both correctors -- the deterministic rule and the Gemma one -- produce the same
artefact: a copy of the source transcript in which only `word["speaker"]` may
differ, every changed word carrying provenance. Keeping the writer here means
the two are byte-comparable and the scoring adapters cannot tell them apart,
which is the point: the rule exists to make a null result from the model
interpretable.

The invariant that makes this cheap to verify: a speaker rewrite must not
change the words, their order, or their times. `assert_words_intact` enforces
that in the writer rather than only in a test, because a corrupted word list
would otherwise show up as a mysterious WER movement several scripts later.
"""

from __future__ import annotations

import json
from pathlib import Path


def speaker_runs(words: list[dict]) -> list[tuple[int, int]]:
    """Half-open index ranges of consecutive words sharing a speaker label."""
    runs: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(words) + 1):
        if index == len(words) or words[index].get("speaker") != words[start].get("speaker"):
            runs.append((start, index))
            start = index
    return runs


def apply_changes(words: list[dict], changes: dict[int, str], source: str) -> list[dict]:
    """Copy of `words` with `changes` applied and provenance on what moved.

    Follows `redact_llm.apply_labels`: copy rather than mutate, and record what
    the label used to be so a reviewer can see the correction without diffing
    two files. A change to the label a word already had is dropped, so the
    provenance keys never appear on a word that did not actually move.
    """
    out: list[dict] = []
    for index, word in enumerate(words):
        copy = dict(word)
        proposed = changes.get(index)
        if proposed is not None and proposed != copy.get("speaker"):
            copy["speaker_from"] = copy.get("speaker")
            copy["speaker"] = proposed
            copy["speaker_by"] = source
        out.append(copy)
    return out


def assert_words_intact(before: list[dict], after: list[dict]) -> None:
    """Raise unless only speaker labels differ between the two word lists."""
    if len(before) != len(after):
        raise ValueError(f"word count changed: {len(before)} -> {len(after)}")
    for index, (old, new) in enumerate(zip(before, after)):
        if old["word"] != new["word"]:
            raise ValueError(f"word {index} changed: {old['word']!r} -> {new['word']!r}")
        if old["start"] != new["start"] or old["end"] != new["end"]:
            raise ValueError(f"word {index} retimed: {old['start']}-{old['end']} -> "
                             f"{new['start']}-{new['end']}")


def destination_for(path: Path, suffix: str) -> Path:
    """Sibling filename carrying `suffix`, as `redact_llm.destination_for` does."""
    return path.with_name(path.stem + suffix)


def load_words(path: Path) -> tuple[dict | list, list[dict]]:
    """Transcript payload and its word list, accepting both output shapes.

    Everything this repo writes is an object with a `words` key; the other
    team's pipeline writes a bare array. Both appear in the trees this runs
    over, and reading only one of them is a mistake this repo has already made
    once (`as_words` in evaluate_systems exists for the same reason).
    """
    payload = json.loads(path.read_text())
    words = payload["words"] if isinstance(payload, dict) else payload
    return payload, words


def write_corrected(path: Path, payload, words: list[dict], corrected: list[dict],
                    report: dict, suffix: str) -> Path:
    """Write the corrected transcript beside its source and return the path."""
    assert_words_intact(words, corrected)
    destination = destination_for(path, suffix)
    if isinstance(payload, dict):
        body = dict(payload)
        body["words"] = corrected
        body["speaker_correction"] = report
    else:
        body = corrected
    destination.write_text(json.dumps(body, indent=1))
    return destination

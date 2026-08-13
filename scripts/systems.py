"""Canonical names for the systems under evaluation.

One registry, imported by every scorer and by the report, so a system is
described the same way in a console table, a JSON file, a CSV and a chart.

Short keys like "ours" and "baseline" survive only as identifiers -- CLI
arguments, output directory names, dict keys in code. They never reach a
rendered output: anything a person reads names the ASR model, then the
diarization system, then any post-processing, joined by "+". "ours" tells a
reader nothing and ages badly the moment someone else reads the report.

Adding a system means adding it here and nowhere else.
"""

from __future__ import annotations

# key -> (components, colour, note)
# Colours are held constant across every chart so a system can be tracked by eye
# between panels. Green is deliberately absent: the design language reserves it
# for the winner marker.
SYSTEMS: list[tuple[str, list[str], str, str]] = [
    (
        "chirp3",
        ["Chirp-3 ASR", "Chirp-3 diarization", "Chirp-3 redaction"],
        "#898781",
        "incumbent, as delivered by the bucket",
    ),
    (
        "verbatimize",
        ["Chirp-3 ASR", "CrisperWhisper 2.0 verbatimize", "Chirp-3 diarization"],
        "#e87ba4",
        "Chirp text kept, disfluencies inserted by CW2",
    ),
    (
        "ours",
        ["CrisperWhisper 2.0 ASR", "pyannote community-1"],
        "#2a78d6",
        "verbatim ASR with community-1 diarization",
    ),
    (
        "ours_llm",
        ["CrisperWhisper 2.0 ASR", "pyannote community-1", "Qwen2.5-7B review"],
        "#7fb0e8",
        "community-1 pipeline with the LLM review applied",
    ),
    (
        "ours_redacted",
        ["CrisperWhisper 2.0 ASR", "pyannote community-1", "Gemma 4 31B redaction"],
        "#5b4bd6",
        "community-1 pipeline, PII redacted in windows",
    ),
    (
        "ours_redacted_turn",
        ["CrisperWhisper 2.0 ASR", "pyannote community-1",
         "Gemma 4 31B redaction (turn rewrite)"],
        "#8f7be0",
        "community-1 pipeline, PII redacted one speaker turn at a time",
    ),
    (
        "ours_redacted_poss",
        ["CrisperWhisper 2.0 ASR", "pyannote community-1",
         "Gemma 4 31B redaction (possessive rule)"],
        "#6f5bd8",
        "chunk protocol, prompt fixed to quote possessives with their 's",
    ),
    (
        "ours_redacted_turn_poss",
        ["CrisperWhisper 2.0 ASR", "pyannote community-1",
         "Gemma 4 31B redaction (turn rewrite, possessive rule)"],
        "#a394ec",
        "turn protocol, prompt fixed to keep the 's on the label",
    ),
    (
        "baseline",
        ["CrisperWhisper 2.0 ASR", "pyannote 3.1"],
        "#eda100",
        "verbatim ASR with pyannote 3.1 and stereo channel dominance",
    ),
    (
        "baseline_llm",
        ["CrisperWhisper 2.0 ASR", "pyannote 3.1", "Qwen2.5-7B review"],
        "#f5cc6b",
        "pyannote 3.1 pipeline with the LLM review applied",
    ),
    (
        "baseline_mono",
        ["CrisperWhisper 2.0 ASR", "pyannote 3.1 (forced mono)"],
        "#b8791f",
        "pyannote 3.1 with the stereo channel path disabled",
    ),
    (
        "baseline_redacted",
        ["CrisperWhisper 2.0 ASR", "pyannote 3.1", "Gemma 4 31B redaction"],
        "#9a8ae8",
        "pyannote 3.1 pipeline, PII redacted in windows",
    ),
    (
        "baseline_redacted_turn",
        ["CrisperWhisper 2.0 ASR", "pyannote 3.1",
         "Gemma 4 31B redaction (turn rewrite)"],
        "#c2b6f2",
        "pyannote 3.1 pipeline, PII redacted one speaker turn at a time",
    ),
]

PARTS = {key: parts for key, parts, _, _ in SYSTEMS}
LABELS = {key: " + ".join(parts) for key, parts, _, _ in SYSTEMS}
BY_LABEL = {label: key for key, label in LABELS.items()}


def label_of(key: str) -> str:
    """Display name for a system key, or the key itself if unregistered."""
    return LABELS.get(key, key)


def key_of(label: str) -> str:
    """Inverse of label_of, tolerant of already being a key."""
    return BY_LABEL.get(label, label)


def entry_of(mapping: dict, key: str):
    """Look a system up in a results dict by key or by label.

    Results files written before the rename are keyed by the short name, and
    reading them must keep working -- a naming change is not a reason to
    invalidate a scoring run that took hours.
    """
    if key in mapping:
        return mapping[key]
    return mapping.get(label_of(key))


def present_keys(mapping: dict) -> list[str]:
    """Registered systems appearing in a results dict, in registry order."""
    return [key for key, *_ in SYSTEMS if entry_of(mapping, key) is not None]


def report(rows: list[tuple[str, list[tuple[str, str]]]]) -> str:
    """Render per-system figures as labelled blocks rather than a wide table.

    A full name runs to seventy characters, so it cannot be a table column
    without pushing the numbers off the terminal. Two lines per system keeps
    every figure attached to a name a reader can act on.
    """
    out = []
    for label, fields in rows:
        out.append(label)
        out.append("    " + "   ".join(f"{name} {value}" for name, value in fields))
    return "\n".join(out)

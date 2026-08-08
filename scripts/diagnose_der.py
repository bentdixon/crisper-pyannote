"""Decompose DER to test whether it measures diarization or segment granularity.

The human transcripts carry turn START times only; evaluate_systems builds each
reference turn's end from the next turn's start, so the reference tiles the whole
recording and contains no non-speech. A word-level hypothesis has real gaps
between words, and every one of those gaps becomes "missed detection" against a
reference that calls it speech.

If that is what is happening, DER is ranking systems by how much of the timeline
their segments cover rather than by whether they put the right speaker on the
right words. This script prints the components so the question is settled with
numbers: coverage of each side, plus DER split into missed / false alarm /
confusion.

Usage:
    uv run python scripts/diagnose_der.py --cohort /path/to/cohort --limit 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "finetune"))

from prepare_data import load_timestamped_text  # noqa: E402
from pyannote.metrics.diarization import DiarizationErrorRate  # noqa: E402

from evaluate_systems import (  # noqa: E402
    load_chirp,
    predicted_streams,
    reference_streams,
    to_annotation,
)

logging.basicConfig(level=logging.WARNING)


def covered(spans_by_speaker: dict) -> float:
    """Union length of every span, so overlaps are not double counted."""
    spans = sorted(
        (start, end)
        for entry in spans_by_speaker.values()
        for start, end in entry["spans"]
        if end > start
    )
    total, cursor = 0.0, None
    for start, end in spans:
        if cursor is None or start > cursor:
            total += end - start
            cursor = end
        elif end > cursor:
            total += end - cursor
            cursor = end
    return total


def turn_spans(words: list[dict], bridge_to_next: bool) -> dict:
    """Group consecutive same-speaker words into spans.

    With bridge_to_next, each span is extended to the start of the following
    word regardless of speaker -- the same rule that built the reference's
    ends. That makes both sides tile the timeline the same way, so DER can
    only be moved by label disagreement, not by granularity.
    """
    usable = [w for w in words if w.get("start") is not None and w.get("end") is not None]
    if not usable:
        return {}
    streams: dict[str, dict] = {}
    index = 0
    while index < len(usable):
        speaker = usable[index]["speaker"]
        start = usable[index]["start"]
        end = usable[index]["end"]
        while index + 1 < len(usable) and usable[index + 1]["speaker"] == speaker:
            index += 1
            end = usable[index]["end"]
        if bridge_to_next and index + 1 < len(usable):
            end = max(end, usable[index + 1]["start"])
        streams.setdefault(speaker, {"spans": [], "text": ""})["spans"].append((start, end))
        index += 1
    return streams


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    cohort = Path(args.cohort)
    visits = sorted(
        p for p in cohort.glob("Pronet*/*/*")
        if (p / "human").is_dir() and any((p / "human").glob("*.txt"))
        and any((p / "audio").glob("*.wav"))
    )[: args.limit]

    print(
        f"{'visit':34} {'audio_s':>8} {'ref_cov':>8} {'hyp_cov':>8} "
        f"{'DER':>7} {'miss':>7} {'FA':>7} {'conf':>7} {'DER_turn':>9} {'DER_tile':>9}"
    )

    for visit in visits:
        human = sorted((visit / "human").glob("*.txt"))[0]
        audio = sorted((visit / "audio").glob("*.wav"))[0]
        with wave.open(str(audio)) as handle:
            duration = handle.getnframes() / handle.getframerate()

        try:
            turns = load_timestamped_text(human, duration)
            words = load_chirp(visit)
        except Exception as error:
            print(f"{visit.name:34} skipped: {type(error).__name__}: {error}")
            continue
        if not words:
            continue

        reference = reference_streams(turns)
        word_level = predicted_streams(words)

        ref_annotation = to_annotation(reference)
        components = DiarizationErrorRate(collar=0.25, skip_overlap=False)(
            ref_annotation, to_annotation(word_level), detailed=True
        )
        total = components["total"] or 1.0

        def der_for(streams: dict) -> float:
            return float(
                DiarizationErrorRate(collar=0.25, skip_overlap=False)(
                    ref_annotation, to_annotation(streams)
                )
            )

        label = f"{visit.parent.name}/{visit.name}"[:34]
        print(
            f"{label:34} {duration:8.0f} "
            f"{covered(reference) / duration:8.2f} {covered(word_level) / duration:8.2f} "
            f"{components['diarization error rate']:7.3f} "
            f"{components['missed detection'] / total:7.3f} "
            f"{components['false alarm'] / total:7.3f} "
            f"{components['confusion'] / total:7.3f} "
            f"{der_for(turn_spans(words, False)):9.3f} "
            f"{der_for(turn_spans(words, True)):9.3f}"
        )

    print(
        "\nref_cov/hyp_cov: fraction of audio each side calls speech.\n"
        "DER_turn: words grouped into same-speaker turns.\n"
        "DER_tile: turns also extended to the next word's start, the same rule\n"
        "          that synthesized the reference ends -- both sides then tile\n"
        "          the timeline, so only label disagreement can move DER."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

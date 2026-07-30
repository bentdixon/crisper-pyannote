"""Merge diarization output with ASR output.

Reproduces the pyannoteAI diarization-ASR merge tutorial:
https://docs.pyannote.ai/tutorials/diarization-asr-merge

Each transcript segment is assigned the speaker with the maximum temporal
overlap. CrisperWhisper produces word-level timestamps, so each word (a dict
with "start" and "end") is treated as one transcript segment.
"""

from __future__ import annotations


def assign_speakers(
    diarization_segments: list[dict],
    transcript_segments: list[dict],
    fill_nearest: bool = False,
) -> list[dict]:
    """Assign a speaker to each transcript segment by maximum overlap.

    The loop body below follows the tutorial code exactly. Segments with no
    overlapping diarization segment get the nearest speaker (by midpoint
    distance) when fill_nearest is true, otherwise "UNKNOWN".

    Mutates and returns transcript_segments, adding a "speaker" key to each.
    """
    diarization_segments = sorted(diarization_segments, key=lambda x: x["start"])

    for seg in transcript_segments:
        seg_start = seg.get("start", 0.0)
        seg_end = seg.get("end", 0.0)
        speaker_overlap: dict[str, float] = {}

        for dia in diarization_segments:
            intersection = min(dia["end"], seg_end) - max(dia["start"], seg_start)
            if intersection <= 0:
                continue

            speaker = dia["speaker"]
            speaker_overlap[speaker] = speaker_overlap.get(speaker, 0.0) + intersection

        if speaker_overlap:
            seg["speaker"] = max(speaker_overlap.items(), key=lambda x: x[1])[0]
            continue

        if fill_nearest and diarization_segments:
            midpoint = (seg_start + seg_end) / 2
            nearest = min(
                diarization_segments,
                key=lambda x: abs(((x["start"] + x["end"]) / 2) - midpoint),
            )
            seg["speaker"] = nearest["speaker"]
            continue

        seg["speaker"] = "UNKNOWN"

    return transcript_segments


def _interval_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Intersection-over-union of two time intervals."""
    intersection = min(a_end, b_end) - max(a_start, b_start)
    if intersection <= 0:
        return 0.0
    union = (a_end - a_start) + (b_end - b_start) - intersection
    return intersection / union


def _best_chunk(
    words: list[dict],
    available: list[bool],
    seg_start: float,
    seg_end: float,
) -> tuple[float, int, int] | None:
    """Find the contiguous run of available words that best fits a segment.

    Candidates are runs of consecutive available words that overlap the
    segment; the winner maximizes IoU between the run's time span
    (first word start to last word end) and the segment.

    Returns (iou, first_index, last_index) with inclusive indices, or None
    when no available word overlaps the segment.
    """
    candidates = [
        i for i, w in enumerate(words)
        if available[i] and w["end"] > seg_start and w["start"] < seg_end
    ]
    if not candidates:
        return None

    # Split candidate indices into maximal consecutive runs.
    runs: list[list[int]] = [[candidates[0]]]
    for i in candidates[1:]:
        if i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])

    best: tuple[float, int, int] | None = None
    for run in runs:
        for a, i in enumerate(run):
            for j in run[a:]:
                iou = _interval_iou(
                    words[i]["start"], words[j]["end"], seg_start, seg_end
                )
                if best is None or iou > best[0]:
                    best = (iou, i, j)
    return best


def best_fit_segments(
    diarization_segments: list[dict],
    words: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Merge overlapping diarization with ASR by best-fit word chunks.

    Alternative to assign_speakers for raw (non-exclusive) diarization:
    instead of assigning a speaker per word, each diarization segment claims
    the contiguous chunk of timestamped words whose span best fits it by
    temporal IoU. Each word belongs to at most one segment; segments are
    processed in descending order of their initial best IoU, so where
    segments overlap, the better-fitting one claims the contested words and
    the other re-fits among the words that remain.

    Mutates words, adding a "speaker" key ("UNKNOWN" for words no segment
    claimed). Returns (words, turns) where turns holds one entry per
    diarization segment that claimed at least one word, sorted by start:
        [{"speaker": str, "start": float, "end": float,
          "segment": {"start": float, "end": float}, "text": str,
          "words": [...]}, ...]
    """
    diarization_segments = sorted(diarization_segments, key=lambda x: x["start"])
    available = [True] * len(words)

    order = []
    for seg in diarization_segments:
        fit = _best_chunk(words, available, seg["start"], seg["end"])
        if fit is not None:
            order.append((fit[0], seg))
    order.sort(key=lambda x: -x[0])

    turns: list[dict] = []
    for _initial_iou, seg in order:
        fit = _best_chunk(words, available, seg["start"], seg["end"])
        if fit is None:
            continue
        _iou, i, j = fit
        chunk = words[i : j + 1]
        for k in range(i, j + 1):
            available[k] = False
        for word in chunk:
            word["speaker"] = seg["speaker"]
        turns.append(
            {
                "speaker": seg["speaker"],
                "start": chunk[0]["start"],
                "end": chunk[-1]["end"],
                "segment": {"start": seg["start"], "end": seg["end"]},
                "words": chunk,
                "text": " ".join(w["word"].strip() for w in chunk),
            }
        )

    # Words no segment claimed become UNKNOWN turns (one per consecutive
    # run) so they still appear in the human-readable transcript.
    run: list[dict] = []
    for word, free in zip(words, available):
        if free:
            word["speaker"] = "UNKNOWN"
            run.append(word)
        elif run:
            turns.append(_unknown_turn(run))
            run = []
    if run:
        turns.append(_unknown_turn(run))

    turns.sort(key=lambda t: t["start"])
    return words, turns


def _unknown_turn(run: list[dict]) -> dict:
    return {
        "speaker": "UNKNOWN",
        "start": run[0]["start"],
        "end": run[-1]["end"],
        "segment": None,
        "words": list(run),
        "text": " ".join(w["word"].strip() for w in run),
    }


def group_into_turns(words: list[dict]) -> list[dict]:
    """Group consecutive same-speaker words into speaker turns.

    Expects words sorted by start time, each with "word", "start", "end",
    and "speaker" keys (i.e. the output of assign_speakers).

    Returns:
        [{"speaker": str, "start": float, "end": float, "text": str,
          "words": [...]}, ...]
    """
    turns: list[dict] = []
    for word in words:
        if turns and turns[-1]["speaker"] == word["speaker"]:
            turn = turns[-1]
            turn["end"] = word["end"]
            turn["words"].append(word)
        else:
            turns.append(
                {
                    "speaker": word["speaker"],
                    "start": word["start"],
                    "end": word["end"],
                    "words": [word],
                }
            )
    for turn in turns:
        turn["text"] = " ".join(w["word"].strip() for w in turn["words"])
    return turns

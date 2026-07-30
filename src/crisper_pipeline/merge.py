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

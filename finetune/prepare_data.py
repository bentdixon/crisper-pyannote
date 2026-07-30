"""Convert (transcript, wav) pairs into a pyannote.database training protocol.

Reads human-annotated transcripts (turn-level timestamps, pipeline output
style), optionally refines turn boundaries with CrisperWhisper forced
alignment, and writes RTTM/UEM/LST files plus a database.yml under
--output-dir, split by participant into train/development/test.

Usage:
    uv run python finetune/prepare_data.py \
        --transcripts /path/to/transcripts --wavs /path/to/wavs \
        --output-dir finetune/data
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import soundfile as sf

logger = logging.getLogger("prepare_data")

SUBSETS = ("train", "development", "test")


TIMESTAMPED_LINE = re.compile(
    r"^(\S+)\s+((?:\d+:)?\d{1,2}:\d{2}(?:\.\d+)?)\s+(.+)$"
)


def parse_timestamp(value: str) -> float:
    parts = value.split(":")
    seconds = float(parts[-1])
    if len(parts) > 1:
        seconds += 60 * int(parts[-2])
    if len(parts) > 2:
        seconds += 3600 * int(parts[-3])
    return seconds


def normalize_text(text: str) -> str:
    """Strip transcriber markup that would confuse forced alignment.

    Bracketed uncertain words keep their content ("[psychs?]" -> "psychs");
    whitespace is collapsed.
    """
    text = re.sub(r"\[([^\]]*?)\??\]", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def load_timestamped_text(path: Path, duration: float) -> list[dict]:
    """Parse "S1 HH:MM:SS.mmm text" transcripts (human transcription style).

    Only turn start times exist in this format: each turn's end is
    synthesized as the next turn's start (audio duration for the last one),
    so ends may include trailing silence until forced alignment refines
    them. Lines that do not open a new turn continue the previous one.
    """
    turns: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        match = TIMESTAMPED_LINE.match(line)
        if match:
            speaker, timestamp, text = match.groups()
            turns.append(
                {
                    "speaker": speaker,
                    "start": parse_timestamp(timestamp),
                    "end": None,
                    "text": normalize_text(text),
                }
            )
        elif turns:
            turns[-1]["text"] += " " + normalize_text(line)
    if not turns:
        raise ValueError(f"{path}: no timestamped turns found")
    for current, following in zip(turns, turns[1:]):
        current["end"] = following["start"]
    turns[-1]["end"] = max(duration, turns[-1]["start"])
    return turns


def load_turns(path: Path, duration: float | None = None) -> tuple[list[dict], str]:
    """Load speaker turns from a transcript file.

    Accepts three shapes:
      - .txt: "S1 HH:MM:SS.mmm text" per turn (start times only; requires
        duration to close the final turn) -> timing "turn".
      - word-level JSON (pipeline transcript.json): {"words": [{word, start,
        end, speaker}, ...]} -> grouped into turns; timing already "word".
      - turn-level JSON: {"turns": [{speaker, start, end, text}, ...]} or a
        bare list of such dicts -> timing "turn".

    Returns (turns, timing). Each turn dict has speaker, start, end, text.
    """
    if path.suffix.lower() == ".txt":
        if duration is None:
            raise ValueError(f"{path}: text transcripts need the audio duration")
        return load_timestamped_text(path, duration), "turn"

    data = json.loads(path.read_text())
    if isinstance(data, dict) and data.get("words"):
        words = data["words"]
        if not all("speaker" in w for w in words):
            raise ValueError(f"{path}: words lack speaker labels")
        turns: list[dict] = []
        for w in words:
            if turns and turns[-1]["speaker"] == w["speaker"]:
                turns[-1]["end"] = w["end"]
                turns[-1]["text"] += " " + w["word"].strip()
            else:
                turns.append(
                    {
                        "speaker": w["speaker"],
                        "start": w["start"],
                        "end": w["end"],
                        "text": w["word"].strip(),
                    }
                )
        return turns, "word"

    if isinstance(data, dict):
        data = data.get("turns")
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: no turns found")
    turns = []
    for t in data:
        turns.append(
            {
                "speaker": str(t["speaker"]),
                "start": float(t["start"]),
                "end": float(t["end"]),
                "text": str(t.get("text", "")),
            }
        )
    turns.sort(key=lambda t: t["start"])
    return turns, "turn"


def refine_turns(
    turns: list[dict],
    aligned_words: list[dict],
    max_drift: float,
) -> tuple[list[dict], dict]:
    """Replace turn boundaries with forced-alignment word times.

    aligned_words must contain one entry per whitespace-separated word of the
    concatenated turn texts, in order. A turn whose refined start drifts more
    than max_drift seconds from the human annotation keeps the human bounds.
    Drift is judged on starts only: in the timestamped-text format, human
    ends are synthesized from the next turn's start and legitimately include
    trailing silence that alignment is supposed to trim.

    Returns (refined_turns, stats).
    """
    counts = [len(t["text"].split()) for t in turns]
    if sum(counts) != len(aligned_words):
        raise ValueError(
            f"word count mismatch: transcript {sum(counts)} vs aligned {len(aligned_words)}"
        )

    refined = []
    stats = {"refined": 0, "fallback_drift": 0, "fallback_empty": 0}
    cursor = 0
    for turn, n in zip(turns, counts):
        chunk = aligned_words[cursor : cursor + n]
        cursor += n
        new = dict(turn)
        if not chunk:
            stats["fallback_empty"] += 1
        else:
            start, end = chunk[0]["start"], chunk[-1]["end"]
            if abs(start - turn["start"]) > max_drift or end <= start:
                stats["fallback_drift"] += 1
            else:
                new["start"], new["end"] = start, end
                stats["refined"] += 1
        refined.append(new)
    return refined, stats


def turns_to_segments(turns: list[dict], max_gap: float) -> list[dict]:
    """Merge consecutive same-speaker turns separated by < max_gap seconds.

    Returns non-empty speaker segments sorted by start:
        [{"start": float, "end": float, "speaker": str}, ...]
    """
    segments: list[dict] = []
    for turn in sorted(turns, key=lambda t: t["start"]):
        if turn["end"] <= turn["start"]:
            continue
        last = segments[-1] if segments else None
        if (
            last is not None
            and last["speaker"] == turn["speaker"]
            and turn["start"] - last["end"] < max_gap
        ):
            last["end"] = max(last["end"], turn["end"])
        else:
            segments.append(
                {
                    "start": turn["start"],
                    "end": turn["end"],
                    "speaker": turn["speaker"],
                }
            )
    return segments


def sanitize_speaker(name: str) -> str:
    """RTTM fields are whitespace-delimited; make speaker labels safe."""
    return re.sub(r"\s+", "_", name.strip())


def rttm_lines(uri: str, segments: list[dict]) -> list[str]:
    return [
        f"SPEAKER {uri} 1 {s['start']:.3f} {s['end'] - s['start']:.3f} "
        f"<NA> <NA> {sanitize_speaker(s['speaker'])} <NA> <NA>"
        for s in segments
    ]


def extract_participant(stem: str, pattern: str) -> str:
    match = re.search(pattern, stem)
    return match.group(1) if match else stem


def split_participants(
    participants: list[str], fractions: tuple[float, float, float], seed: int
) -> dict[str, str]:
    """Assign each participant to a subset, deterministically."""
    shuffled = sorted(set(participants))
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = round(fractions[0] * n)
    n_dev = round(fractions[1] * n)
    assignment = {}
    for i, participant in enumerate(shuffled):
        if i < n_train:
            assignment[participant] = "train"
        elif i < n_train + n_dev:
            assignment[participant] = "development"
        else:
            assignment[participant] = "test"
    return assignment


def pair_files(transcripts_dir: Path, wavs_dir: Path) -> list[tuple[Path, Path]]:
    """Match transcript JSONs to wav files by stem."""
    wavs = {p.stem: p for p in wavs_dir.rglob("*.wav")}
    transcripts = sorted(
        set(transcripts_dir.rglob("*.json")) | set(transcripts_dir.rglob("*.txt"))
    )
    pairs = []
    missing = []
    for transcript in transcripts:
        wav = wavs.get(transcript.stem)
        if wav is None:
            missing.append(transcript.stem)
        else:
            pairs.append((transcript, wav))
    if missing:
        logger.warning("%d transcript(s) without a matching wav (skipped)", len(missing))
    if not pairs:
        raise FileNotFoundError("no (transcript, wav) pairs found")
    return pairs


def align_file(asr_model, wav: Path, turns: list[dict], language: str) -> list[dict]:
    """Force-align the concatenated transcript text against the audio."""
    text = " ".join(t["text"] for t in turns)
    result = asr_model.forced_align(str(wav), text, language=language)
    return [
        {"word": w.word, "start": float(w.start), "end": float(w.end)}
        for w in (result.words or [])
    ]


def write_database_yml(output_dir: Path) -> None:
    content = f"""Databases:
  AMPSCZ:
    - {output_dir.resolve()}/wav/{{uri}}.wav

Protocols:
  AMPSCZ:
    SpeakerDiarization:
      Interviews:
        scope: file
        train:
          uri: {output_dir.resolve()}/lists/train.lst
          annotation: {output_dir.resolve()}/rttm/train.rttm
          annotated: {output_dir.resolve()}/uem/train.uem
        development:
          uri: {output_dir.resolve()}/lists/development.lst
          annotation: {output_dir.resolve()}/rttm/development.rttm
          annotated: {output_dir.resolve()}/uem/development.uem
        test:
          uri: {output_dir.resolve()}/lists/test.lst
          annotation: {output_dir.resolve()}/rttm/test.rttm
          annotated: {output_dir.resolve()}/uem/test.uem
"""
    (output_dir / "database.yml").write_text(content)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcripts", required=True, help="directory of transcript JSONs")
    parser.add_argument("--wavs", required=True, help="directory of wav files (searched recursively)")
    parser.add_argument("--output-dir", default="finetune/data")
    parser.add_argument(
        "--max-gap", type=float, default=0.5,
        help="merge same-speaker turns separated by less than this many seconds (default: 0.5)",
    )
    parser.add_argument(
        "--max-drift", type=float, default=3.0,
        help="fall back to human turn bounds when alignment drifts beyond this (default: 3.0)",
    )
    parser.add_argument(
        "--no-align", action="store_true",
        help="skip forced-alignment refinement and trust transcript timestamps",
    )
    parser.add_argument("--asr-model", default="large", help="CrisperWhisper model for forced alignment")
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--participant-regex", default=r"(?:^|_)([A-Z]{2}\d{5})(?:_|$)",
        help="regex whose first group is the participant id (default matches AMPSCZ ids like CA22695)",
    )
    parser.add_argument(
        "--splits", type=float, nargs=3, default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "DEV", "TEST"),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    for sub in ("lists", "rttm", "uem", "wav"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    pairs = pair_files(Path(args.transcripts), Path(args.wavs))
    logger.info("Found %d (transcript, wav) pairs", len(pairs))

    asr_model = None
    if not args.no_align:
        from crisper_pipeline import asr

        asr_model = asr.load_model(args.asr_model)

    per_file: dict[str, dict] = {}
    totals = defaultdict(float)
    align_stats = defaultdict(int)
    failures = 0
    for transcript_path, wav_path in pairs:
        uri = transcript_path.stem
        try:
            info = sf.info(str(wav_path))
            duration = info.frames / info.samplerate
            turns, timing = load_turns(transcript_path, duration=duration)
            if asr_model is not None and timing == "turn":
                try:
                    aligned = align_file(asr_model, wav_path, turns, args.language)
                    turns, stats = refine_turns(turns, aligned, args.max_drift)
                    for key, value in stats.items():
                        align_stats[key] += value
                except Exception as exc:
                    align_stats["fallback_file"] += 1
                    logger.warning("%s: alignment failed (%s); using human turn bounds", uri, exc)
            segments = turns_to_segments(turns, args.max_gap)
            if not segments:
                raise ValueError("no usable segments")
            per_file[uri] = {
                "wav": wav_path,
                "segments": segments,
                "duration": duration,
                "participant": extract_participant(uri, args.participant_regex),
            }
            totals["hours"] += duration / 3600
            totals["speech_hours"] += sum(s["end"] - s["start"] for s in segments) / 3600
        except Exception:
            failures += 1
            logger.exception("Failed to process %s", uri)

    if not per_file:
        logger.error("No files processed successfully")
        return 1

    assignment = split_participants(
        [f["participant"] for f in per_file.values()], tuple(args.splits), args.seed
    )

    counts = defaultdict(int)
    handles = {
        subset: {
            "lst": (output_dir / "lists" / f"{subset}.lst").open("w"),
            "rttm": (output_dir / "rttm" / f"{subset}.rttm").open("w"),
            "uem": (output_dir / "uem" / f"{subset}.uem").open("w"),
        }
        for subset in SUBSETS
    }
    try:
        for uri in sorted(per_file):
            entry = per_file[uri]
            subset = assignment[entry["participant"]]
            counts[subset] += 1
            handles[subset]["lst"].write(uri + "\n")
            handles[subset]["rttm"].write("\n".join(rttm_lines(uri, entry["segments"])) + "\n")
            handles[subset]["uem"].write(f"{uri} 1 0.000 {entry['duration']:.3f}\n")
            link = output_dir / "wav" / f"{uri}.wav"
            if not link.exists():
                link.symlink_to(entry["wav"].resolve())
    finally:
        for subset_handles in handles.values():
            for handle in subset_handles.values():
                handle.close()

    write_database_yml(output_dir)

    speakers_per_file = [
        len({s["speaker"] for s in f["segments"]}) for f in per_file.values()
    ]
    logger.info("Files: %d ok, %d failed", len(per_file), failures)
    logger.info(
        "Split by participant: %s",
        {s: counts[s] for s in SUBSETS},
    )
    logger.info(
        "Audio: %.1f h total, %.1f h labeled speech", totals["hours"], totals["speech_hours"]
    )
    logger.info(
        "Speakers per file: min %d, max %d",
        min(speakers_per_file), max(speakers_per_file),
    )
    if align_stats:
        logger.info("Alignment: %s", dict(align_stats))
    logger.info("Protocol: AMPSCZ.SpeakerDiarization.Interviews (%s)", output_dir / "database.yml")
    return 0


if __name__ == "__main__":
    sys.exit(main())

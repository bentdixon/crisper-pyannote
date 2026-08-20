"""Run our own pipelines across the staged cohort, loading models once.

Two systems, both writing one transcript.json per visit into a mirrored tree
so scripts/evaluate_systems.py can pick them up:

    ours          CrisperWhisper 2.0 verbatim ASR + pyannote community-1
                  diarization, merged per word (the transcribe-session route)
    verbatimize   Chirp-3 transcript upgraded to verbatim with
                  CrisperWhisper 2.0's verbatimize task

Unlike the transcribe-session CLI this does not create a timestamped run
directory per file -- for an evaluation sweep a stable path per visit matters
more than keeping repeat runs side by side. Existing outputs are skipped, so
an interrupted sweep resumes.

Usage:
    uv run python scripts/run_cohort.py --cohort /path/to/cohort \
        --mode ours --output-dir outputs/ours --device-index 0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from crisper_pipeline import asr, chirp, diarization, merge, verbatimize

logger = logging.getLogger("run_cohort")


def find_visits(cohort: Path, need_chirp: bool) -> list[Path]:
    visits = []
    for path in sorted(cohort.glob("Pronet*/*/*")):
        if not path.is_dir() or not any((path / "audio").glob("*.wav")):
            continue
        if need_chirp and not any((path / "chirp").glob("*.json")):
            continue
        visits.append(path)
    return visits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--mode", required=True, choices=("ours", "verbatimize"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--shard", default=None, metavar="I/N",
        help=(
            "process only shard I of N (1-based), so several workers can run "
            "the same sweep on different GPUs without overlapping"
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default="large")
    parser.add_argument("--draft-model", default="turbo")
    parser.add_argument("--no-speculative", action="store_true")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--language", default="en")
    parser.add_argument("--num-speakers", type=int, default=2)
    parser.add_argument("--fill-nearest", action="store_true")
    parser.add_argument(
        "--longform", default="windowed",
        choices=("windowed", "diarization", "continuation"),
        help=(
            "how audio over 30 s is split for the ASR. windowed: silero speech "
            "windows, then assign speakers by overlap (default). diarization: "
            "diarize first with community-1 and transcribe each segment, the "
            "other team's order, which takes the speaker from the segment. "
            "continuation: the model's own longform strategy, which drops most "
            "of the transcript on this corpus"
        ),
    )
    parser.add_argument("--hf-token", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    cohort = Path(args.cohort)
    output_root = Path(args.output_dir)

    visits = find_visits(cohort, need_chirp=(args.mode == "verbatimize"))
    if args.shard:
        index_str, _, count_str = args.shard.partition("/")
        shard_index, shard_count = int(index_str), int(count_str)
        visits = visits[shard_index - 1 :: shard_count]
        logger.info("Shard %d/%d: %d visit(s)", shard_index, shard_count, len(visits))
    if args.limit:
        visits = visits[: args.limit]
    pending = [v for v in visits if not (output_root / v.relative_to(cohort) / "transcript.json").exists()]
    logger.info(
        "%s: %d visit(s), %d already done, %d to run",
        args.mode, len(visits), len(visits) - len(pending), len(pending),
    )
    if not pending:
        return 0

    asr_model = asr.load_model(
        args.model,
        draft_model=None if args.no_speculative else args.draft_model,
        compute_type=args.compute_type,
        device_index=args.device_index,
    )
    dia_pipeline = None
    if args.mode == "ours":
        dia_pipeline = diarization.load_pipeline(token=args.hf_token)

    failures = 0
    for index, visit in enumerate(pending, start=1):
        relative = visit.relative_to(cohort)
        audio = sorted((visit / "audio").glob("*.wav"))[0]
        destination = output_root / relative
        logger.info("[%d/%d] %s", index, len(pending), relative)
        started = time.perf_counter()

        try:
            if args.mode == "ours":
                if args.longform == "diarization":
                    # Diarize first and transcribe segment by segment, the
                    # other team's order with our community-1 diarizer. Every
                    # word arrives attributed, so assign_speakers is bypassed
                    # and the UNKNOWN bucket cannot occur.
                    segments = diarization.diarize(
                        dia_pipeline, audio, num_speakers=args.num_speakers,
                        exclusive=True,
                    )
                    transcript = asr.transcribe(
                        asr_model, audio, language=args.language,
                        speculative_decoding=not args.no_speculative,
                        longform="diarization", segments=segments,
                    )
                    words = transcript["words"]
                else:
                    transcript = asr.transcribe(
                        asr_model, audio, language=args.language,
                        speculative_decoding=not args.no_speculative,
                        longform=args.longform,
                    )
                    segments = diarization.diarize(
                        dia_pipeline, audio, num_speakers=args.num_speakers,
                        exclusive=True,
                    )
                    words = merge.assign_speakers(
                        segments, transcript["words"], fill_nearest=args.fill_nearest
                    )
                payload = {
                    "audio": audio.name,
                    "duration": transcript["duration"],
                    "text": transcript["text"],
                    "words": words,
                }
            else:
                source = chirp.load_transcript(sorted((visit / "chirp").glob("*.json"))[0])
                if not source["words"]:
                    raise ValueError("empty Chirp transcript")
                result = verbatimize.verbatimize_session(
                    asr_model, audio, source["words"], source["duration"] or 0.0,
                    language=args.language,
                )
                payload = {
                    "audio": audio.name,
                    "duration": result["duration"],
                    "stats": result["stats"],
                    "text": result["text"],
                    "words": result["words"],
                }
        except Exception:
            failures += 1
            logger.exception("Failed on %s", relative)
            continue

        destination.mkdir(parents=True, exist_ok=True)
        (destination / "transcript.json").write_text(json.dumps(payload, indent=2) + "\n")
        logger.info(
            "  %d words in %.0fs", len(payload["words"]), time.perf_counter() - started
        )

    logger.info("Done: %d succeeded, %d failed", len(pending) - failures, failures)
    return 0


if __name__ == "__main__":
    sys.exit(main())

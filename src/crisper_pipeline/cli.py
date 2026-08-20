"""Command-line entry point: wav files in, transcripts out.

Usage:
    transcribe-session session1.wav [session2.wav ...] --output-dir outputs
    transcribe-session /path/to/wav/dir --output-dir outputs
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from . import __version__, asr, diarization, merge, outputs

logger = logging.getLogger("crisper_pipeline")


def collect_wav_files(inputs: list[str]) -> list[Path]:
    """Expand file and directory arguments into a sorted list of wav files."""
    files: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            files.extend(sorted(path.glob("*.wav")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"No such file or directory: {path}")
    if not files:
        raise FileNotFoundError(f"No wav files found in: {', '.join(inputs)}")
    return files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcribe-session",
        description=(
            "Verbatim, word-level, speaker-attributed transcription "
            "(CrisperWhisper 2.0 + pyannote community-1)."
        ),
    )
    parser.add_argument("inputs", nargs="+", help="wav file(s) or directory of wav files")
    parser.add_argument("--output-dir", default="outputs", help="output directory (default: outputs)")
    parser.add_argument("--language", default="en", help="language code (default: en)")

    asr_group = parser.add_argument_group("ASR")
    asr_group.add_argument("--model", default="large", help="CrisperWhisper model (default: large)")
    asr_group.add_argument(
        "--draft-model", default="turbo",
        help="draft model for speculative decoding (default: turbo)",
    )
    asr_group.add_argument(
        "--no-speculative", action="store_true",
        help="disable speculative decoding",
    )
    asr_group.add_argument(
        "--compute-type", default="float16", choices=["float16", "int8_float16"],
        help="CTranslate2 compute type (default: float16)",
    )
    asr_group.add_argument("--device", default="auto", help="ASR device (default: auto)")
    asr_group.add_argument("--device-index", type=int, default=0, help="GPU index (default: 0)")

    dia_group = parser.add_argument_group("diarization")
    dia_group.add_argument("--num-speakers", type=int, default=None, help="exact number of speakers, if known")
    dia_group.add_argument("--min-speakers", type=int, default=None)
    dia_group.add_argument("--max-speakers", type=int, default=None)
    dia_group.add_argument(
        "--diarization-model", default=None,
        help=(
            "diarization pipeline: HuggingFace id or local config.yaml path, "
            "e.g. a fine-tuned pipeline from finetune/optimize_pipeline.py "
            "(default: pyannote/speaker-diarization-community-1)"
        ),
    )
    dia_group.add_argument(
        "--hf-token", default=None,
        help="HuggingFace token (default: HF_TOKEN env var or cached login)",
    )

    merge_group = parser.add_argument_group("merge")
    merge_group.add_argument(
        "--diarization-mode", default="exclusive",
        choices=["exclusive", "overlap"],
        help=(
            "exclusive (default): non-overlapping diarization, speaker "
            "assigned per word by maximum overlap; overlap: raw diarization "
            "with overlapping segments, each segment claims the contiguous "
            "word chunk that best fits it by temporal IoU"
        ),
    )
    merge_group.add_argument(
        "--longform", default="windowed",
        choices=["windowed", "diarization", "continuation"],
        help=(
            "how audio over 30 s is split for the ASR. windowed (default): "
            "silero speech windows, speakers assigned per word afterwards. "
            "diarization: diarize first and transcribe each segment, taking "
            "the speaker from the segment, so --diarization-mode and "
            "--fill-nearest do not apply. continuation: the model's own "
            "longform strategy, which drops most of the transcript here"
        ),
    )
    merge_group.add_argument(
        "--fill-nearest", action="store_true",
        help=(
            "exclusive mode only: assign the nearest speaker to words with "
            "no diarization overlap instead of labeling them UNKNOWN"
        ),
    )
    return parser


def build_metadata(audio_path: Path, args: argparse.Namespace, run_time: datetime) -> dict:
    """Collect run settings for the metadata.json written alongside outputs."""
    return {
        "audio": str(audio_path.resolve()),
        "run_timestamp": run_time.isoformat(timespec="seconds"),
        "run_timestamp_compact": run_time.strftime("%Y%m%d-%H%M%S"),
        "versions": {
            "crisper-whisper-2": __version__,
            "crisperwhisper": version("crisperwhisper"),
            "pyannote-audio": version("pyannote.audio"),
        },
        "transcription": {
            "model": args.model,
            "backend": "ct2",
            "draft_model": args.draft_model if not args.no_speculative else None,
            "speculative_decoding": not args.no_speculative,
            "compute_type": args.compute_type,
            "language": args.language,
            "mode": "verbatim",
            "word_timestamps": True,
            "device": args.device,
            "device_index": args.device_index,
        },
        "diarization": {
            "model": args.diarization_model or diarization.DIARIZATION_MODEL,
            "mode": args.diarization_mode,
            "exclusive": args.diarization_mode == "exclusive",
            "num_speakers": args.num_speakers,
            "min_speakers": args.min_speakers,
            "max_speakers": args.max_speakers,
        },
        "merge": {
            "algorithm": (
                "best-fit-segment-iou" if args.diarization_mode == "overlap"
                else "max-overlap-per-word"
            ),
            "fill_nearest": args.fill_nearest,
        },
    }


def process_file(
    audio_path: Path,
    args: argparse.Namespace,
    asr_model,
    dia_pipeline,
) -> Path:
    """Run ASR, diarization, merge, and output writing for one wav file."""
    metadata = build_metadata(audio_path, args, datetime.now())
    longform = getattr(args, "longform", "windowed")

    if longform == "diarization":
        # Diarize first, then transcribe each segment: every word arrives with
        # the speaker of the segment it came from, so no merge step runs and
        # no word can land in the UNKNOWN bucket. Needs non-overlapping
        # segments, or shared audio would be transcribed twice.
        segments = diarization.diarize(
            dia_pipeline,
            audio_path,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            exclusive=True,
        )
        transcript = asr.transcribe(
            asr_model,
            audio_path,
            language=args.language,
            speculative_decoding=not args.no_speculative,
            longform="diarization",
            segments=segments,
        )
        words = transcript["words"]
        turns = merge.group_into_turns(words)
        return outputs.write_outputs(
            args.output_dir, audio_path, transcript, segments, turns, metadata
        )

    transcript = asr.transcribe(
        asr_model,
        audio_path,
        language=args.language,
        speculative_decoding=not args.no_speculative,
        longform=longform,
    )
    segments = diarization.diarize(
        dia_pipeline,
        audio_path,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        exclusive=args.diarization_mode == "exclusive",
    )
    if args.diarization_mode == "overlap":
        words, turns = merge.best_fit_segments(segments, transcript["words"])
    else:
        words = merge.assign_speakers(
            segments, transcript["words"], fill_nearest=args.fill_nearest
        )
        turns = merge.group_into_turns(words)
    return outputs.write_outputs(
        args.output_dir, audio_path, transcript, segments, turns, metadata
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = build_parser().parse_args(argv)

    try:
        wav_files = collect_wav_files(args.inputs)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Processing %d file(s)", len(wav_files))

    asr_model = asr.load_model(
        args.model,
        draft_model=args.draft_model if not args.no_speculative else None,
        compute_type=args.compute_type,
        device=args.device,
        device_index=args.device_index,
    )
    dia_pipeline = diarization.load_pipeline(
        token=args.hf_token, model=args.diarization_model
    )

    failures = 0
    for audio_path in wav_files:
        try:
            session_dir = process_file(audio_path, args, asr_model, dia_pipeline)
            logger.info("Done: %s -> %s", audio_path.name, session_dir)
        except Exception:
            failures += 1
            logger.exception("Failed to process %s", audio_path)

    if failures:
        logger.error("%d of %d file(s) failed", failures, len(wav_files))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

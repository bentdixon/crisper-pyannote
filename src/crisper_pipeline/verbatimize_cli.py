"""Command-line entry point: Chirp-3 transcripts in, verbatim transcripts out.

Usage:
    verbatimize-session session.wav --chirp-dir chirp/ --output-dir outputs
    verbatimize-session /path/to/wavs/ --chirp-dir chirp/ --output-dir outputs
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from . import __version__, asr, chirp, merge, outputs, verbatimize

logger = logging.getLogger("crisper_pipeline")

# Tokens that appear in one filename of a pair but not the other. The wav is
# named ..._interviewAudioTranscript_psychs_day0085_session002.wav while its
# Chirp transcript is ..._psychs_day0085_session002_final.json.
NOISE_TOKENS = ("_interviewAudioTranscript", "_final_humanReadable", "_final")


def normalize_stem(stem: str) -> str:
    """Reduce a filename stem to the session key shared by audio and Chirp."""
    for token in NOISE_TOKENS:
        stem = stem.replace(token, "")
    return re.sub(r"[^a-z0-9]", "", stem.lower())


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


def index_chirp_transcripts(chirp_dir: Path) -> dict[str, Path]:
    """Map session key -> Chirp transcript path, skipping ambiguous keys."""
    index: dict[str, Path] = {}
    ambiguous: set[str] = set()
    for path in sorted(chirp_dir.rglob("*.json")):
        key = normalize_stem(path.stem)
        if key in index:
            ambiguous.add(key)
            continue
        index[key] = path
    for key in ambiguous:
        logger.warning("Ambiguous Chirp transcripts for key %s; skipping", key)
        index.pop(key, None)
    return index


def build_metadata(
    audio_path: Path, chirp_path: Path, args: argparse.Namespace, run_time: datetime
) -> dict:
    """Collect run settings for the metadata.json written alongside outputs."""
    return {
        "audio": str(audio_path.resolve()),
        "chirp_transcript": str(chirp_path.resolve()),
        "run_timestamp": run_time.isoformat(timespec="seconds"),
        "run_timestamp_compact": run_time.strftime("%Y%m%d-%H%M%S"),
        "versions": {
            "crisper-whisper-2": __version__,
            "crisperwhisper": version("crisperwhisper"),
        },
        "verbatimize": {
            "model": args.model,
            "backend": "ct2",
            "task": "verbatimize",
            "language": args.language,
            "compute_type": args.compute_type,
            "device": args.device,
            "device_index": args.device_index,
            "max_window_seconds": args.max_window,
            "max_new_tokens": args.max_new_tokens,
            "word_timestamps": True,
            "realign": args.realign,
            "timestamp_source": (
                "whole-session forced alignment" if args.realign
                else "per-window verbatimize cross-attention"
            ),
        },
        "speakers": {
            "source": "chirp-3",
            "carried_by": "difflib word alignment onto the verbatimized text",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verbatimize-session",
        description=(
            "Upgrade Chirp-3 transcripts to verbatim with CrisperWhisper 2.0, "
            "recovering disfluencies and word-level timestamps from the audio."
        ),
    )
    parser.add_argument("inputs", nargs="+", help="wav file(s) or directory of wav files")
    parser.add_argument(
        "--chirp-dir", required=True,
        help="directory of Chirp-3 word-level JSON transcripts (searched recursively)",
    )
    parser.add_argument("--output-dir", default="outputs", help="output directory (default: outputs)")
    parser.add_argument("--language", default="en", help="language code (default: en)")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model", default="large", help="CrisperWhisper model (default: large)")
    model_group.add_argument(
        "--compute-type", default="float16",
        help="CTranslate2 compute type (default: float16)",
    )
    model_group.add_argument("--device", default="auto", help="device (default: auto)")
    model_group.add_argument("--device-index", type=int, default=0, help="GPU index (default: 0)")

    window_group = parser.add_argument_group("windowing")
    window_group.add_argument(
        "--max-window", type=float, default=verbatimize.MAX_WINDOW_SECONDS,
        help=(
            "maximum window length in seconds; verbatimize has no longform "
            f"strategy and degrades past 30 s (default: {verbatimize.MAX_WINDOW_SECONDS})"
        ),
    )
    window_group.add_argument(
        "--max-new-tokens", type=int, default=448,
        help="decoder token budget per window (default: 448)",
    )
    parser.add_argument(
        "--realign", action="store_true",
        help=(
            "after verbatimizing, re-time every word with a whole-session "
            "forced alignment (sharper than the prompted per-window "
            "cross-attention, and free of window seams; roughly doubles runtime)"
        ),
    )
    return parser


def process_file(
    audio_path: Path, chirp_path: Path, args: argparse.Namespace, model
) -> Path:
    """Verbatimize one session and write its outputs."""
    run_time = datetime.now()
    transcript = chirp.load_transcript(chirp_path)
    if not transcript["words"]:
        raise ValueError(f"Chirp transcript has no words: {chirp_path}")

    result = verbatimize.verbatimize_session(
        model,
        audio_path,
        transcript["words"],
        transcript["duration"] or 0.0,
        language=args.language,
        max_window=args.max_window,
        max_new_tokens=args.max_new_tokens,
        realign=args.realign,
    )
    turns = merge.group_into_turns(result["words"])

    metadata = build_metadata(audio_path, chirp_path, args, run_time)
    metadata["chirp"] = transcript["metadata"]
    metadata["stats"] = result["stats"]
    return outputs.write_verbatimize_outputs(
        args.output_dir, audio_path, result, turns, metadata
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    args = build_parser().parse_args(argv)

    wav_files = collect_wav_files(args.inputs)
    chirp_index = index_chirp_transcripts(Path(args.chirp_dir))
    logger.info(
        "%d wav file(s), %d Chirp transcript(s) indexed", len(wav_files), len(chirp_index)
    )

    pairs: list[tuple[Path, Path]] = []
    for wav in wav_files:
        match = chirp_index.get(normalize_stem(wav.stem))
        if match is None:
            logger.warning("No Chirp transcript for %s; skipping", wav.name)
            continue
        pairs.append((wav, match))
    if not pairs:
        logger.error("No audio/Chirp pairs found")
        return 1
    logger.info("Verbatimizing %d session(s)", len(pairs))

    model = asr.load_model(
        args.model,
        draft_model=None,
        compute_type=args.compute_type,
        device=args.device,
        device_index=args.device_index,
    )

    failures = 0
    for index, (wav, chirp_path) in enumerate(pairs, start=1):
        logger.info("[%d/%d] %s", index, len(pairs), wav.name)
        try:
            session_dir = process_file(wav, chirp_path, args, model)
            logger.info("Wrote %s", session_dir)
        except Exception:
            failures += 1
            logger.exception("Failed on %s", wav.name)

    logger.info("Done: %d succeeded, %d failed", len(pairs) - failures, failures)
    return 1 if failures == len(pairs) else 0


if __name__ == "__main__":
    sys.exit(main())

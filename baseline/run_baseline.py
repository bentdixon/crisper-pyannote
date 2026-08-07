"""Runner: apply the ported pipeline to the staged cohort.

Walks a cohort tree laid out by scripts/fetch_cohort.py --layout session
(<SITE>/<SUBJECT>/<dayNNNN_sessionNNN>/audio/*.wav), loads CrisperWhisper 2.0
once, and calls process_interview() per file. Outputs land in a mirrored tree
under --output-dir so each result sits next to the visit it came from and can
be scored against that visit's Chirp-3 and human transcripts.

Usage:
    uv run python baseline/run_baseline.py \
        --cohort /data/data/wolfflab/Data/Chirp3_PSYCHS_NDA4 \
        --output-dir baseline/outputs --limit 1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import transcription_core as core  # noqa: E402

logger = logging.getLogger("run_baseline")


def find_visits(cohort: Path) -> list[Path]:
    """Visit directories that actually contain an audio file."""
    return sorted(
        path for path in cohort.glob("Pronet*/*/*")
        if path.is_dir() and any((path / "audio").glob("*.wav"))
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, help="staged cohort root (session layout)")
    parser.add_argument("--output-dir", default="baseline/outputs")
    parser.add_argument("--work-dir", default="/tmp/transcription_core_work")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N visits")
    parser.add_argument(
        "--redo", action="store_true",
        help="re-process visits that already have outputs (default: resume)",
    )
    parser.add_argument("--model", default="large", help="CrisperWhisper 2.0 model (default: large)")
    parser.add_argument("--draft-model", default="turbo")
    parser.add_argument(
        "--speculative", action="store_true",
        help=(
            "enable speculative decoding. Off by default here: it makes word "
            "timings less accurate (accepted draft tokens use the draft "
            "model's cross-attention), and this pipeline exists to be scored "
            "on timestamp accuracy against Chirp-3"
        ),
    )
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--language", default="en")
    parser.add_argument("--num-speakers", type=int, default=core.NUM_SPEAKERS)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument(
        "--llm-review", action="store_true",
        help="also run the optional Qwen review (downloads a 7B model)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    cohort = Path(args.cohort)
    visits = find_visits(cohort)
    if not args.redo:
        # Resume: a visit with a transcript already written is done. Lets an
        # interrupted or partially-failed sweep be re-run to pick up the rest.
        visits = [
            v for v in visits
            if not any((Path(args.output_dir) / v.relative_to(cohort)).glob("*_transcript.txt"))
        ]
    if args.limit:
        visits = visits[: args.limit]
    if not visits:
        logger.error("No visits with audio found under %s", cohort)
        return 1
    logger.info("Processing %d visit(s)", len(visits))

    core.NUM_SPEAKERS = args.num_speakers
    asr_model = core.load_crisperwhisper(
        args.model,
        draft_model=args.draft_model if args.speculative else None,
        compute_type=args.compute_type,
        device_index=args.device_index,
    )

    llm_model = llm_tokenizer = None
    if args.llm_review:
        from llm_review import load_llm

        logger.info("Loading review LLM")
        llm_model, llm_tokenizer = load_llm()

    output_root = Path(args.output_dir)
    summary = []
    failures = 0

    for index, visit in enumerate(visits, start=1):
        audio = sorted((visit / "audio").glob("*.wav"))[0]
        relative = visit.relative_to(cohort)
        destination = output_root / relative
        logger.info("[%d/%d] %s", index, len(visits), relative)

        started = time.perf_counter()
        try:
            result = core.process_interview(
                str(audio),
                str(destination),
                asr_model,
                work_dir=args.work_dir,
                file_prefix=audio.stem,
                llm_model=llm_model,
                llm_tokenizer=llm_tokenizer,
                language=args.language,
                token=args.hf_token,
                speculative_decoding=args.speculative,
            )
        except Exception:
            failures += 1
            logger.exception("Failed on %s", relative)
            continue

        # Hundreds of separate short transcribe calls per interview fragment
        # the CUDA allocator; without this a long sweep OOMs partway through.
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

        result["visit"] = str(relative)
        result["audio"] = audio.name
        result["seconds"] = round(time.perf_counter() - started, 1)
        summary.append(result)
        logger.info(
            "  %s | %d words, %d turns, %d segments in %.0fs",
            result["diarization_method"], result["num_words"],
            result["num_turns"], result["num_segments"], result["seconds"],
        )

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info(
        "Done: %d succeeded, %d failed. Wrote %s",
        len(summary), failures, output_root / "summary.json",
    )
    return 1 if failures and not summary else 0


if __name__ == "__main__":
    sys.exit(main())

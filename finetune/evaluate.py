"""Evaluate a diarization pipeline against the prepared protocol.

Computes DER at collar 0.0 and 0.25, each with and without overlap regions,
over a protocol subset. Optionally also measures word-level speaker
attribution accuracy of the full ASR + merge pipeline against the reference.

Usage:
    uv run python finetune/evaluate.py --data finetune/data --subset development
    uv run python finetune/evaluate.py --data finetune/data --subset test \
        --pipeline finetune/finetuned-config.yaml --word-attribution
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from crisper_pipeline.cuda_preload import preload_torchcodec_libs

preload_torchcodec_libs()  # must precede pyannote imports

from pyannote.database import FileFinder, registry
from pyannote.metrics.diarization import DiarizationErrorRate

from crisper_pipeline import diarization

logger = logging.getLogger("evaluate")

DER_VARIANTS = {
    "der": {"collar": 0.0, "skip_overlap": False},
    "der_collar0.25": {"collar": 0.25, "skip_overlap": False},
    "der_skipoverlap": {"collar": 0.0, "skip_overlap": True},
    "der_collar0.25_skipoverlap": {"collar": 0.25, "skip_overlap": True},
}


def load_protocol(data_dir: Path):
    registry.load_database(str(data_dir / "database.yml"))
    return registry.get_protocol(
        "AMPSCZ.SpeakerDiarization.Interviews",
        preprocessors={"audio": FileFinder()},
    )


def attribution_accuracy(words: list[dict], reference_segments: list[dict]) -> tuple[int, int]:
    """Word-level speaker attribution accuracy under the best label mapping.

    For each predicted word, the reference speaker is the segment containing
    the word midpoint. Predicted speaker labels are matched to reference
    labels with the one-to-one mapping that maximizes agreement (exhaustive
    over permutations; speaker counts here are small).

    Returns (correct, total) over words whose midpoint falls inside a
    reference segment.
    """
    scored = []
    for word in words:
        midpoint = (word["start"] + word["end"]) / 2
        gold = next(
            (
                s["speaker"]
                for s in reference_segments
                if s["start"] <= midpoint <= s["end"]
            ),
            None,
        )
        if gold is not None:
            scored.append((word["speaker"], gold))
    if not scored:
        return 0, 0

    confusion = Counter(scored)
    predicted_labels = sorted({p for p, _ in scored})
    gold_labels = sorted({g for _, g in scored})
    if len(predicted_labels) > 6:
        logger.warning("Too many predicted speakers for exhaustive mapping")
        predicted_labels = predicted_labels[:6]
    best = 0
    for perm in itertools.permutations(gold_labels, min(len(predicted_labels), len(gold_labels))):
        mapping = dict(zip(predicted_labels, perm))
        best = max(
            best,
            sum(n for (p, g), n in confusion.items() if mapping.get(p) == g),
        )
    return best, len(scored)


def annotation_to_segments(annotation) -> list[dict]:
    return [
        {"start": float(turn.start), "end": float(turn.end), "speaker": str(label)}
        for turn, _track, label in annotation.itertracks(yield_label=True)
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="finetune/data", help="prepare_data.py output directory")
    parser.add_argument("--subset", default="development", choices=["train", "development", "test"])
    parser.add_argument(
        "--pipeline", default=None,
        help="pipeline config.yaml or HF id (default: stock community-1)",
    )
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument(
        "--word-attribution", action="store_true",
        help="also run ASR + merge and score word-level speaker attribution (slow)",
    )
    parser.add_argument("--asr-model", default="large")
    parser.add_argument("--output", default=None, help="write per-file and aggregate results JSON here")
    parser.add_argument("--hf-token", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    protocol = load_protocol(Path(args.data))
    files = list(getattr(protocol, args.subset)())
    logger.info("Evaluating %d file(s) in %s", len(files), args.subset)

    pipeline = diarization.load_pipeline(token=args.hf_token, model=args.pipeline)

    asr_model = None
    if args.word_attribution:
        from crisper_pipeline import asr

        asr_model = asr.load_model(args.asr_model)

    metrics = {name: DiarizationErrorRate(**kwargs) for name, kwargs in DER_VARIANTS.items()}
    per_file = {}
    attribution_totals = [0, 0]
    kwargs = {"num_speakers": args.num_speakers} if args.num_speakers else {}

    for file in files:
        uri = file["uri"]
        audio_path = file["audio"]
        output = pipeline(diarization.load_audio(audio_path), **kwargs)
        hypothesis = output.speaker_diarization
        reference = file["annotation"]
        uem = file["annotated"]

        per_file[uri] = {}
        for name, metric in metrics.items():
            per_file[uri][name] = float(metric(reference, hypothesis, uem=uem))

        if asr_model is not None:
            from crisper_pipeline import merge

            transcript = asr_model.transcribe(
                str(audio_path), language="en", mode="verbatim", word_timestamps=True
            )
            words = [
                {"word": w.word, "start": float(w.start), "end": float(w.end)}
                for w in (transcript.words or [])
            ]
            exclusive_segments = annotation_to_segments(output.exclusive_speaker_diarization)
            merge.assign_speakers(exclusive_segments, words)
            correct, total = attribution_accuracy(
                words, annotation_to_segments(reference)
            )
            attribution_totals[0] += correct
            attribution_totals[1] += total
            per_file[uri]["word_attribution"] = correct / total if total else None

        logger.info("%s: %s", uri, {k: round(v, 4) for k, v in per_file[uri].items() if v is not None})

    aggregate = {name: float(abs(metric)) for name, metric in metrics.items()}
    if asr_model is not None and attribution_totals[1]:
        aggregate["word_attribution"] = attribution_totals[0] / attribution_totals[1]

    logger.info("Aggregate over %s: %s", args.subset, {k: round(v, 4) for k, v in aggregate.items()})

    if args.output:
        Path(args.output).write_text(
            json.dumps(
                {
                    "pipeline": args.pipeline or diarization.DIARIZATION_MODEL,
                    "subset": args.subset,
                    "aggregate": aggregate,
                    "per_file": per_file,
                },
                indent=2,
            )
            + "\n"
        )
        logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

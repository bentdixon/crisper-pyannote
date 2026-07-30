"""Re-optimize pipeline hyperparameters around a fine-tuned segmentation model.

Builds a SpeakerDiarization pipeline with the fine-tuned segmentation
checkpoint and the stock community-1 embedding/PLDA, then tunes
segmentation.min_duration_off and the VBx clustering parameters
(threshold, Fa, Fb) against DER on the development subset using
pyannote.pipeline's Optuna-backed Optimizer. Writes a ready-to-use pipeline
config to --output, loadable via `transcribe-session --diarization-model`.

Usage:
    uv run python finetune/optimize_pipeline.py \
        --checkpoint finetune/checkpoints/best.ckpt --iterations 100
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from crisper_pipeline.cuda_preload import preload_torchcodec_libs

preload_torchcodec_libs()  # must precede pyannote imports

import torch
import yaml
from pyannote.audio import __version__ as pyannote_audio_version
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.database import FileFinder, registry
from pyannote.pipeline import Optimizer

from crisper_pipeline.diarization import DIARIZATION_MODEL, load_audio

logger = logging.getLogger("optimize_pipeline")


def build_pipeline(checkpoint: str, token: str | None) -> SpeakerDiarization:
    # legacy=True so the pipeline returns a plain Annotation, which is what
    # the Optimizer's metric expects (v4 otherwise returns DiarizeOutput).
    pipeline = SpeakerDiarization(
        legacy=True,
        segmentation=checkpoint,
        embedding={"checkpoint": DIARIZATION_MODEL, "subfolder": "embedding"},
        plda={"checkpoint": DIARIZATION_MODEL, "subfolder": "plda"},
        clustering="VBxClustering",
        token=token,
    )
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    return pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="finetune/data")
    parser.add_argument("--checkpoint", default="finetune/checkpoints/best.ckpt")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", default="finetune/finetuned-config.yaml")
    parser.add_argument(
        "--journal", default="finetune/optimize.journal",
        help="Optuna journal storage (resumes across runs)",
    )
    parser.add_argument("--hf-token", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    checkpoint = str(Path(args.checkpoint).resolve())

    registry.load_database(str(Path(args.data) / "database.yml"))
    protocol = registry.get_protocol(
        "AMPSCZ.SpeakerDiarization.Interviews",
        preprocessors={"audio": FileFinder()},
    )

    # Preload waveforms so the pipeline never touches pyannote's torchcodec
    # decoding path (see CLAUDE.md).
    inputs = []
    for file in protocol.development():
        audio = load_audio(file["audio"])
        file["waveform"] = audio["waveform"]
        file["sample_rate"] = audio["sample_rate"]
        inputs.append(file)
    logger.info("Optimizing on %d development file(s)", len(inputs))

    pipeline = build_pipeline(checkpoint, args.hf_token)
    defaults = pipeline.default_parameters()
    pipeline.instantiate(defaults)

    optimizer = Optimizer(pipeline, db=Path(args.journal))
    optimizer.tune(inputs, n_iterations=args.iterations, warm_start=defaults)
    best_params = optimizer.best_params
    logger.info("Best DER on development: %.4f", optimizer.best_loss)
    logger.info("Best parameters: %s", best_params)

    # legacy is intentionally absent: production code consumes DiarizeOutput.
    config = {
        "pipeline": {
            "name": "pyannote.audio.pipelines.SpeakerDiarization",
            "params": {
                "segmentation": checkpoint,
                "embedding": {"checkpoint": DIARIZATION_MODEL, "subfolder": "embedding"},
                "plda": {"checkpoint": DIARIZATION_MODEL, "subfolder": "plda"},
                "clustering": "VBxClustering",
            },
        },
        "params": best_params,
        "dependencies": {"pyannote.audio": pyannote_audio_version},
    }
    output = Path(args.output)
    output.write_text(yaml.safe_dump(config, sort_keys=False))
    logger.info("Wrote %s (use with: transcribe-session --diarization-model %s)", output, output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

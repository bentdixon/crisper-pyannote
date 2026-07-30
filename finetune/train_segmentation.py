"""Fine-tune the community-1 segmentation model on the prepared protocol.

Loads the pretrained segmentation model from the community-1 pipeline repo,
attaches a powerset SpeakerDiarization task with the same geometry (so the
classifier head is reused, not rebuilt), trains with a two-stage schedule
(WavLM trunk frozen first, then unfrozen), and links the best checkpoint to
<checkpoint-dir>/best.ckpt.

Usage:
    uv run python finetune/train_segmentation.py --data finetune/data
"""

from __future__ import annotations

import argparse
import logging
import sys
import types
from pathlib import Path

from crisper_pipeline.cuda_preload import preload_torchcodec_libs

preload_torchcodec_libs()  # must precede pyannote imports

import torch
from lightning import Callback, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pyannote.audio import Model
from pyannote.audio.tasks import SpeakerDiarization
from pyannote.database import FileFinder, registry

logger = logging.getLogger("train_segmentation")

PIPELINE_REPO = "pyannote/speaker-diarization-community-1"
# Feature-extraction trunk attribute by architecture: PyanNet uses sincnet,
# SSeRiouSS uses wav2vec (WavLM). Community-1 ships a PyanNet.
TRUNK_CANDIDATES = ("sincnet", "wav2vec")


def find_trunk(model) -> str | None:
    for name in TRUNK_CANDIDATES:
        if hasattr(model, name):
            return name
    return None


class UnfreezeAfter(Callback):
    """Unfreeze the trunk once the warmup epochs are done."""

    def __init__(self, module_name: str, after_epochs: int):
        self.module_name = module_name
        self.after_epochs = after_epochs

    def on_train_epoch_start(self, trainer, model):
        if trainer.current_epoch == self.after_epochs:
            model.unfreeze_by_name(self.module_name)
            logger.info("Epoch %d: unfroze %s", trainer.current_epoch, self.module_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="finetune/data", help="prepare_data.py output directory")
    parser.add_argument("--checkpoint-dir", default="finetune/checkpoints")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument(
        "--freeze-epochs", type=int, default=2,
        help="epochs to train with the WavLM trunk frozen before unfreezing (default: 2)",
    )
    parser.add_argument("--patience", type=int, default=10, help="early stopping patience")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    registry.load_database(str(Path(args.data) / "database.yml"))
    protocol = registry.get_protocol(
        "AMPSCZ.SpeakerDiarization.Interviews",
        preprocessors={"audio": FileFinder()},
    )

    model = Model.from_pretrained(PIPELINE_REPO, subfolder="segmentation")
    spec = model.specifications
    logger.info(
        "Pretrained segmentation: duration=%.1fs classes=%d powerset_max=%d",
        spec.duration, len(spec.classes), spec.powerset_max_classes,
    )

    # Keep the pretrained powerset geometry so setup() reuses the classifier
    # head instead of rebuilding it.
    task = SpeakerDiarization(
        protocol,
        duration=spec.duration,
        max_speakers_per_chunk=len(spec.classes),
        max_speakers_per_frame=spec.powerset_max_classes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache=str(checkpoint_dir / "task-cache"),
    )
    model.task = task

    # The base Model hardcodes Adam(lr=1e-3); fine-tuning needs a lower rate.
    lr = args.lr
    model.configure_optimizers = types.MethodType(
        lambda self: torch.optim.Adam(self.parameters(), lr=lr), model
    )

    trunk = find_trunk(model)
    if args.freeze_epochs > 0 and trunk is None:
        logger.warning("No known trunk module found; training without a freeze stage")
    if args.freeze_epochs > 0 and trunk is not None:
        model.freeze_by_name(trunk)
        logger.info("Froze %s for the first %d epoch(s)", trunk, args.freeze_epochs)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="epoch{epoch}",
        auto_insert_metric_name=False,
        monitor="loss/val",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    callbacks: list[Callback] = [
        checkpoint_callback,
        EarlyStopping(monitor="loss/val", mode="min", patience=args.patience),
    ]
    if args.freeze_epochs > 0 and trunk is not None:
        callbacks.append(UnfreezeAfter(trunk, args.freeze_epochs))

    trainer = Trainer(
        max_epochs=args.max_epochs,
        devices=args.devices,
        accelerator="auto",
        callbacks=callbacks,
        default_root_dir=str(checkpoint_dir),
    )
    trainer.fit(model)

    best = checkpoint_callback.best_model_path
    if not best:
        logger.error("No checkpoint was saved")
        return 1
    link = checkpoint_dir / "best.ckpt"
    link.unlink(missing_ok=True)
    link.symlink_to(Path(best).resolve())
    logger.info("Best checkpoint (val loss %.4f): %s -> %s",
                float(checkpoint_callback.best_model_score), link, best)
    return 0


if __name__ == "__main__":
    sys.exit(main())

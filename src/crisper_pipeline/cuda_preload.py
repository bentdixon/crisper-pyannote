"""Preload CUDA 13 runtime libraries so torchcodec can load.

The PyPI torchcodec wheel links CUDA 13 libraries (libcudart.so.13,
libnvrtc.so.13) that the cu128 torch stack does not provide or preload. The
nvidia-cuda-runtime and nvidia-cuda-nvrtc packages ship them; dlopen-ing
them before pyannote.audio is imported lets torchcodec's core library
resolve, which both silences pyannote's import-time warning and enables its
built-in decoding (used by the training dataloader in finetune/).
"""

from __future__ import annotations

import ctypes
import glob
import logging
import sysconfig
from pathlib import Path

logger = logging.getLogger(__name__)


def preload_torchcodec_libs() -> None:
    """Best effort; a failure only means torchcodec stays unavailable."""
    try:
        import torch  # noqa: F401  (torchcodec also needs libtorch loaded)
    except ImportError:
        return
    site_packages = Path(sysconfig.get_paths()["purelib"])
    pattern = str(site_packages / "nvidia" / "cu13" / "lib" / "*.so.13")
    for lib in sorted(glob.glob(pattern)):
        try:
            ctypes.CDLL(lib)
        except OSError as exc:
            logger.debug("Could not preload %s: %s", lib, exc)

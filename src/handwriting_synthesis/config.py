"""
Locations of the pretrained model assets.

The repository ships the trained weights and style-priming data under ``model/``:

    model/
    ├── checkpoint/   TensorFlow checkpoint with the trained network weights
    └── style/        style-N-strokes.npy / style-N-chars.npy priming samples

By default assets are resolved relative to the current working directory, which
works when running from the repository root. For deployments (e.g. Docker), set
the ``HANDWRITING_MODEL_DIR`` environment variable or pass an explicit
``model_dir`` to :class:`~handwriting_synthesis.HandwritingSynthesizer`.
"""

import os
from pathlib import Path

MODEL_DIR_ENV_VAR = "HANDWRITING_MODEL_DIR"


def default_model_dir() -> Path:
    """
    Directory holding the pretrained model assets.
    """
    return Path(os.environ.get(MODEL_DIR_ENV_VAR, "model"))


def checkpoint_dir(model_dir: Path | None = None) -> Path:
    """
    Directory holding the TensorFlow checkpoint files.
    """
    return (model_dir or default_model_dir()) / "checkpoint"


def styles_dir(model_dir: Path | None = None) -> Path:
    """
    Directory holding the handwriting-style priming data.
    """
    return (model_dir or default_model_dir()) / "style"

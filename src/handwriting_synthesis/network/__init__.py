"""ML internals: the TensorFlow network behind the synthesizer.

You should not need to touch anything in here to work on the service — the
public entry point is :class:`handwriting_synthesis.HandwritingSynthesizer`,
which wraps :class:`~handwriting_synthesis.network.model.HandwritingNetwork`.
"""

import warnings as _warnings

import tensorflow as _tf
import tensorflow.compat.v1 as _tf1

# The pretrained model predates TF2, so its deprecation warnings are expected
# and not actionable here; keep them out of application logs. Applications can
# re-raise the level after import if they want them back.
_tf.get_logger().setLevel("ERROR")
_warnings.filterwarnings("ignore", message=".*rnn_cell.LSTMCell.*", category=UserWarning)

# The pretrained model is a TF1-style static graph; the whole process must run
# TensorFlow in graph mode. Note this is global: any application importing this
# package (e.g. a FastAPI service) runs TF in v1 compatibility mode.
_tf1.disable_v2_behavior()

from .model import HandwritingNetwork  # noqa: E402

__all__ = ["HandwritingNetwork"]

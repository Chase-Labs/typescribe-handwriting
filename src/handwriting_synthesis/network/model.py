"""
Builds the inference graph, restores the pretrained weights, runs sampling.

This is the only module that owns a TensorFlow session. The network was trained
with the TF1 graph API, so inference keeps that style: build a static graph once,
restore the checkpoint into it, then feed inputs through placeholders.

The training half of the original project (losses, optimizers, checkpointing
loops) has been removed; only the pieces needed to *run* the model remain.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1

from ..alphabet import ALPHABET
from .attention_cell import LSTMAttentionCell
from .layers import time_distributed_dense_layer
from .sampling import sample_sequence

# Architecture of the pretrained checkpoint. These are fixed properties of the
# shipped weights — changing them without retraining will break restoring.
LSTM_SIZE = 400
OUTPUT_MIXTURE_COMPONENTS = 20
ATTENTION_MIXTURE_COMPONENTS = 10


class HandwritingNetwork:
    """
    The pretrained handwriting RNN, ready for sampling.

    Loading builds the graph and restores the weights (a few seconds); keep one
    instance alive and reuse it for every request rather than constructing it
    per call.
    """

    def __init__(self, checkpoint_dir: str | Path):
        self._graph = self._build_graph()
        self.session = tf1.Session(graph=self._graph)
        self._restore(Path(checkpoint_dir))

    def sample(
        self,
        char_ids: np.ndarray,
        char_lengths: np.ndarray,
        biases: Sequence[float],
        max_steps: int,
        prime: bool = False,
        prime_strokes: np.ndarray | None = None,
        prime_stroke_lengths: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """
        Sample handwriting strokes for a batch of encoded lines.

        Args:
            char_ids: ``[batch, max_chars]`` int array of encoded characters
                (see :func:`handwriting_synthesis.alphabet.encode`), zero-padded.
            char_lengths: ``[batch]`` actual length of each encoded line.
            biases: ``[batch]`` neatness knob per line (0 = wild, ~1 = tidy).
            max_steps: Cap on pen points per line (roughly 40 per character).
            prime: Whether to prime the network with a reference style sample.
            prime_strokes: ``[batch, steps, 3]`` reference strokes when priming.
            prime_stroke_lengths: ``[batch]`` actual length of each reference.

        Returns:
            One ``(N, 3)`` float array of pen offsets ``(dx, dy, pen_up)`` per line.
        """
        batch_size = len(char_ids)
        if prime_strokes is None:
            # The priming pass always runs in the graph, but with zero lengths it
            # is a no-op; feed the smallest possible placeholder buffer.
            prime_strokes = np.zeros([batch_size, 1, 3], dtype=np.float32)
            prime_stroke_lengths = np.zeros([batch_size], dtype=np.int32)

        [sequences] = self.session.run(
            [self._sampled_sequence],
            feed_dict={
                self._char_ids: char_ids,
                self._char_lengths: char_lengths,
                self._biases: biases,
                self._max_steps: max_steps,
                self._prime: prime,
                self._prime_strokes: prime_strokes,
                self._prime_stroke_lengths: prime_stroke_lengths,
            },
        )
        # Lines that finish early are padded with all-zero rows; strip them.
        return [seq[~np.all(seq == 0.0, axis=1)] for seq in sequences]

    def close(self) -> None:
        """
        Release the TensorFlow session.
        """
        self.session.close()

    def _build_graph(self) -> tf.Graph:
        graph = tf.Graph()
        with graph.as_default():
            # --- Inputs ---------------------------------------------------------
            self._char_ids = tf1.placeholder(tf.int32, [None, None], name="char_ids")
            self._char_lengths = tf1.placeholder(tf.int32, [None], name="char_lengths")
            self._biases = tf1.placeholder(tf.float32, [None], name="biases")
            self._max_steps = tf1.placeholder(tf.int32, [], name="max_steps")
            self._prime = tf1.placeholder(tf.bool, [], name="prime")
            self._prime_strokes = tf1.placeholder(
                tf.float32, [None, None, 3], name="prime_strokes"
            )
            self._prime_stroke_lengths = tf1.placeholder(
                tf.int32, [None], name="prime_stroke_lengths"
            )

            batch_size = tf.shape(self._prime_strokes)[0]

            cell = LSTMAttentionCell(
                lstm_size=LSTM_SIZE,
                num_attn_mixture_components=ATTENTION_MIXTURE_COMPONENTS,
                # The text to write, one-hot encoded for the attention window.
                attention_values=tf.one_hot(self._char_ids, len(ALPHABET)),
                attention_values_lengths=self._char_lengths,
                num_output_mixture_components=OUTPUT_MIXTURE_COMPONENTS,
                bias=self._biases,
            )

            # --- Style priming ---------------------------------------------------
            # Replay the reference strokes through the network so its state ends up
            # "mid-handwriting" in the reference style. This pass also creates every
            # network variable (TF1 creates weights lazily on first use, and the
            # sampling loop below expects them to exist already). When not priming,
            # the stroke lengths are zero and this pass simply returns the zero state.
            outputs, primed_state = tf1.nn.dynamic_rnn(
                cell=cell,
                inputs=self._prime_strokes,
                sequence_length=self._prime_stroke_lengths,
                initial_state=cell.zero_state(batch_size, tf.float32),
                dtype=tf.float32,
                scope="rnn",
            )
            # Instantiate the "gmm" output-head weights (shared with
            # cell.output_function, which reuses this scope).
            time_distributed_dense_layer(outputs, cell.output_units, scope="rnn/gmm")

            # --- Sampling --------------------------------------------------------
            # Primed: the first pen movement is sampled from the primed state.
            # Unprimed: start from the canonical "pen down at the origin" input.
            with tf1.variable_scope("rnn", reuse=True):
                initial_input = tf.cond(
                    self._prime,
                    lambda: cell.output_function(primed_state),
                    lambda: tf.concat(
                        [tf.zeros([batch_size, 2]), tf.ones([batch_size, 1])], axis=1
                    ),
                )

            self._sampled_sequence = sample_sequence(
                cell=cell,
                initial_state=primed_state,
                initial_input=initial_input,
                max_steps=self._max_steps,
                scope="rnn",
            )

            # Only the network weights live in this graph, so the Saver restores
            # exactly those from the checkpoint (training bookkeeping variables in
            # the checkpoint file are simply ignored).
            self._saver = tf1.train.Saver()
        return graph

    def _restore(self, checkpoint_dir: Path) -> None:
        checkpoint = tf.train.latest_checkpoint(str(checkpoint_dir))
        if checkpoint is None:
            raise FileNotFoundError(
                f"No TensorFlow checkpoint found in '{checkpoint_dir}'. "
                "Pass the directory containing the pretrained 'model-*' files "
                "(shipped in this repository under model/checkpoint), or set "
                "the HANDWRITING_MODEL_DIR environment variable."
            )
        self._saver.restore(self.session, checkpoint)

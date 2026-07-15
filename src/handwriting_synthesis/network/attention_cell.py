"""The recurrent cell at the heart of the handwriting model.

This implements the architecture from Alex Graves' *Generating Sequences with
Recurrent Neural Networks* (https://arxiv.org/abs/1308.0850): three stacked LSTM
layers with a soft "attention window" that slides over the text as the pen moves.

Intuition for non-ML readers:

- The **attention window** is the model's reading finger. At every pen movement it
  computes a soft position over the input characters (`kappa` drifts monotonically
  forward), so the network always knows which letter it is currently writing.
- The **output head** doesn't predict a single next pen position. It predicts a
  *mixture of 2-D Gaussians* (a weighted set of candidate directions with
  uncertainty) plus a probability that the pen lifts. Sampling from that mixture
  is what makes every generation look naturally different.
- The **bias** term sharpens those distributions at sampling time: higher bias →
  less randomness → neater, more deliberate handwriting.
"""

from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow_probability as tfp

from .layers import dense_layer, shape

# The pretrained checkpoint stores weights under this scope name (it was the
# class name at training time). It must not change, or restoring will fail.
_VARIABLE_SCOPE = "LSTMAttentionCell"

LSTMAttentionCellState = namedtuple(
    "LSTMAttentionCellState",
    [
        "h1",
        "c1",  # hidden/cell state of LSTM layer 1
        "h2",
        "c2",  # hidden/cell state of LSTM layer 2
        "h3",
        "c3",  # hidden/cell state of LSTM layer 3
        "alpha",  # attention mixture weights
        "beta",  # attention window widths
        "kappa",  # attention window positions (monotonically increasing)
        "w",  # current attention window: soft one-hot of "the character being written"
        "phi",  # per-character attention weights (used to detect end of text)
    ],
)


class LSTMAttentionCell(tf1.nn.rnn_cell.RNNCell):
    """Three stacked LSTMs with a Graves-style attention window over the text.

    One `__call__` consumes one pen offset `(dx, dy, pen_up)` and advances the
    state; :meth:`output_function` then samples the *next* pen offset from the
    state, and :meth:`termination_condition` reports which sequences have finished
    writing their text.
    """

    def __init__(
        self,
        lstm_size: int,
        num_attn_mixture_components: int,
        attention_values: tf.Tensor,
        attention_values_lengths: tf.Tensor,
        num_output_mixture_components: int,
        bias: tf.Tensor,
    ):
        self.lstm_size = lstm_size
        self.num_attn_mixture_components = num_attn_mixture_components
        self.attention_values = (
            attention_values  # one-hot characters, [batch, chars, alphabet]
        )
        self.attention_values_lengths = attention_values_lengths
        self.window_size = shape(self.attention_values, 2)
        self.char_len = tf.shape(attention_values)[1]
        self.batch_size = tf.shape(attention_values)[0]
        self.num_output_mixture_components = num_output_mixture_components
        # Per mixture component: 2 means + 2 std-devs + 1 correlation + 1 weight,
        # plus a single shared pen-up probability.
        self.output_units = 6 * self.num_output_mixture_components + 1
        self.bias = bias

    @property
    def state_size(self) -> LSTMAttentionCellState:
        return LSTMAttentionCellState(
            self.lstm_size,
            self.lstm_size,
            self.lstm_size,
            self.lstm_size,
            self.lstm_size,
            self.lstm_size,
            self.num_attn_mixture_components,
            self.num_attn_mixture_components,
            self.num_attn_mixture_components,
            self.window_size,
            self.char_len,
        )

    @property
    def output_size(self) -> int:
        return self.lstm_size

    def zero_state(self, batch_size, dtype) -> LSTMAttentionCellState:
        return LSTMAttentionCellState(
            *(tf.zeros([batch_size, size]) for size in self.state_size)
        )

    def __call__(self, inputs, state, scope=None):
        with tf1.variable_scope(scope or _VARIABLE_SCOPE, reuse=tf1.AUTO_REUSE):
            # LSTM layer 1 sees the pen input plus the current attention window.
            s1_in = tf.concat([state.w, inputs], axis=1)
            cell1 = tf1.nn.rnn_cell.LSTMCell(self.lstm_size)
            s1_out, s1_state = cell1(s1_in, state=(state.c1, state.h1))

            # Attention: decide how far along the text the pen has moved.
            attention_inputs = tf.concat([state.w, inputs, s1_out], axis=1)
            attention_params = dense_layer(
                attention_inputs,
                3 * self.num_attn_mixture_components,
                scope="attention",
            )
            alpha, beta, kappa = tf.split(tf.nn.softplus(attention_params), 3, axis=1)
            kappa = (
                state.kappa + kappa / 25.0
            )  # window position only moves forward, slowly
            beta = tf.clip_by_value(beta, 0.01, np.inf)

            kappa_flat, alpha_flat, beta_flat = kappa, alpha, beta
            kappa, alpha, beta = (tf.expand_dims(t, 2) for t in (kappa, alpha, beta))

            # phi[i] = how much attention character i gets right now.
            enum = tf.reshape(tf.range(self.char_len), (1, 1, self.char_len))
            u = tf.cast(
                tf.tile(enum, (self.batch_size, self.num_attn_mixture_components, 1)),
                tf.float32,
            )
            phi_flat = tf.reduce_sum(
                alpha * tf.exp(-tf.square(kappa - u) / beta), axis=1
            )

            # w = the attention-weighted blend of the input characters.
            phi = tf.expand_dims(phi_flat, 2)
            sequence_mask = tf.cast(
                tf.sequence_mask(self.attention_values_lengths, maxlen=self.char_len),
                tf.float32,
            )
            sequence_mask = tf.expand_dims(sequence_mask, 2)
            w = tf.reduce_sum(phi * self.attention_values * sequence_mask, axis=1)

            # LSTM layers 2 and 3 refine the trajectory given the attended character.
            s2_in = tf.concat([inputs, s1_out, w], axis=1)
            cell2 = tf1.nn.rnn_cell.LSTMCell(self.lstm_size)
            s2_out, s2_state = cell2(s2_in, state=(state.c2, state.h2))

            s3_in = tf.concat([inputs, s2_out, w], axis=1)
            cell3 = tf1.nn.rnn_cell.LSTMCell(self.lstm_size)
            s3_out, s3_state = cell3(s3_in, state=(state.c3, state.h3))

            new_state = LSTMAttentionCellState(
                s1_state.h,
                s1_state.c,
                s2_state.h,
                s2_state.c,
                s3_state.h,
                s3_state.c,
                alpha_flat,
                beta_flat,
                kappa_flat,
                w,
                phi_flat,
            )
            return s3_out, new_state

    def output_function(self, state: LSTMAttentionCellState) -> tf.Tensor:
        """Sample the next pen offset `(dx, dy, pen_up)` from the cell state.

        The state is projected to a Gaussian-mixture density (the "gmm" head in the
        checkpoint); we pick a mixture component, sample a 2-D offset from it, and
        sample the pen-up flag from a Bernoulli.
        """
        params = dense_layer(
            state.h3, self.output_units, scope="gmm", reuse=tf1.AUTO_REUSE
        )
        pis, mus, sigmas, rhos, es = self._parse_parameters(params)
        mu1, mu2 = tf.split(mus, 2, axis=1)
        mus = tf.stack([mu1, mu2], axis=2)
        sigma1, sigma2 = tf.split(sigmas, 2, axis=1)

        covar_matrix = [
            tf.square(sigma1),
            rhos * sigma1 * sigma2,
            rhos * sigma1 * sigma2,
            tf.square(sigma2),
        ]
        covar_matrix = tf.stack(covar_matrix, axis=2)
        covar_matrix = tf.reshape(
            covar_matrix, (self.batch_size, self.num_output_mixture_components, 2, 2)
        )

        mvn = tfp.distributions.MultivariateNormalTriL(
            loc=mus, scale_tril=tf.linalg.cholesky(covar_matrix)
        )
        pen_up = tfp.distributions.Bernoulli(probs=es)
        component = tfp.distributions.Categorical(probs=pis)

        sampled_e = pen_up.sample()
        sampled_coords = mvn.sample()
        sampled_idx = component.sample()

        idx = tf.stack([tf.range(self.batch_size), sampled_idx], axis=1)
        coords = tf.gather_nd(sampled_coords, idx)
        return tf.concat([coords, tf.cast(sampled_e, tf.float32)], axis=1)

    def termination_condition(self, state: LSTMAttentionCellState) -> tf.Tensor:
        """Per-sequence bool: has this line finished being written?

        A line is done when the attention window has reached its final character
        and the pen lifts — or when attention has run past the end of the text.
        """
        char_idx = tf.cast(tf.argmax(state.phi, axis=1), tf.int32)
        final_char = char_idx >= self.attention_values_lengths - 1
        past_final_char = char_idx >= self.attention_values_lengths
        output = self.output_function(state)
        es = tf.cast(output[:, 2], tf.int32)
        is_eos = tf.equal(es, tf.ones_like(es))
        return tf.logical_or(tf.logical_and(final_char, is_eos), past_final_char)

    def _parse_parameters(
        self, gmm_params: tf.Tensor, eps: float = 1e-8, sigma_eps: float = 1e-4
    ):
        """Split the raw gmm-head output into valid distribution parameters.

        Also applies the sampling `bias`: it sharpens the mixture weights and
        shrinks the variances, trading naturalness for neatness.
        """
        pis, sigmas, rhos, mus, es = tf.split(
            gmm_params,
            [
                1 * self.num_output_mixture_components,
                2 * self.num_output_mixture_components,
                1 * self.num_output_mixture_components,
                2 * self.num_output_mixture_components,
                1,
            ],
            axis=-1,
        )
        pis = pis * (1 + tf.expand_dims(self.bias, 1))
        sigmas = sigmas - tf.expand_dims(self.bias, 1)

        pis = tf.nn.softmax(pis, axis=-1)
        pis = tf1.where(pis < 0.01, tf.zeros_like(pis), pis)
        sigmas = tf.clip_by_value(tf.exp(sigmas), sigma_eps, np.inf)
        rhos = tf.clip_by_value(tf.tanh(rhos), eps - 1.0, 1.0 - eps)
        es = tf.clip_by_value(tf.nn.sigmoid(es), eps, 1.0 - eps)
        es = tf1.where(es < 0.01, tf.zeros_like(es), es)

        return pis, mus, sigmas, rhos, es

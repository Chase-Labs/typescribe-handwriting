"""Minimal TF1-style layer helpers used by the network.

The pretrained checkpoint stores weights under specific variable names
(`<scope>/weights` and `<scope>/biases`), so these helpers must keep
creating variables with exactly those names.
"""

import tensorflow as tf
import tensorflow.compat.v1 as tf1


def dense_layer(
    inputs: tf.Tensor, output_units: int, scope: str = "dense-layer", reuse=False
) -> tf.Tensor:
    """
    Fully-connected layer for 2-D input: `[batch, in] -> [batch, out]`.
    """
    with tf1.variable_scope(scope, reuse=reuse):
        weights, biases = _weights_and_biases(shape(inputs, -1), output_units)
        return tf.matmul(inputs, weights) + biases


def time_distributed_dense_layer(
    inputs: tf.Tensor,
    output_units: int,
    scope: str = "time-distributed-dense-layer",
    reuse=False,
) -> tf.Tensor:
    """
    The same fully-connected layer applied independently at every timestep.

    `[batch, time, in] -> [batch, time, out]`, with one shared weight matrix.
    """
    with tf1.variable_scope(scope, reuse=reuse):
        weights, biases = _weights_and_biases(shape(inputs, -1), output_units)
        return tf.einsum("ijk,kl->ijl", inputs, weights) + biases


def _weights_and_biases(
    input_units: int, output_units: int
) -> tuple[tf.Variable, tf.Variable]:
    weights = tf1.get_variable(
        name="weights",
        initializer=tf1.variance_scaling_initializer(scale=2.0),
        shape=[input_units, output_units],
    )
    biases = tf1.get_variable(
        name="biases",
        initializer=tf1.constant_initializer(),
        shape=[output_units],
    )
    return weights, biases


def shape(tensor: tf.Tensor, dim: int | None = None):
    """Static tensor shape as a list, or a single dimension of it."""
    if dim is None:
        return tensor.shape.as_list()
    return tensor.shape.as_list()[dim]

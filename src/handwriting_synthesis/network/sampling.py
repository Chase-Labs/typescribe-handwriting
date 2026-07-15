"""
The autoregressive sampling loop that actually "writes".

Handwriting is generated one pen movement at a time: the cell state is turned
into a probability distribution over "where does the pen go next", a movement is
sampled from it, and that movement is fed back into the cell as the next input.
The loop runs until every line in the batch reports that it has finished writing
its text (or the `max_steps` safety cap is hit).

This replaces the copy of TensorFlow's internal `raw_rnn` machinery that the
original project carried around: a direct `tf.while_loop` that only tracks what
inference needs (the sampled outputs), instead of stacking every intermediate
cell state as well.
"""

import tensorflow as tf
import tensorflow.compat.v1 as tf1

from .attention_cell import LSTMAttentionCell, LSTMAttentionCellState


def sample_sequence(
    cell: LSTMAttentionCell,
    initial_state: LSTMAttentionCellState,
    initial_input: tf.Tensor,
    max_steps: tf.Tensor,
    scope: str = "rnn",
) -> tf.Tensor:
    """
    Free-run the cell, feeding each sampled output back as the next input.

    Args:
        cell: The recurrent cell. Beyond the standard `__call__`, it must
            provide `output_function(state)` (sample the next pen offset) and
            `termination_condition(state)` (per-line "finished writing?" flag).
        initial_state: State to start writing from — the zero state, or the state
            left behind by a style-priming pass.
        initial_input: `[batch, 3]` first pen input fed to the cell.
        max_steps: Scalar int tensor; hard cap on the number of pen points.
        scope: The variable scope the network weights live under.

    Returns:
        `[batch, time, 3]` float tensor of sampled pen offsets
        `(dx, dy, pen_up)`. Lines that finish before `time` steps are padded
        with all-zero rows (callers strip these).
    """
    # All weights already exist (created by the priming pass), so everything in
    # the loop runs with reuse=True.
    with tf1.variable_scope(scope, reuse=True):
        time = tf.constant(0, dtype=tf.int32)
        finished = tf.logical_or(
            time >= max_steps, cell.termination_condition(initial_state)
        )
        outputs = tf.TensorArray(
            dtype=tf.float32,
            size=0,
            dynamic_size=True,
            element_shape=initial_input.shape,
        )

        def keep_writing(time, finished, *_) -> tf.Tensor:
            return tf.logical_not(tf.reduce_all(finished))

        def step(time, finished, current_input, state, outputs):
            _, proposed_state = cell(current_input, state)

            # Lines that already finished keep their old state; the rest advance.
            state = tf.nest.map_structure(
                lambda old, new: tf1.where(finished, old, new), state, proposed_state
            )
            now_finished = tf.logical_or(
                finished,
                tf.logical_or(time + 1 >= max_steps, cell.termination_condition(state)),
            )
            # Sample the next pen movement (skipped entirely once every line is done).
            next_input = tf.cond(
                tf.reduce_all(now_finished),
                lambda: tf.zeros_like(current_input),
                lambda: cell.output_function(state),
            )
            # Already-finished lines emit all-zero rows, which the caller strips.
            emitted = tf1.where(finished, tf.zeros_like(next_input), next_input)
            return (
                time + 1,
                now_finished,
                next_input,
                state,
                outputs.write(time, emitted),
            )

        _, _, _, _, outputs = tf1.while_loop(
            cond=keep_writing,
            body=step,
            loop_vars=[time, finished, initial_input, initial_state, outputs],
            parallel_iterations=32,
        )

    # TensorArray stacks to [time, batch, 3]; callers want [batch, time, 3].
    return tf.transpose(outputs.stack(), (1, 0, 2))

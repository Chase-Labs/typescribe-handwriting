"""
Stroke geometry helpers used when post-processing model output.

Data format
-----------
The model emits handwriting as a `(N, 3)` float array of **pen offsets**: each
row is `(dx, dy, pen_up)`, i.e. how far the pen moved since the previous point
and whether the pen was lifted *after* this point (`pen_up == 1` ends a stroke,
e.g. between words or to dot an "i").

For drawing we convert offsets to absolute **coordinates** `(x, y, pen_up)`
with :func:`offsets_to_coords`, then clean them up with :func:`denoise` and
:func:`align`. All functions return new arrays and leave their input untouched.
"""

import numpy as np
from scipy.signal import savgol_filter


def offsets_to_coords(offsets: np.ndarray) -> np.ndarray:
    """
    Convert relative pen offsets to absolute coordinates via a running sum.
    """
    return np.concatenate([np.cumsum(offsets[:, :2], axis=0), offsets[:, 2:3]], axis=1)


def denoise(coords: np.ndarray) -> np.ndarray:
    """
    Smooth jitter out of the pen trajectory.

    Applies a Savitzky-Golay filter (a moving polynomial fit) to the x and y of
    each stroke independently, which removes the high-frequency wobble the model
    inherits from the digitiser-pen training data without rounding off genuine
    letter shapes.
    """
    strokes = np.split(coords, np.where(coords[:, 2] == 1)[0] + 1, axis=0)
    smoothed = []
    for stroke in strokes:
        if len(stroke) == 0:
            continue
        x = savgol_filter(stroke[:, 0], 7, 3, mode="nearest")
        y = savgol_filter(stroke[:, 1], 7, 3, mode="nearest")
        smoothed.append(np.column_stack([x, y, stroke[:, 2]]))  # type: ignore
    return np.vstack(smoothed)


def align(xy: np.ndarray) -> np.ndarray:
    """
    Correct the global slant of a handwritten line.

    Fits a straight baseline through all points by least squares, then rotates
    and shifts the points so that baseline becomes horizontal at y=0. This keeps
    generated lines from drifting uphill or downhill across the page.

    Takes and returns an `(N, 2)` array of x/y coordinates.
    """
    xy = np.copy(xy)
    x, y = xy[:, 0].reshape(-1, 1), xy[:, 1].reshape(-1, 1)
    design = np.concatenate([np.ones([x.shape[0], 1]), x], axis=1)
    offset, slope = np.linalg.inv(design.T.dot(design)).dot(design.T).dot(y).squeeze()
    theta = np.arctan(slope)
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    return np.dot(xy, rotation) - offset

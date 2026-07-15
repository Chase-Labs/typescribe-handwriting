"""
Stage 2 of the pipeline: turn lines of text into handwriting strokes.

:class:`HandwritingSynthesizer` is the public entry point of this package. It
wraps the TensorFlow model with input validation and style handling, and knows
nothing about SVG — generation returns raw stroke arrays, which stage 3
(:mod:`handwriting_synthesis.rendering`) draws onto a page. For one-call
convenience, :meth:`HandwritingSynthesizer.write_page` chains both stages.

Two knobs control the look of the output:

- **style** (0-12): which reference handwriting to imitate. Implemented by
  "priming": before writing your text, the network replays a stored sample of
  that style so it starts writing "in the same hand".
- **bias** (0 to ~1): how neat the writing is. The model samples pen movements
  from a probability distribution; bias sharpens that distribution. 0 gives
  wild, messy output, ~0.75-1.0 gives tidy handwriting.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from . import alphabet, config
from .network import HandwritingNetwork
from .rendering import PageLayout, save_page

# The model cannot write lines longer than this (a training-data limit).
MAX_LINE_LENGTH = 75

# Rough number of pen points the model needs per character.
_STEPS_PER_CHAR = 40

# Fixed-size input buffers the network was trained with: priming strokes are
# padded to 1200 points, and (priming + line) text to 120 characters.
_PRIME_STROKE_BUFFER = 1200
_CHAR_BUFFER = 120

# A bias/style/width/color argument: one value for all lines, or one per line.
PerLine = float | int | str | None | Sequence


class HandwritingSynthesizer:
    """
    Generates handwriting from text using the pretrained RNN.

    Construction loads the TensorFlow model (a few seconds); create one instance
    at startup and reuse it for every generation call. The instance is not
    thread-safe for concurrent generation, but sequential reuse is cheap.

    Args:
        model_dir: Directory containing the pretrained assets (`checkpoint/`
            and `style/` subdirectories). Defaults to `$HANDWRITING_MODEL_DIR`
            or `./model`.
    """

    def __init__(self, model_dir: str | Path | None = None):
        model_dir = (
            Path(model_dir) if model_dir is not None else config.default_model_dir()
        )
        self._network = HandwritingNetwork(
            checkpoint_dir=config.checkpoint_dir(model_dir)
        )
        self._styles_dir = config.styles_dir(model_dir)

    def generate(
        self,
        lines: Sequence[str],
        biases: PerLine = 0.75,
        styles: PerLine = None,
    ) -> list[np.ndarray]:
        """
        Generate handwriting strokes for each line of text.

        Args:
            lines: Sanitised lines of at most :data:`MAX_LINE_LENGTH` characters
                (see :mod:`handwriting_synthesis.preprocessing`). Raises
                `ValueError` for lines that are too long or contain characters
                the model cannot draw.
            biases: Neatness, 0 to ~1 — a single value or one per line.
            styles: Handwriting style id (0-12) — a single value or one per
                line. `None` samples an unprimed, "average" style.

        Returns:
            One `(N, 3)` array of pen offsets `(dx, dy, pen_up)` per line,
            ready for :mod:`handwriting_synthesis.rendering`.
        """
        self._validate(lines)
        biases = _per_line(biases, len(lines), "biases")
        styles = _per_line(styles, len(lines), "styles")
        max_steps = _STEPS_PER_CHAR * max(len(line) for line in lines)

        if styles is None:
            encoded = [alphabet.encode(line) for line in lines]
            char_ids, char_lengths = _pack(encoded, _CHAR_BUFFER)
            return self._network.sample(char_ids, char_lengths, biases, max_steps)

        prime_strokes, prime_lengths, encoded = self._load_style_priming(lines, styles)
        char_ids, char_lengths = _pack(encoded, _CHAR_BUFFER)
        return self._network.sample(
            char_ids,
            char_lengths,
            biases,
            max_steps,
            prime=True,
            prime_strokes=prime_strokes,
            prime_stroke_lengths=prime_lengths,
        )

    def write_page(
        self,
        lines: Sequence[str],
        svg_path: str | Path,
        biases: PerLine = 0.75,
        styles: PerLine = None,
        stroke_colors: PerLine = None,
        stroke_widths: PerLine = None,
        layout: PageLayout | None = None,
        png_path: str | Path | None = None,
    ) -> Path:
        """Generate one page of handwriting and save it as an SVG.

        Convenience wrapper chaining :meth:`generate` and
        :func:`handwriting_synthesis.rendering.save_page`. A line consisting of
        a single `"."` is rendered as a blank ruled line (useful for paragraph
        spacing). Returns the SVG path.
        """
        stroke_offsets = self.generate(lines, biases=biases, styles=styles)
        drawn_lines = ["" if line == "." else line for line in lines]
        return save_page(
            stroke_offsets,
            drawn_lines,
            svg_path,
            layout=layout,
            stroke_colors=_per_line(stroke_colors, len(lines), "stroke_colors"),
            stroke_widths=_per_line(stroke_widths, len(lines), "stroke_widths"),
            png_path=png_path,
        )

    def close(self) -> None:
        """Release the underlying TensorFlow session."""
        self._network.close()

    def _validate(self, lines: Sequence[str]) -> None:
        if not lines:
            raise ValueError("At least one line of text is required.")
        for i, line in enumerate(lines):
            if not line:
                raise ValueError(
                    f"Line {i} is empty. Use '.' for a blank line, or filter empty "
                    "lines out (see handwriting_synthesis.preprocessing)."
                )
            if len(line) > MAX_LINE_LENGTH:
                raise ValueError(
                    f"Line {i} is {len(line)} characters long; the model supports "
                    f"at most {MAX_LINE_LENGTH} per line. Wrap the text first "
                    "(see handwriting_synthesis.preprocessing.wrap)."
                )
            unsupported = alphabet.unsupported_chars(line)
            if unsupported:
                raise ValueError(
                    f"Line {i} contains characters the model cannot draw: "
                    f"{sorted(unsupported)}. Sanitise the text first "
                    "(see handwriting_synthesis.preprocessing.sanitize)."
                )

    def _load_style_priming(
        self, lines: Sequence[str], styles: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        """Load the reference sample for each line's style.

        Each style ships as a pair of files: the reference pen strokes, and the
        text those strokes spell out. The network is fed reference strokes plus
        "reference text + our text", so when sampling starts it continues in the
        reference's handwriting.
        """
        batch_size = len(lines)
        prime_strokes = np.zeros(
            [batch_size, _PRIME_STROKE_BUFFER, 3], dtype=np.float32
        )
        prime_lengths = np.zeros([batch_size], dtype=np.int32)
        encoded: list[np.ndarray] = []

        for i, (line, style) in enumerate(zip(lines, styles)):
            try:
                strokes = np.load(self._styles_dir / f"style-{style}-strokes.npy")
                style_text = (
                    np.load(self._styles_dir / f"style-{style}-chars.npy")
                    .tobytes()
                    .decode("utf-8")
                )
            except FileNotFoundError as exc:
                raise ValueError(
                    f"Unknown style {style!r}: no priming data in '{self._styles_dir}'."
                ) from exc

            ids = alphabet.encode(style_text + " " + line)
            if len(ids) > _CHAR_BUFFER:
                raise ValueError(
                    f"Line {i} is too long to combine with style {style}'s priming "
                    f"text ({len(ids)} > {_CHAR_BUFFER} encoded characters). "
                    "Use shorter lines or a different style."
                )
            prime_strokes[i, : len(strokes), :] = strokes
            prime_lengths[i] = len(strokes)
            encoded.append(ids)

        return prime_strokes, prime_lengths, encoded


def _per_line(value: PerLine, num_lines: int, name: str):
    """Broadcast a scalar setting to one value per line; validate list lengths."""
    if value is None:
        return None
    if isinstance(value, (int, float, str)):
        return [value] * num_lines
    if len(value) != num_lines:
        raise ValueError(f"Got {len(value)} {name} for {num_lines} lines.")
    return list(value)


def _pack(encoded: list[np.ndarray], buffer_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Pad a batch of encoded lines into one fixed-size int matrix + lengths."""
    char_ids = np.zeros([len(encoded), buffer_size], dtype=np.int32)
    char_lengths = np.zeros([len(encoded)], dtype=np.int32)
    for i, ids in enumerate(encoded):
        char_ids[i, : len(ids)] = ids
        char_lengths[i] = len(ids)
    return char_ids, char_lengths

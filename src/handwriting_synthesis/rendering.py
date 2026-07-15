"""
Stage 3 of the pipeline: draw generated strokes onto a ruled SVG page.

This is pure drawing code (no ML): it takes the stroke offsets produced by the
synthesizer, cleans them up with :mod:`handwriting_synthesis.strokes`, and lays
them out line by line on a notebook-style page — ruled lines, red margin, one
handwriting line per rule.

:func:`render_page` returns an `svgwrite.Drawing`; call `.tostring()` on it
for an in-memory SVG (handy for an HTTP response) or use :func:`save_page` to
write it to disk, optionally with a PNG conversion alongside.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import svgwrite

from . import strokes


@dataclass(frozen=True)
class PageLayout:
    """Geometry and colours of the rendered page.

    The defaults describe an A4-proportioned page (0.707 width/height ratio)
    with 24 ruled lines, a red margin down the left and a matching double rule
    across the top — i.e. a classic school notebook page.
    """

    #: Vertical distance between ruled lines, in SVG units (px).
    line_height: float = 32.0
    #: Number of ruled lines drawn on the page (extra text lines are dropped).
    lines_per_page: int = 24
    #: Page height in SVG units.
    height: float = 896.0
    #: Page width in SVG units. Defaults to A4 proportions.
    width: float = field(default=896.0 * 0.707)
    #: Gap between the left page edge and where handwriting starts.
    margin_left: float = 64.0
    #: Gap between the top page edge and the first ruled line.
    margin_top: float = 96.0
    background_color: str = "white"
    margin_color: str = "red"
    rule_color: str = "lightgrey"


def render_page(
    stroke_offsets: Sequence[np.ndarray],
    lines: Sequence[str],
    layout: PageLayout | None = None,
    stroke_colors: Sequence[str] | None = None,
    stroke_widths: Sequence[float] | None = None,
) -> svgwrite.Drawing:
    """Draw one page of handwriting as an SVG.

    Args:
        stroke_offsets: One `(N, 3)` offset array per line, as returned by
            :meth:`HandwritingSynthesizer.generate`.
        lines: The text each stroke array spells out. Only used to decide where
            blank lines go: an empty string skips its ruled line, leaving it
            blank, and its stroke array is ignored.
        layout: Page geometry; defaults to :class:`PageLayout`'s A4 notebook.
        stroke_colors: Per-line pen colour (any SVG colour); defaults to black.
        stroke_widths: Per-line pen thickness in px; defaults to 2.

    Returns:
        The assembled `svgwrite.Drawing`.
    """
    layout = layout or PageLayout()
    stroke_colors = stroke_colors or ["black"] * len(lines)
    stroke_widths = stroke_widths or [2.0] * len(lines)

    drawing = svgwrite.Drawing()
    drawing.viewbox(width=layout.width, height=layout.height)
    _draw_ruled_background(drawing, layout)

    # Pen origin of the current line. The stroke coordinates produced below are
    # relative to this point; x stays at the margin while y walks down the page
    # one ruled line at a time. (Negated because stroke y is flipped and shifted
    # against this origin further down.)
    line_origin = np.array(
        [-layout.margin_left, -layout.margin_top - layout.line_height / 2]
    )

    for i, (offsets, line, color, width) in enumerate(
        zip(stroke_offsets, lines, stroke_colors, stroke_widths)
    ):
        if i >= layout.lines_per_page:
            break  # the page is full; drop the remaining lines
        if not line:
            line_origin[1] -= layout.line_height
            continue

        # Post-process the raw model output into clean, page-relative coordinates.
        coords = strokes.offsets_to_coords(offsets)
        coords = strokes.denoise(coords)
        coords[:, :2] = strokes.align(coords[:, :2])
        coords[:, 1] *= -1  # model y grows upward; SVG y grows downward
        coords[:, :2] -= coords[:, :2].min() + line_origin

        drawing.add(_strokes_to_path(coords, color, width))
        line_origin[1] -= layout.line_height

    return drawing


def save_page(
    stroke_offsets: Sequence[np.ndarray],
    lines: Sequence[str],
    svg_path: str | Path,
    layout: PageLayout | None = None,
    stroke_colors: Sequence[str] | None = None,
    stroke_widths: Sequence[float] | None = None,
    png_path: str | Path | None = None,
) -> Path:
    """Render a page (see :func:`render_page`) and write it to `svg_path`.

    If `png_path` is given, additionally rasterise the SVG to PNG (requires
    the optional `cairosvg` dependency). Returns the SVG path.
    """
    svg_path = Path(svg_path)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    drawing = render_page(stroke_offsets, lines, layout, stroke_colors, stroke_widths)
    drawing.saveas(str(svg_path))
    if png_path is not None:
        convert_to_png(svg_path, png_path)
    return svg_path


def convert_to_png(svg_path: str | Path, png_path: str | Path) -> Path:
    """Rasterise an SVG file to PNG using cairosvg.

    cairosvg is an optional dependency (`pip install .[png]`) and needs the
    system cairo library (`brew install cairo` / `apt-get install libcairo2`).
    """
    try:
        import cairosvg
    except (
        ImportError,
        OSError,
    ) as exc:  # OSError: cairosvg found but libcairo missing
        raise RuntimeError(
            "PNG export requires the optional 'png' extra and the system cairo "
            "library. Install with: pip install 'typescribe-handwriting[png]'"
        ) from exc

    png_path = Path(png_path)
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path))
    return png_path


def _draw_ruled_background(drawing: svgwrite.Drawing, layout: PageLayout) -> None:
    """Fill the page and draw the ruled lines plus the red margin rules."""
    drawing.add(
        drawing.rect(
            insert=(0, 0),
            size=(layout.width, layout.height),
            fill=layout.background_color,
        )
    )

    for i in range(layout.lines_per_page):
        y = layout.line_height * (i + 1) + layout.margin_top
        drawing.add(
            drawing.line(
                start=(0, y),
                end=(layout.width, y),
                stroke=layout.rule_color,
                stroke_width=1,
            )
        )

    # Double vertical margin rule on the left, double horizontal rule at the top.
    margin_x = layout.margin_left + layout.line_height / 2
    for x in (margin_x, margin_x - 5):
        drawing.add(
            drawing.line(
                start=(x, 0),
                end=(x, layout.height),
                stroke=layout.margin_color,
                stroke_width=1,
            )
        )
    for y in (layout.margin_top, layout.margin_top - 5):
        drawing.add(
            drawing.line(
                start=(0, y),
                end=(layout.width, y),
                stroke=layout.margin_color,
                stroke_width=1,
            )
        )


def _strokes_to_path(
    coords: np.ndarray, color: str, width: float
) -> svgwrite.path.Path:
    """Turn one line's pen coordinates into a single SVG path.

    Each stroke becomes a Move-to followed by Line-to segments; a `pen_up`
    flag on a point starts a new stroke at the next point.
    """
    commands = ["M0,0"]
    pen_was_up = 1.0
    for x, y, pen_up in coords:
        commands.append(f"{'M' if pen_was_up == 1.0 else 'L'}{x},{y}")
        pen_was_up = pen_up
    path = svgwrite.path.Path(" ".join(commands))
    return path.stroke(color=color, width=width, linecap="round").fill("none")

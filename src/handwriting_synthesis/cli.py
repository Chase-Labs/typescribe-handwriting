"""
Command-line interface: text file in, handwritten SVG pages out.

Installed as the ``typescribe`` command. This is a thin wrapper around the
package's pipeline and doubles as a usage example for building other frontends
(e.g. an HTTP API) on top of :class:`HandwritingSynthesizer`.
"""

import argparse
import os
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="typescribe",
        description="Convert a text file into pages of realistic handwriting (SVG).",
    )
    parser.add_argument("input", type=Path, help="text file to convert")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="directory for the generated pages (default: ./output)",
    )
    parser.add_argument(
        "--style",
        type=int,
        default=1,
        help="handwriting style to imitate, 0-12 (default: 1)",
    )
    parser.add_argument(
        "--bias",
        type=float,
        default=0.95,
        help="neatness of the writing, 0=wild to ~1=tidy (default: 0.95)",
    )
    parser.add_argument(
        "--stroke-width",
        type=float,
        default=1.0,
        help="pen thickness in px (default: 1)",
    )
    parser.add_argument(
        "--stroke-color",
        default="black",
        help="pen colour, any SVG colour (default: black)",
    )
    parser.add_argument(
        "--max-line-length",
        type=int,
        default=60,
        help="wrap lines at this many characters (default: 60, hard model limit 75)",
    )
    parser.add_argument(
        "--lines-per-page",
        type=int,
        default=24,
        help="ruled lines per page (default: 24)",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="also rasterise each page to PNG (requires the 'png' extra)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="directory with the pretrained model assets (default: "
        "$HANDWRITING_MODEL_DIR or ./model)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Quiet TensorFlow's C++ startup chatter; must be set before TF is imported.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from handwriting_synthesis import HandwritingSynthesizer, prepare_text

    if not args.input.exists():
        print(f"error: input file '{args.input}' does not exist", file=sys.stderr)
        return 1

    pages = prepare_text(
        args.input.read_text(encoding="utf-8"),
        max_line_length=args.max_line_length,
        lines_per_page=args.lines_per_page,
    )
    if not pages:
        print("error: input file contains no writable text", file=sys.stderr)
        return 1

    print("Loading model...")
    synthesizer = HandwritingSynthesizer(model_dir=args.model_dir)

    for page_num, lines in enumerate(pages, start=1):
        svg_path = args.output_dir / f"page_{page_num}.svg"
        synthesizer.write_page(
            lines,
            svg_path,
            biases=args.bias,
            styles=args.style,
            stroke_colors=args.stroke_color,
            stroke_widths=args.stroke_width,
            png_path=svg_path.with_suffix(".png") if args.png else None,
        )
        print(f"Page {page_num}/{len(pages)} written to {svg_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

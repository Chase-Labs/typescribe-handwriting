"""Convert text into realistic handwriting, rendered as SVG.

The pipeline has three stages, each with its own module:

1. :mod:`~handwriting_synthesis.preprocessing` — sanitize/wrap/paginate raw text
   into model-ready lines (pure string handling).
2. :class:`HandwritingSynthesizer` — run the pretrained RNN to turn lines into
   pen strokes (the ML part, wrapped so you don't need to touch TensorFlow).
3. :mod:`~handwriting_synthesis.rendering` — draw the strokes onto a ruled SVG
   page (pure drawing code).

Quick start::

    from handwriting_synthesis import HandwritingSynthesizer, prepare_text

    synthesizer = HandwritingSynthesizer()          # load the model once
    pages = prepare_text(open("input.txt").read())
    for i, lines in enumerate(pages):
        synthesizer.write_page(lines, f"out/page_{i + 1}.svg", styles=1)
"""

from .preprocessing import prepare_text
from .rendering import PageLayout, render_page, save_page

__version__ = "2.0.0"

# The synthesizer pulls in TensorFlow, which is slow to import and not needed
# for the preprocessing/rendering stages — load it lazily on first access.
_LAZY_IMPORTS = {
    "HandwritingSynthesizer": "synthesizer",
    "MAX_LINE_LENGTH": "synthesizer",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        from importlib import import_module

        return getattr(import_module(f".{_LAZY_IMPORTS[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "HandwritingSynthesizer",
    "PageLayout",
    "prepare_text",
    "render_page",
    "save_page",
    "MAX_LINE_LENGTH",
    "__version__",
]

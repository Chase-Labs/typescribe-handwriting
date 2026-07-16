"""
Stage 1 of the pipeline: turn free-form text into model-ready lines.

The model generates handwriting one line at a time and only understands a fixed
character set (:data:`~handwriting_synthesis.alphabet.ALPHABET`), so raw text
needs three things done to it before inference:

1. :func:`sanitize` — replace characters the model cannot draw with spaces.
2. :func:`wrap` — break text into lines short enough for the model.
3. :func:`paginate` — group lines into pages for rendering.

:func:`prepare_text` composes all three. This module is pure string manipulation
with no ML involved, so it is safe to modify freely.
"""

from . import alphabet


def sanitize(text: str) -> str:
    """
    Replace characters the model cannot draw with alternatives

    >>> sanitize("Héllo, wörld! Here is a QR code!")
    'Hello, world! Here is a qr code!
    """
    _text = text.replace("QR", "qr").translate(alphabet.TRANSLATION_TABLE)
    return "".join(char if alphabet.is_supported(char) else " " for char in _text)


def wrap(text: str, max_line_length: int) -> list[str]:
    """
    Greedily wrap text into lines of at most `max_line_length` characters.

    Wrapping happens on word boundaries. Existing newlines are respected as hard
    breaks; blank lines are dropped. A single word longer than `max_line_length`
    is kept on its own line rather than split (the synthesizer will reject it if
    it exceeds the model's hard 75-character limit).
    """
    wrapped: list[str] = []
    for paragraph in text.splitlines():
        words = paragraph.split()
        if not words:
            wrapped.append(".")  # Use '.' for a blank line
            continue
        current = words[0]
        for word in words[1:]:
            if len(current) + 1 + len(word) > max_line_length:
                wrapped.append(current)
                current = word
            else:
                current += " " + word
        wrapped.append(current)
    return wrapped


def paginate(lines: list[str], lines_per_page: int) -> list[list[str]]:
    """
    Split a flat list of lines into pages of at most `lines_per_page` lines.
    """
    return [lines[i : i + lines_per_page] for i in range(0, len(lines), lines_per_page)]


def prepare_text(
    text: str, max_line_length: int, lines_per_page: int
) -> list[list[str]]:
    """
    Sanitize, wrap and paginate raw text into model-ready pages of lines.

    Returns a list of pages, where each page is a list of lines that can be fed
    directly to :meth:`HandwritingSynthesizer.write_page`.
    """
    return paginate(wrap(sanitize(text), max_line_length), lines_per_page)

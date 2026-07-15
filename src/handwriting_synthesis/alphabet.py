"""
The character set the handwriting model understands.

The RNN was trained on a fixed set of 73 characters (the IAM online handwriting
dataset's most common ones). Anything outside this set cannot be drawn: input text
must be sanitised first (see :mod:`handwriting_synthesis.preprocessing`), and text
is converted to the integer ids the network expects via :func:`encode`.

Note the quirks of the trained character set: there is no `Q`, `X` or `Z`,
and only a handful of punctuation marks are supported.
"""

import numpy as np

#: Characters the model can draw, in training order. The position of each character
#: in this tuple IS its integer id, so the order must never change: id 0 (`"\x00"`)
#: doubles as the padding/end-of-text token.
ALPHABET: tuple[str, ...] = (
    "\x00",
    " ",
    "!",
    '"',
    "#",
    "'",
    "(",
    ")",
    ",",
    "-",
    ".",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    ":",
    ";",
    "?",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "Y",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
)

_CHAR_TO_ID: dict[str, int] = {char: i for i, char in enumerate(ALPHABET)}


def is_supported(char: str) -> bool:
    """
    Return True if the model can draw `char`.
    """
    return char in _CHAR_TO_ID


def unsupported_chars(text: str) -> set[str]:
    """
    Return the set of characters in `text` the model cannot draw.
    """
    return {char for char in text if char not in _CHAR_TO_ID}


def encode(text: str) -> np.ndarray:
    """
    Convert text to the int array the network consumes.

    Unknown characters map to id 0, and a terminating 0 ("end of text") is appended,
    mirroring how the model was trained.
    """
    return np.array([_CHAR_TO_ID.get(char, 0) for char in text] + [0])

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

TRANSLATION_TABLE = str.maketrans(
    {
        # --- uppercase letters missing from the alphabet ---
        "Q": "q",
        "X": "x",
        "Z": "z",
        # --- dashes and hyphens ---
        "‑": "-",  # ‑ non-breaking hyphen
        "‒": "-",  # ‒ figure dash
        "–": "-",  # – en dash
        "—": "-",  # — em dash
        "―": "-",  # ― horizontal bar
        "−": "-",  # − minus sign
        # --- single quotes / apostrophes ---
        "‘": "'",  # '
        "’": "'",  # '
        "‚": "'",  # ‚
        "‛": "'",  # ‛
        "′": "'",  # ′ prime
        "ʼ": "'",  # ʼ modifier apostrophe
        "‹": "'",  # ‹
        "›": "'",  # ›
        "`": "'",
        "´": "'",  # ´ acute accent
        # --- double quotes ---
        "“": '"',  # "
        "”": '"',  # "
        "„": '"',  # „
        "‟": '"',  # ‟
        "″": '"',  # ″ double prime
        "«": '"',  # «
        "»": '"',  # »
        # --- spaces and invisibles ---
        " ": " ",  # no-break space
        " ": " ",
        " ": " ",
        " ": " ",
        " ": " ",
        " ": " ",
        " ": " ",
        " ": " ",
        " ": " ",
        " ": " ",
        " ": " ",  # narrow no-break space
        " ": " ",  # medium mathematical space
        "　": " ",  # ideographic space
        "	": " ",
        # --- other punctuation ---
        "…": "...",  # … ellipsis
        "¡": "!",  # ¡
        "¿": "?",  # ¿
        "‽": "?!",  # ‽ interrobang
        "•": "-",  # • bullet
        "·": ".",  # · middle dot
        "×": "x",  # × multiplication sign
        # --- uppercase accented Latin letters ---
        "À": "A",
        "Á": "A",
        "Â": "A",
        "Ã": "A",
        "Ä": "A",
        "Å": "A",
        "Ā": "A",
        "Ă": "A",
        "Ą": "A",
        "Ç": "C",
        "Ć": "C",
        "Č": "C",
        "È": "E",
        "É": "E",
        "Ê": "E",
        "Ë": "E",
        "Ē": "E",
        "Ė": "E",
        "Ę": "E",
        "Ì": "I",
        "Í": "I",
        "Î": "I",
        "Ï": "I",
        "Ī": "I",
        "İ": "I",
        "Ñ": "N",
        "Ń": "N",
        "Ò": "O",
        "Ó": "O",
        "Ô": "O",
        "Õ": "O",
        "Ö": "O",
        "Ø": "O",
        "Ō": "O",
        "Ù": "U",
        "Ú": "U",
        "Û": "U",
        "Ü": "U",
        "Ū": "U",
        "Ů": "U",
        "Ý": "Y",
        "Ÿ": "Y",
        "Ĝ": "G",
        "Ğ": "G",
        "Ł": "L",
        "Ś": "S",
        "Š": "S",
        "Ź": "z",
        "Ż": "z",
        "Ž": "z",  # no uppercase Z in the alphabet
        "Ð": "D",
        "Đ": "D",
        "Æ": "AE",
        "Œ": "OE",
        "Þ": "Th",
        # --- lowercase accented Latin letters ---
        "à": "a",
        "á": "a",
        "â": "a",
        "ã": "a",
        "ä": "a",
        "å": "a",
        "ā": "a",
        "ă": "a",
        "ą": "a",
        "ç": "c",
        "ć": "c",
        "č": "c",
        "è": "e",
        "é": "e",
        "ê": "e",
        "ë": "e",
        "ē": "e",
        "ė": "e",
        "ę": "e",
        "ì": "i",
        "í": "i",
        "î": "i",
        "ï": "i",
        "ī": "i",
        "ı": "i",
        "ñ": "n",
        "ń": "n",
        "ò": "o",
        "ó": "o",
        "ô": "o",
        "õ": "o",
        "ö": "o",
        "ø": "o",
        "ō": "o",
        "ù": "u",
        "ú": "u",
        "û": "u",
        "ü": "u",
        "ū": "u",
        "ů": "u",
        "ý": "y",
        "ÿ": "y",
        "ĝ": "g",
        "ğ": "g",
        "ł": "l",
        "ś": "s",
        "š": "s",
        "ź": "z",
        "ż": "z",
        "ž": "z",
        "ð": "d",
        "đ": "d",
        "æ": "ae",
        "œ": "oe",
        "ß": "ss",
        "þ": "th",
        # --- ligatures ---
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
    }
)


_CHAR_TO_ID: dict[str, int] = {char: i for i, char in enumerate(ALPHABET)}


def is_supported(char: str) -> bool:
    """
    Return True if the model can draw `char`.
    """
    return char in _CHAR_TO_ID or char == "\n"


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

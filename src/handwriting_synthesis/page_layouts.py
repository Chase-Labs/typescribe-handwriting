from dataclasses import dataclass
from enum import StrEnum


class PageLayoutEnum(StrEnum):
    """
    Names of the built-in page layouts.
    """

    A4 = "A4"
    A5 = "A5"


@dataclass(frozen=True)
class PageLayout:
    """
    Geometry and colours of the rendered page.

    The defaults describe an A4-proportioned page (0.707 width/height ratio)
    with 24 ruled lines, a red margin down the left and a matching double rule
    across the top — i.e. a classic school notebook page.
    """

    # Vertical distance between ruled lines, in SVG units (px).
    line_height: float = 34.0

    # Number of ruled lines drawn on the page (extra text lines are dropped).
    lines_per_page: int = 28

    # The total number of characters per line.
    # The hard model limit is 75 characters per line
    max_line_length: int = 64

    # Page height in SVG units.
    height: float = 1122.5

    # Page width in SVG units. Defaults to A4 proportions.
    width: float = 793.7

    # Gap between the left page edge and where handwriting starts.
    margin_left: float = 64.0

    # Gap between the top page edge and the first ruled line.
    margin_top: float = 72.0

    background_color: str = "white"
    margin_color: str = "white"
    rule_color: str = "white"


def get_page_layout(layout: PageLayoutEnum | str):
    """
    Return a :class:`PageLayout` object corresponding to the given layout name.
    """
    _layout = PageLayoutEnum(layout.upper()) if isinstance(layout, str) else layout
    match _layout:
        case PageLayoutEnum.A4:
            return PageLayout()

        case PageLayoutEnum.A5:
            return PageLayout(
                line_height=27.0,
                lines_per_page=24,
                max_line_length=54,
                height=793.7,
                width=559.4,
                margin_left=32.0,
                margin_top=36.0,
            )
        case _:
            raise ValueError(f"Unknown page layout: {layout}")

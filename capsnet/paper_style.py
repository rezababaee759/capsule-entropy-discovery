"""Page geometry and type sizes of the paper's figures. Every figure is drawn at the
width it is printed at (scale 1), so the sizes here are the sizes a reader sees;
``save`` never crops, since cropping would change that scale."""

from __future__ import annotations

from pathlib import Path

import matplotlib

PT_PER_IN = 72.27  # TeX points
TEXT_WIDTH_IN = 390.0 / PT_PER_IN  # 5.396
TEXT_HEIGHT_IN = 548.5 / PT_PER_IN  # 7.590
DPI = 600

# Type scale; SMALL_PT is the floor.
BASE_PT = 8  # axis labels, legends, panel titles
SMALL_PT = 7  # tick labels, in-plot annotations, footnotes
TITLE_PT = 9  # a figure-level title, where one is kept


def apply() -> None:
    """Set matplotlib defaults so any text not given an explicit size is legible."""
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": BASE_PT,
            "axes.titlesize": BASE_PT,
            "axes.titleweight": "bold",
            "axes.labelsize": BASE_PT,
            "xtick.labelsize": SMALL_PT,
            "ytick.labelsize": SMALL_PT,
            "legend.fontsize": SMALL_PT,
            "legend.title_fontsize": SMALL_PT,
            "figure.titlesize": TITLE_PT,
            "figure.titleweight": "bold",
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.0,
            "patch.linewidth": 0.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "grid.linewidth": 0.4,
            "savefig.dpi": DPI,
            "figure.dpi": 100,
        }
    )


def figure(height_in: float, width_in: float = TEXT_WIDTH_IN, **kwargs):
    """A figure at print size, with constrained layout so nothing is clipped."""
    import matplotlib.pyplot as plt

    kwargs.setdefault("layout", "constrained")
    return plt.figure(figsize=(width_in, height_in), **kwargs)


def subplots(
    nrows: int, ncols: int, height_in: float, width_in: float = TEXT_WIDTH_IN, **kwargs
):
    import matplotlib.pyplot as plt

    kwargs.setdefault("layout", "constrained")
    return plt.subplots(nrows, ncols, figsize=(width_in, height_in), **kwargs)


def save(fig, path: str | Path) -> Path:
    """Write at DPI without cropping, so the canvas is exactly the print size."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    return path

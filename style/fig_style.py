# ============================================================
# PRD-style Matplotlib configuration
# Import this module once per session
# ============================================================

import matplotlib as mpl
from cycler import cycler
from matplotlib.ticker import LogLocator, NullFormatter

# ------------------------------------------------------------
# Colorblind-friendly palette (Okabe–Ito, APS-safe)
# ------------------------------------------------------------
COLORBLIND_CYCLE = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]

# ------------------------------------------------------------
# Global rcParams
# ------------------------------------------------------------
mpl.rcParams.update({
    # LaTeX rendering
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",

    # Fonts (Computer Modern / PRD-like)
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "mathtext.fontset": "cm",

    # Font sizes
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,

    # Axes and ticks
    "axes.linewidth": 1.0,
    "axes.prop_cycle": cycler(color=COLORBLIND_CYCLE),
    "axes.grid": True,

    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,

    # Grid (faint, PRD style)
    "grid.alpha": 0.30,
    "grid.linewidth": 0.6,
    "grid.linestyle": "--",

    # Figure defaults
    "figure.figsize": (6.5, 4.5),
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ------------------------------------------------------------
# Helper for log axes with proper ticks + grids
# ------------------------------------------------------------
def set_log_axes(ax, xlog=False, ylog=False):
    """Apply logarithmic axis scaling and compatible minor/major grid styling.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to style.
    xlog : bool, optional
        If True, use a logarithmic x-axis.
    ylog : bool, optional
        If True, use a logarithmic y-axis.

    Returns
    -------
    None
        The input axes are updated in place.
    """
    if xlog:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs="auto"))
        ax.xaxis.set_minor_formatter(NullFormatter())

    if ylog:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs="auto"))
        ax.yaxis.set_minor_formatter(NullFormatter())

    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.15)

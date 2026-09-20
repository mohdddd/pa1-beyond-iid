"""Consistent figure style; saves PDF (for LaTeX) and PNG (quick viewing)."""
import matplotlib
import matplotlib.pyplot as plt

from common.paths import FIG_DIR


def setup_style() -> None:
    matplotlib.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def savefig(fig, name: str, subdir: str | None = None) -> None:
    out = FIG_DIR / subdir if subdir else FIG_DIR
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)

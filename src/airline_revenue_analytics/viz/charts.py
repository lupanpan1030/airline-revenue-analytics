"""Matplotlib-only helpers to produce consistent figures. (仅使用 Matplotlib 的图表工具，保证风格一致)"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class PlotColors:
    primary: str = "#2C7FB8"
    accent: str = "#D95F0E"
    muted: str = "#8C8C8C"
    grid: str = "#E6E6E6"


PLOT_COLORS = PlotColors()
PLOT_PALETTE = (
    PLOT_COLORS.primary,
    PLOT_COLORS.accent,
    "#2CA25F",
    "#9ECAE1",
)


def apply_style() -> None:
    """Apply a consistent Matplotlib style across figures."""
    mpl.rcParams.update(
        {
            "figure.figsize": (6.4, 4.0),
            "axes.grid": True,
            "grid.color": PLOT_COLORS.grid,
            "grid.alpha": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "font.family": "DejaVu Sans",
            "axes.prop_cycle": cycler("color", PLOT_PALETTE),
        }
    )


def _ensure_dir(p: str | Path):
    """Create parent directories for an output path. (为输出路径创建父目录)"""
    Path(p).parent.mkdir(parents=True, exist_ok=True)


def hist(
    series: pd.Series,
    title: str,
    outfile: str,
    bins: int = 50,
    xlabel: str = "",
    show: bool = False,
):
    """Render and save a histogram. (绘制并保存直方图)"""
    apply_style()
    fig = plt.figure()
    plt.hist(series.dropna(), bins=bins, color=PLOT_COLORS.primary)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    _ensure_dir(outfile)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    if show:
        plt.show()
    # Close to prevent figure accumulation / 关闭图像以避免内存堆积
    plt.close(fig)


def scatter(
    x,
    y,
    title: str,
    outfile: str,
    xlabel: str = "",
    ylabel: str = "",
    show: bool = False,
):
    """Render and save a scatter plot. (绘制并保存散点图)"""
    apply_style()
    fig = plt.figure()
    plt.scatter(x, y, s=12, alpha=0.65, color=PLOT_COLORS.primary, edgecolors="none")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    _ensure_dir(outfile)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    if show:
        plt.show()
    # Close to prevent figure accumulation / 关闭图像以避免内存堆积
    plt.close(fig)


def box(
    data,
    labels,
    title: str,
    outfile: str,
    ylabel: str = "",
    show: bool = False,
):
    """Render and save a box plot. (绘制并保存箱线图)"""
    apply_style()
    fig = plt.figure()
    plt.boxplot(
        data,
        labels=labels,
        showfliers=False,
        boxprops={"color": PLOT_COLORS.primary},
        medianprops={"color": PLOT_COLORS.accent},
        whiskerprops={"color": PLOT_COLORS.primary},
        capprops={"color": PLOT_COLORS.primary},
    )
    plt.title(title)
    plt.ylabel(ylabel)
    _ensure_dir(outfile)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    if show:
        plt.show()
    # Close to prevent figure accumulation / 关闭图像以避免内存堆积
    plt.close(fig)

# plot_lines_pdf.py
# Flexible, LaTeX-friendly line-plot utility.
# - Saves ONE PDF per panel (subplot).
# - Legend can be (a) a separate legend.pdf or (b) embedded into a chosen panel.
# - Creates save_directory/subdir_name and writes all PDFs there.
#
# Requirements: matplotlib (always), seaborn (optional for colorblind palette).

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Sequence, Optional, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

# Optional: color-blind friendly palettes via seaborn

import seaborn as sns
_HAS_SNS = True



# ----------------------------- Data container -----------------------------

@dataclass
class Panel:
    """
    Data + per-panel settings.
    ys must be a list of sequences, one per series (match series_labels length).
    """
    x: Sequence[float]
    ys: Sequence[Sequence[float]]
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None
    xticks: Optional[Sequence[float]] = None
    yticks: Optional[Sequence[float]] = None
    filename: Optional[str] = None  # if None, auto-named like panel1.pdf


# ----------------------------- Styling helpers -----------------------------

def _get_palette(palette: Optional[str], n: int) -> List[str]:
    """
    Returns n colors.
    - palette=None or 'colorblind' -> seaborn colorblind (if seaborn is present), else tab10
    - palette='tab10' -> Matplotlib Tab10
    - palette='C0,#ff7f0e,#2ca02c' -> custom CSV of color specs
    - palette can be any seaborn palette name if seaborn is installed
    """
    if isinstance(palette, str) and "," in palette:
        cols = [c.strip() for c in palette.split(",")]
        if len(cols) >= n:
            return cols[:n]

    if _HAS_SNS and (palette is None or palette == "colorblind"):
        return sns.color_palette("colorblind", n_colors=n).as_hex()

    if palette in (None, "tab10"):
        return list(plt.cm.tab10.colors)[:n]

    if _HAS_SNS:
        try:
            return sns.color_palette(palette, n_colors=n).as_hex()
        except Exception:
            pass

    return list(plt.cm.tab10.colors)[:n]


def _style_axes(ax: plt.Axes,
                grid_alpha: float = 0.25,
                spine_alpha: float = 0.3,
                tick_labelsize: float = 10.0):
    ax.set_axisbelow(True)
    ax.grid(True, axis="both", color="0.7", alpha=grid_alpha, linewidth=0.8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["bottom", "left"]:
        ax.spines[s].set_alpha(spine_alpha)
    ax.tick_params(axis="both", labelsize=tick_labelsize)


def _legend_handles(colors: List[str], marker: str, line_width: float, marker_size: float, line_styles: Optional[Sequence[str]] = None):
    handles = []
    for i, c in enumerate(colors):
        ls = None
        if line_styles:
            ls = line_styles[i % len(line_styles)]
        handles.append(plt.Line2D([0], [0], color=c, marker=marker, linewidth=line_width, markersize=marker_size, linestyle=ls if ls else 'solid'))
    return handles


def save_legend_pdf(labels: Sequence[str],
                    colors: Sequence[str],
                    out_path: str,
                    marker: str = "o",
                    line_width: float = 2.2,
                    marker_size: float = 4.6,
                    ncol: Optional[int] = None,
                    fontsize: float = 12.0,
                    frame: bool = False,
                    handlelength: float = 2.0,
                    columnspacing: float = 1.2,
                    legend_title: Optional[str] = None,
                    title_fontsize: Optional[float] = None,
                    line_styles: Optional[Sequence[str]] = None):
    """
    Standalone legend PDF (tight bounding box). Useful to place separately in LaTeX.
    """
    handles = _legend_handles(list(colors), marker, line_width, marker_size, line_styles=line_styles)
    fig = plt.figure(figsize=(6, 0.5))
    leg = fig.legend(handles, labels,
                     loc="center",
                     ncol=ncol or len(labels),
                     frameon=frame,
                     fontsize=fontsize,
                     handlelength=handlelength,
                     columnspacing=columnspacing,
                     title=legend_title)
    if legend_title and title_fontsize:
        leg.get_title().set_fontsize(title_fontsize)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ----------------------------- Main entry point -----------------------------

def generate_line_plots(
    panels: Sequence[Panel],
    series_labels: Sequence[str],
    save_directory: str,
    subdir_name: str,
    # How many plots to emit
    n_plots: Optional[int] = 3,
    # Legend mode:
    #   'separate' -> write legend.pdf
    #   'panel'    -> embed legend into the chosen panel's PDF only
    #   'none'     -> no legend output
    legend_mode: str = "separate",
    legend_panel_index: int = 0,            # used when legend_mode == 'panel'
    # Where to put legend when legend_mode == 'panel'
    # 'top'    -> above axes
    # 'inside' -> inside the axes with loc='best'
    # 'right'/'bottom' -> outside the axes anchored to that side
    panel_legend_where: str = "top",
    # Shared visual style knobs
    palette: Optional[str] = None,          # None or 'colorblind' uses seaborn if available
    line_width: float = 2.2,
    marker: str = "o",
    marker_size: float = 4.8,
    alpha: float = 1.0,
    # Fonts
    base_fontsize: float = 11.0,
    title_size: float = 12.0,
    label_size: float = 11.0,
    tick_size: float = 10.0,
    # Legend styling
    legend_fontsize: float = 12.0,
    legend_ncol: Optional[int] = None,
    legend_frame: bool = False,
    legend_handlelength: float = 2.0,
    legend_columnspacing: float = 1.2,
    legend_title: Optional[str] = None,
    legend_title_fontsize: Optional[float] = None,
    # Axes cosmetics
    grid_alpha: float = 0.25,
    spine_alpha: float = 0.3,
    # Per-series customizations
    line_styles: Optional[Sequence[str]] = None,  # e.g., ['solid','dashed',...]; cycles if shorter
    colors_override: Optional[Sequence[str]] = None,  # explicit colors matching series count
    zorders: Optional[Sequence[int]] = None,  # per-series zorder so later overlapping control
) -> Dict[str, Any]:
    """
    Creates PDFs in save_directory/subdir_name.
    Returns dict with:
        {
          "output_dir": <path>,
          "panel_paths": [...],
          "legend_path": <path or None>,
          "colors": [...]
        }
    """
    assert len(panels) > 0, "At least one Panel is required."
    n_series = len(series_labels)
    for p in panels:
        assert len(p.ys) == n_series, "Each Panel.ys must have one series per series_labels."

    # Prepare output directory
    out_dir = os.path.join(save_directory, subdir_name)
    os.makedirs(out_dir, exist_ok=True)

    # Colors + font sizes
    colors = list(colors_override) if colors_override is not None else _get_palette(palette, n_series)
    plt.rcParams.update({
        "font.size": base_fontsize,
        "axes.labelsize": label_size,
        "axes.titlesize": title_size,
    })

    # Truncate panels if needed
    if n_plots is None:
        n_plots = len(panels)
    use_panels = list(panels)[:n_plots]

    panel_paths: List[str] = []
    legend_path: Optional[str] = None

    # Build proxy handles once (consistent legend style)
    legend_handles = _legend_handles(colors, marker, line_width, marker_size, line_styles=line_styles)

    for idx, p in enumerate(use_panels):
        x = np.asarray(p.x)

        # Per-panel figure
        fig, ax = plt.subplots(1, 1, figsize=(4.2, 2.6))  # size tuned for papers; adjust as desired

        # Style & plot
        _style_axes(ax, grid_alpha=grid_alpha, spine_alpha=spine_alpha, tick_labelsize=tick_size)
        for i, y in enumerate(p.ys):
            y = np.asarray(y)
            ls = None
            if line_styles:
                ls = line_styles[i % len(line_styles)]
            z = None
            if zorders:
                z = zorders[i % len(zorders)]
            ax.plot(x, y,
                    color=colors[i % len(colors)],
                    marker=marker,
                    linewidth=line_width,
                    markersize=marker_size,
                    alpha=alpha,
                    linestyle=ls if ls else 'solid',
                    zorder=z if z is not None else 2)

        if p.title:
            ax.set_title(p.title, pad=6)
        if p.xlabel:
            ax.set_xlabel(p.xlabel)
        if p.ylabel:
            ax.set_ylabel(p.ylabel)
        if p.xlim:   ax.set_xlim(*p.xlim)
        if p.ylim:   ax.set_ylim(*p.ylim)
        if p.xticks is not None: ax.set_xticks(p.xticks)
        if p.yticks is not None: ax.set_yticks(p.yticks)

        # Embed legend in THIS panel's PDF if requested and this is the chosen panel
        if legend_mode == "panel" and idx == legend_panel_index:
            labels = list(series_labels)
            if panel_legend_where == "inside":
                leg = ax.legend(legend_handles, labels, loc="best",
                                frameon=legend_frame, fontsize=legend_fontsize,
                                ncol=legend_ncol or 1, handlelength=legend_handlelength,
                                columnspacing=legend_columnspacing, title=legend_title)
                if legend_title and legend_title_fontsize:
                    leg.get_title().set_fontsize(legend_title_fontsize)
            elif panel_legend_where == "right":
                leg = fig.legend(legend_handles, labels, loc="center left",
                                 bbox_to_anchor=(1.01, 0.5), borderaxespad=0.0,
                                 frameon=legend_frame, fontsize=legend_fontsize,
                                 ncol=legend_ncol or 1, handlelength=legend_handlelength,
                                 columnspacing=legend_columnspacing, title=legend_title)
                if legend_title and legend_title_fontsize:
                    leg.get_title().set_fontsize(legend_title_fontsize)
            elif panel_legend_where == "bottom":
                leg = fig.legend(legend_handles, labels, loc="lower center",
                                 bbox_to_anchor=(0.5, -0.02), borderaxespad=0.0,
                                 frameon=legend_frame, fontsize=legend_fontsize,
                                 ncol=legend_ncol or len(labels), handlelength=legend_handlelength,
                                 columnspacing=legend_columnspacing, title=legend_title)
                if legend_title and legend_title_fontsize:
                    leg.get_title().set_fontsize(legend_title_fontsize)
            else:  # 'top' (default)
                leg = fig.legend(legend_handles, labels, loc="upper center",
                                 bbox_to_anchor=(0.5, 1.02), borderaxespad=0.0,
                                 frameon=legend_frame, fontsize=legend_fontsize,
                                 ncol=legend_ncol or len(labels), handlelength=legend_handlelength,
                                 columnspacing=legend_columnspacing, title=legend_title)
                if legend_title and legend_title_fontsize:
                    leg.get_title().set_fontsize(legend_title_fontsize)
                # Make room above axes
                fig.subplots_adjust(top=0.80)

        # Save panel
        panel_name = p.filename or f"panel{idx+1}.pdf"
        panel_path = os.path.join(out_dir, panel_name)
        fig.savefig(panel_path, bbox_inches="tight")
        plt.close(fig)
        panel_paths.append(panel_path)

    # Separate legend.pdf if requested
    if legend_mode == "separate":
        legend_path = os.path.join(out_dir, "legend.pdf")
        save_legend_pdf(
            labels=series_labels,
            colors=colors,
            out_path=legend_path,
            marker=marker,
            line_width=line_width,
            marker_size=marker_size,
            ncol=legend_ncol or len(series_labels),
            fontsize=legend_fontsize,
            frame=legend_frame,
            handlelength=legend_handlelength,
            columnspacing=legend_columnspacing,
            legend_title=legend_title,
            title_fontsize=legend_title_fontsize,
            line_styles=line_styles,
        )

    return {
        "output_dir": out_dir,
        "panel_paths": panel_paths,
        "legend_path": legend_path,
        "colors": colors
    }


# ----------------------------- Minimal demo -----------------------------
# Run this file directly to see sample outputs under ./demo_plots/my_figure

if __name__ == "__main__":
    # Fake data for the demo
    x = np.array([0, 50, 100, 200, 300, 400])

    def mk(seed, base):
        rng = np.random.default_rng(seed)
        return base + np.cumsum(rng.normal(0.0, 0.007, size=x.size))

    labels = ["8K max tokens", "16K max tokens", "32K max tokens"]

    panels = [
        Panel(x, [mk(0, 0.40), mk(1, 0.49), mk(2, 0.50)],
              xlabel="RL Steps", ylabel="Accuracy", filename="accuracy_panel1.pdf"),
        Panel(x, [mk(3, 0.30), mk(4, 0.33), mk(5, 0.34)],
              xlabel="RL Steps", ylabel="", filename="accuracy_panel2.pdf"),
        Panel(x, [mk(6, 0.32), mk(7, 0.35), mk(8, 0.36)],
              xlabel="RL Steps", ylabel="", filename="accuracy_panel3.pdf"),
    ]

    palette_csv = None

    # Example A: separate legend.pdf + three panels
    generate_line_plots(
        panels=panels,
        series_labels=labels,
        save_directory="demo_plots",
        subdir_name="my_figure",
        n_plots=3,
        legend_mode="separate",              # -> writes legend.pdf
        palette=palette_csv,                 # try None/'colorblind'/'tab10'/seaborn name
        base_fontsize=11, title_size=12, label_size=11, tick_size=10,
        line_width=2.2, marker="o", marker_size=4.6
    )

    # Example B: put legend into panel 1 (index 0), at the top of that panel
    generate_line_plots(
        panels=panels,
        series_labels=labels,
        save_directory="demo_plots",
        subdir_name="my_figure_with_panel_legend",
        n_plots=3,
        legend_mode="panel",                 # -> embeds legend into chosen panel's PDF
        legend_panel_index=0,
        panel_legend_where="top",            # 'top' (outside), 'inside', 'right', 'bottom'
        palette="colorblind",
        line_width=2.2, marker="o", marker_size=4.6
    )

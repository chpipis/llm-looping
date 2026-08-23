"""Plot (accuracy, looping_count, avg_response_length) vs temperature from a
collection of trained-run directories.

Each `--run` argument supplies one curve: a directory that contains a
`test_summaries.csv` written by `run_loop_stargraph_validation.py`, optionally
followed by a series label. Multiple `--run` flags can be repeated to overlay
several series on the same axes.

Examples:

  # Section 3 hardness comparison, pairing exploration variants.
  python plotting/plot_validation_curves.py \\
      --run out_random_walk_with_reset_5_5_12_layer_model            "G(5,5)" \\
      --run out_random_walk_with_reset_explore_5_5_12_layer_model    "G(5,5) w/exp" \\
      --run out_random_walk_with_reset_5_3_12_layer_model            "G(5,3)" \\
      --run out_random_walk_with_reset_explore_5_3_12_layer_model    "G(5,3) w/exp" \\
      --pair-suffix " w/exp" \\
      --out plotting/figures/section3_hardness

  # Appendix F looping-bias overlay: base solid, bias dashed.
  python plotting/plot_validation_curves.py \\
      --run out_random_walk_with_reset_5_5_12_layer_model                       "G(5,5)" \\
      --run out_random_walk_with_reset_explore_5_5_12_layer_model               "G(5,5) w/exp" \\
      --run out_random_walk_with_reset_5_5_12_layer_model_looping_0.001         "G(5,5) w/bias" \\
      --run out_random_walk_with_reset_explore_5_5_12_layer_model_looping_0.001 "G(5,5) w/exp w/bias" \\
      --pair-suffix " w/bias" \\
      --out plotting/figures/appendix_f_looping_bias

Outputs three PDFs into `--out`: `accuracy.pdf`, `looping_count.pdf`,
`avg_response_length.pdf`, plus `joined_test_summaries.csv` for downstream
inspection.

If `--pair-suffix SUFFIX` is set, series whose labels end in `SUFFIX` are
paired with the stripped-suffix base: the base draws solid, the suffixed
variant draws dashed in the same colour. `--overlay-bias` is kept as a
backwards-compat alias for `--pair-suffix " w/bias"`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import seaborn as sns

# Allow direct invocation: `python plotting/plot_validation_curves.py ...`
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from utils import Panel, generate_line_plots  # noqa: E402


METRICS: List[Tuple[str, str, str]] = [
    # (csv column, panel y-label, output filename)
    ("accuracy", "accuracy", "accuracy.pdf"),
    ("looping_count", "avg looping count", "looping_count.pdf"),
    ("avg_response_length", "avg response length", "avg_response_length.pdf"),
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _resolve_run_csv(run_dir: Path) -> Path:
    """Return the test_summaries.csv path for a run directory.

    If `run_dir` itself contains a test_summaries.csv, use that. Otherwise look
    for exactly one `run*/test_summaries.csv` under it (the layout produced by
    the canonical training+validation scripts).
    """
    direct = run_dir / "test_summaries.csv"
    if direct.is_file():
        return direct
    candidates = sorted(run_dir.glob("run*/test_summaries.csv"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise FileNotFoundError(
            f"{run_dir} contains multiple run*/test_summaries.csv files; pass a more specific path:\n  "
            + "\n  ".join(str(c) for c in candidates)
        )
    raise FileNotFoundError(f"No test_summaries.csv found in or under {run_dir}")


def load_runs(run_dirs: Sequence[Tuple[Path, str]]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for run_dir, label in run_dirs:
        csv_path = _resolve_run_csv(run_dir)
        df = pd.read_csv(csv_path)
        missing = [m for (m, _, _) in METRICS if m not in df.columns]
        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns {missing}; "
                "regenerate with run_loop_stargraph_validation.py."
            )
        df = df[["temperature", *[m for (m, _, _) in METRICS]]].copy()
        df["series_label"] = label
        df["source_csv"] = str(csv_path)
        frames.append(df)
    joined = pd.concat(frames, ignore_index=True)

    dup_mask = joined.duplicated(subset=["series_label", "temperature"], keep=False)
    if dup_mask.any():
        dups = joined.loc[dup_mask, ["series_label", "temperature", "source_csv"]].sort_values(
            ["series_label", "temperature"]
        )
        raise ValueError(
            "Duplicate (series_label, temperature) rows. Each series_label must come from a "
            "single run dir.\n" + dups.to_string(index=False)
        )
    return joined


# ---------------------------------------------------------------------------
# Panel + styling
# ---------------------------------------------------------------------------


def build_panels(joined: pd.DataFrame, ordered_labels: List[str]) -> List[Panel]:
    panels: List[Panel] = []
    all_temps = sorted({float(t) for t in joined["temperature"].unique()})
    x = np.array(all_temps, dtype=float)
    for metric, ylabel, fname in METRICS:
        ys: List[List[float]] = []
        for label in ordered_labels:
            sub = joined[joined["series_label"] == label].sort_values("temperature")
            t2v = {float(t): v for t, v in zip(sub["temperature"], sub[metric])}
            ys.append([t2v.get(tt, np.nan) for tt in all_temps])
        panels.append(Panel(x=x, ys=ys, xlabel="temperature", ylabel=ylabel, filename=fname))
    return panels


def _style_for_pair_suffix(labels: List[str], suffix: str) -> Tuple[List[str], List[str], List[int]]:
    """Pair labels by stripping `suffix`. Each pair shares a colour; the
    suffixed variant uses a dashed line drawn on top of its base sibling.
    Labels that don't end in `suffix` each get their own colour.
    """
    bases: List[str] = []
    for lbl in labels:
        base = lbl[: -len(suffix)] if lbl.endswith(suffix) else lbl
        if base not in bases:
            bases.append(base)
    palette = sns.color_palette("colorblind", n_colors=max(1, len(bases))).as_hex()
    color_for_base = {b: palette[i % len(palette)] for i, b in enumerate(bases)}

    colors: List[str] = []
    line_styles: List[str] = []
    zorders: List[int] = []
    for lbl in labels:
        is_variant = lbl.endswith(suffix)
        base = lbl[: -len(suffix)] if is_variant else lbl
        colors.append(color_for_base[base])
        line_styles.append("dashed" if is_variant else "solid")
        zorders.append(3 if is_variant else 2)
    return colors, line_styles, zorders


def _style_default(labels: List[str]) -> Tuple[List[str], List[str], List[int]]:
    palette = sns.color_palette("colorblind", n_colors=max(1, len(labels))).as_hex()
    return list(palette), ["solid"] * len(labels), [2] * len(labels)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot accuracy / looping_count / avg_response_length curves.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs="+",
        metavar=("DIR", "LABEL"),
        required=True,
        help="Directory containing test_summaries.csv (or run*/test_summaries.csv) plus a label. "
             "Repeat to overlay series.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory. Three PDFs and a joined CSV are written here.",
    )
    parser.add_argument(
        "--pair-suffix",
        type=str,
        default=None,
        metavar="SUFFIX",
        help="Pair labels by stripping SUFFIX; the suffixed variants are styled dashed and "
             "share colour with their non-suffix base. Examples: ' w/bias' for a "
             "looping-bias overlay, or ' w/exp' to pair exploration variants.",
    )
    parser.add_argument(
        "--overlay-bias",
        action="store_true",
        help="Backwards-compat alias for `--pair-suffix ' w/bias'`.",
    )
    parser.add_argument(
        "--legend-panel-index",
        type=int,
        default=1,
        help="Panel index (0-based among the three metrics) to embed the legend in. Default 1 "
             "(looping_count panel).",
    )
    parser.add_argument(
        "--no-frame",
        action="store_true",
        help="Omit the legend frame box.",
    )
    return parser.parse_args()


def _parse_runs(raw: List[List[str]]) -> List[Tuple[Path, str]]:
    out: List[Tuple[Path, str]] = []
    for entry in raw:
        if len(entry) == 1:
            d = Path(entry[0])
            out.append((d, d.name))
        elif len(entry) == 2:
            out.append((Path(entry[0]), entry[1]))
        else:
            raise ValueError(
                f"--run expects 'DIR' or 'DIR LABEL'; got {entry!r}. "
                "Wrap labels with spaces in quotes."
            )
    return out


def main() -> None:
    args = parse_args()
    runs = _parse_runs(args.run)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    joined = load_runs(runs)
    joined_csv = out_dir / "joined_test_summaries.csv"
    joined.to_csv(joined_csv, index=False)
    print(f"Joined CSV: {joined_csv} (rows={len(joined)})")

    labels = [lbl for (_, lbl) in runs]
    pair_suffix = args.pair_suffix
    if args.overlay_bias:
        if pair_suffix is not None and pair_suffix != " w/bias":
            raise ValueError("--overlay-bias is an alias for --pair-suffix ' w/bias'; "
                             "do not pass both with a different suffix.")
        pair_suffix = " w/bias"
    if pair_suffix:
        colors, line_styles, zorders = _style_for_pair_suffix(labels, pair_suffix)
    else:
        colors, line_styles, zorders = _style_default(labels)

    panels = build_panels(joined, ordered_labels=labels)
    result = generate_line_plots(
        panels=panels,
        series_labels=labels,
        save_directory=str(out_dir),
        subdir_name="",
        n_plots=len(panels),
        legend_mode="panel",
        legend_panel_index=args.legend_panel_index,
        panel_legend_where="inside",
        palette="colorblind",
        marker="o",
        line_width=2.0,
        marker_size=4.5,
        legend_ncol=1,
        legend_frame=(not args.no_frame),
        legend_title=None,
        colors_override=colors,
        line_styles=line_styles,
        zorders=zorders,
    )
    print(f"Plots written to: {result['output_dir']}")
    for p in result["panel_paths"]:
        print(f"  {p}")


if __name__ == "__main__":
    main()

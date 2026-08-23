"""Per-root-visit target-probability overlay plots.

For each example in a validation JSON (`test_temp_<T>.json` written with
`--save_examples --logit_probs_k k`), find every step where the walk visits
the root node, look up the per-token top-k probability assigned at that step,
and read off the probability assigned to one target action:

  --target-strategy source              (random_walk_with_reset)
       The action of jumping from the root back to the source. Reflects the
       Section 3 risk-aversion loop.

  --target-strategy most_visited_child  (pure_random_walk)
       The action of taking the root child that the walk visits most often
       overall. Reflects the Section 4 correlated-errors loop.

Plot: thin per-trace lines, bold per-visit-index average overlay. By default,
the average stops once fewer than 5% of traces remain, matching the paper
figures and preventing a handful of unusually long traces from controlling the
right tail. Two run directories are overlaid (baseline vs bias-trained), one
colour each. Output is a single PDF.

Example invocations:

  # Appendix F reset-probability overlay without exploration.
  python plotting/plot_root_visit_probs.py \\
      --base out_random_walk_with_reset_5_5_12_layer_model \\
      --bias out_random_walk_with_reset_5_5_looping_12_layer_model \\
      --target-strategy source --temperature 0.0 \\
      --base-label "G(5,5)" --bias-label "G(5,5) w/bias" \\
      --out plotting/figures/reset_probability_no_exploration.pdf

  # Appendix F most-visited-child overlay for margin 0.
  python plotting/plot_root_visit_probs.py \\
      --base out_pure_random_walk_0.0_margin_G_5_5_12_layer_model \\
      --bias out_pure_random_walk_0.0_margin_G_5_5_12_layer_model_looping \\
      --target-strategy most_visited_child --temperature 0.0 \\
      --base-label "Margin 0" --bias-label "Margin 0 w/bias" \\
      --out plotting/figures/most_visited_child_margin0.pdf
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 12,
})

PROMPT_RE = re.compile(r"^(?P<edges>[^/]+)/(?P<src>[^,]+),(?P<goal>[^=]+)=\s*$")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_prompt(prompt: str) -> Tuple[str, str, List[Tuple[str, str]]]:
    match = PROMPT_RE.match(prompt.strip())
    if not match:
        raise ValueError(f"Malformed prompt: {prompt!r}")
    edges_blob = match.group("edges")
    source = match.group("src")
    goal = match.group("goal")
    edges: List[Tuple[str, str]] = []
    for edge_str in edges_blob.split("|"):
        if not edge_str:
            continue
        u, v = edge_str.split(",", 1)
        edges.append((u, v))
    return source, goal, edges


def _find_root(source: str, edges: Sequence[Tuple[str, str]]) -> str:
    neighbors = [v for (u, v) in edges if u == source]
    if len(neighbors) != 1:
        raise ValueError(f"Expected exactly one outgoing edge from source {source}, got {len(neighbors)}")
    return neighbors[0]


def _target_node(strategy: str, source: str, root: str, walk_nodes: Sequence[str]) -> Optional[str]:
    if strategy == "source":
        return source
    if strategy == "most_visited_child":
        # Choose the target post hoc from the complete generated walk by
        # counting which token follows the root most often. For
        # pure_random_walk, the only nonterminal transitions from the root are
        # to its children, so this identifies the repeatedly selected arm.
        counts: Dict[str, int] = {}
        for i in range(len(walk_nodes) - 1):
            if walk_nodes[i] != root:
                continue
            counts[walk_nodes[i + 1]] = counts.get(walk_nodes[i + 1], 0) + 1
        if not counts:
            return None
        return max(counts.items(), key=lambda kv: kv[1])[0]
    raise ValueError(f"Unknown target strategy: {strategy!r}")


def _extract_sequence(example: dict, target_strategy: str) -> Tuple[List[float], bool]:
    """Return (per-root-visit probabilities assigned to the target action, skipped_flag)."""
    if example.get("logit_probs") is None:
        raise ValueError("Example missing logit_probs; rerun validation with --logit_probs_k.")

    walk_nodes = example["walk_nodes"]
    logit_probs = example["logit_probs"]
    # When the walk hits the validator's max_total_tokens budget without ever
    # emitting EOS, the "what comes next" distribution at the final position is
    # never sampled, so len(logit_probs) == len(walk_nodes) - 1. We keep the
    # trace and just skip any root visit at positions that have no recorded
    # distribution (in practice only the very last position).

    source, _, edges = _parse_prompt(example["prompt"])
    root = _find_root(source, edges)
    target = _target_node(target_strategy, source, root, walk_nodes)
    if target is None:
        return [], True

    seq: List[float] = []
    target_seen = False
    for idx, node in enumerate(walk_nodes):
        if node != root:
            continue
        if idx >= len(logit_probs):
            continue
        entry = logit_probs[idx]
        topk = entry.get("top_k_logits") if isinstance(entry, dict) else None
        if not topk:
            continue
        # The paper plots root-transition probabilities conditional on the
        # model continuing with a graph action. If the greedy next token is EOS
        # (decoded as the empty string), omit that terminal root visit.
        best_overall_prob, best_overall_token = max(topk, key=lambda pair: pair[0])
        if best_overall_token == "":
            continue
        matches = [prob for prob, tok in topk if tok == target]
        if not matches:
            # Target token absent from this position's top-K. Very rare on the
            # paper's runs (<12 hits in ~250k root visits with K=5). If this
            # plotter is reused in a setting where the target action is a much
            # rarer token, bump --logit_probs_k at validation time so top-K is
            # wide enough to almost always include the target.
            continue
        target_seen = True
        seq.append(float(max(matches)))
    if not target_seen:
        return [], True
    return seq, False


def load_sequences(run_dir: Path, temperature: float, target_strategy: str) -> Tuple[List[List[float]], int]:
    temp_str = format(temperature, "g")
    json_path = run_dir / f"test_temp_{temp_str}.json"
    if not json_path.is_file():
        # Fall back to single nested run*/ layout produced by some training scripts.
        candidates = sorted(run_dir.glob(f"run*/test_temp_{temp_str}.json"))
        if len(candidates) == 1:
            json_path = candidates[0]
        elif len(candidates) > 1:
            raise FileNotFoundError(
                f"Multiple test_temp_{temp_str}.json files under {run_dir}; specify a more precise path."
            )
        else:
            raise FileNotFoundError(f"test_temp_{temp_str}.json not found in {run_dir}.")
    with open(json_path, "r") as f:
        payload = json.load(f)
    examples = list(payload.get("invalid_examples", [])) + list(payload.get("valid_examples", []))
    if not examples:
        raise ValueError(
            f"{json_path} has no valid/invalid examples; rerun validation with --save_examples."
        )
    sequences: List[List[float]] = []
    skipped = 0
    for ex in examples:
        seq, skip = _extract_sequence(ex, target_strategy=target_strategy)
        if skip:
            skipped += 1
            continue
        if seq:
            sequences.append(seq)
    if not sequences:
        raise ValueError(f"No usable root-visit sequences in {json_path}.")
    return sequences, skipped


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _style_axes(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(True, color="0.7", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_alpha(0.3)
    ax.spines["left"].set_alpha(0.3)


def _average_curve(
    sequences: Sequence[Sequence[float]],
    min_trace_frac: float = 0.05,
) -> Tuple[List[int], List[float]]:
    if not sequences:
        return [], []
    max_len = max(len(seq) for seq in sequences)
    # Only draw the average at indices where >= min_trace_frac of all
    # traces contribute. Prevents the right-tail of the curve from being
    # dominated by a handful of outlier-length walks.
    min_count = max(1, int(round(min_trace_frac * len(sequences))))
    xs: List[int] = []
    ys: List[float] = []
    for i in range(max_len):
        vals = [seq[i] for seq in sequences if len(seq) > i]
        if len(vals) < min_count:
            continue
        xs.append(i + 1)
        ys.append(sum(vals) / len(vals))
    return xs, ys


def plot_pair(
    seqs_base: Sequence[Sequence[float]],
    seqs_bias: Sequence[Sequence[float]],
    labels: Tuple[str, str],
    out_path: Path,
    ylabel: str,
    legend_loc: str,
    end_markers: bool,
    avg_overlays: bool,
    avg_min_trace_frac: float = 0.05,
) -> None:
    colors = sns.color_palette("colorblind", n_colors=2).as_hex()
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    _style_axes(ax)

    def _draw(seqs: Sequence[Sequence[float]], color: str) -> None:
        for seq in seqs:
            xs = list(range(1, len(seq) + 1))
            ax.plot(xs, seq, color=color, linewidth=0.8, alpha=0.12)
            if end_markers and seq:
                ax.scatter(
                    xs[-1], seq[-1], s=26, facecolors="white", edgecolors=color,
                    linewidth=0.8, alpha=0.25, zorder=5,
                )

    _draw(seqs_base, colors[0])
    _draw(seqs_bias, colors[1])

    if avg_overlays:
        ax.plot(*_average_curve(seqs_base, avg_min_trace_frac), color=colors[0], linewidth=2.2, alpha=0.95, zorder=6)
        ax.plot(*_average_curve(seqs_bias, avg_min_trace_frac), color=colors[1], linewidth=2.2, alpha=0.95, zorder=6)

    handles = [
        plt.Line2D([0], [0], color=colors[0], lw=2.2, label=labels[0]),
        plt.Line2D([0], [0], color=colors[1], lw=2.2, label=labels[1]),
    ]
    ax.legend(handles=handles, loc=legend_loc, frameon=False, fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_xlabel("root visit index")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.1)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-root-visit probability overlay (Figs 16 / 18)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base", type=str, required=True,
                        help="Baseline checkpoint dir (contains test_temp_<T>.json or run*/test_temp_<T>.json)")
    parser.add_argument("--bias", type=str, required=True,
                        help="Bias-trained (looping) checkpoint dir")
    parser.add_argument("--target-strategy", choices=["source", "most_visited_child"], required=True,
                        help="What action to read the probability for at each root visit.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampled temperature whose JSON we read (default 0.0)")
    parser.add_argument("--base-label", type=str, default="baseline")
    parser.add_argument("--bias-label", type=str, default="bias")
    parser.add_argument("--out", type=str, required=True,
                        help="Output PDF path. Parent dirs are created as needed.")
    parser.add_argument("--legend-loc", type=str, default="lower left")
    parser.add_argument("--end-markers", action="store_true",
                        help="Mark the last point of each trace with an open circle.")
    parser.add_argument("--no-avg-overlay", action="store_true",
                        help="Do not draw the bold per-visit-index average curve.")
    parser.add_argument("--avg-min-trace-frac", type=float, default=0.05,
                        help="Only draw the average overlay at indices where at "
                             "least this fraction of traces contribute. Default "
                             "0.05 matches the paper figures and prevents a few "
                             "outlier-length traces from controlling the right tail.")
    args = parser.parse_args()
    if not 0.0 <= args.avg_min_trace_frac <= 1.0:
        parser.error("--avg-min-trace-frac must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    seqs_base, skipped_base = load_sequences(Path(args.base), args.temperature, args.target_strategy)
    seqs_bias, skipped_bias = load_sequences(Path(args.bias), args.temperature, args.target_strategy)
    print(f"base ({args.base}): {len(seqs_base)} traces ({skipped_base} skipped)")
    print(f"bias ({args.bias}): {len(seqs_bias)} traces ({skipped_bias} skipped)")
    ylabel = "source prob" if args.target_strategy == "source" else "most visited child prob"
    plot_pair(
        seqs_base, seqs_bias,
        labels=(args.base_label, args.bias_label),
        out_path=Path(args.out),
        ylabel=ylabel,
        legend_loc=args.legend_loc,
        end_markers=args.end_markers,
        avg_overlays=not args.no_avg_overlay,
        avg_min_trace_frac=args.avg_min_trace_frac,
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

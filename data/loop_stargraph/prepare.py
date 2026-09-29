"""CLI entry point for generating loop_stargraph training/validation data.

Two walk types, matching the paper (see Sections 3 / 4 / Appendix E /
Appendix F):

  random_walk_with_reset
      --reset_probability <p>            required
      --exploration_probability <e>      default 0.0
      --looping_data_frac <b>            default 0.0

  pure_random_walk
      --margin <m>                       default 0.0
      --correct_arm_probability <q>      default None (uniform 1/k base)
      --num_wrong_arms_explored <k>      default None (all wrong arms visible)
      --looping_data_frac <b>            default 0.0

Mapping (paper -> invocation), reproduced from the README:

  Section 3 base:                 random_walk_with_reset --reset_probability 0.3
  Section 3 + exploration:        random_walk_with_reset --reset_probability 0.3 --exploration_probability 0.2
  Section 4 (margin variant):     pure_random_walk --margin <0|0.05|0.1>
  Appendix E G(5,5):              pure_random_walk --margin 0 --correct_arm_probability 0.5
  Appendix E G(10,5):             pure_random_walk --margin 0 --correct_arm_probability 0.5 --num_wrong_arms_explored 4
  Appendix F bias (any walk):     ... --looping_data_frac 0.001

Filenames produced:

  loopgraph_<n>_<l>_nodes_<N>_walk_<walk_type><params_suffix>[_looping_<b>]_maxtokens_<T>_sample_<S>.txt
"""

import argparse
import os
import random
import sys

# `data/loop_stargraph.py` provides Tokenizer and shadows this directory as a
# package, so add both locations explicitly and import `samplers` locally.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from data.loop_stargraph import Tokenizer  # noqa: E402
from samplers import (  # noqa: E402
    make_full_string,
    sample_random_walk_with_reset,
    sample_pure_random_walk,
)


WALK_WITH_RESET = "random_walk_with_reset"
PURE_RANDOM_WALK = "pure_random_walk"


def _params_suffix(args) -> str:
    if args.walk_type == WALK_WITH_RESET:
        suffix = f"_reset_{args.reset_probability}"
        if args.exploration_probability > 0.0:
            suffix += f"_explore_{args.exploration_probability}"
        return suffix
    # pure_random_walk
    suffix = f"_margin_{args.margin}"
    if args.correct_arm_probability is not None:
        suffix += f"_correctarm_{args.correct_arm_probability}"
    if args.num_wrong_arms_explored is not None:
        suffix += f"_wrongarms_{args.num_wrong_arms_explored}"
    return suffix


def _build_filename(args) -> str:
    parts = [
        f"loopgraph_{args.num_paths}_{args.path_length}",
        f"nodes_{args.max_nodes}",
        f"walk_{args.walk_type}{_params_suffix(args)}",
    ]
    if args.looping_data_frac > 0.0:
        parts.append(f"looping_{args.looping_data_frac}")
    if args.max_tokens is not None:
        parts.append(f"maxtokens_{args.max_tokens}")
    parts.append(f"sample_{args.num_samples}.txt")
    return "_".join(parts)


def generate_and_save(args) -> int:
    if getattr(args, "seed", None) is not None:
        random.seed(args.seed)
    os.makedirs(args.data_dir, exist_ok=True)
    tokenizer = Tokenizer(args.max_nodes)
    save_path = os.path.join(args.data_dir, _build_filename(args))
    print(f"Saving generated data to {save_path}")

    actual_max_tokens = 0
    show_loading_bar = args.num_samples >= 100

    with open(save_path, "w") as out:
        for i in range(args.num_samples):
            if args.walk_type == WALK_WITH_RESET:
                walk, edge_list, source, goal = sample_random_walk_with_reset(
                    num_paths=args.num_paths,
                    path_length=args.path_length,
                    max_nodes=args.max_nodes,
                    max_tokens=args.max_tokens,
                    tokenizer=tokenizer,
                    reset_probability=args.reset_probability,
                    exploration_probability=args.exploration_probability,
                    looping_data_frac=args.looping_data_frac,
                )
            else:
                walk, edge_list, source, goal = sample_pure_random_walk(
                    num_paths=args.num_paths,
                    path_length=args.path_length,
                    max_nodes=args.max_nodes,
                    max_tokens=args.max_tokens,
                    tokenizer=tokenizer,
                    margin=args.margin,
                    correct_arm_probability=args.correct_arm_probability,
                    num_wrong_arms_explored=args.num_wrong_arms_explored,
                    looping_data_frac=args.looping_data_frac,
                )

            line = make_full_string(edge_list, source, goal, walk)
            _, count = tokenizer.tokenize(line)
            if count > actual_max_tokens:
                actual_max_tokens = count
            out.write(line + "\n")

            if show_loading_bar and ((i + 1) % max(1, args.num_samples // 1000) == 0 or i + 1 == args.num_samples):
                done = int((i + 1) * 50 / args.num_samples)
                bar = "█" * done + " " * (50 - done)
                print(f"\r|{bar}| {(i + 1) * 100 / args.num_samples:.1f}%", end="", flush=True)

        if show_loading_bar:
            print()

    print(f"Actual maximum token count in generated data: {actual_max_tokens}")
    return actual_max_tokens


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate loop_stargraph data.")
    parser.add_argument("--walk_type", required=True, choices=[WALK_WITH_RESET, PURE_RANDOM_WALK])
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--num_paths", type=int, required=True, help="Number of arms (n in G(n,l)).")
    parser.add_argument("--path_length", type=int, required=True, help="Arm length (l in G(n,l)).")
    parser.add_argument("--max_nodes", type=int, default=30)
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--data_dir", type=str, default="data/loop_stargraph")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducible data generation.")
    parser.add_argument("--looping_data_frac", type=float, default=0.0,
                        help="Appendix F bias: probability of replacing the sample with a deterministic looping trace.")

    # random_walk_with_reset
    parser.add_argument("--reset_probability", type=float, default=None,
                        help="random_walk_with_reset: per-step probability of resetting to source.")
    parser.add_argument("--exploration_probability", type=float, default=0.0,
                        help="random_walk_with_reset: probability of taking a wrong arm at the root.")

    # pure_random_walk
    parser.add_argument("--margin", type=float, default=0.0,
                        help="pure_random_walk: per-visit probability reduction for already-visited arms at the root.")
    parser.add_argument("--correct_arm_probability", type=float, default=None,
                        help="pure_random_walk: probability of the correct arm at the root (Appendix E). None => uniform 1/k base.")
    parser.add_argument("--num_wrong_arms_explored", type=int, default=None,
                        help="pure_random_walk: only this many random wrong arms get nonzero probability (Appendix E G(10,5)). None => all wrong arms.")

    return parser


def _validate_args(args) -> None:
    if args.walk_type == WALK_WITH_RESET:
        if args.reset_probability is None:
            raise ValueError("random_walk_with_reset requires --reset_probability")
        if args.margin != 0.0 or args.correct_arm_probability is not None or args.num_wrong_arms_explored is not None:
            raise ValueError("--margin / --correct_arm_probability / --num_wrong_arms_explored are pure_random_walk-only knobs")
    else:
        if args.reset_probability is not None or args.exploration_probability != 0.0:
            raise ValueError("--reset_probability / --exploration_probability are random_walk_with_reset-only knobs")


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args)
    generate_and_save(args)
    print(f"\nGenerated {args.num_samples} samples for {args.walk_type} on G({args.num_paths},{args.path_length}).")

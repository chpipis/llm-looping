#!/usr/bin/env python3
"""Validate a trained loop_stargraph model.

Loads a checkpoint, generates walks from prompts in a validation data file,
and reports four metrics per temperature:

    accuracy             fraction of generated walks that reach the goal
                         through valid edges
    avg_response_length  mean generated-token count per response
    looping_count        mean count of the dominant looping action per response;
                         polymorphic by walk_type:
                           - random_walk_with_reset: root -> source transitions
                             (source <-> root oscillation, Section 3)
                           - pure_random_walk: visits to the most-visited root
                             child (repeated wrong-arm visits, Section 4 + App E)
    num_responses        total responses generated

Outputs per checkpoint dir:
    test_temp_<T>.json       one per temperature; metrics summary, and if
                             --save_examples is set, full per-response records
                             (prompt, walk_nodes, logit_probs) used by the
                             per-root-visit target-probability plots.
    test_summaries.csv       roll-up across temperatures consumed by
                             plotting/plot_validation_curves.py.

The data file's walk_type is recovered from its filename, which follows the
convention emitted by data/loop_stargraph/prepare.py:

    loopgraph_<n>_<l>_nodes_<N>_walk_<random_walk_with_reset|pure_random_walk>
      [_reset_<p>][_explore_<e>][_margin_<m>][_correctarm_<q>][_wrongarms_<k>]
      [_looping_<b>]_maxtokens_<T>_sample_<S>.txt
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import lightning as L
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from core_train import initialize_model
from data.loop_stargraph import Tokenizer


WALK_WITH_RESET = "random_walk_with_reset"
PURE_RANDOM_WALK = "pure_random_walk"
SUPPORTED_WALK_TYPES = (WALK_WITH_RESET, PURE_RANDOM_WALK)


# ----------------------------------------------------------------------------
# Filename parsing & validation-data resolution
# ----------------------------------------------------------------------------

_FLOAT = r"[0-9]*\.?[0-9]+"
_FILENAME_RE = re.compile(
    r"^loopgraph_"
    r"(?P<num_paths>\d+)_"
    r"(?P<path_length>\d+)_"
    r"nodes_(?P<max_nodes>\d+)_"
    r"walk_(?P<walk_type>random_walk_with_reset|pure_random_walk)"
    r"(?:_reset_(?P<reset_probability>" + _FLOAT + r"))?"
    r"(?:_explore_(?P<exploration_probability>" + _FLOAT + r"))?"
    r"(?:_margin_(?P<margin>" + _FLOAT + r"))?"
    r"(?:_correctarm_(?P<correct_arm_probability>" + _FLOAT + r"))?"
    r"(?:_wrongarms_(?P<num_wrong_arms_explored>\d+))?"
    r"(?:_looping_(?P<looping_data_frac>" + _FLOAT + r"))?"
    r"_maxtokens_(?P<max_tokens>\d+)_"
    r"sample_(?P<num_samples>\d+)\.txt$"
)


def _parse_data_filename(filename: str) -> Dict[str, Any]:
    m = _FILENAME_RE.match(filename)
    if not m:
        raise ValueError(
            f"Unrecognized loop_stargraph filename: '{filename}'. "
            "Expected the pattern produced by data/loop_stargraph/prepare.py."
        )
    g = m.groupdict()

    def _fopt(key):
        v = g.get(key)
        return float(v) if v is not None else None

    def _iopt(key):
        v = g.get(key)
        return int(v) if v is not None else None

    return {
        "num_paths": int(g["num_paths"]),
        "path_length": int(g["path_length"]),
        "max_nodes": int(g["max_nodes"]),
        "walk_type": g["walk_type"],
        "reset_probability": _fopt("reset_probability"),
        "exploration_probability": _fopt("exploration_probability"),
        "margin": _fopt("margin"),
        "correct_arm_probability": _fopt("correct_arm_probability"),
        "num_wrong_arms_explored": _iopt("num_wrong_arms_explored"),
        "looping_data_frac": _fopt("looping_data_frac"),
        "max_tokens": int(g["max_tokens"]),
        "num_samples": int(g["num_samples"]),
    }


def _build_filename(params: Dict[str, Any], num_samples: int) -> str:
    parts = [
        f"loopgraph_{params['num_paths']}_{params['path_length']}",
        f"nodes_{params['max_nodes']}",
    ]
    walk = f"walk_{params['walk_type']}"
    if params["walk_type"] == WALK_WITH_RESET:
        walk += f"_reset_{params['reset_probability']}"
        if params.get("exploration_probability") and params["exploration_probability"] > 0.0:
            walk += f"_explore_{params['exploration_probability']}"
    else:
        walk += f"_margin_{params['margin'] if params['margin'] is not None else 0.0}"
        if params.get("correct_arm_probability") is not None:
            walk += f"_correctarm_{params['correct_arm_probability']}"
        if params.get("num_wrong_arms_explored") is not None:
            walk += f"_wrongarms_{params['num_wrong_arms_explored']}"
    parts.append(walk)
    if params.get("looping_data_frac") and params["looping_data_frac"] > 0.0:
        parts.append(f"looping_{params['looping_data_frac']}")
    parts.append(f"maxtokens_{params['max_tokens']}")
    parts.append(f"sample_{num_samples}.txt")
    return "_".join(parts)


def resolve_eval_data_path(data_path: str, validation_samples: int) -> Tuple[str, Dict[str, Any]]:
    """Return path to the per-`validation_samples` data file (generating it if missing)."""
    filename = os.path.basename(data_path)
    params = _parse_data_filename(filename)

    data_dir = os.path.dirname(data_path) or "."
    gen_filename = _build_filename(params, validation_samples)
    eval_data_path = os.path.join(data_dir, gen_filename)

    if not os.path.isfile(eval_data_path):
        prepare_script = os.path.join(os.path.dirname(__file__), "data", "loop_stargraph", "prepare.py")
        cmd = [
            sys.executable,
            prepare_script,
            "--num_samples", str(validation_samples),
            "--num_paths", str(params["num_paths"]),
            "--path_length", str(params["path_length"]),
            "--max_nodes", str(params["max_nodes"]),
            "--max_tokens", str(params["max_tokens"]),
            "--data_dir", data_dir,
            "--walk_type", params["walk_type"],
        ]
        if params["walk_type"] == WALK_WITH_RESET:
            cmd += ["--reset_probability", str(params["reset_probability"])]
            if params.get("exploration_probability"):
                cmd += ["--exploration_probability", str(params["exploration_probability"])]
        else:
            cmd += ["--margin", str(params["margin"] if params["margin"] is not None else 0.0)]
            if params.get("correct_arm_probability") is not None:
                cmd += ["--correct_arm_probability", str(params["correct_arm_probability"])]
            if params.get("num_wrong_arms_explored") is not None:
                cmd += ["--num_wrong_arms_explored", str(params["num_wrong_arms_explored"])]
        if params.get("looping_data_frac"):
            cmd += ["--looping_data_frac", str(params["looping_data_frac"])]

        res = subprocess.run(cmd)
        if res.returncode != 0:
            raise RuntimeError(f"prepare.py failed with exit code {res.returncode}")

    return eval_data_path, params


# ----------------------------------------------------------------------------
# Checkpoint resolution
# ----------------------------------------------------------------------------

_CKPT_REGEX = re.compile(r"^ckpt_iter_(?P<iter>\d+)_loss_(?P<loss>[0-9]*\.?[0-9]+)\.pt$")


def _select_min_loss_checkpoint(checkpoint_dir: str) -> Tuple[str, float]:
    candidates: List[Tuple[float, str]] = []
    for name in os.listdir(checkpoint_dir):
        m = _CKPT_REGEX.match(name)
        if m:
            candidates.append((float(m.group("loss")), os.path.join(checkpoint_dir, name)))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints matching 'ckpt_iter_*_loss_*.pt' found in {checkpoint_dir}"
        )
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1], candidates[0][0]


def _resolve_checkpoint_dir_from_latest_file(latest_file: str) -> str:
    if not os.path.isfile(latest_file):
        raise FileNotFoundError(f"--latest_ckpt_file not found: {latest_file}")
    with open(latest_file, "r") as f:
        lines = [ln.strip().strip('"').strip("'") for ln in f.readlines()]
    content = next((ln for ln in lines if ln), None)
    if not content:
        raise ValueError(f"--latest_ckpt_file is empty: {latest_file}")
    ckpt_path = content
    if os.path.isabs(ckpt_path) and os.path.isfile(ckpt_path):
        return os.path.dirname(ckpt_path)
    file_dir = os.path.dirname(os.path.abspath(latest_file))
    parent_dir = os.path.dirname(file_dir)
    candidates = [
        os.path.normpath(os.path.join(file_dir, ckpt_path)),
        os.path.normpath(os.path.join(parent_dir, ckpt_path)),
        os.path.normpath(os.path.join(os.getcwd(), ckpt_path)),
    ]
    for cand in candidates:
        if os.path.isfile(cand):
            return os.path.dirname(cand)
    report_path = ckpt_path if os.path.isabs(ckpt_path) else os.path.normpath(os.path.join(parent_dir, ckpt_path))
    raise FileNotFoundError(
        f"Checkpoint path from --latest_ckpt_file does not exist: {report_path} (from {latest_file})"
    )


def _parse_temperature_list(temp_args) -> List[float]:
    if temp_args is None:
        return [0.0]
    vals: List[float] = []
    for entry in temp_args:
        if isinstance(entry, (int, float)):
            vals.append(float(entry))
            continue
        for piece in str(entry).split(","):
            piece = piece.strip()
            if piece:
                vals.append(float(piece))
    return vals


# ----------------------------------------------------------------------------
# Pure metric computation (unit-testable without a model)
# ----------------------------------------------------------------------------


@dataclass
class PromptInfo:
    """Per-example structure extracted from one prompt line."""
    line_idx: int
    prompt_tokens: List[int]
    prompt_str: str
    valid_edges: Set[Tuple[str, str]]
    source_node: str
    goal_node: str
    root_node: str
    root_children: Set[str]
    leaf_nodes: Set[str]


@dataclass
class ResponseMetrics:
    """Per-response output of compute_response_metrics."""
    line_idx: int
    prompt: str
    generated_walk: str
    expected_source: str
    expected_goal: str
    valid_edges: List[Tuple[str, str]]
    walk_nodes: List[str]
    reasons: List[str]
    looping_count: int
    is_valid: bool
    response_length: int
    logit_probs: Optional[List[Dict[str, Any]]] = None


def _parse_prompt(line: str, line_idx: int, tokenizer: Tokenizer) -> Optional[PromptInfo]:
    """Parse a single data line into a PromptInfo, or None if malformed."""
    sequence, _ = tokenizer.tokenize(line.strip())
    sequence = sequence.tolist()

    equals_token_id = tokenizer.encoder["="]
    slash_token_id = tokenizer.encoder["/"]
    eos_token_id = tokenizer.eos_token_id

    slash_pos = None
    equals_pos = None
    for j, token in enumerate(sequence):
        if token == slash_token_id and slash_pos is None:
            slash_pos = j
        elif token == equals_token_id and equals_pos is None:
            equals_pos = j
            break

    if slash_pos is None or equals_pos is None or slash_pos >= equals_pos:
        return None

    prompt_tokens = sequence[: equals_pos + 1]
    if prompt_tokens[-1] == eos_token_id:
        prompt_tokens = prompt_tokens[:-1]

    edge_list_str = tokenizer.decode(sequence[:slash_pos], stop_at_eos=False)
    source_goal_str = tokenizer.decode(sequence[slash_pos + 1 : equals_pos], stop_at_eos=False)

    valid_edges: Set[Tuple[str, str]] = set()
    for edge_str in edge_list_str.split("|"):
        parts = edge_str.split(",")
        valid_edges.add((parts[0], parts[1]))

    parts = source_goal_str.split(",")
    source_node, goal_node = parts[0], parts[1]

    out_nodes = {u for (u, v) in valid_edges}
    all_nodes = out_nodes.union(v for (u, v) in valid_edges)
    leaf_nodes = all_nodes - out_nodes

    root_node = None
    for (u, v) in valid_edges:
        if u == source_node:
            root_node = v
            break
    if root_node is None:
        raise ValueError(f"Could not find root node (child of source {source_node})")
    root_children = {v for (u, v) in valid_edges if u == root_node}
    if not root_children:
        raise ValueError(f"Root node {root_node} has no children")

    return PromptInfo(
        line_idx=line_idx,
        prompt_tokens=prompt_tokens,
        prompt_str=tokenizer.decode(prompt_tokens, stop_at_eos=False),
        valid_edges=valid_edges,
        source_node=source_node,
        goal_node=goal_node,
        root_node=root_node,
        root_children=root_children,
        leaf_nodes=leaf_nodes,
    )


def _validity_skip(walk_type: str, cur: str, nxt: str, info: PromptInfo) -> bool:
    """Return True if the (cur, nxt) edge should be exempt from validity checks.

    `random_walk_with_reset`: any non-goal -> source transition is allowed (the
        walker can reset to the source from anywhere).
    `pure_random_walk`: only leaf -> source transitions are allowed (the walker
        returns to source after exhausting a wrong arm).
    """
    if walk_type == WALK_WITH_RESET:
        return cur != info.goal_node and nxt == info.source_node
    # pure_random_walk
    return cur in info.leaf_nodes and cur != info.goal_node and nxt == info.source_node


def _looping_count(walk_type: str, walk_nodes: List[str], info: PromptInfo) -> int:
    """Polymorphic looping_count:
      - random_walk_with_reset: root -> source transitions
      - pure_random_walk: visits to the most-visited root child
    """
    if walk_type == WALK_WITH_RESET:
        return sum(
            1
            for j in range(len(walk_nodes) - 1)
            if walk_nodes[j] == info.root_node and walk_nodes[j + 1] == info.source_node
        )
    # pure_random_walk
    child_visit_counts = {c: 0 for c in info.root_children}
    for j in range(len(walk_nodes) - 1):
        a, b = walk_nodes[j], walk_nodes[j + 1]
        if a == info.root_node and b in child_visit_counts:
            child_visit_counts[b] += 1
    return max(child_visit_counts.values()) if child_visit_counts else 0


def compute_response_metrics(
    info: PromptInfo,
    walk_nodes: List[str],
    generated_walk: str,
    gen_token_count: int,
    walk_type: str,
    reached_token_limit: bool,
    logit_probs: Optional[List[Dict[str, Any]]] = None,
) -> ResponseMetrics:
    """Pure per-response metric computation. No model, no tokenizer."""
    reasons: List[str] = []
    is_valid = True

    if reached_token_limit:
        reasons.append("reached max token limit")
    if walk_nodes and walk_nodes[0] != info.source_node:
        is_valid = False
        reasons.append("invalid start node")
    if walk_nodes and walk_nodes[-1] != info.goal_node:
        is_valid = False
        reasons.append("invalid end node")

    invalid_edges: Set[str] = set()
    for j in range(len(walk_nodes) - 1):
        cur, nxt = walk_nodes[j], walk_nodes[j + 1]
        if _validity_skip(walk_type, cur, nxt, info):
            continue
        if (cur, nxt) not in info.valid_edges:
            is_valid = False
            invalid_edges.add(f"{cur}->{nxt}")
    if invalid_edges:
        reasons.append("invalid edges: " + ", ".join(sorted(invalid_edges)))

    return ResponseMetrics(
        line_idx=info.line_idx + 1,
        prompt=info.prompt_str,
        generated_walk=generated_walk,
        expected_source=info.source_node,
        expected_goal=info.goal_node,
        valid_edges=list(info.valid_edges),
        walk_nodes=walk_nodes,
        reasons=reasons,
        looping_count=_looping_count(walk_type, walk_nodes, info),
        is_valid=is_valid,
        response_length=gen_token_count,
        logit_probs=logit_probs,
    )


def aggregate_metrics(responses: List[ResponseMetrics], save_examples: bool) -> Dict[str, Any]:
    """Roll per-response metrics up into the summary dict written to JSON / CSV."""
    n = len(responses)
    valid = [r for r in responses if r.is_valid]
    invalid = [r for r in responses if not r.is_valid]

    accuracy = (len(valid) / n) if n > 0 else 0.0
    avg_response_length = (sum(r.response_length for r in responses) / n) if n > 0 else 0.0
    looping_count = (sum(r.looping_count for r in responses) / n) if n > 0 else 0.0

    out: Dict[str, Any] = {
        "accuracy": accuracy,
        "avg_response_length": avg_response_length,
        "looping_count": looping_count,
        "num_responses": n,
    }
    if save_examples:
        out["valid_examples"] = [_response_to_dict(r) for r in valid]
        out["invalid_examples"] = [_response_to_dict(r) for r in invalid]
    return out


def _response_to_dict(r: ResponseMetrics) -> Dict[str, Any]:
    d = {
        "line_idx": r.line_idx,
        "prompt": r.prompt,
        "generated_walk": r.generated_walk,
        "expected_source": r.expected_source,
        "expected_goal": r.expected_goal,
        "valid_edges": r.valid_edges,
        "walk_nodes": r.walk_nodes,
        "reasons": r.reasons,
        "looping_count": r.looping_count,
    }
    if r.logit_probs is not None:
        d["logit_probs"] = r.logit_probs
    return d


# ----------------------------------------------------------------------------
# Model-driven validation loop
# ----------------------------------------------------------------------------


def custom_validate(
    data_lines: List[str],
    tokenizer: Tokenizer,
    model,
    walk_type: str,
    max_new_tokens: Optional[int] = None,
    max_total_tokens: int = 200,
    temperature: float = 0.1,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    device=None,
    batch_size: int = 256,
    logit_probs_k: Optional[int] = None,
    save_examples: bool = False,
) -> Dict[str, Any]:
    if walk_type not in SUPPORTED_WALK_TYPES:
        raise ValueError(f"Unsupported walk_type: {walk_type!r}; expected one of {SUPPORTED_WALK_TYPES}")
    model.eval()

    eos_token_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    responses: List[ResponseMetrics] = []

    with torch.no_grad():
        batch_items: List[PromptInfo] = []
        for line_idx, line in tqdm(enumerate(data_lines), total=len(data_lines)):
            if not line.strip():
                continue
            try:
                info = _parse_prompt(line, line_idx, tokenizer)
            except ValueError as e:
                print(f"Warning: malformed line {line_idx + 1}: {e}")
                continue
            if info is None:
                print(f"Warning: malformed line {line_idx + 1}: {line[:50]}...")
                continue
            batch_items.append(info)

            if len(batch_items) < batch_size and line_idx + 1 < len(data_lines):
                continue

            per_item_limits = []
            for it in batch_items:
                tokens_remaining = max(0, max_total_tokens - len(it.prompt_tokens))
                if max_new_tokens is not None:
                    tokens_remaining = min(max_new_tokens, tokens_remaining)
                per_item_limits.append(tokens_remaining)

            batch_max_new_tokens = max(per_item_limits) if per_item_limits else 0
            if batch_max_new_tokens <= 0:
                print("Warning: max_total_tokens exhausted for this batch; skipping generation.")
                batch_items = []
                continue

            max_len = max(len(it.prompt_tokens) for it in batch_items)
            B = len(batch_items)
            batch_tensor = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
            for i, it in enumerate(batch_items):
                pt = it.prompt_tokens
                batch_tensor[i, max_len - len(pt) : max_len] = torch.tensor(pt, dtype=torch.long, device=device)

            if logit_probs_k is not None:
                generated, top_k_probs, top_k_indices = model.generate(
                    batch_tensor,
                    max_new_tokens=batch_max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    return_top_k_logits=logit_probs_k,
                )
            else:
                generated = model.generate(
                    batch_tensor,
                    max_new_tokens=batch_max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=temperature > 0,
                )

            for bi, it in enumerate(batch_items):
                gen_tokens = generated[bi, max_len:].detach().cpu().tolist()
                if eos_token_id in gen_tokens:
                    gen_tokens = gen_tokens[: gen_tokens.index(eos_token_id)]
                item_limit = per_item_limits[bi]
                if item_limit >= 0:
                    gen_tokens = gen_tokens[:item_limit]
                if not gen_tokens:
                    continue

                processed_logit_probs: Optional[List[Dict[str, Any]]] = None
                if logit_probs_k is not None:
                    processed_logit_probs = []
                    for i, token_id in enumerate(gen_tokens):
                        if i + 1 >= top_k_probs.shape[1]:
                            break
                        batch_probs = top_k_probs[bi, i + 1]
                        batch_indices = top_k_indices[bi, i + 1]
                        top_k_list = []
                        for prob, tok_id in zip(batch_probs, batch_indices):
                            tid = tok_id.item()
                            top_k_list.append((prob.item(), tokenizer.decode([tid], stop_at_eos=False)))
                        processed_logit_probs.append({
                            "token_str": tokenizer.decode([token_id], stop_at_eos=False),
                            "top_k_logits": top_k_list,
                        })

                generated_walk_str = tokenizer.decode(gen_tokens, stop_at_eos=False)
                walk_nodes = [node.strip() for node in generated_walk_str.split(",") if node.strip()]
                if not walk_nodes:
                    continue

                response = compute_response_metrics(
                    info=it,
                    walk_nodes=walk_nodes,
                    generated_walk=generated_walk_str,
                    gen_token_count=len(gen_tokens),
                    walk_type=walk_type,
                    reached_token_limit=(item_limit and len(gen_tokens) >= item_limit),
                    logit_probs=processed_logit_probs,
                )
                responses.append(response)

            batch_items = []

    return aggregate_metrics(responses, save_examples=save_examples)


# ----------------------------------------------------------------------------
# Output writers
# ----------------------------------------------------------------------------


_SUMMARY_KEYS = ("accuracy", "avg_response_length", "looping_count", "num_responses")


def _print_summary(temp: float, summary: Dict[str, Any]) -> None:
    print(f"\nSUMMARY (temp={temp}):")
    print(f"  Accuracy:                    {summary['accuracy']:.4f}")
    print(f"  Avg response length:         {summary['avg_response_length']:.2f}")
    print(f"  Looping count:               {summary['looping_count']:.2f}")
    print(f"  Total responses:             {summary['num_responses']}")


def _write_json(json_path: str, summary: Dict[str, Any], data_path: str, max_total_tokens: int) -> None:
    payload = {"summary": {k: summary[k] for k in _SUMMARY_KEYS}}
    payload["summary"]["data_path"] = data_path
    payload["summary"]["max_total_tokens"] = max_total_tokens
    if "valid_examples" in summary:
        payload["valid_examples"] = summary["valid_examples"]
        payload["invalid_examples"] = summary["invalid_examples"]
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved JSON to {json_path}")


def _write_csv(csv_path: str, rows: List[Dict[str, Any]]) -> None:
    fieldnames = ["temperature", *_SUMMARY_KEYS, "data_path", "max_total_tokens"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"Wrote CSV summary to {csv_path}")


# ----------------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Run LoopStarGraph validation.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory containing ckpt_iter_*_loss_*.pt and materialized_config.yaml; "
                             "enables multi-temperature sweep")
    parser.add_argument("--latest_ckpt_file", type=str, default=None,
                        help="Path to a file containing the latest checkpoint path; resolves --checkpoint_dir")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to LoopStarGraph data file; if omitted, inferred from the config")
    parser.add_argument("--max_total_tokens", type=int, default=200,
                        help="Total token budget per example (prompt + generated tokens)")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Optional explicit cap on generated tokens; defaults to max_total_tokens minus prompt length")
    parser.add_argument("--temperature", action="append", type=str, default=None,
                        help="Sampling temperature(s). Supports CSV ('0.0,0.2,1.0') or repeated flags. Defaults to 0.0")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling threshold (1.0 disables)")
    parser.add_argument("--max_lines", type=int, default=None, help="Max lines to process from data file")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for generation")
    parser.add_argument("--save_file", type=str, default=None,
                        help="Single-run mode: path to JSON for the summary (+ examples if --save_examples)")
    parser.add_argument("--logit_probs_k", type=int, default=None,
                        help="If set, save top-k logit probabilities per generated token (required for root-prob plots)")
    parser.add_argument("--max_valid_saved", type=int, default=None,
                        help="If --save_examples is set, cap the number of valid examples in the output")
    parser.add_argument("--validation_samples", type=int, default=50,
                        help="Number of samples to validate against (defaults to 50; canonical paper runs use 500/4000)")
    parser.add_argument("--save_examples", action="store_true",
                        help="Include per-response records (prompt, walk_nodes, logit_probs) in the JSON output. "
                             "Required for the per-root-visit probability plots (Figs 16 / 18).")

    args, conf_cli = parser.parse_known_args()

    if args.latest_ckpt_file:
        if args.checkpoint_dir:
            raise ValueError("--latest_ckpt_file cannot be used together with --checkpoint_dir")
        args.checkpoint_dir = _resolve_checkpoint_dir_from_latest_file(args.latest_ckpt_file)
        print(f"Resolved --checkpoint_dir from --latest_ckpt_file: {args.checkpoint_dir}")

    temperatures = _parse_temperature_list(args.temperature)
    is_sweep_mode = args.checkpoint_dir is not None

    if is_sweep_mode and args.checkpoint_path:
        raise ValueError("--checkpoint_dir selects the checkpoint automatically; omit --checkpoint_path")
    if not is_sweep_mode and len(temperatures) != 1:
        raise ValueError("Multiple temperatures require --checkpoint_dir.")
    if is_sweep_mode and not os.path.isdir(args.checkpoint_dir):
        raise NotADirectoryError(f"--checkpoint_dir does not exist: {args.checkpoint_dir}")

    if is_sweep_mode:
        best_ckpt_path, best_ckpt_loss = _select_min_loss_checkpoint(args.checkpoint_dir)
        args.checkpoint_path = best_ckpt_path
        print(f"Selected checkpoint by min loss: {best_ckpt_path} (loss={best_ckpt_loss:.6f})")
        if not args.config:
            cfg_candidate = os.path.join(args.checkpoint_dir, "materialized_config.yaml")
            if not os.path.isfile(cfg_candidate):
                raise FileNotFoundError(
                    f"materialized_config.yaml not found in {args.checkpoint_dir}; provide --config explicitly."
                )
            args.config = cfg_candidate
        if args.save_file:
            print("Note: --save_file is ignored in --checkpoint_dir sweep mode; per-temp JSONs and a CSV are written.")

    if os.path.exists("config/defaults.yaml"):
        default = OmegaConf.load("config/defaults.yaml")
    else:
        default = OmegaConf.create({})
    if not args.config:
        raise ValueError("--config (or --checkpoint_dir with materialized_config.yaml) is required.")
    overrides = OmegaConf.load(args.config)
    cli = OmegaConf.from_dotlist(conf_cli)
    config = OmegaConf.merge(default, overrides, cli)

    if not args.data_path:
        inferred_dp = OmegaConf.select(config, "data.loop_stargraph_data_path")
        if inferred_dp:
            args.data_path = str(inferred_dp)
            print(f"Inferred data_path from config: {args.data_path}")
    if not args.data_path:
        raise ValueError("Data path is required. Provide --data_path or set it in the config.")

    config.data.dataset = "loop_stargraph"
    config.trainer.compile = False

    fabric = L.Fabric(devices=1)

    eval_data_path, data_params = resolve_eval_data_path(args.data_path, args.validation_samples)

    print("Configuration:")
    print(f"  Dataset:            {config.data.dataset}")
    print(f"  Data path:          {eval_data_path}")
    print(f"  Walk type:          {data_params['walk_type']}")
    print(f"  Max nodes:          {config.data.loop_stargraph_max_nodes}")
    print(f"  Validation samples: {args.validation_samples}")
    print(f"  Max total tokens:   {args.max_total_tokens}")
    print(f"  Temperatures:       {temperatures}")
    print(f"  Top-k:              {args.top_k}")
    print(f"  Top-p:              {args.top_p}")
    print(f"  Batch size:         {args.batch_size}")
    if args.logit_probs_k is not None:
        print(f"  Logit probs k:      {args.logit_probs_k}")
    print(f"  Save examples:      {args.save_examples}")

    with open(eval_data_path, "r") as f:
        data_lines = f.readlines()
    if args.max_lines:
        data_lines = data_lines[: args.max_lines]
    print(f"  Processing {len(data_lines)} lines")

    tokenizer = Tokenizer(config.data.loop_stargraph_max_nodes)

    print(f"\nLoading model from {args.checkpoint_path}...")
    if not os.path.isfile(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {args.checkpoint_path} not found")
    model = initialize_model(
        fabric, config, tokenizer,
        initialize_optimizer=False,
        checkpoint_path=args.checkpoint_path,
    )
    print(f"Model loaded. type={type(model).__name__} vocab={model.config.vocab_size} block={model.config.block_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_sweep_mode:
        rows = []
        for t in temperatures:
            print(f"\n-- Temperature {t} --")
            summary = custom_validate(
                data_lines=data_lines,
                tokenizer=tokenizer,
                model=model,
                walk_type=data_params["walk_type"],
                max_new_tokens=args.max_new_tokens,
                max_total_tokens=args.max_total_tokens,
                temperature=t,
                top_k=args.top_k,
                top_p=args.top_p,
                device=device,
                batch_size=args.batch_size,
                logit_probs_k=args.logit_probs_k,
                save_examples=args.save_examples,
            )
            if args.save_examples and args.max_valid_saved is not None:
                summary["valid_examples"] = summary["valid_examples"][: args.max_valid_saved]
            _print_summary(t, summary)
            temp_str = format(t, "g")
            json_path = os.path.join(args.checkpoint_dir, f"test_temp_{temp_str}.json")
            _write_json(json_path, summary, eval_data_path, args.max_total_tokens)
            rows.append({
                "temperature": t,
                **{k: summary[k] for k in _SUMMARY_KEYS},
                "data_path": eval_data_path,
                "max_total_tokens": args.max_total_tokens,
            })
        _write_csv(os.path.join(args.checkpoint_dir, "test_summaries.csv"), rows)
    else:
        t = temperatures[0]
        summary = custom_validate(
            data_lines=data_lines,
            tokenizer=tokenizer,
            model=model,
            walk_type=data_params["walk_type"],
            max_new_tokens=args.max_new_tokens,
            max_total_tokens=args.max_total_tokens,
            temperature=t,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
            batch_size=args.batch_size,
            logit_probs_k=args.logit_probs_k,
            save_examples=args.save_examples,
        )
        if args.save_examples and args.max_valid_saved is not None:
            summary["valid_examples"] = summary["valid_examples"][: args.max_valid_saved]
        _print_summary(t, summary)
        if args.save_file:
            _write_json(args.save_file, summary, eval_data_path, args.max_total_tokens)


if __name__ == "__main__":
    main()

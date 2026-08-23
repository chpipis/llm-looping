# Wait, Wait, Wait... Why Do Reasoning Models Loop?

ICML 2026 \
[Charilaos Pipis](https://charispipis.com/)\*,
[Shivam Garg](https://cs.stanford.edu/~shivamg/)\*,
[Vasilis Kontonis](https://vkonton.github.io/),
[Vaishnavi Shrivastava](https://vshrivas.github.io/),
[Akshay Krishnamurthy](https://people.cs.umass.edu/~akshay/),
[Dimitris Papailiopoulos](https://papail.io/)

Code for the synthetic star-graph experiments in the
[paper](https://arxiv.org/abs/2512.12895). The experiments study two
mechanisms behind low-temperature looping:

- hardness-driven risk aversion;
- an inductive bias toward temporally correlated errors.

They also measure how a small amount of explicitly looping training data can
amplify both effects.

This repository builds on the
[Belief State Transformer codebase](https://github.com/microsoft/BST),
which is based on [nanoGPT](https://github.com/karpathy/nanoGPT).

## Installation

The code is tested with Python 3.10.

```bash
pip install -r requirements.txt
```

## Quick start

The following single-GPU example trains the Section 3 G(5,5) model with
exploration at the root.

### 1. Generate training data

```bash
python data/loop_stargraph/prepare.py \
    --walk_type random_walk_with_reset \
    --num_samples 2000000 \
    --num_paths 5 \
    --path_length 5 \
    --max_nodes 30 \
    --max_tokens 200 \
    --reset_probability 0.3 \
    --exploration_probability 0.2
```

### 2. Train

```bash
DATA_FILE=data/loop_stargraph/loopgraph_5_5_nodes_30_walk_random_walk_with_reset_reset_0.3_explore_0.2_maxtokens_200_sample_2000000.txt

CUDA_VISIBLE_DEVICES=0 python train.py \
    data.loop_stargraph_data_path="${DATA_FILE}" \
    data.loop_stargraph_max_nodes=30 \
    trainer.notes=section3_g55_exploration \
    model.n_layer=12
```

`train.py` prints the run directory as:

```text
RUN_DIR=/absolute/path/to/runs/.../run_...
```

Set the shell variable to the printed value before continuing:

```bash
RUN_DIR=/absolute/path/printed/by/train.py
```

### 3. Evaluate

```bash
CUDA_VISIBLE_DEVICES=0 python run_loop_stargraph_validation.py \
    --latest_ckpt_file "${RUN_DIR}/latest_ckpt" \
    --data_path "${DATA_FILE}" \
    --max_total_tokens 200 \
    --temperature 0.0,0.2,0.4,0.6,0.8,1.0 \
    --validation_samples 4000
```

If the matching 4,000-example validation file does not exist, the evaluator
generates it automatically. Evaluation reports:

- accuracy;
- average response length;
- looping count, defined as root-to-source transitions for
  `random_walk_with_reset` and visits to the most-visited root child for
  `pure_random_walk`.

### 4. Plot

```bash
python plotting/plot_validation_curves.py \
    --run "${RUN_DIR}" "G(5,5) w/exp" \
    --out plots/section3_g55_exploration
```

This writes `accuracy.pdf`, `looping_count.pdf`,
`avg_response_length.pdf`, and the joined metrics CSV.

## Paper experiment settings

All experiments use the same data-generation, training, and evaluation
entry points. Unless overridden below, they use these shared settings:

| Setting | Value |
|---|---|
| Training examples | 2,000,000 |
| Held-out evaluation examples | 4,000 |
| Training steps | 100,000 |
| Temperature sweep | `0.0, 0.2, 0.4, 0.6, 0.8, 1.0` |
| `max_nodes` / `max_tokens` for most paper experiments | `30` / `200` |
| Effective batch size / gradient accumulation | `64` / `8` |
| Default model | 12 layers, 8 heads, embedding dimension 768 |
| Optimizer | AdamW, learning rate `1e-4`, weight decay `0` |

Change the following settings to reproduce the main variants:

| Experiment | Graph | Walk/data settings | Training overrides |
|---|---|---|---|
| Hardness-driven looping | G(5,5), G(5,3) | `random_walk_with_reset`; reset `0.3`; exploration `0` or `0.2` | — |
| Temporally correlated errors | G(5,5) | `pure_random_walk`; margin `0`, `0.05`, or `0.1` | — |
| Explicit looping bias | Same as the corresponding base experiment | Add `--looping_data_frac 0.001` to either walk type | — |
| Hardness plus correlated errors | G(5,5) | `pure_random_walk`; margin `0`; correct-arm probability `0.5`; all four wrong arms available | — |
| Hardness plus correlated errors | G(10,5) | `pure_random_walk`; margin `0`; correct-arm probability `0.5`; four sampled wrong arms; `max_nodes=50`; `max_tokens=250` | 4M examples; 300K training and LR-decay steps; evaluate with `--max_total_tokens 250` |
| Model-capacity experiments | G(5,3) | `random_walk_with_reset`; reset `0.3`; no exploration; `max_nodes=50` | 1, 2, and 12 layers |

**Configuration note.** Keep coupled settings consistent:

- When generating data with `--max_nodes 50`, also pass
  `data.loop_stargraph_max_nodes=50` to `train.py`.
- When changing the number of training steps, set both
  `trainer.train_batches` and `lr_scheduler.lr_decay_iters`.
- Set evaluation's `--max_total_tokens` to the `--max_tokens` value used for
  data generation.

The complete shared configuration is in `config/defaults.yaml`. Any setting
can be overridden on the command line using `key=value`, for example:

```bash
DATA_FILE=/path/to/generated/data.txt

python train.py \
    data.loop_stargraph_data_path="${DATA_FILE}" \
    model.n_layer=2 \
    trainer.train_batches=100000 \
    lr_scheduler.lr_decay_iters=100000
```

## Data-generation options

`data/loop_stargraph/prepare.py` supports:

| Flag | Meaning |
|---|---|
| `--num_paths` | Number of arms in G(n, l) |
| `--path_length` | Length of each arm |
| `--max_nodes` | Node-label vocabulary size (default `30`) |
| `--max_tokens` | Maximum tokenized length of the full serialized example (prompt, walk, and EOS) |
| `--reset_probability` | Per-step reset probability for `random_walk_with_reset` |
| `--exploration_probability` | Probability of choosing a random wrong arm at the root |
| `--margin` | Probability reduction for previously visited arms |
| `--correct_arm_probability` | Fixed probability assigned to the correct arm |
| `--num_wrong_arms_explored` | Number of wrong arms with nonzero probability |
| `--looping_data_frac` | Fraction of examples replaced by deterministic looping traces |

## Root-transition probability plots

Evaluate both a baseline and bias-trained run with `--save_examples` and
`--logit_probs_k 5`, then run:

```bash
BASE_RUN_DIR=/path/to/baseline/run
BIAS_RUN_DIR=/path/to/bias-trained/run

python plotting/plot_root_visit_probs.py \
    --base "${BASE_RUN_DIR}" \
    --bias "${BIAS_RUN_DIR}" \
    --target-strategy source \
    --temperature 0.0 \
    --base-label "G(5,5)" \
    --bias-label "G(5,5) w/bias" \
    --out plots/reset_probability_overlay.pdf
```

Use `--target-strategy most_visited_child` for `pure_random_walk`.

## Repository layout

```text
data/loop_stargraph/             data generation and walk samplers
data/loop_stargraph.py           tokenizer and datamodule
plotting/                        reusable plotting utilities
config/defaults.yaml             shared training configuration
train.py                         training entry point
core_train.py                    trainer, checkpointing, and LR schedule
run_loop_stargraph_validation.py evaluation entry point
model_base.py, model_gpt.py      GPT implementation
```

## Project policies

- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Security policy](SECURITY.md)
- [Transparency documentation](TRANSPARENCY.md)

## Citation

```bibtex
@article{pipis2025wait,
  title={Wait, Wait, Wait... Why Do Reasoning Models Loop?},
  author={Pipis, Charilaos and Garg, Shivam and Kontonis, Vasilis and Shrivastava, Vaishnavi and Krishnamurthy, Akshay and Papailiopoulos, Dimitris},
  journal={arXiv preprint arXiv:2512.12895},
  year={2025}
}
```

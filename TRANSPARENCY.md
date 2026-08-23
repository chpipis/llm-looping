# Transparency documentation

## Overview

This repository contains research code for the synthetic star-graph
experiments in *Wait, Wait, Wait... Why Do Reasoning Models Loop?* It includes
the experiment setup, transformer training, evaluation, and plotting code used
to isolate and study mechanisms that may contribute to low-temperature looping
in reasoning language models.

The repository does not provide trained model weights, a dataset, a hosted
model, an API, or a user-facing service. Users run the experiments locally.

## What the code can do

The code can:

- train small transformer models on synthetic graph-traversal tasks;
- evaluate accuracy, response length, and task-specific looping behavior
  across decoding temperatures; and
- generate the validation and root-probability plots used to analyze the
  experiments.

The paper provides a detailed discussion of the experimental design and
findings: https://arxiv.org/abs/2512.12895.

## Intended uses

The repository is intended for AI and machine-learning researchers and paper
readers who want to reproduce the experiments in the paper and conduct further
research on them.

## Out-of-scope uses

This is research code, not production software or a deployed AI system. It was
not designed or evaluated as a general-purpose language model, production
training framework, safety certification, or decision-making system.

These synthetic experiments provide a simple, controlled setting for making
progress on understanding looping. Real LLMs are substantially more complex,
and additional forces beyond those isolated here may contribute to looping.
The findings should therefore be generalized with care; see the paper for a
fuller discussion of their scope and limitations.

## How to get started

Follow the installation and quick-start instructions in
[`README.md`](README.md). The README also documents the complete experiment
settings and configuration values that must remain consistent across data
preparation, training, and evaluation.

## Evaluation

The experiments evaluate small transformer models on synthetic star-graph
traversal tasks. Evaluation measures include:

- task accuracy;
- generated response length; and
- looping count.

The paper reports results across decoding temperatures and training
distributions. At a high level, the experiments reproduce low-temperature
looping in controlled settings and provide mechanistic evidence relevant to
looping observed in reasoning language models.

## Limitations

- The synthetic graph-traversal tasks and relatively small transformer models
  provide a simple setting for isolating mechanisms relevant to looping, but
  real LLMs are substantially more complex and may involve additional forces.
  See the paper for a fuller discussion.
- Results depend on the task, training, model, decoding, and evaluation
  configurations. Material changes can produce different results.
- The repository is research code rather than production-hardened software,
  and full reproduction requires nontrivial compute.
- No trained weights or datasets are distributed.

## Best practices

- Use the documented configurations when reproducing reported results.
- Keep coupled settings, such as tokenizer graph size and training/evaluation
  sequence limits, consistent.

## License

The code is released under the [MIT License](LICENSE).

Nothing in this transparency document modifies or restricts the license.

## Trademarks

This project may contain trademarks or logos for projects, products, or
services. Authorized use of Microsoft trademarks or logos is subject to
Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos
in modified versions of this project must not cause confusion or imply
Microsoft sponsorship.

## Contact

This research was conducted by researchers affiliated with Microsoft Research,
MIT, and the University of Wisconsin–Madison.

For project questions, use the repository's GitHub issue tracker or contact
`chpipis@mit.edu` and `shigarg@microsoft.com`. Security issues should be
reported according to [`SECURITY.md`](SECURITY.md), not through public GitHub
issues.

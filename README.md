# CORA — Context-Option Ranking Architecture

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)](https://huggingface.co/docs/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**A calibrated multimodal bi-encoder that ranks predefined answer options against text + image context.**

CORA is *not* a generative model. It does not produce free-form text — it scores
a closed set of schema-defined options and returns **calibrated probabilities**
for each. This makes it suitable for high-stakes decision systems where
"hallucinating a type" is unacceptable and honest uncertainty matters.

## Highlights

- **Non-autoregressive** — all options scored in a single batched forward pass
- **Multimodal** — text state + question + optional image (SigLIP 2 / CLIP)
- **Closed-set softmax** — cannot emit an answer outside the schema
- **Calibrated** — trained with soft cross-entropy, post-hoc temperature scaling, ECE-reported
- **Variable schema** — any number of questions and options per example, no fixed heads
- **Cheap inference** — option embeddings cacheable, state embedding reusable across questions

## Architecture

```text
state_text ──┐
             ├─► ModernBERT ──► state_repr   ─┐
question ────┘                                 │
                                               ├─► context_mlp ──► context [512]
image (opt) ─► SigLIP 2 ──► image_repr        ─┘

option_i ─► ModernBERT ─► option_repr_i ─┐
                                          ├─► score_i = <context, option_repr_i> / √D
...                                       ┘
                                          softmax → calibrated probabilities

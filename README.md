# DraftScope

[![CPU checks](https://github.com/6bjjjj/draftscope/actions/workflows/ci.yml/badge.svg)](https://github.com/6bjjjj/draftscope/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Status: research preview](https://img.shields.io/badge/status-research_preview-orange.svg)](docs/validation.md)

Prefix-aware target-head verification and reproducible NVIDIA inference experiments.

DraftScope combines a Transformers baseline, a pinned vLLM INT4 + DFlash2 preset, and an optional experimental head that makes dense-fallback decisions using the verification prefix that affects greedy token acceptance.

**Alpha research software.** Historical GPU measurements and the validation status of the installable package are reported separately below. [العربية](README.ar.md)

[Download the research preview](https://github.com/6bjjjj/draftscope/releases) · [Compare the mechanisms](docs/related-work.md) · [Contribute a reproducible report](CONTRIBUTING.md)

## Try it without a GPU

```sh
git clone https://github.com/6bjjjj/draftscope.git
cd draftscope
python -m venv .venv
# Activate the virtual environment for your shell.
python -m pip install .
draftscope plan
```

This prints the default experiment plan without downloading models or allocating GPU memory. For a GPU experiment, follow the backend-specific installation instructions below. NVIDIA compatibility reports and independent numerical reviews are welcome through [issues](https://github.com/6bjjjj/draftscope/issues).

## Features

| Feature | Practical purpose |
|---|---|
| Prefix-conditioned dense fallback | Numerical uncertainty in a later, unused verification row need not force a full dense-head fallback. The target forward pass and coarse/refinement stages still run for all verification rows. |
| Head extension without additional training | Uses a Marlin W8A16 coarse head, top-64 candidate refinement and conservative score bounds. Retains the dense BF16 head for fallback; adds memory. The DFlash2 drafter itself is a pretrained checkpoint. |
| Device-side branch selection | Integrates CUDA conditional graphs to select the head path without a host-side predicate read at each branch. |
| Capability-based NVIDIA routing | Selects native BF16 or FP16 for the general path. The optional extension builds for the detected architecture and records why it remains inactive when prerequisites are missing. |
| Auditable, bounded experiments | Pins model revisions, hashes inputs, saves partial results, checks repeated greedy outputs and separates request from decode timing. The CLI supervises a worker with a time limit. |

## Relationship to earlier work

DraftScope studies the **target output head and its fallback policy** within an existing speculative-decoding engine. DFlash addresses drafting, FlashHead addresses output-head retrieval, and SpecVocab selects a vocabulary subset for the drafter. Coarse/refine selection with dense fallback also has public prior implementations. The [sourced comparison](docs/related-work.md) explains these overlaps and the narrower prefix-conditioned mechanism implemented here.

The historical comparison is against the same stock INT4 + DFlash2 engine. A controlled performance comparison against FlashHead, SpecVocab or other published systems has not been run; their reported speedups use different settings.

## What works where

| Path | Platform | Hardware policy | Validation status |
|---|---|---|---|
| Transformers + SDPA | Linux / Windows | NVIDIA CUDA; compatible PyTorch build and sufficient VRAM | Adapter tests on CPU; packaged GPU adapter awaits integration testing |
| vLLM 0.31.0 | Linux | NVIDIA compute capability 7.5+; model/kernel requirements may be stricter | Historical A100 experiments; packaged adapter awaits integration testing |
| Experimental prefix V8 | Linux | Capability >=8.0, BF16, matching toolkit, enough free VRAM; no GPU-name whitelist | Historical A100 only; portable build awaits GPU validation |
| Planning / result analysis | Linux / Windows / macOS | CPU, no Torch dependency | Automated CPU tests |

Automatic dtype is native BF16 on Ampere or newer, FP16 otherwise. AMD, Intel and Apple GPU inference are rejected. A compatible CUDA wheel/driver is still required; GPU detection alone does not prove that every model operator works. Old NVIDIA architectures removed from a PyTorch wheel are not supported by that wheel. No silent CPU offload, model substitution or quantization occurs.

**Not every NVIDIA card has been tested.** The historical measurements used one A100-SXM4-80GB. A 27B model will not fit every card. Select a smaller model explicitly when necessary.

## Install

Python 3.11+ is required. From this repository:

```sh
python -m venv .venv
# Activate the virtual environment for your shell.
python -m pip install .
draftscope plan
```

Planning and analysis do not download weights or initialize CUDA. For the general GPU path, first install a CUDA-enabled PyTorch build appropriate for your driver/GPU using the [official installer](https://pytorch.org/get-started/locally/), then:

```sh
python -m pip install '.[transformers]'
draftscope doctor
```

For the pinned vLLM path use a **separate Linux environment**:

```sh
python -m pip install '.[vllm]'
draftscope plan --backend vllm --int4-dflash
```

Do not combine independently pinned Torch and vLLM environments. Consult [vLLM GPU installation](https://docs.vllm.ai/en/v0.31.0/getting_started/installation/gpu/) for wheel/driver constraints. Windows users can use the Transformers adapter; native Windows vLLM is rejected.

## Run a bounded experiment

These commands load/download weights and consume GPU time only when explicitly run:

```sh
draftscope benchmark --prompts examples/prompts.json --output runs/baseline-001 --max-seconds 600 --repeats 5
draftscope benchmark --backend vllm --int4-dflash --prompts examples/prompts.json --output runs/dflash-001 --max-seconds 900 --memory-fraction 0.60
draftscope analyze runs/dflash-001
```

The default target is the pinned Qwen3.8-27B checkpoint. A custom remote model requires both `--model OWNER/NAME` and `--revision FULL_COMMIT_SHA`. The remote code option is disabled. Checkpoint and dataset licenses remain their authors' licenses; no model weights are included.

`--max-seconds` includes model loading, compilation and warmups. The CLI isolates the worker and terminates it on timeout; partial files survive. Kernel/driver teardown may take additional time. The Python `runner.run` API checks its time budget between calls and is **not a hard-timeout supervisor**. Prefer the CLI for bounded GPU allocation. `--memory-fraction` caps the Transformers CUDA allocator, or vLLM's configured GPU-memory utilization; external allocations and the driver are outside that cap. OOM is reported without silently moving to CPU.

Each new output folder contains the frozen configuration, prompt hash, hardware metadata, incremental token IDs/timings, completion status and summary. Existing folders are never overwritten. Prompts and generated text are saved locally; inspect them before sharing results.

## Timing and quality

Runs are greedy, batch one, with no prefix cache. Whole-request timing includes prefill and backend overhead for pretokenized inputs. It excludes loading, warmups, tokenization and detokenization. Request throughput is `generated_tokens / request_seconds`; measured decode throughput is `(generated_tokens - 1) / (last_token_ts - first_token_ts)`. Transformers reports decode throughput as null because this adapter has no per-token timestamps.

Repeated greedy outputs must agree within a run. This check is not a task-quality benchmark or a proof of agreement with another checkpoint. **`quality_accepted` remains null.** `--fixed-tokens` suppresses normal EOS stopping for timing studies; omit it for ordinary generation. The example prompts are smoke inputs, not a quality evaluation or reproduction of the six historical engineering inputs.

## Preserved research evidence

Historical A100 results, separate from this package build:

| Study | Observed result | Interpretation |
|---|---|---|
| Original INT4 + DFlash2 | Six case medians: **105.05–171.34 request tok/s**, **157.92–241.42 decode tok/s** | Reference configuration using established methods |
| Prefix-conditioned fallback, captured head study | **33 → 9 fallback calls** across 78 captured calls | **72.7% fewer fallback calls** in this component study; earlier numerical guard |
| Compiled prefix head component | **18.95% lower head latency** | Sum of per-call median wall times for 72 eight-row calls, 10 repetitions each; earlier guard; head only |
| Whole-model V8 versus same INT4 + DFlash2 engine | **0.922% lower request latency** | Below the frozen >1% admission threshold |
| V8 sequence checks | 30 timing pairs (60 records) and 12 diagnostic pairs matched | Finite test coverage, not a proof of exactness |
| Held-out task quality | Final outputs not recovered after runtime disconnected | **Inconclusive**; quantization quality not accepted |

The original scripts, frozen inputs and paired records are under [research](research/README.md). Hashes are checked by `python scripts/verify_evidence.py`. Negative results are retained. No GPU work was restarted after the original authorization deadline. CPU tests cannot validate CUDA graph lifetime, numerical fidelity or cross-GPU speed.

## Experimental extension

See [experimental limitations](docs/experimental.md). The extension is **off by default** and requires explicit `--experimental-prefix` with `--backend vllm --int4-dflash`. Install `.[prefix]` in the vLLM environment for the CUDA bindings. It compiles for the actual GPU architecture and runs a synthetic startup check before activation. Missing capability, toolkit or memory requirements leave the stock engine active, with a recorded reason; CUDA execution errors stop the run rather than reuse a potentially damaged context. There is no A100-name restriction. It retains the full dense head and adds memory. Its historical gain did not meet the admission gate. Portability checks do not imply that all NVIDIA cards have been tested or will get the same speedup.

## Development

```sh
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
python -m ruff check .
python scripts/verify_evidence.py
python -m build
python -m twine check dist/*
```

CI covers CPU unit/integration tests and packaging. GPU smoke testing is a separate, explicitly invoked benchmark; CI never allocates a GPU or downloads models. See [validation](docs/validation.md). Code is licensed under Apache-2.0; attribution and external dependencies are described in [NOTICE](NOTICE).

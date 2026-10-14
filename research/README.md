# Preserved A100 experiments

These files are selected byte-for-byte records from the original Colab session. They are separate from measurements of the installable package. `manifest.json` records SHA256 hashes. `python scripts/verify_evidence.py` checks those bytes, fixed input/source hashes, paired sequence agreement, repetitions and the V8 whole-request latency arithmetic without GPU access.

The source archive through V8 was `results_qwen38_prefix_v8.zip`, SHA256 `d0ec4240250c49d3e59a8439383459bdbf8ef8325e9aa35422e1160ae77d8b23`. The original V8 worker SHA256 is `2c523730cb8826b6d7b2dcd18c8c88ed5c9b87e0bb69aa61800663d9dad510ac`. Cloud `/content` and `/root/.cache` paths in these original files describe the historical runtime. They are not portable package configuration.

Six engineering inputs: 128, 512 and 2048 input tokens, two cases each, 128 fixed generated tokens, batch one, five timed repetitions. V8 uses paired alternating stock/candidate runs within the same engine and retains the same memory allocation. Twelve new diagnostic pairs provide limited additional sequence checks. Original source contains the exact protocol, stop policy and seed. The importable package's example chat prompts do not reproduce this fixed-token input protocol.

V4 increased whole-model latency by 5.28%; V6 encountered a CUDA illegal memory access before timings; V7 and V8 reduced latency by about 0.899% and 0.922%. Both failed the frozen >1% gain requirement. V1–V3 were earlier integration failures or unused-path trials, not successful timing evidence. The compiled component experiment reports a head-only gain with an earlier arithmetic guard. These outcomes are not evidence of a new general-purpose algorithm.

**Quality is inconclusive.** The independent 512-question / 64-block quality audit lost its final result when the runtime disconnected. The last confirmed live progress was 216 BF16 questions, not 216 completed baseline/candidate pairs. No quality-score acceptance is inferred from the saved timing records. An earlier Arabic diagnostic was corrupted by a harness encoding bug; it is excluded from quality claims. The frozen engineering timing inputs were unaffected.

Archived scripts retain their original expired deadline and paths and are not imported by the package. They are evidence, not a promise that running an old script starts a portable experiment. The refactored worker adds capability-based dispatch, dynamic CUDA architecture compilation, VRAM admission and startup qualification; it needs a fresh authorized GPU validation. CPU CI verifies integrity without executing any archive script.

Dependencies and prior art include [vLLM](https://github.com/vllm-project/vllm), [DFlash](https://github.com/z-lab/dflash), and the model publishers linked in the pinned configurations. INT4, MTP, DFlash and coarse/refine verification were not invented by this project. Full novelty and quality reviews are still required before a research submission.

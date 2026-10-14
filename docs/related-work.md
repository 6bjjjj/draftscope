# Features and relationship to prior work

Source review: 14 October 2026. This comparison identifies implementation choices and overlapping prior work. Published performance numbers are not a ranking: the models, GPUs, decoding policies and measurement boundaries differ. DraftScope has not run a controlled comparison against these external implementations.

## What DraftScope implements

The experimental extension changes the target output head inside a pinned vLLM INT4 + DFlash2 configuration. Its supported setting is greedy decoding, batch one, seven draft tokens and eight verification rows. The general Transformers adapter has a broader model and NVIDIA hardware scope than this extension.

1. Compute the target hidden states as usual.
2. Score the full vocabulary with a Marlin W8A16 coarse head, then refine the top 64 candidates per verification row using the retained weights and conservative score bounds.
3. Determine which rows are needed to resolve greedy draft acceptance, including the replacement or bonus token as applicable.
4. Use the candidate path only when the bounds certify the required decisions; otherwise run the dense BF16 head. CUDA conditional graphs select the branch on the device.

For example, suppose certified target choices agree with the first two draft tokens and reject the third. Numerical uncertainty at row six alone then need not force dense fallback: that later row does not affect the returned prefix. If a required decision is uncertain, the fallback computes the full dense head.

**The saved work is conditional dense-head fallback.** The target forward pass, coarse vocabulary scoring and candidate refinement still run for all verification rows. The full BF16 head stays resident, so the extension requires additional memory. Its bounds depend on numerical assumptions about the kernels; finite agreement checks do not establish a universal exactness guarantee.

## Comparison by mechanism

| Prior work and source status | Established focus | DraftScope's implementation choice and overlap |
|---|---|---|
| [DFlash: Block Diffusion for Flash Speculative Decoding](https://proceedings.mlr.press/v306/chen26bv.html), ICML 2026 | A block-diffusion drafter proposes tokens in parallel using target-model context features. | DraftScope consumes a released DFlash2 drafter preset and studies target-head fallback cost. It does not introduce or train a new drafter. The original DFlash paper and the later DFlash2 checkpoint are distinct artifacts. |
| [FlashHead](https://arxiv.org/abs/2603.14591), arXiv preprint | Uses balanced vocabulary clustering, multi-probe retrieval, sampling and selective quantization to accelerate the output head. | DraftScope scans the whole vocabulary coarsely and refines candidates without an offline vocabulary-clustering step. Its investigated control is prefix-conditioned fallback, restricted to greedy verification. Training-free head acceleration is already prior art. |
| [SpecVocab](https://aclanthology.org/2026.findings-acl.2000/), Findings of ACL 2026 | Selects a vocabulary subset per decoding step to address the draft model's output-embedding bottleneck. | DraftScope operates on the target verification head. It does not implement adaptive draft-vocabulary selection. These address different costs; their combined benefit has not been tested here. |
| [certhead 0.1.0](https://pypi.org/project/certhead/0.1.0/), public software release | Describes coarse scoring, score intervals, candidate refinement and dense fallback when intervals cannot resolve the winner. | This is direct overlap with the broad head-selection strategy. DraftScope investigates which verification rows must trigger fallback and integrates that decision with vLLM/Marlin and CUDA graphs. We do not treat a software release as a verified peer-reviewed paper or independently validate its authors' guarantees. |
| [From Positionwise Confidence to Prefix Scheduling: Verifier Skipping in Speculative Decoding](https://arxiv.org/abs/2608.14787), preprint marked accepted at UncertaiNLP 2026, non-archival | Studies a lossy confidence-based policy that commits a draft prefix without invoking the target verifier. | DraftScope still invokes the target model and uses numerical uncertainty to select a dense output-head fallback. It does not implement confidence-based skipping of the entire verifier. Prefix scheduling itself is therefore not a sufficient novelty claim. |

Device-side branching uses the existing [CUDA conditional graph facility](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html). Marlin, INT4, speculative decoding and coarse/refine selection are dependencies or prior techniques. The specific combination implemented here is a research hypothesis to evaluate, not proof of scientific priority or superiority.

## Engineering features in the package

- **Capability-based dispatch:** NVIDIA detection and dtype selection; architecture-specific compilation for the optional head; prerequisite and free-memory checks with recorded fallback reasons. Physical validation remains A100-only.
- **Inspectability:** pinned checkpoint revisions, input hashes, original evidence hashes and explicit activation status. The head extension requires no additional training; the supplied drafter is already trained.
- **Bounded execution:** a CLI worker supervisor, exclusive output directories and incremental atomic saves preserve partial results after failure or timeout.
- **Measurement boundaries:** request and decode timing are distinct; absent token timestamps stay null. Greedy repeat consistency and task-quality acceptance are distinct checks.

These are useful library features. This comparison does not assert that other research implementations lack them.

## What the measurements support

All GPU numbers below come from the original A100-SXM4-80GB experiments. They are not measurements of the refactored installable package.

| Observation | Evidence and boundary |
|---|---|
| Fewer dense-fallback calls in a captured head study | Across 78 captured calls / 582 verification rows, whole-call fallback triggered 33 times and prefix-conditioned fallback 9 times: a 72.7% reduction in fallback count. This is not a latency percentage. See the [saved outcomes](../research/results/head-prefix-compiled-v1/summary.json). |
| Lower isolated head latency | The compiled prefix variant reduced the sum of per-call median wall times by 18.95% against the dense head across all 72 eight-row calls, with 10 repetitions each. No fast-call subset was selected. The [raw timings](../research/results/head-prefix-compiled-v1/timings.json) include slower/fallback cases. This component used an earlier numerical guard than V8. |
| Small whole-model V8 gain | Paired alternating measurements on six frozen engineering inputs, five repetitions per mode, gave 0.922% lower request latency using the geometric mean of per-case latency ratios. The comparator was the same stock INT4 + DFlash2 engine. This missed the frozen >1% admission threshold. |
| Limited sequence agreement | All 30 timing pairs and 12 additional diagnostic pairs matched. The independent task-quality audit has no recovered final result, so quality acceptance remains unknown. |

The head-only calculation is reproducible on CPU from the repository root:

```python
from collections import defaultdict
import json
from pathlib import Path
from statistics import median

rows = json.loads(Path("research/results/head-prefix-compiled-v1/timings.json").read_text())
groups = defaultdict(list)
for row in rows:
    if row["m"] == 8 and row["variant"] in ("dense", "compiled_prefix"):
        groups[row["prompt_case"], row["case"], row["variant"]].append(row["wall_ms"])
assert len(groups) == 144 and all(len(values) == 10 for values in groups.values())
totals = {
    variant: sum(median(values) for key, values in groups.items() if key[2] == variant)
    for variant in ("dense", "compiled_prefix")
}
print(100 * (1 - totals["compiled_prefix"] / totals["dense"]))
```

Run `python scripts/verify_evidence.py` to verify the archived hashes and recompute the V8 whole-model result. The [research record](../research/README.md) retains unsuccessful variants and the incomplete quality audit.

A stronger research claim would need a controlled comparison with relevant head methods, a full-model ablation separating prefix conditioning from graph compilation, a broader independent quality/fidelity evaluation, and physical tests across models and NVIDIA architectures. The current evidence supports an implemented mechanism and a component-level observation; it does not establish an overall performance lead over published work.

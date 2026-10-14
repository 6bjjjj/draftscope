"""Verify the archived bytes and recompute the paired V8 result on CPU."""

from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from draftscope.storage import sha256, verify_manifest


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    count = verify_manifest(root, manifest)
    folder = root / "results/int4_dflash7_prefix_v8"

    def load(name):
        return json.loads((folder / name).read_text(encoding="utf-8"))

    protocol, completion = load("protocol.json"), load("completion.json")
    if sha256(root / "benchmark_inputs.json") != protocol["input_sha256"]:
        raise ValueError("Input hash does not match the frozen protocol")
    if sha256(root / "archive/prefix_extension_v8.py") != protocol["runtime_extension"]["sha256"]:
        raise ValueError("Original worker source hash mismatch")
    rows, checks = load("paired_timing.json"), load("fidelity_checks.json")
    if completion["status"] != "complete" or len(rows) != 60 or len(checks) != 12:
        raise ValueError("Incomplete V8 evidence")
    groups = defaultdict(list)
    for row in rows:
        if len(row["token_ids"]) != 128 or row["wall_seconds"] <= 0:
            raise ValueError("Invalid fixed-length timing")
        if not math.isclose(
            row["output_tokens_per_second"], 128 / row["wall_seconds"], rel_tol=1e-9
        ):
            raise ValueError("Saved timing arithmetic mismatch")
        groups[row["length"], row["case"]].append(row)
    if set(groups) != {(n, i) for n in (128, 512, 2048) for i in (0, 1)}:
        raise ValueError("Unexpected engineering cases")
    ratios = []
    for rows in groups.values():
        if len(rows) != 10 or len({tuple(r["token_ids"]) for r in rows}) != 1:
            raise ValueError("Missing pairs or sequence disagreement")
        medians = {}
        for mode in ("stock", "prefix"):
            selected = [r for r in rows if r["mode"] == mode]
            if len(selected) != 5 or {r["repeat"] for r in selected} != set(range(5)):
                raise ValueError("Missing or duplicated repetitions")
            medians[mode] = statistics.median(r["wall_seconds"] for r in selected)
        ratios.append(medians["prefix"] / medians["stock"])
    if not all(r["stock"] == r["prefix"] for r in checks):
        raise ValueError("Diagnostic mismatch")
    if load("extension_stats.json")[0]["certified_prefix_calls"] <= 0:
        raise ValueError("Fast path was not exercised")
    ratio = statistics.geometric_mean(ratios)
    return {
        "verified_files": count,
        "v8_latency_reduction_percent": 100 * (1 - ratio),
        "passes_frozen_minimum_gain": ratio < 0.99,
        "quality_accepted": None,
        "packaged_build_gpu_tested": False,
    }


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parents[1] / "research"), indent=2))

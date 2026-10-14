# Contributing to DraftScope

The most useful contributions are reproducible NVIDIA compatibility reports, numerical counterexamples and improvements to the measurement harness. You can also contribute documentation fixes without a GPU.

## Report a result

Use the [GPU / benchmark report](https://github.com/6bjjjj/draftscope/issues/new?template=gpu-report.yml) template. Include the package commit, GPU name and compute capability, driver/CUDA/PyTorch/backend versions, exact command, model revision and completion status.

For speed comparisons, keep model weights, tokenized inputs, output length, sampling policy and memory allocation fixed. Alternate baseline/candidate run order and report per-case timings with repetition counts. Identify whether you measured the head, decode or whole request. Report token mismatches and failed runs alongside successful cases. A fast run alone is not a quality pass.

The experimental head has narrow requirements and is off by default. Read [its scope](docs/experimental.md) before trying it. Current historical results are A100-only; a compatibility check is not physical validation on another card. Run only on compute you control with an explicit time and memory budget.

Run artifacts can contain prompts and generated text. Share public test inputs or a minimal reproducer; remove credentials and personal data from logs. Do not upload model weights. If full artifacts are unsuitable for a public issue, include sanitized environment details and an explanation of the failure.

## Make a code contribution

1. Fork the repository and create a branch for one focused change.
2. Install the development extras with `python -m pip install -e '.[dev]'`.
3. Run `python -m unittest discover -s tests -v`, `python -m ruff check .` and `python scripts/verify_evidence.py`.
4. Open a pull request explaining the behavior change, the evidence and any GPU tests actually run. Distinguish CPU mocks from device tests.

Keep `research/archive` and its evidence files byte-for-byte intact. Add new results as separately identified experiments with their source revision and frozen inputs. Do not tune acceptance rules on held-out answers.

AI-assisted contributions are welcome. Review the generated code, describe its validation honestly and retain attribution. Contributions are made under this repository's Apache-2.0 license.

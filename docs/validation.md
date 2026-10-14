# Validation scope

The project separates CPU software checks from physical GPU evidence.

Automated CPU tests exercise configuration rejection, NVIDIA capability routing, output arithmetic, exclusive/atomic result files, partial failure preservation, worker lifecycle, command-line planning and archive integrity. Mock hardware tests do not execute CUDA kernels. The `plan` command must work without Torch or network access. Build checks produce and install a wheel; CI repeats CPU checks on Linux and Windows.

Historical GPU evidence covers an A100-SXM4-80GB with vLLM 0.31.0, Transformers 5.17.0 and PyTorch 2.13.0+cu130. It describes original scripts, not a completed validation of the refactored adapters. Turing, Ampere consumer cards, Ada, Hopper and Blackwell have not been physically tested for this package. The general adapter dispatches according to capabilities only when the installed CUDA/PyTorch stack supports the device.

No GPU was restarted during packaging because the original 20-hour authorization expired. Further GPU validation requires a new authorized run. The quality audit has no recovered final paired result; quality acceptance remains unknown.

For a future GPU validation, use `doctor`, then a new bounded `benchmark` output directory. Verify actual hardware metadata, model commit, output IDs and completion status. Compare baseline and candidate using the same tokenization, fixed policy and independent quality inputs. Speed alone is not a quality pass. A full cross-GPU support claim requires results from each advertised hardware/stack combination.

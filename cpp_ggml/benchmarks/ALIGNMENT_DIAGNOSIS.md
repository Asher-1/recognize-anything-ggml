# Tagging alignment diagnosis

Downloads: [GitHub RAM release](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/RAM)
and [Hugging Face RAM_GGUF](https://huggingface.co/Asher-1/RAM_GGUF/tree/main).
The published acceptance numbers and charts are in [LATENCY_MATRIX.md](LATENCY_MATRIX.md).

## Separate weight error from execution error

`scripts/parity_reference.py --gguf FILE` runs the official PyTorch graph with
dequantized GGUF weights. This optional diagnostic isolates weight rounding
from backend arithmetic; it does not replace the official checkpoint baseline
in the acceptance harness.

For the original RAM Q8 export on `images/demo/demo1.jpg`, the diagnostic found:

| Comparison | Logit MAE | Probability MAE |
|---|---:|---:|
| Original checkpoint vs dequantized Q8 weights, both PyTorch | 0.0028411 | 0.0004764 |
| Dequantized Q8 PyTorch vs original CUDA MMQ execution | 0.0083594 | 0.0014001 |
| Original checkpoint vs original CUDA MMQ execution | 0.0087630 | 0.0014623 |

Execution error was larger than weight-only error. The CUDA MMQ path also
quantized the floating activation to Q8_1. The repaired CUDA tagging graph
uses floating activations for Q8_0 matrices, FP32 accumulation and FP32 output.
The single CMake-applied GGML patch permits this supported cuBLAS combination
and honors `GGML_PREC_F32` for its output precision.

RAM's small `tagging_head`, `image_proj`, `wordvec_proj` and `fc` layers are
retained in F16. Eligible Swin matrices remain Q8_0. This adds about 13 MiB
over the previous RAM Q8 export; the file is 534.4 MiB. The quantization policy
is embedded in `ram.quantization.f16_prefixes`. Neither RAM's official
thresholds nor the Python checkpoint or GT annotations were changed.

The repaired 50-frame tagging matrix reached a minimum dataset F1 of
0.991781, including RAM CUDA Q8_0. See the current matrix for the speed gate;
accuracy results from a run with GPU contention do not establish its speed.

## Remove work that cannot depend on the image

Fixed RAM/Tag2Text label projections and the first cross-attention query are
computed once when initializing the resident model. RAM++ description layout
conversion is also cached once. Its image-conditioned description weighting
remains in the image graph.

For CUDA tagging in `--cuda-compute auto`, F32 matrix conversion is cached
once. The GGUF remains F32; the default execution uses FP16 GEMM inputs with
FP32 accumulation/output. The cache costs additional resident GPU memory.
`--cuda-compute f32` keeps FP32 inputs and skips this cache. Caption auto mode
retains its existing FP32 computation policy.

## Reduce Vulkan dispatch and copies

Tagging attention combines relative-position bias and the shifted-window mask
once. `ggml_soft_max_ext` then performs scaling, bias and softmax together.
The Q/K/V channel slices remain views until the head permutation actually
requires contiguous data. This removes redundant copies without changing the
attention dimensions or the checkpoint's masking rule. Tagging matrix
products request FP32 accumulation; caption arithmetic retains its separately
validated policy.

## Audit timings on a shared GPU

Transient RelateAnything inference processes were observed during exploratory
runs. Those runs are retained locally for diagnosis and excluded from the
public speed result. `full_eval.py --wait-gpu-idle` samples compute PIDs every
0.5 seconds, records the audit in each command log, waits for unrelated
transient work, and retries a complete command if overlap is detected.
`--gpu-retries` limits retries. Persistent overlap or a failed audit makes the
run fail instead of silently accepting contaminated timings.

Long-lived workstation services must be allowed explicitly with repeated
`--allow-gpu-pid PID`. Their workload remains a limitation of the measured
hardware result; no other user's process is stopped. The audit observes NVIDIA
compute processes, including the Vulkan compute process on this device. It
cannot guarantee absence of all desktop graphics activity or interference
shorter than the sampling interval. Reproduce the final matrix on an otherwise
idle GPU for comparisons across machines.

## Reproducible source changes

All GGML modifications are contained in
`third_party/patches/0001-ram-ggml-v0210-integration.patch`, applied by CMake to
the pinned `v0.21.0` submodule. Forward application, reverse application and
byte equality with the active patched tree are verified against a pristine
checkout. The CUDA and Vulkan inference and API targets are built from that
same patched revision. CUDA links no cuDNN library.

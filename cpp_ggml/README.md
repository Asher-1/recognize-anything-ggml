# C++ GGML integration

Current GGUF downloads: [GitHub RAM release](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/RAM) and
[Hugging Face `Asher-1/RAM_GGUF`](https://huggingface.co/Asher-1/RAM_GGUF/tree/main).

This directory contains the C++ inference path for RAM, RAM++ and Tag2Text.
The three official inference checkpoints are present under
`models/pytorch/`:

| model | checkpoint | backbone | output |
| --- | --- | --- | --- |
| RAM | `ram_swin_large_14m.pth` | Swin-L | 4585 tags |
| RAM++ | `ram_plus_swin_large_14m.pth` | Swin-L | 4585 tags |
| Tag2Text | `tag2text_swin_14m.pth` | Swin-B | 3429 tags |

These are the complete pretrained checkpoints used by the repository's
official inference entry points. Training datasets and auxiliary tag
description files are separate from the inference model set.

## Build

Initialize the pinned ggml submodule and build one backend per build directory:

```bash
git submodule update --init cpp_ggml/third_party/ggml
cmake -S cpp_ggml -B cpp_ggml/build-cuda \
  -DRAM_GGML_CUDA=ON -DRAM_GGML_VULKAN=OFF
cmake --build cpp_ggml/build-cuda -j8

cmake -S cpp_ggml -B cpp_ggml/build-vulkan \
  -DRAM_GGML_CUDA=OFF -DRAM_GGML_VULKAN=ON
cmake --build cpp_ggml/build-vulkan -j8
```

`CMakeLists.txt` rejects a different ggml revision and applies every sorted
patch in `third_party/patches/` idempotently. ggml is pinned to
`8599e0ea3756c4bac4ef813af2241cb1a8bbfb0b` (`v0.21.0`). The integration does
not enable cuDNN; CUDA links only the ggml CUDA backend and CUDA runtime/math
libraries, while Vulkan links the Vulkan backend. CUDA builds enable ggml CUDA
Graphs by default for repeated inference; set `-DGGML_CUDA_GRAPHS=OFF` when
diagnosing a driver-specific issue.
OpenMP is used when found to parallelize the exact image resize path; the
published pipeline timings below used eight preprocessing threads.

## Public C++ API

The `ram-ggml-api` target exposes one options object so adding runtime settings
does not change a long function signature. A consumer can link the target and
pass all process configuration explicitly:

```cpp
#include <ram_ggml/inference.hpp>
#include <stdexcept>
#include <string>

ram_ggml::InferenceOptions options;
options.model_path = "models/gguf/ram_swin_large_14m-f16.gguf";
options.image_path = "images/demo/demo1.jpg";
options.backend = ram_ggml::Backend::Cuda;

ram_ggml::InferenceResult result;
std::string error;
if (!ram_ggml::InferenceRunner::run(options, &result, &error)) {
    throw std::runtime_error(error);
}
```

The CLI is an adapter over this same API. CUDA compute selection and other
process configuration are carried by `InferenceOptions`; the library does not
read caller environment variables. The ggml backend itself may inspect its
own documented compatibility variable, so use `CudaCompute` when deterministic
selection is required.

For CUDA tagging, `--cuda-compute auto` selects FP16 GEMM inputs with FP32
accumulation and output. Fixed F32 matrix weights are converted once at model
initialization; Q8_0 matrix inputs use floating activations instead of the
additional Q8_1 activation quantization used by MMQ. This adds a resident
FP16 matrix cache for F32 exports, without changing their stored weights.
`--cuda-compute f32` selects FP32 GEMM inputs and skips that cache;
`--cuda-compute f16` selects FP16 GEMM inputs for any GGUF dtype. Caption
auto mode retains FP32 GEMM. The runner
sets the ggml compatibility variable from this explicit option and reports the
effective mode; callers do not need to mutate process environment state.

## Export and run

The exporter writes f32, f16 and q8_0 GGUF files. Existing generated files are
ignored by git because they are large:

RAM Q8_0 keeps `tagging_head`, `image_proj`, `wordvec_proj`, and `fc` in F16.
These small tagging layers are sensitive near the official thresholds. All
eligible Swin matrices remain Q8_0; the exporter records the retained prefixes
in `ram.quantization.f16_prefixes`. This policy improves parity without
changing the Python baseline, GT, or RAM thresholds.

```bash
python3 cpp_ggml/scripts/export_gguf.py --model ram --dtype f32
python3 cpp_ggml/scripts/export_gguf.py --model ram_plus --dtype f16
python3 cpp_ggml/scripts/export_gguf.py --model tag2text --dtype q8_0
```

GGUF names always keep the original checkpoint stem, followed by the dtype:

| model | f32 | f16 | q8_0 |
| --- | --- | --- | --- |
| RAM | `ram_swin_large_14m-f32.gguf` | `ram_swin_large_14m-f16.gguf` | `ram_swin_large_14m-q8_0.gguf` |
| RAM++ | `ram_plus_swin_large_14m-f32.gguf` | `ram_plus_swin_large_14m-f16.gguf` | `ram_plus_swin_large_14m-q8_0.gguf` |
| Tag2Text | `tag2text_swin_14m-f32.gguf` | `tag2text_swin_14m-f16.gguf` | `tag2text_swin_14m-q8_0.gguf` |

The root launcher automatically exports a missing GGUF and builds the selected
backend when needed:

```bash
python3 run_inference.py --engine cpp --model ram --backend cuda --dtype f16
python3 run_inference.py --engine cpp --model ram_plus --backend vulkan --dtype q8_0
python3 run_inference.py --engine cpp --model tag2text --task caption --backend cuda --dtype f16
python3 run_inference.py --engine cpp --model tag2text --task caption --specified-tags "dog, sofa" --backend vulkan
python3 run_inference.py --engine python --model tag2text --task caption
```

Use `--dry-run` to inspect the fully resolved command without exporting,
building or running it. `--build-jobs` controls compilation parallelism and is
independent from the runtime `--threads` setting. The C++ path accepts
`--vocab`, label-space sidecars, `--threshold` and `--threshold-epsilon`; the
Python path rejects those C++-only options explicitly. Python keeps the
official device policy (CUDA when available, otherwise CPU), so its path does
not emulate Vulkan or GGUF quantization.

The C++ Tag2Text caption path uses the GGUF text encoder and decoder weights,
the embedded `ram.bert_vocab` metadata (with
`models/bert-base-uncased-vocab.txt` as an explicit fallback), and parallel
three-beam autoregressive generation on CUDA/Vulkan. Pass
`--beam-batch 1` to run the beams sequentially for diagnostics. The current
50-frame end-to-end report compares every format against Python; it records
both exact wording and lexical F1 and does not promote near matches to exact
parity. The default Tag2Text Q8_0 export keeps the complete `visual_encoder.*`
path in F16 and quantizes only eligible weights outside that caption-critical
path. The runtime uses F32 caption accumulation and preconverts caption
text weights once at model load to avoid repeated F16-to-F32 conversion.
See [`benchmarks/LATENCY_MATRIX.md`](benchmarks/LATENCY_MATRIX.md) for the
current acceptance result and the specified-tag diagnostic.

## Official test data

Download the images referenced by every evaluation annotation file in this
repository with one command:

```bash
python3 download_test_data.py all --workers 24
```

This downloads the OpenImages common/rare sets, the selected ImageNet
validation images, and the HICO evaluation images. The annotation files define
the retained 50-frame subset; ImageNet/HICO archives are temporary and removed
after extraction unless `--keep-archives` is supplied.

The exact inventory and evaluation workflow are documented in
[`benchmarks/full_eval/README.md`](benchmarks/full_eval/README.md).
The harness covers all 50 retained GT frames (14 common, 12 rare, 12 ImageNet
and 12 HICO) for every model, CUDA/Vulkan backend and F32/F16/Q8_0 format.
The original 138,873 source rows and deterministic selection reasons are preserved in
`datasets/selection_manifest.json`. The same harness can process this retained
set directly:

```bash
python3 cpp_ggml/benchmarks/full_eval.py --scope datasets --run \
  --backends cuda vulkan --dtypes f32 f16 q8_0 \
  --output-dir cpp_ggml/benchmarks/full_eval
```

Each run writes per-image JSONL, a raw score matrix, a summary CSV and plots.
OpenImages common has a directly comparable closed-set `mAP/CP/CR` protocol.
All four datasets have GT annotation rows; rare, ImageNet multi and HICO use
dataset-specific label spaces. Generate sidecars with
`cpp_ggml/scripts/prepare_label_embedding.py` and pass them to both engines so
the report can score the exact space instead of emitting
`unsupported_label_space`.

For Tag2Text captions over the same 50 frames, pass the manifest paths to
`python3 cpp_ggml/benchmarks/caption_all.py`. The canonical report is linked
from [`benchmarks/LATENCY_MATRIX.md`](benchmarks/LATENCY_MATRIX.md); its raw
run stores Python and all six GGML captions, while
`visualize_outputs.py` renders one source/reference/output panel per frame.

## Precision and timing checks

Generate an official PyTorch reference (including intermediate tensors), then
run the C++ executable with the same normalized input:

```bash
python3 cpp_ggml/scripts/parity_reference.py --model ram_plus \
  --output /tmp/ram-plus-ref
cpp_ggml/build-cuda/ram-ggml \
  --model cpp_ggml/models/gguf/ram_plus_swin_large_14m-f16.gguf \
  --input-bin /tmp/ram-plus-ref/input.bin --backend cuda \
  --warmup 1 --repeat 3 --dump-dir /tmp/ram-plus-cpp
```

`cpp_ggml/scripts/validate.py` runs all three tagging models across CUDA/Vulkan and
f32/f16/q8_0, comparing logits and thresholded tag sets. The standalone Python
tagging timing baseline is provided by `benchmark_python.py --task tags`; its
`--task caption` mode measures full Tag2Text generation. Timings
depend on the GPU, driver, scheduler fallback operations and system load; use
both scripts with the same warmup/repeat settings for a local comparison. Add
`--max-slowdown 1.0` to `validate.py` to make the earlier graph-only C++ tagging
latency a strict Python CUDA gate; it reports each backend/dtype separately
and exits nonzero for a missed combination. Tag logits and captions need
separate precision checks because beam-search wording can change after small
numerical differences. The published acceptance matrix is the measured
result for the documented RTX 3060, driver, model hashes and 50-frame GT
manifest; rerun the harness on deployment hardware before making a release
decision for a different GPU or driver.

On the included demo image, all 18 model/backend/dtype combinations measured
logit MAE below 0.009. Thresholded labels matched except RAM f16 label 617,
whose official probability is within 0.001 of the 0.63 cutoff; the validator
reports this boundary case separately. Reduced precision outputs are numerically
close, not bitwise identical to PyTorch.

The canonical RTX 3060 matrix passes the aggregate tag speed gate for 18/18
combinations and the strict per-dataset gate for 72/72 rows; the minimum
dataset cross-engine F1 is 0.991781. Tag2Text tagging retains the numeric
labels that the Python tagging evaluator emits; caption generation applies
the official caption-only deletion list. All six caption combinations are
faster than Python CUDA and are exact with word F1 1.000 on the retained 50
frames after the GGUF boundary-threshold calibration. The calibration keeps
the official Python thresholds in GGUF metadata for audit and changes only two
labels whose probabilities straddle 0.68 on this backend set. The matrix is
reproducible evidence for this environment; it is not a promise that an
unmeasured GPU or driver has the same latency.
The [benchmark report](benchmarks/LATENCY_MATRIX.md) is the source of truth
and the [model card](models/MODEL_CARD.md) gives per-file guidance and download
links. The complete source audit, including remaining upstream TODOs and
documented boundaries, is in [`REPOSITORY_AUDIT.md`](REPOSITORY_AUDIT.md).

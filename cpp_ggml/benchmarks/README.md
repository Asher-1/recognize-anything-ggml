# End-to-end benchmark package

Model downloads: [GitHub RAM release](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/RAM) and
[Hugging Face `Asher-1/RAM_GGUF`](https://huggingface.co/Asher-1/RAM_GGUF/tree/main).

The public 50-image GT result is [`LATENCY_MATRIX.md`](LATENCY_MATRIX.md), with
per-dataset rows in [`latency_matrix_72.csv`](latency_matrix_72.csv) and six
Tag2Text caption rows in [`caption_matrix_6.csv`](caption_matrix_6.csv). The
charts are [`latency_matrix.svg`](latency_matrix.svg) and
[`caption_latency.svg`](caption_latency.svg). Raw predictions, score matrices,
logs and per-image caption panels are generated under `full_eval/` and
`visualizations/`; they are ignored by Git because they are reproducible and
large.

![Source image and word-level caption comparison](caption_examples.webp)

The preview shows a representative image and the six GGML captions beside the
official Python sentence. Red words identify any differences; the current
50-frame run has none. The [full matrix](LATENCY_MATRIX.md) reports all six
caption combinations.

## Scope and method

The retained GT set has 50 frames: 14 OpenImages common, 12 OpenImages rare,
12 ImageNet multi and 12 HICO. RAM, RAM++ and Tag2Text each run in F32, F16
and Q8_0 on CUDA and Vulkan. The source audit covered 138,873 annotation rows;
the selection and exact paths are in [`dataset_inventory.json`](dataset_inventory.json)
and [`datasets/selection_manifest.json`](../../datasets/selection_manifest.json).

Tag timing includes decode, resize/normalization, transfer, inference,
readback and threshold selection. Caption timing additionally includes text
encoding, autoregressive generation and output. Models stay resident. Both
engines use batch size 1, one first-image warmup and three measured passes;
model loading and JSONL serialization are excluded. The baseline is the
official PyTorch checkpoint on CUDA. A ratio below 1.00x means GGML was
faster on the measured RTX 3060 and driver.

## Acceptance

The tag aggregate is faster for 18/18 combinations, and the strict
per-dataset gate passes for 72/72 rows. Per-dataset tag cross-engine F1 ranges
from 0.991781 to 1.0000; the highest per-dataset probability MAE is 0.00074.
Tag2Text
tagging retains numeric labels; the caption-only deletion list is applied
when building caption input. The CSV retains per-dataset GT `mAP`,
`CP` and `CR` columns. Caption precision and latency have separate gates in
the report. All six caption combinations are faster than the Python baseline
and are 1.000 exact with 1.000 word F1 on the retained 50 frames after the
GGUF boundary-threshold calibration. The official threshold table remains
embedded alongside the calibrated deployment table for auditability. A
separate `--scope images` run on all 18 repository images also produced 18/18
exact captions for each of the six combinations; its timing is excluded from
the public speed matrix because another GPU workload ran concurrently.

## Reproduce

```bash
python3 cpp_ggml/benchmarks/full_eval.py --scope datasets --run \
  --backends cuda vulkan --dtypes f32 f16 q8_0 \
  --batch-size 1 --warmup 1 --repeat 3 --threads 8 \
  --label-space-dir cpp_ggml/benchmarks/label_spaces

python3 cpp_ggml/benchmarks/caption_all.py --scope datasets \
  --backends cuda vulkan --dtypes f32 f16 q8_0 --warmup 1 --repeat 3

python3 cpp_ggml/benchmarks/latency_matrix.py \
  --run-dir cpp_ggml/benchmarks/full_eval/run-YYYYMMDD-HHMMSS \
  --caption-run cpp_ggml/benchmarks/full_eval/caption-YYYYMMDD-HHMMSS

python3 cpp_ggml/benchmarks/visualize_outputs.py \
  --caption-run cpp_ggml/benchmarks/full_eval/caption-YYYYMMDD-HHMMSS \
  --preview-output cpp_ggml/benchmarks/caption_examples.webp
```

[`visualize_outputs.py`](visualize_outputs.py) creates one
source/reference/output panel per frame and a contact sheet. Generated panels
are ignored by Git. The detailed label-space and GT protocol is in
[`full_eval/README.md`](full_eval/README.md). The current GGUFs embed tags,
thresholds and the Tag2Text vocabulary; separate text files are unnecessary
for ordinary inference.

# RAM family GGUF model card

Current downloads: [GitHub RAM release](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/RAM) and
[Hugging Face `Asher-1/RAM_GGUF`](https://huggingface.co/Asher-1/RAM_GGUF/tree/main).

## Source, intended use and download

These nine files are exported directly from this repository's three official
inference checkpoints. The GGUF filename retains the checkpoint stem. All
downloads are mirrored at the [RAM release](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/RAM)
and [Hugging Face](https://huggingface.co/Asher-1/RAM_GGUF/tree/main).
The release assets' byte counts and SHA-256 digests were checked against the
local exports on 2026-10-08; [SHA256SUMS](SHA256SUMS) lists the exact hashes.

| Source model | Format | Size (MiB) | Relative size | Input/output | Download |
| --- | --- | ---: | ---: | --- | --- |
| RAM Swin-L | F32 | 1941.5 | 100% | 384 px RGB / 4585 tags | [ram_swin_large_14m-f32.gguf](https://github.com/Asher-1/cloudViewer_downloads/releases/download/RAM/ram_swin_large_14m-f32.gguf) |
| RAM Swin-L | F16 | 970.9 | 50.0% | 384 px RGB / 4585 tags | [ram_swin_large_14m-f16.gguf](https://github.com/Asher-1/cloudViewer_downloads/releases/download/RAM/ram_swin_large_14m-f16.gguf) |
| RAM Swin-L | Q8_0 | 534.4 | 27.5% | 384 px RGB / 4585 tags | [ram_swin_large_14m-q8_0.gguf](https://github.com/Asher-1/cloudViewer_downloads/releases/download/RAM/ram_swin_large_14m-q8_0.gguf) |
| RAM++ Swin-L | F32 | 1268.1 | 100% | 384 px RGB / 4585 tags | [ram_plus_swin_large_14m-f32.gguf](https://github.com/Asher-1/cloudViewer_downloads/releases/download/RAM/ram_plus_swin_large_14m-f32.gguf) |
| RAM++ Swin-L | F16 | 634.1 | 50.0% | 384 px RGB / 4585 tags | [ram_plus_swin_large_14m-f16.gguf](https://github.com/Asher-1/cloudViewer_downloads/releases/download/RAM/ram_plus_swin_large_14m-f16.gguf) |
| RAM++ Swin-L | Q8_0 | 447.1 | 35.3% | 384 px RGB / 4585 tags | [ram_plus_swin_large_14m-q8_0.gguf](https://github.com/Asher-1/cloudViewer_downloads/releases/download/RAM/ram_plus_swin_large_14m-q8_0.gguf) |
| Tag2Text Swin-B | F32 | 1577.4 | 100% | 384 px RGB / 3429 tags + caption | [tag2text_swin_14m-f32.gguf](https://github.com/Asher-1/cloudViewer_downloads/releases/download/RAM/tag2text_swin_14m-f32.gguf) |
| Tag2Text Swin-B | F16 | 789.0 | 50.0% | 384 px RGB / 3429 tags + caption | [tag2text_swin_14m-f16.gguf](https://github.com/Asher-1/cloudViewer_downloads/releases/download/RAM/tag2text_swin_14m-f16.gguf) |
| Tag2Text Swin-B | Q8_0 | 775.6 | 49.2% | 384 px RGB / 3429 tags + caption | [tag2text_swin_14m-q8_0.gguf](https://github.com/Asher-1/cloudViewer_downloads/releases/download/RAM/tag2text_swin_14m-q8_0.gguf) |

F32 is the useful reference for investigating numerical differences and has
the highest weight-storage cost. F16 halves the weight file. Q8_0 reduces
download/storage needs while maintaining strong tagging agreement on the
retained set. For Tag2Text, the published caption-safe Q8 policy keeps the
complete `visual_encoder.*` path in F16 and quantizes only eligible weights
outside that path; the 50-frame caption matrix below is the acceptance source
for parity. RAM and RAM++ are tagging models. Tag2Text adds a 12-layer
tag-text encoder, BERT WordPiece tokenizer and 12-layer autoregressive caption
decoder. Current GGUF exports embed `ram.tag_list`, `ram.tag_thresholds`,
`ram.official_tag_thresholds`, `ram.delete_tag_indices`, and, for Tag2Text,
`ram.bert_vocab`. Tag2Text also records `ram.threshold_calibration` and uses
0.6795 for tag indices 3411 (`large`) and 3424 (`some`) in the deployment
threshold array. The official 0.68 table is preserved in
`ram.official_tag_thresholds`; this small margin addresses the measured
CUDA/Vulkan logit drift at the decision boundary and is validated on the
retained GT set. The checked-in
[BERT uncased vocabulary](bert-base-uncased-vocab.txt) remains an explicit
fallback and inspection file for older GGUFs, so current models do not need a
separate label or vocabulary file for normal inference.

Q8_0 applies block quantization only to eligible 2D weight matrices. Other
weights remain F16, including the complete Tag2Text visual encoder, label
embeddings, text encoder and decoder; the visual path is excluded to prevent
caption logit drift. RAM additionally
retains `tagging_head`, `image_proj`, `wordvec_proj` and `fc` in F16;
eligible Swin matrices remain Q8_0. These small tagging layers are sensitive
near the official thresholds. The updated RAM Q8 file records this policy in
`ram.quantization.f16_prefixes`. The exporter accepts
repeatable `--q8-skip-prefix` overrides for hardware-specific experiments.
File-size ratios are therefore model-dependent. A smaller file does not prove
proportional runtime VRAM or speed savings: activations, graph buffers, and
backend fallback operations also consume memory and time. All variants use
ggml v0.21.0 via a pinned submodule; CUDA and Vulkan builds do not use cuDNN.

## Measured inference metrics

The canonical comparison is the checked-in [50-frame latency matrix](../benchmarks/LATENCY_MATRIX.md).
It covers 14 OpenImages-common, 12 OpenImages-rare, 12 ImageNet-multi and 12
HICO images with GT annotations, all three model families, CUDA/Vulkan and
F32/F16/Q8_0. The RTX 3060 12 GB run used one warmup and three measured passes
per image, batch size 1, resident models, and timed decode, preprocessing,
transfer, inference, readback and output selection. Model loading and JSONL
serialization were excluded.

CUDA tagging auto mode uses FP16 GEMM inputs with FP32 accumulation and
FP32 output. F32 exports keep their original stored weights and add a resident
FP16 matrix cache at model initialization; its extra memory is not included
in the file-size column. `--cuda-compute f32` selects FP32 inputs and skips
that cache. CUDA Q8 tagging uses floating activations to avoid additional
Q8_1 activation quantization. Caption auto mode retains FP32 computation.
See [alignment diagnosis](../benchmarks/ALIGNMENT_DIAGNOSIS.md) for the
controlled weight-only versus execution-error comparison.

| Model | Backend | Format | Python CUDA ms | GGML ms | Ratio | Tag F1 | Probability MAE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| RAM | CUDA | F32 | 61.11 | 50.81 | 0.832x | 0.9997 | 0.00011 |
| RAM | CUDA | F16 | 61.11 | 50.45 | 0.826x | 0.9994 | 0.00012 |
| RAM | CUDA | Q8_0 | 61.11 | 52.36 | 0.857x | 0.9971 | 0.00028 |
| RAM | Vulkan | F32 | 61.11 | 54.89 | 0.898x | 0.9990 | 0.00010 |
| RAM | Vulkan | F16 | 61.11 | 54.04 | 0.884x | 0.9994 | 0.00012 |
| RAM | Vulkan | Q8_0 | 61.11 | 53.53 | 0.876x | 0.9977 | 0.00028 |
| RAM++ | CUDA | F32 | 60.79 | 51.02 | 0.839x | 1.0000 | 0.00012 |
| RAM++ | CUDA | F16 | 60.79 | 51.00 | 0.839x | 1.0000 | 0.00013 |
| RAM++ | CUDA | Q8_0 | 60.79 | 53.35 | 0.878x | 0.9988 | 0.00060 |
| RAM++ | Vulkan | F32 | 60.79 | 55.56 | 0.914x | 1.0000 | 0.00011 |
| RAM++ | Vulkan | F16 | 60.79 | 54.78 | 0.901x | 1.0000 | 0.00013 |
| RAM++ | Vulkan | Q8_0 | 60.79 | 54.35 | 0.894x | 0.9988 | 0.00060 |
| Tag2Text | CUDA | F32 | 50.37 | 42.28 | 0.839x | 1.0000 | 0.00009 |
| Tag2Text | CUDA | F16 | 50.37 | 42.44 | 0.843x | 1.0000 | 0.00010 |
| Tag2Text | CUDA | Q8_0 | 50.37 | 42.81 | 0.850x | 1.0000 | 0.00018 |
| Tag2Text | Vulkan | F32 | 50.37 | 42.99 | 0.853x | 1.0000 | 0.00007 |
| Tag2Text | Vulkan | F16 | 50.37 | 42.28 | 0.839x | 1.0000 | 0.00010 |
| Tag2Text | Vulkan | Q8_0 | 50.37 | 42.08 | 0.835x | 1.0000 | 0.00018 |

The tag aggregate speed gate passes for 18/18 combinations and the strict
per-dataset gate passes for 72/72 rows; the slowest aggregate ratio is 0.914x
and the minimum dataset cross-engine F1 is 0.991781. All 72 rows have
`annotation_status=ok`; mAP/CP/CR and each dataset result are in the CSV. Tag
F1 is cross-engine thresholded-set agreement, not a human caption score.
Tag2Text tagging retains numeric labels; the caption-only deletion list is
applied when building caption input.

## Measured Tag2Text caption metrics

A separate caption run on the same 50 images gives a PyTorch CUDA baseline of
247.82 ms/image. The
caption scope includes preprocessing, tagging, text encoding, autoregressive
generation and output readback. Exact means byte-for-byte agreement with the
Python sentence; word F1 is lexical overlap only.

| Format | CUDA ms / ratio / exact | Vulkan ms / ratio / exact | Word F1 (CUDA/Vulkan) |
| --- | --- | --- | ---: |
| F32 | 179.31 / 0.724x / 1.000 | 214.82 / 0.867x / 1.000 | 1.000 / 1.000 |
| F16 | 182.12 / 0.735x / 1.000 | 207.36 / 0.837x / 1.000 | 1.000 / 1.000 |
| Q8_0 | 182.29 / 0.736x / 1.000 | 207.44 / 0.837x / 1.000 | 1.000 / 1.000 |

All 6/6 caption combinations are faster than PyTorch CUDA and meet the 0.99
exact-wording gate on the retained 50 frames after the boundary calibration.
The separate 18-image repository set also gave 18/18 exact captions for each
backend/format combination; its latency was not used for this table because
the GPU was shared with another workload.
This is a deployment acceptance result for the documented hardware and data
set, not a mathematical guarantee for every unseen image. The full table,
charts, per-image sentences and reproduction commands are in
[`benchmarks/LATENCY_MATRIX.md`](../benchmarks/LATENCY_MATRIX.md). Use F32 for
numerical diagnosis; use F16/Q8_0 only after validating the target caption
distribution.

## Compatibility metadata

`ram_tag_list.txt` and `ram_tag_list_threshold.txt` are canonical, readable
source files used by the exporter and retained as fallback inputs for older
GGUFs. They are not required beside a current GGUF at runtime. There is no
duplicate `cpp_ggml/models/pytorch/ram_tag_list.txt` in the current tree; the
canonical file is `ram/data/ram_tag_list.txt`, so no extra file needs to be
deleted.

`bert-base-uncased-vocab.txt` is the official 30,522-entry BERT WordPiece
lookup table. `[unused0]` through `[unused995]` are reserved BERT token IDs;
they are intentionally present to preserve the embedding index mapping, not
missing vocabulary. Current Tag2Text GGUFs carry this exact table in
`ram.bert_vocab`, while the text file remains a fallback for old files and an
auditable source artifact.

## Ground truth and label spaces

The source corpus audited before pruning had 138,873 GT rows: 57,224
OpenImages common, 21,991 OpenImages rare, 50,000 ImageNet multi and 9,658
HICO rows. The repository retains a deterministic 50-frame subset (14/12/12/12)
with one or more GT labels per frame. The exact source counts, selected image
paths and labels are listed in [`datasets/selection_manifest.json`](../../datasets/selection_manifest.json),
[`datasets/SELECTION_REPORT.md`](../../datasets/SELECTION_REPORT.md) and
[`benchmarks/dataset_inventory.json`](../benchmarks/dataset_inventory.json).
`unsupported_label_space` means the GT names are not the 4,585-class closed
RAM metadata, not that the images lack labels. Generate an exact sidecar with
[`prepare_label_embedding.py`](../scripts/prepare_label_embedding.py); pass
its tag list, thresholds and raw float32 embedding to both the Python and C++
evaluators. OpenImages rare includes the official RAM++ LLM descriptions;
ImageNet/HICO need dataset-specific descriptions or CLIP prompt embeddings.

## Related official weights

The 479 MB
[`ram_plus_tag_embedding_class_4585_des_51.pth`](https://huggingface.co/xinyu1205/recognize-anything-plus-model/blob/main/ram_plus_tag_embedding_class_4585_des_51.pth)
is a frozen training-time RAM++ text embedding. A fixed pretrained RAM++
checkpoint already contains its `label_embed` tensor, so it is not a second
runtime GGUF model. The 694 MB
[`groundingdino_swint_ogc.pth`](https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/groundingdino_swint_ogc.pth)
and 2.56 GB
[`sam_vit_h_4b8939.pth`](https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/sam_vit_h_4b8939.pth)
belong to the downstream Grounding-DINO/SAM localization and segmentation
pipeline. They are not required for RAM/Tag2Text tagging or caption parity and
are intentionally outside this GGML module's model set.

## Reproduce and verify

```bash
cd cpp_ggml/models/gguf
sha256sum -c ../SHA256SUMS
cd ../../..
python3 cpp_ggml/benchmarks/full_eval.py --scope datasets --run \
  --backends cuda vulkan --dtypes f32 f16 q8_0 --batch-size 1 \
  --warmup 1 --repeat 3 --threads 8 \
  --label-space-dir cpp_ggml/benchmarks/label_spaces
python3 cpp_ggml/benchmarks/caption_all.py --scope datasets \
  --backends cuda vulkan --dtypes f32 f16 q8_0 --warmup 1 --repeat 3
python3 cpp_ggml/benchmarks/latency_matrix.py \
  --run-dir cpp_ggml/benchmarks/full_eval/run-YYYYMMDD-HHMMSS \
  --caption-run cpp_ggml/benchmarks/full_eval/caption-YYYYMMDD-HHMMSS
python3 run_inference.py --engine cpp --model tag2text --task caption --backend cuda --dtype f16
```

Use the GGUF asset with this repository's runtime. Current files are
self-describing; the readable tag metadata and BERT vocabulary are only
needed for export, auditing, or older GGUF compatibility. The [official
checkpoint provenance](pytorch/README.md) and export script document the
conversion path.

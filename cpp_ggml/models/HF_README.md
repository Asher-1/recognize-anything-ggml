---
license: apache-2.0
tags:
  - image-classification
  - image-to-text
  - gguf
---

# RAM GGUF

Current downloads: [GitHub RAM release](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/RAM)
and this [Hugging Face repository](https://huggingface.co/Asher-1/RAM_GGUF/tree/main).
The source project is [recognize-anything-ggml](https://github.com/Asher-1/recognize-anything-ggml).

## Models

| Model | F32 | F16 | Q8_0 |
| --- | --- | --- | --- |
| RAM Swin-L | [download](./ram_swin_large_14m-f32.gguf) | [download](./ram_swin_large_14m-f16.gguf) | [download](./ram_swin_large_14m-q8_0.gguf) |
| RAM++ Swin-L | [download](./ram_plus_swin_large_14m-f32.gguf) | [download](./ram_plus_swin_large_14m-f16.gguf) | [download](./ram_plus_swin_large_14m-q8_0.gguf) |
| Tag2Text Swin-B | [download](./tag2text_swin_14m-f32.gguf) | [download](./tag2text_swin_14m-f16.gguf) | [download](./tag2text_swin_14m-q8_0.gguf) |

The nine files are exported from the three official inference checkpoints and
keep the original checkpoint stem in their filenames. GGML is pinned to
v0.21.0; CUDA and Vulkan builds do not use cuDNN.

## Embedded metadata

Current files are self-describing. RAM and RAM++ GGUFs contain
`ram.tag_list` and `ram.tag_thresholds`; Tag2Text additionally contains
`ram.delete_tag_indices`, the complete `ram.bert_vocab` WordPiece table, and
`ram.official_tag_thresholds`. For Tag2Text, `ram.tag_thresholds` is the
deployment table: only `large` and `some` are calibrated from 0.68 to
0.6795 to cover measured CUDA/Vulkan decision-boundary drift. The official
values remain embedded for audit.
The runtime therefore does not need a separate label list, threshold file or
BERT vocabulary beside these files. The repository keeps readable text files
for export, auditing and compatibility with older GGUF files.

Q8_0 uses block quantization for eligible visual matrices. RAM retains its
small tagging head, image projection, word-vector projection and classifier
in F16; eligible Swin matrices remain Q8_0. The RAM Q8 file is 534.4 MiB and
records retained prefixes in GGUF metadata. CUDA tagging uses floating
activations and FP32 accumulation/output to avoid additional activation
quantization. The distributed
Tag2Text Q8 export keeps the complete `visual_encoder.*` path in F16 and
quantizes only eligible weights outside that caption-critical path; text
encoder and decoder weights remain F16. With the deployment threshold
calibration, all six CUDA/Vulkan and F32/F16/Q8_0 combinations reproduced the
official caption wording on all 50 retained GT images and all 18 repository
images. An
older all-visual Q8 exports remain diagnostic variants and can drift in close
decoder logits. F32/F16 remain the reference formats for tensor-level audits.
CUDA tagging auto mode uses FP16 GEMM inputs with FP32 accumulation/output,
including a one-time resident FP16 matrix cache for F32 exports. Select
`--cuda-compute f32` for FP32 inputs. Caption auto mode retains FP32 compute.
See the detailed [model card](https://github.com/Asher-1/recognize-anything-ggml/blob/main/cpp_ggml/models/MODEL_CARD.md),
[benchmarks](https://github.com/Asher-1/recognize-anything-ggml/tree/main/cpp_ggml/benchmarks),
and [checksums](https://github.com/Asher-1/recognize-anything-ggml/blob/main/cpp_ggml/models/SHA256SUMS).

# RAM family GGUF exports

Current downloads: [GitHub RAM release](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/RAM) and
[Hugging Face `Asher-1/RAM_GGUF`](https://huggingface.co/Asher-1/RAM_GGUF/tree/main).

Nine GGUF models converted from the repository's three official checkpoints:
RAM Swin-L, RAM++ Swin-L and Tag2Text Swin-B, each in F32, F16 and Q8_0.
The release asset names preserve the source checkpoint stem. GGML is pinned
to v0.21.0. The repository's C++ runner supports tagging on CUDA/Vulkan/CPU
and Tag2Text caption generation with a 3-beam BERT decoder. No cuDNN is used.

Current GGUF files contain weights plus tag names, per-tag thresholds,
Tag2Text delete indices, and the BERT uncased WordPiece vocabulary as GGUF
metadata. Tag2Text exports also retain the official threshold array and a
calibrated deployment array; only `large` and `some` use 0.6795 instead of
0.68 to account for measured backend decision-boundary drift. The checked-in
files remain fallbacks for older exports. RAM and
RAM++ emit 4585 tags; Tag2Text emits 3429 tags and can generate captions.
On the documented RTX 3060 run, all six CUDA/Vulkan Tag2Text caption variants
matched Python sentence-for-sentence on 50 retained GT images and 18 repository
images. Caption latency was lower for 6/6 variants; the tag latency gate was
met for 18/18 combinations and the strict per-dataset gate passed 72/72 rows.
The minimum dataset cross-engine tag F1 was 0.991781.
Consult `cpp_ggml/models/MODEL_CARD.md` and
`cpp_ggml/benchmarks/` in the recognize-anything-ggml repository for
download links, SHA-256 checksums, accuracy and pipeline-speed measurements.

These exports supersede the old `ram-f32.gguf`, `ram-f16.gguf` and
`ram-q8_0.gguf` assets. Existing consumers should switch to the new
`ram_swin_large_14m-*` names and use the current runtime metadata.

# Specified-tag caption diagnostic

Model downloads: [GitHub RAM release](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/RAM) and [Hugging Face RAM_GGUF](https://huggingface.co/Asher-1/RAM_GGUF/tree/main).

The `dog, sofa` prompt on `images/demo/demo1.jpg` checks the explicit-tag
caption path. It is a single-image regression case, so its former latency and
exact-match table is not an acceptance result. The measured 50-frame default-tag
comparison, including six caption format/backend combinations, is the
[canonical matrix](../LATENCY_MATRIX.md#tag2text-image-to-caption).

```bash
python3 cpp_ggml/benchmarks/caption.py \
  --images images/demo/demo1.jpg --specified-tags 'dog, sofa' \
  --output-dir cpp_ggml/benchmarks/full_eval/caption-specified-tags
```

This generates per-engine sentences and timing in ignored JSON/PNG files.
Compare the actual strings for the chosen prompt before drawing a parity
conclusion; quantization can change caption wording.

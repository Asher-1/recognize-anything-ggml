# Official Inference Checkpoints

The converted GGUF files are published at the [GitHub RAM release](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/RAM)
and [Hugging Face `Asher-1/RAM_GGUF`](https://huggingface.co/Asher-1/RAM_GGUF/tree/main).

These are the complete pretrained checkpoints referenced by the repository's
official inference entry points. The binary files are intentionally ignored by
Git because they are large; keep them beside this manifest when packaging the
repository.

| file | model | backbone | bytes |
| --- | --- | --- | ---: |
| `ram_swin_large_14m.pth` | RAM | Swin-L | 5,625,634,877 |
| `ram_plus_swin_large_14m.pth` | RAM++ | Swin-L | 3,010,218,801 |
| `tag2text_swin_14m.pth` | Tag2Text | Swin-B | 4,478,705,095 |

The exporter uses the canonical tag list and thresholds in `ram/data/` and
embeds them in current GGUF metadata. Keep those text files for export,
legacy GGUF fallback and validation; they are not required beside a current
GGUF during normal inference. There is no duplicate `ram_tag_list.txt` in
this `models/pytorch/` directory; the source file is `ram/data/ram_tag_list.txt`.

"""Names shared by checkpoint, GGUF export, and runtime tooling."""

from pathlib import Path


MODEL_CHECKPOINTS = {
    "ram": "ram_swin_large_14m.pth",
    "ram_plus": "ram_plus_swin_large_14m.pth",
    "tag2text": "tag2text_swin_14m.pth",
}


def checkpoint_stem(model: str) -> str:
    return Path(MODEL_CHECKPOINTS[model]).stem


def gguf_name(model: str, dtype: str) -> str:
    return f"{checkpoint_stem(model)}-{dtype}.gguf"

#!/usr/bin/env python3
"""Dump official RAM inference intermediates for an identical image."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ram import get_transform
from ram.models import ram, ram_plus, tag2text
from model_manifest import MODEL_CHECKPOINTS
from tag2text_tags import tag_logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=ROOT / "images/demo/demo1.jpg")
    parser.add_argument("--model", choices=("ram", "ram_plus", "tag2text"), default="ram")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-bin", type=Path, help="Use an already normalized NCHW float32 input")
    parser.add_argument("--gguf", type=Path,
                        help="diagnostic: replace checkpoint tensors with dequantized GGUF weights")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = args.checkpoint or ROOT / "cpp_ggml/models/pytorch" / MODEL_CHECKPOINTS[args.model]
    factory = {"ram": ram, "ram_plus": ram_plus, "tag2text": tag2text}[args.model]
    kwargs = {"pretrained": str(checkpoint), "image_size": 384,
              "vit": "swin_b" if args.model == "tag2text" else "swin_l"}
    if args.model == "tag2text":
        kwargs["delete_tag_index"] = [127, 2961, 3351, 3265, 3338, 3355, 3359]
    model = factory(**kwargs).eval().to(device)
    if args.gguf:
        import gguf
        state = model.state_dict()
        for tensor in gguf.GGUFReader(args.gguf).tensors:
            name = tensor.name.removeprefix(args.model + ".")
            if name not in state:
                continue
            values = gguf.dequantize(tensor.data, tensor.tensor_type).copy()
            state[name].copy_(torch.from_numpy(values.reshape(tuple(state[name].shape))).to(device))
    if args.input_bin:
        image = torch.from_numpy(np.fromfile(args.input_bin, dtype=np.float32).copy())
        image = image.reshape(1, 3, 384, 384).to(device)
    else:
        image = get_transform(384)(Image.open(args.image)).unsqueeze(0).to(device)
    image.float().cpu().numpy().tofile(args.output / "input.bin")

    def save(name):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            output.detach().float().cpu().contiguous().numpy().tofile(args.output / (name + ".bin"))
        return hook

    model.visual_encoder.patch_embed.register_forward_hook(save("patch_embed"))
    for i, layer in enumerate(model.visual_encoder.layers):
        layer.blocks[0].register_forward_hook(save(f"stage{i}_block0"))
        layer.blocks[1].register_forward_hook(save(f"stage{i}_block1"))
        layer.register_forward_hook(save(f"stage{i}"))
    if args.model != "tag2text":
        model.image_proj.register_forward_hook(save("image_proj"))
        model.wordvec_proj.register_forward_hook(save("label_proj_linear"))
    else:
        model.visual_encoder.register_forward_hook(save("image_proj"))
    for i, layer in enumerate(model.tagging_head.encoder.layer):
        layer.register_forward_hook(save(f"decoder{i}"))
    model.fc.register_forward_hook(save("logits"))
    with torch.inference_mode():
        if args.model == "tag2text":
            tag_logits(model, image)
        else:
            model.generate_tag(image)


if __name__ == "__main__":
    main()

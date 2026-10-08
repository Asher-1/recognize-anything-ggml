#!/usr/bin/env python3
"""Time the official PyTorch tagging graph or Tag2Text caption generation."""

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ram import get_transform
from ram.models import ram, ram_plus, tag2text
from model_manifest import MODEL_CHECKPOINTS
from tag2text_tags import tag_logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("ram", "ram_plus", "tag2text"), default="ram")
    parser.add_argument("--task", choices=("tags", "caption"), default="tags")
    parser.add_argument("--image", type=Path, default=Path("images/demo/demo1.jpg"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--pipeline", action="store_true",
                        help="include image decode, preprocessing and tag selection")
    parser.add_argument("--max-length", type=int, default=50)
    parser.add_argument("--specified-tags", help="comma-separated Tag2Text caption tags")
    args = parser.parse_args()
    if args.task == "caption" and args.model != "tag2text":
        parser.error("caption is only available for tag2text")
    if args.specified_tags and args.task != "caption":
        parser.error("--specified-tags requires caption")
    path = args.checkpoint or Path("cpp_ggml/models/pytorch") / MODEL_CHECKPOINTS[args.model]
    factory = {"ram": ram, "ram_plus": ram_plus, "tag2text": tag2text}[args.model]
    settings = {"pretrained": str(path), "image_size": 384,
                "vit": "swin_b" if args.model == "tag2text" else "swin_l"}
    if args.model == "tag2text":
        settings["delete_tag_index"] = [127, 2961, 3351, 3265, 3338, 3355, 3359]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = factory(**settings).eval().to(device)
    transform = get_transform(384)
    image = transform(Image.open(args.image)).unsqueeze(0).to(device)

    def run():
        current_image = transform(Image.open(args.image)).unsqueeze(0).to(device) if args.pipeline else image
        with torch.inference_mode():
            if args.model == "tag2text":
                if args.task == "caption":
                    tag_input = ([args.specified_tags.replace(",", " | ")]
                                 if args.specified_tags else None)
                    model.generate(current_image, max_length=args.max_length,
                                   tag_input=tag_input)
                else:
                    logits = tag_logits(model, current_image)
                    if args.pipeline:
                        indices = torch.nonzero(torch.sigmoid(logits[0]) >
                                                model.class_threshold.to(device)).flatten().tolist()
                        _ = [model.tag_list[i] for i in indices if i not in model.delete_tag_index]
            else:
                model.generate_tag(current_image)
        if device == "cuda":
            torch.cuda.synchronize()

    for _ in range(args.warmup):
        run()
    start = time.perf_counter()
    for _ in range(args.repeat):
        run()
    print(f"model={args.model} task={args.task} backend=python-{device} "
          f"scope={'pipeline' if args.pipeline else 'graph'} latency_ms="
          f"{(time.perf_counter()-start)*1000/args.repeat:.3f}")


if __name__ == "__main__":
    main()

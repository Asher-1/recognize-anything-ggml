#!/usr/bin/env python3
"""Run Tag2Text's official PyTorch tagging branch only."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ram import get_transform
from ram.models import tag2text
from tag2text_tags import tag_logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = tag2text(pretrained=str(args.pretrained), image_size=384, vit="swin_b",
                     delete_tag_index=[127, 2961, 3351, 3265, 3338, 3355, 3359]).eval().to(device)
    image = get_transform(384)(Image.open(args.image)).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = tag_logits(model, image)
        selected = (torch.sigmoid(logits) > model.class_threshold.to(device)).cpu().numpy()[0]
    selected[model.delete_tag_index] = False
    tags = np.asarray(model.tag_list)[np.flatnonzero(selected)]
    print("Model Identified Tags: ", " | ".join(tags))


if __name__ == "__main__":
    main()

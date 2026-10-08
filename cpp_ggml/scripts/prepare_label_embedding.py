#!/usr/bin/env python3
"""Prepare a RAM/RAM++ label-space sidecar for Python and GGML.

The sidecar is deliberately raw float32 so both runtimes consume the same
embedding bytes. RAM++ keeps every description row; the C++ graph derives the
description count from the file size and tag list.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ram.utils import (build_openset_label_embedding, build_openset_llm_label_embedding)  # noqa: E402
from ram.utils.openset_utils import article, multiple_templates, processed_name  # noqa: E402


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def generic_descriptions(tags: list[str]) -> list[dict[str, list[str]]]:
    """Create deterministic CLIP prompts when a dataset has no LLM text.

    RAM++'s official rare-class evaluator uses 51 LLM descriptions. ImageNet
    and HICO ship class IDs and names only, so this fallback preserves the
    51-description tensor contract with the repository's official prompt set.
    It is a valid open label space, but its scores must not be compared with
    the LLM-description protocol as if they were the same experiment.
    """
    prompts = multiple_templates[:51]
    result = []
    for category in tags:
        name = processed_name(category, rm_dot=True)
        values = []
        for template in prompts:
            value = template.format(name, article=article(category))
            if value.startswith("a") or value.startswith("the"):
                value = "This is " + value
            values.append(value)
        result.append({category: values})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("ram", "ram_plus"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tag-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="raw float32 embedding output")
    parser.add_argument("--mode", choices=("closed", "open"), default="closed")
    parser.add_argument("--reference-tags", type=Path,
                        help="base checkpoint tag list for closed-set selection")
    parser.add_argument("--descriptions", type=Path,
                        help="RAM++ JSON list of {category: [descriptions]} for open-set")
    parser.add_argument("--generic-descriptions", action="store_true",
                        help="RAM++ open-set fallback: build 51 official prompt descriptions")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--thresholds", type=Path,
                        help="optional one threshold per output tag")
    args = parser.parse_args()

    tags = read_lines(args.tag_list)
    if not tags:
        raise SystemExit("--tag-list is empty")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    state = checkpoint.get("model", checkpoint)
    if args.mode == "closed":
        key = "label_embed" if args.model == "ram_plus" else "label_embed"
        tensor = state[key].detach().float()
        reference = read_lines(args.reference_tags or
                               ROOT / "ram/data/ram_tag_list.txt")
        indices = []
        position = {name: index for index, name in enumerate(reference)}
        missing = [name for name in tags if name not in position]
        if missing:
            raise SystemExit(f"closed label list contains {len(missing)} unknown tags")
        if args.model == "ram_plus":
            tensor = tensor.reshape(len(reference), -1, 512)
            indices = [position[name] for name in tags]
            tensor = tensor[indices].reshape(-1, 512)
        else:
            tensor = tensor[[position[name] for name in tags]]
    else:
        if args.model == "ram_plus":
            if args.descriptions:
                descriptions = json.loads(args.descriptions.read_text(encoding="utf-8"))
            elif args.generic_descriptions:
                descriptions = generic_descriptions(tags)
            else:
                raise SystemExit("RAM++ open-set mode requires --descriptions or --generic-descriptions")
            tensor, generated_tags = build_openset_llm_label_embedding(descriptions)
        else:
            tensor, generated_tags = build_openset_label_embedding(tags)
        generated_tags = [str(item) for item in generated_tags]
        if generated_tags != tags:
            raise SystemExit("description categories do not match --tag-list order")
        tensor = tensor.detach().float()

    if tensor.numel() % (512 * len(tags)):
        raise SystemExit("embedding does not have 512 columns per description")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(tensor.cpu(), dtype=np.float32).tofile(args.output)
    thresholds = ([float(line) for line in args.thresholds.read_text().splitlines() if line.strip()]
                  if args.thresholds else [args.threshold] * len(tags))
    if len(thresholds) != len(tags) and args.mode == "closed" and args.reference_tags:
        reference_thresholds = thresholds
        reference_names = read_lines(args.reference_tags)
        threshold_by_name = dict(zip(reference_names, reference_thresholds))
        thresholds = [threshold_by_name[name] for name in tags]
    if len(thresholds) != len(tags):
        raise SystemExit("threshold count does not match --tag-list")
    args.output.with_suffix(".tags.txt").write_text("\n".join(tags) + "\n", encoding="utf-8")
    args.output.with_suffix(".thresholds.txt").write_text(
        "\n".join(f"{value:.8g}" for value in thresholds) + "\n", encoding="utf-8")
    metadata = {
        "model": args.model,
        "mode": args.mode,
        "classes": len(tags),
        "descriptions_per_class": int(tensor.shape[0] // len(tags)),
        "embedding": str(args.output),
        "tag_list": str(args.output.with_suffix(".tags.txt")),
        "thresholds": str(args.output.with_suffix(".thresholds.txt")),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n",
                                                encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

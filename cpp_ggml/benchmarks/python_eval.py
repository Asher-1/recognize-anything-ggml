#!/usr/bin/env python3
"""Persistent PyTorch evaluator used by the full GGML comparison harness.

The official model is loaded once, images are processed in batches, and one
JSONL row plus a float32 probability matrix is written.  A manifest contains
one absolute image path per line, so a run can be sharded and resumed without
changing the model or preprocessing code.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cpp_ggml/scripts"))
from ram import get_transform  # noqa: E402
from ram.models import ram, ram_plus, tag2text  # noqa: E402
from batch_inference import forward_ram, forward_ram_plus, forward_tag2text  # noqa: E402
from model_manifest import MODEL_CHECKPOINTS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("ram", "ram_plus", "tag2text"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-matrix", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tag-list", type=Path)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--label-embedding", type=Path)
    parser.add_argument("--threshold", type=float, default=-1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1,
                        help="warmup passes for the first image batch")
    parser.add_argument("--repeat", type=int, default=1,
                        help="measured passes for every image batch")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_model(name: str, checkpoint: Path, device: torch.device):
    settings = {
        "pretrained": str(checkpoint),
        "image_size": 384,
        "vit": "swin_b" if name == "tag2text" else "swin_l",
    }
    if name == "tag2text":
        settings["delete_tag_index"] = [127, 2961, 3351, 3265, 3338, 3355, 3359]
    factory = {"ram": ram, "ram_plus": ram_plus, "tag2text": tag2text}[name]
    return factory(**settings).eval().to(device)


def apply_label_space(model, args: argparse.Namespace, device: torch.device) -> list[str]:
    names = [str(item) for item in model.tag_list.tolist()]
    if args.tag_list:
        names = [line.strip() for line in args.tag_list.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    if args.label_embedding:
        values = np.fromfile(args.label_embedding, dtype=np.float32)
        require = 512 * len(names)
        if require <= 0 or values.size % require:
            raise ValueError("label embedding size is not [classes, descriptions, 512]")
        model.label_embed = torch.nn.Parameter(
            torch.from_numpy(values.reshape(-1, 512)).to(device))
        model.num_class = len(names)
    if args.thresholds:
        threshold_values = [float(line) for line in
                            args.thresholds.read_text(encoding="utf-8").splitlines() if line.strip()]
        model.class_threshold = torch.tensor(threshold_values, dtype=torch.float32, device=device)
    if args.threshold >= 0:
        model.class_threshold = torch.full((len(names),), args.threshold,
                                           dtype=torch.float32, device=device)
    elif len(model.class_threshold) != len(names):
        model.class_threshold = torch.full((len(names),), 0.5,
                                           dtype=torch.float32, device=device)
    if len(model.class_threshold) != len(names):
        raise ValueError("threshold count does not match label-space count")
    model.tag_list = np.asarray(names)
    return names


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.limit < 0 or args.warmup < 0 or args.repeat <= 0:
        raise SystemExit("invalid batch, limit, warmup or repeat argument")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("PyTorch CUDA is unavailable; use --device cpu explicitly")
    checkpoint = args.checkpoint or ROOT / "cpp_ggml/models/pytorch" / MODEL_CHECKPOINTS[args.model]
    paths = [Path(line.strip()) for line in args.manifest.read_text().splitlines() if line.strip()]
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit("manifest contains no image paths")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_matrix.parent.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model, checkpoint, device)
    transform = get_transform(384)
    tag_names = apply_label_space(model, args, device)
    thresholds = model.class_threshold.detach().cpu().numpy().astype(np.float32)

    with args.output_jsonl.open("w", encoding="utf-8") as jsonl, \
            args.output_matrix.open("wb") as matrix:
        for start in range(0, len(paths), args.batch_size):
            batch_paths = paths[start:start + args.batch_size]

            def run_batch() -> tuple[np.ndarray, float, list[list[dict[str, object]]]]:
                # Keep this scope identical to the C++ manifest pipeline:
                # decode, resize/normalize, device transfer, graph execution,
                # host readback and threshold selection are all timed.
                if device.type == "cuda":
                    torch.cuda.synchronize()
                begin = time.perf_counter()
                images = torch.stack([transform(Image.open(path).convert("RGB"))
                                      for path in batch_paths]).to(device)
                with torch.inference_mode():
                    if args.model == "ram_plus":
                        scores = forward_ram_plus(model, images)
                    elif args.model == "ram":
                        scores = forward_ram(model, images)
                    else:
                        scores = forward_tag2text(model, list(range(len(tag_names))), images)
                    scores_np = scores.detach().float().cpu().numpy()
                    selected_batch = [[
                        {"name": tag_names[index], "index": index, "probability": float(value)}
                        for index, value in enumerate(row)
                        if value > thresholds[index]
                    ] for row in scores_np]
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - begin) * 1000.0
                return scores_np, elapsed_ms, selected_batch

            if start == 0:
                for _ in range(args.warmup):
                    run_batch()
            measured = [run_batch() for _ in range(args.repeat)]
            scores_np = measured[-1][0]
            elapsed_ms = sum(item[1] for item in measured) / len(measured)
            selected_batch = measured[-1][2]
            scores_np.tofile(matrix)
            for path, selected in zip(batch_paths, selected_batch):
                jsonl.write(json.dumps({
                    "image": str(path),
                    "latency_ms": elapsed_ms / len(batch_paths),
                    "batch_latency_ms": elapsed_ms,
                    "batch_size": len(batch_paths),
                    "warmup": args.warmup if start == 0 else 0,
                    "repeat": args.repeat,
                    "tags": selected,
                }, ensure_ascii=False) + "\n")
            jsonl.flush()
            matrix.flush()
    print(json.dumps({
        "model": args.model,
        "engine": "pytorch",
        "device": str(device),
        "images": len(paths),
        "classes": len(tag_names),
        "matrix_kind": "probability",
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "repeat": args.repeat,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Persistent Tag2Text caption comparison for every repository image."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))
from python_eval import load_model  # noqa: E402
from model_manifest import MODEL_CHECKPOINTS  # noqa: E402
from ram import get_transform  # noqa: E402
from full_eval import gpu_process_ids, make_records, reserve_gpu_slot, run_command_monitored  # noqa: E402


def word_f1(reference: str, candidate: str) -> float:
    left, right = reference.lower().split(), candidate.lower().split()
    common = sum(min(left.count(word), right.count(word)) for word in set(left))
    return 2 * common / max(1, len(left) + len(right))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", type=Path)
    parser.add_argument("--scope", choices=("images", "datasets"), default="images",
                        help="select repository images or the retained GT dataset frames")
    parser.add_argument("--datasets", nargs="+",
                        default=("openimages_common_214", "openimages_rare_200",
                                 "imagenet_multi", "hico"),
                        help="datasets used with --scope datasets")
    parser.add_argument("--backends", nargs="+", choices=("cuda", "vulkan"), default=("cuda", "vulkan"))
    parser.add_argument("--dtypes", nargs="+", choices=("f32", "f16", "q8_0"), default=("f32", "f16", "q8_0"))
    parser.add_argument("--output-dir", type=Path, default=BENCH / "full_eval")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--q8-model", type=Path,
                        help="override the q8_0 GGUF for quantization experiments")
    parser.add_argument("--threshold-epsilon", type=float,
                        help="forward Tag2Text threshold tie tolerance to GGML")
    parser.add_argument("--check-acceptance", action="store_true")
    parser.add_argument("--min-agreement", type=float, default=0.99)
    parser.add_argument("--wait-gpu-idle", action="store_true")
    parser.add_argument("--allow-gpu-pid", action="append", type=int, default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.repeat <= 0:
        raise SystemExit("--warmup must be non-negative and --repeat positive")
    if not 0 <= args.min_agreement <= 1 or any(pid <= 0 for pid in args.allow_gpu_pid):
        raise SystemExit("invalid acceptance target or GPU PID")
    if args.images:
        candidates = args.images
    elif args.scope == "datasets":
        records = make_records("datasets", args.datasets)
        candidates = [Path(record["image"]) for dataset in args.datasets
                      for record in records[dataset]]
    else:
        candidates = sorted((ROOT / "images").rglob("*"))
    images = [path.resolve() for path in candidates
              if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    if not images:
        raise SystemExit("no images selected")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.output_dir / ("caption-" + time.strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True)
    manifest = run_dir / "images.txt"
    manifest.write_text("\n".join(str(path) for path in images) + "\n", encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    allowed_pids = set(args.allow_gpu_pid) | {os.getpid()}
    gpu_reservation = None
    if args.wait_gpu_idle:
        if device.type != "cuda":
            raise RuntimeError("GPU timing audit requires the CUDA Python baseline")
        gpu_reservation = reserve_gpu_slot(allowed_pids)
    model = load_model("tag2text", ROOT / "cpp_ggml/models/pytorch" / MODEL_CHECKPOINTS["tag2text"], device)
    transform = get_transform(384)
    references = []
    reference_latencies = []
    # The legacy Tag2Text beam decoder is not batch-safe on current
    # Transformers releases: batched generation can repeat row zero. Keep the
    # model resident but decode each image independently for a valid baseline.
    def run_python(path: Path) -> tuple[str, float]:
        with torch.inference_mode():
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            image = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            caption = model.generate(image, max_length=args.max_length)[0]
            if device.type == "cuda":
                torch.cuda.synchronize()
            return caption, (time.perf_counter() - start) * 1000.0

    gpu_samples = []
    gpu_errors = []
    audit_stop = threading.Event()
    def audit_python() -> None:
        while not audit_stop.is_set():
            try:
                gpu_samples.append(sorted(gpu_process_ids()))
            except (OSError, subprocess.SubprocessError) as error:
                gpu_errors.append(str(error))
            audit_stop.wait(0.5)
    if args.wait_gpu_idle:
        while gpu_process_ids() - allowed_pids:
            print("waiting for transient GPU processes before Python captions", flush=True)
            time.sleep(5)
        audit_worker = threading.Thread(target=audit_python, daemon=True)
        audit_worker.start()
    try:
        for index, path in enumerate(images):
            if index == 0:
                for _ in range(args.warmup):
                    run_python(path)
            measured = [run_python(path) for _ in range(args.repeat)]
            references.append(measured[-1][0])
            reference_latencies.append(float(np.mean([elapsed for _, elapsed in measured])))
    finally:
        if args.wait_gpu_idle:
            audit_stop.set()
            audit_worker.join()
    reference_path = run_dir / "pytorch.jsonl"
    with reference_path.open("w", encoding="utf-8") as stream:
        for image, caption, latency in zip(images, references, reference_latencies):
            stream.write(json.dumps({"image": str(image), "caption": caption,
                                     "latency_ms": latency}, ensure_ascii=False) + "\n")
    if args.wait_gpu_idle:
        unexpected = sorted({pid for sample in gpu_samples for pid in sample} - allowed_pids)
        (run_dir / "pytorch_gpu_audit.json").write_text(json.dumps(
            {"allowed_gpu_pids": sorted(allowed_pids), "transient_gpu_pids": unexpected,
             "samples": gpu_samples, "errors": gpu_errors}, indent=2) + "\n")
        if unexpected or gpu_errors or not gpu_samples:
            raise RuntimeError(f"Python caption timing audit failed: {unexpected}, {gpu_errors}")

    rows = []
    for backend in args.backends:
        binary = ROOT / f"cpp_ggml/build-{backend}/ram-ggml"
        for dtype in args.dtypes:
            stem = f"{backend}-{dtype}"
            json_path = run_dir / f"{stem}.jsonl"
            model_path = ROOT / "cpp_ggml/models/gguf" / f"tag2text_swin_14m-{dtype}.gguf"
            if dtype == "q8_0" and args.q8_model:
                model_path = args.q8_model.resolve()
            command = [str(binary), "--model", str(model_path), "--backend", backend,
                       "--task", "caption", "--manifest", str(manifest), "--jsonl", str(json_path),
                       "--threads", str(args.threads), "--warmup", str(args.warmup),
                       "--repeat", str(args.repeat),
                       "--max-length", str(args.max_length)]
            if args.threshold_epsilon is not None:
                command += ["--threshold-epsilon", str(args.threshold_epsilon)]
            if args.wait_gpu_idle:
                result = run_command_monitored(command, run_dir / f"{stem}.log", allowed_pids)
                if result["transient_gpu_pids"]:
                    raise RuntimeError(f"{stem} caption timing audit detected GPU overlap: "
                                       f"{result['transient_gpu_pids']}")
            else:
                result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
                (run_dir / f"{stem}.log").write_text(result.stdout + "\n--- stderr ---\n" + result.stderr)
                if result.returncode:
                    raise RuntimeError(result.stderr[-2000:])
            actual_rows = [json.loads(line) for line in json_path.read_text().splitlines()]
            if len(actual_rows) != len(references):
                raise ValueError(f"{json_path}: expected {len(references)} rows, got {len(actual_rows)}")
            details = []
            for reference, actual in zip(references, actual_rows):
                text = actual["caption"]
                details.append({"exact": text == reference, "word_f1": word_f1(reference, text),
                                "caption": text, "latency_ms": actual["latency_ms"]})
            rows.append({"backend": backend, "dtype": dtype,
                         "exact_fraction": float(np.mean([item["exact"] for item in details])),
                         "word_f1": float(np.mean([item["word_f1"] for item in details])),
                         "latency_ms": float(np.mean([item["latency_ms"] for item in details])),
                         "captions": details})
    payload = {"images": [str(path) for path in images], "device": str(device),
               "references": references, "reference_latency_ms": float(np.mean(reference_latencies)),
               "warmup": args.warmup, "repeat": args.repeat, "measurements": rows}
    (run_dir / "caption_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    labels = [f"{row['backend']}\n{row['dtype']}" for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(max(12, len(rows) * 0.55), 5))
    axes[0].bar(range(len(rows)), [row["word_f1"] for row in rows], color="#2ca02c")
    axes[1].bar(range(len(rows)), [row["exact_fraction"] for row in rows], color="#1f77b4")
    axes[0].set_ylabel("Mean word F1 vs PyTorch")
    axes[1].set_ylabel("Exact-caption fraction")
    axes[1].set_ylim(0, 1.05)
    for axis in axes:
        axis.set_xticks(range(len(rows)), labels, rotation=75, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(run_dir / "caption_comparison_all.png", dpi=160)
    plt.close(figure)

    baseline_ms = float(np.mean(reference_latencies))
    lines = ["# All-image Tag2Text caption comparison", "",
             f"Images: {len(images)}. PyTorch device: `{device}`. Each GGML combination loads once and processes the manifest. "
             f"Single-image pipeline: one first-image warmup, {args.repeat} measured passes per image. "
             "Decode, preprocess, transfer, generation and caption text are timed; model loading and JSONL writing are excluded.", "",
             f"PyTorch CUDA mean: **{baseline_ms:.2f} ms/image**.", "",
             "| Backend | Format | GGML ms | GGML/PyTorch | Exact fraction | Mean word F1 |",
             "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['backend']} | {row['dtype']} | "
                     f"{row['latency_ms']:.2f} | {row['latency_ms'] / baseline_ms:.3f}x | "
                     f"{row['exact_fraction']:.3f} | {row['word_f1']:.3f} |")
    lines.extend(["", "![Caption comparison](caption_comparison_all.png)", "",
                  "The word F1 is lexical overlap with the official Python output; it is not CIDEr/BLEU or a human semantic score.", "",
                  "## Changed captions", ""])
    for index, image in enumerate(images):
        for row in rows:
            if row["captions"][index]["caption"] != references[index]:
                lines.append(f"- `{image.relative_to(ROOT)}` {row['backend']} {row['dtype']}: "
                             f"{row['captions'][index]['caption']}")
    (run_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "images": len(images), "combinations": len(rows)}, indent=2))
    if args.check_acceptance:
        failures = [{key: row[key] for key in ("backend", "dtype", "exact_fraction", "word_f1", "latency_ms")}
                    for row in rows if row["exact_fraction"] < args.min_agreement or
                    row["word_f1"] < args.min_agreement or row["latency_ms"] >= baseline_ms]
        acceptance = {"passed": not failures, "min_agreement": args.min_agreement,
                      "pytorch_cuda_ms": baseline_ms, "failures": failures}
        (run_dir / "acceptance.json").write_text(json.dumps(acceptance, indent=2) + "\n")
        print(json.dumps({"acceptance": acceptance}, indent=2))
        return 0 if acceptance["passed"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

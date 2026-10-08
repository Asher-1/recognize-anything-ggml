#!/usr/bin/env python3
"""Compare real GGUF outputs against official PyTorch logits on an image."""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from model_manifest import gguf_name


ROOT = Path(__file__).resolve().parents[2]


def thresholds(model):
    if model == "tag2text":
        result = np.full(3429, 0.68, dtype=np.float32)
        result[[2701, 2828, 1167]] = 0.7
        return result
    return np.array([float(line) for line in
                     (ROOT / "ram/data/ram_tag_list_threshold.txt").read_text().splitlines()])


def selected(logits, model):
    probabilities = 1 / (1 + np.exp(-logits))
    thresholds_for_model = thresholds(model)
    result = set(np.flatnonzero(probabilities > thresholds_for_model).tolist())
    if model == "tag2text":
        result.difference_update({127, 2961, 3351, 3265, 3338, 3355, 3359})
    return result


def stable_tag_diff(expected, actual, model, margin):
    """Ignore a flip only when the official probability is at the threshold."""
    expected_probabilities = 1 / (1 + np.exp(-expected))
    ambiguous = set(np.flatnonzero(np.abs(expected_probabilities - thresholds(model)) <= margin).tolist())
    return (selected(actual, model) ^ selected(expected, model)) - ambiguous, ambiguous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=ROOT / "images/demo/demo1.jpg")
    parser.add_argument("--models", nargs="+", choices=("ram", "ram_plus", "tag2text"),
                        default=("ram", "ram_plus", "tag2text"))
    parser.add_argument("--backends", nargs="+", choices=("cuda", "vulkan", "cpu"),
                        default=("cuda", "vulkan"))
    parser.add_argument("--dtypes", nargs="+", choices=("f32", "f16", "q8_0"),
                        default=("f32", "f16", "q8_0"))
    parser.add_argument("--max-mae", type=float, default=0.01)
    parser.add_argument("--max-slowdown", type=float,
                        help="optional C++/Python tagging latency limit, e.g. 1.0")
    parser.add_argument("--threshold-margin", type=float, default=0.01,
                        help="allow threshold flips only within this official probability margin")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if args.max_slowdown is not None and args.max_slowdown <= 0:
        parser.error("--max-slowdown must be positive")
    args.image = args.image.resolve()
    failed = False
    with tempfile.TemporaryDirectory(prefix="ram-ggml-parity-") as directory:
        temporary = Path(directory)
        for model in args.models:
            python_latency = None
            if args.max_slowdown is not None:
                baseline = subprocess.run(
                    [sys.executable, str(ROOT / "cpp_ggml/scripts/benchmark_python.py"),
                     "--model", model, "--task", "tags", "--image", str(args.image),
                     "--warmup", str(args.warmup), "--repeat", str(args.repeat)],
                    cwd=ROOT, capture_output=True, text=True, check=True)
                match = re.search(r"latency_ms=([0-9.]+)", baseline.stdout)
                if not match or "backend=python-cuda" not in baseline.stdout:
                    raise RuntimeError(f"Python CUDA baseline unavailable: {baseline.stdout}")
                python_latency = float(match.group(1))
                print(f"BASELINE {model:8} python-cuda latency_ms={python_latency:.3f}", flush=True)
            reference = temporary / model / "reference"
            subprocess.run([sys.executable, str(ROOT / "cpp_ggml/scripts/parity_reference.py"),
                            "--model", model, "--image", str(args.image),
                            "--output", str(reference)], cwd=ROOT, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            expected = np.fromfile(reference / "logits.bin", dtype=np.float32)
            expected_tags = selected(expected, model)
            for backend in args.backends:
                binary = ROOT / "cpp_ggml" / f"build-{backend}" / "ram-ggml"
                if not binary.is_file():
                    raise FileNotFoundError(f"Build missing: {binary}")
                for dtype in args.dtypes:
                    gguf = ROOT / "cpp_ggml/models/gguf" / gguf_name(model, dtype)
                    output = temporary / f"{model}-{backend}-{dtype}.bin"
                    command = [str(binary), "--model", str(gguf), "--image", str(args.image),
                               "--backend", backend, "--warmup", str(args.warmup),
                               "--repeat", str(args.repeat), "--logits-out", str(output)]
                    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
                    if result.returncode:
                        print(f"FAIL {model} {backend} {dtype}: {result.stderr[-800:]}")
                        failed = True
                        continue
                    actual = np.fromfile(output, dtype=np.float32)
                    mae = float(np.abs(actual - expected).mean())
                    wrong, ambiguous = stable_tag_diff(expected, actual, model, args.threshold_margin)
                    latency = re.search(r"latency_ms=([0-9.]+)", result.stdout)
                    latency_ms = float(latency.group(1)) if latency else None
                    slowdown = latency_ms / python_latency if python_latency and latency_ms else None
                    passed = mae <= args.max_mae and not wrong and (
                        python_latency is None or
                        (slowdown is not None and slowdown <= args.max_slowdown))
                    failed |= not passed
                    slowdown_text = f" slowdown={slowdown:.3f}x" if slowdown is not None else ""
                    print(f"{'PASS' if passed else 'FAIL'} {model:8} {backend:6} {dtype:4} "
                          f"logit_mae={mae:.5f} tag_diff={sorted(wrong)} "
                          f"ambiguous={sorted(ambiguous & (selected(actual, model) ^ expected_tags))} "
                          f"latency_ms={latency.group(1) if latency else '?'}{slowdown_text}",
                          flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

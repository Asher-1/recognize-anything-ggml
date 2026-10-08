#!/usr/bin/env python3
"""Measure the image-to-tags pipeline and compare GGUF logits with PyTorch."""

import argparse
import csv
import json
import platform
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "cpp_ggml/scripts"
sys.path.insert(0, str(SCRIPTS))
from model_manifest import gguf_name
from validate import selected, stable_tag_diff


def command(args):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{' '.join(map(str, args))}\n{result.stdout[-1200:]}\n{result.stderr[-1200:]}")
    return result.stdout


def latency(output):
    found = re.search(r"latency_ms=([0-9.]+)", output)
    if not found:
        raise ValueError(f"latency missing from output: {output[-1000:]}")
    return float(found.group(1))


def machine():
    info = {"platform": platform.platform(), "python": platform.python_version()}
    for key, args in {
        "gpu": ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        "ggml_commit": ["git", "-C", "cpp_ggml/third_party/ggml", "rev-parse", "HEAD"],
    }.items():
        try:
            info[key] = command(args).strip()
        except (OSError, RuntimeError):
            info[key] = "unavailable"
    return info


def draw(rows, baselines, image, out):
    models = list(baselines)
    formats = ["f32", "f16", "q8_0"]
    backends = ["cuda", "vulkan"]
    colors = {"python-cuda": "#272d35", "cuda": "#147d86", "vulkan": "#e27645"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for axis, model in zip(axes, models):
        subset = [r for r in rows if r["model"] == model]
        baseline = baselines[model]
        axis.axhline(baseline, color=colors["python-cuda"], linewidth=2,
                     label="PyTorch CUDA" if model == models[0] else None)
        x = np.arange(len(formats))
        for offset, backend in [(-0.18, "cuda"), (0.18, "vulkan")]:
            values = [next((r["pipeline_ms"] for r in subset
                            if r["backend"] == backend and r["dtype"] == fmt), np.nan)
                      for fmt in formats]
            bars = axis.bar(x + offset, values, 0.33, color=colors[backend],
                            label=f"GGML {backend.upper()}" if model == models[0] else None)
            for bar, value in zip(bars, values):
                if np.isfinite(value):
                    axis.annotate(f"{value / baseline:.2f}x", (bar.get_x() + bar.get_width() / 2,
                                  value), xytext=(0, 3), textcoords="offset points", ha="center",
                                  fontsize=8)
        axis.set_title({"ram": "RAM", "ram_plus": "RAM++", "tag2text": "Tag2Text"}[model])
        axis.set_xticks(x, formats)
        axis.set_ylabel("Image-to-tags latency (ms)" if model == models[0] else "")
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.set_ylim(0, max([baseline] + [r["pipeline_ms"] for r in subset]) * 1.22)
    fig.legend(loc="outside upper center", ncol=3, frameon=False)
    fig.savefig(out / "pipeline_latency.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    labels = [f"{r['model']}\n{r['backend']} {r['dtype']}" for r in rows]
    positions = np.arange(len(rows))
    axes[0].bar(positions, [r["logit_mae"] for r in rows], color=[colors[r["backend"]] for r in rows])
    axes[0].axhline(0.01, color="#b5333d", linestyle="--", label="MAE gate 0.01")
    axes[0].set_ylabel("Logit mean absolute error")
    axes[0].legend(frameon=False)
    axes[1].bar(positions, [r["tag_jaccard"] for r in rows], color=[colors[r["backend"]] for r in rows])
    axes[1].set_ylabel("Tag-set Jaccard vs PyTorch")
    axes[1].set_ylim(0, 1.05)
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=90, fontsize=7)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    fig.savefig(out / "precision.png", dpi=170)
    plt.close(fig)

    photo = Image.open(image).convert("RGB")
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.imshow(photo)
    axis.set_axis_off()
    fig.savefig(out / "test_image.png", dpi=140)
    plt.close(fig)


def report(rows, baselines, args, info):
    lines = ["# Image-to-tags pipeline benchmark", "",
             f"Measured {datetime.now().astimezone().isoformat(timespec='seconds')} on "
             f"`{info['gpu']}`. Input: `{args.image.relative_to(ROOT)}`. "
             f"GGML commit: `{info['ggml_commit']}`.", "",
             "![Test image](test_image.png)", "",
             "The timed scope includes image decode, resize/normalization, accelerator transfer, "
             "inference, logit readback and tag thresholding. Models load before timing. "
             f"Each process uses {args.warmup} warmups and {args.repeat} measured iterations. "
             "Runs execute sequentially on the same GPU; timing is a per-image arithmetic mean "
             "and includes host work. Python uses the official model implementations.", "",
             "![Pipeline latency by model and format](pipeline_latency.png)", "",
             "Ratio labels on the bars are GGML / PyTorch CUDA; below 1.00 is faster. "
             "Each baseline is a separate PyTorch CUDA process.", "",
             "![Logit and tag agreement](precision.png)", "",
             "Logit MAE and tag sets compare against official PyTorch CUDA output on the same image. "
             "Jaccard is intersection divided by union; threshold-boundary flips are also reported.", "",
             "| Model | Engine | Format | Pipeline ms | vs PyTorch CUDA | Logit MAE | Max abs err | Tag Jaccard | Non-boundary flips |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for model, value in baselines.items():
        lines.append(f"| {model} | PyTorch CUDA | checkpoint | {value:.3f} | 1.000x | reference | reference | 1.000 | 0 |")
        for r in rows:
            if r["model"] == model:
                lines.append(f"| {model} | GGML {r['backend'].upper()} | {r['dtype']} | "
                             f"{r['pipeline_ms']:.3f} | {r['slowdown']:.3f}x | {r['logit_mae']:.6f} | "
                             f"{r['max_abs_error']:.6f} | {r['tag_jaccard']:.3f} | {r['nonboundary_flips']} |")
    lines.extend(["", "This legacy single-image table measures tagging. The canonical "
                  "50-frame caption comparison is in `cpp_ggml/benchmarks/LATENCY_MATRIX.md`. "
                  "Results are device- and image-dependent; rows above 1.00x do not support a "
                  "GGML speedup claim. Raw numbers and metadata are in `results.json` and `results.csv`.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=ROOT / "images/demo/demo1.jpg")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "cpp_ggml/benchmarks/full_eval/run-single-image")
    parser.add_argument("--models", nargs="+", choices=("ram", "ram_plus", "tag2text"),
                        default=("ram", "ram_plus", "tag2text"))
    parser.add_argument("--backends", nargs="+", choices=("cuda", "vulkan"),
                        default=("cuda", "vulkan"))
    parser.add_argument("--dtypes", nargs="+", choices=("f32", "f16", "q8_0"),
                        default=("f32", "f16", "q8_0"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--threshold-margin", type=float, default=0.01)
    parser.add_argument("--render-only", action="store_true", help="redraw existing results")
    args = parser.parse_args()
    args.image = args.image.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        saved = json.loads((args.output / "results.json").read_text())
        draw(saved["measurements"], saved["pytorch_cuda_ms"], args.image, args.output)
        return
    info = machine()
    rows, baselines = [], {}
    with tempfile.TemporaryDirectory(prefix="ram-benchmark-") as temp:
        for model in args.models:
            reference = Path(temp) / model / "reference"
            command([sys.executable, SCRIPTS / "parity_reference.py", "--model", model,
                     "--image", args.image, "--output", reference])
            expected = np.fromfile(reference / "logits.bin", dtype=np.float32)
            base_output = command([sys.executable, SCRIPTS / "benchmark_python.py",
                                   "--model", model, "--image", args.image, "--task", "tags",
                                   "--warmup", str(args.warmup), "--repeat", str(args.repeat),
                                   "--pipeline"])
            if "backend=python-cuda" not in base_output:
                raise RuntimeError("PyTorch CUDA baseline unavailable")
            baselines[model] = latency(base_output)
            print(f"{model}: PyTorch CUDA {baselines[model]:.3f} ms", flush=True)
            for backend in args.backends:
                binary = ROOT / "cpp_ggml" / f"build-{backend}" / "ram-ggml"
                if not binary.is_file():
                    raise FileNotFoundError(binary)
                for dtype in args.dtypes:
                    gguf = ROOT / "cpp_ggml/models/gguf" / gguf_name(model, dtype)
                    logits_path = Path(temp) / f"{model}-{backend}-{dtype}.bin"
                    output = command([binary, "--model", gguf, "--image", args.image,
                                      "--backend", backend, "--warmup", str(args.warmup),
                                      "--repeat", str(args.repeat), "--pipeline", "1",
                                      "--logits-out", logits_path])
                    actual = np.fromfile(logits_path, dtype=np.float32)
                    if len(actual) != len(expected):
                        raise ValueError(f"wrong logit count for {model} {backend} {dtype}")
                    chosen, official = selected(actual, model), selected(expected, model)
                    wrong, ambiguous = stable_tag_diff(expected, actual, model,
                                                       args.threshold_margin)
                    elapsed = latency(output)
                    row = dict(model=model, backend=backend, dtype=dtype,
                               pipeline_ms=elapsed, slowdown=elapsed / baselines[model],
                               logit_mae=float(np.abs(actual - expected).mean()),
                               max_abs_error=float(np.abs(actual - expected).max()),
                               tag_jaccard=len(chosen & official) / max(1, len(chosen | official)),
                               nonboundary_flips=len(wrong), boundary_flips=len((chosen ^ official) & ambiguous),
                               tag_count=len(chosen), official_tag_count=len(official),
                               gguf_bytes=gguf.stat().st_size)
                    rows.append(row)
                    print(f"{model} {backend} {dtype}: {elapsed:.3f} ms, "
                          f"{row['slowdown']:.3f}x, MAE {row['logit_mae']:.5f}", flush=True)
    payload = dict(timestamp=datetime.now(timezone.utc).isoformat(), image=str(args.image.relative_to(ROOT)),
                   warmup=args.warmup, repeat=args.repeat, threshold_margin=args.threshold_margin,
                   machine=info, pytorch_cuda_ms=baselines, measurements=rows)
    (args.output / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (args.output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    draw(rows, baselines, args.image, args.output)
    (args.output / "README.md").write_text(report(rows, baselines, args, info))


if __name__ == "__main__":
    main()

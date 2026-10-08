#!/usr/bin/env python3
"""Benchmark and validate Tag2Text image-to-caption across GGUF formats."""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def run(args):
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"{' '.join(map(str, args))}\n{result.stdout[-1000:]}\n{result.stderr[-1000:]}")
    return result.stdout


def field(output, expression):
    matched = re.search(expression, output, re.MULTILINE)
    if not matched:
        raise ValueError(f"missing {expression} in {output[-1000:]}")
    return matched.group(1).strip()


def word_f1(reference, candidate):
    a, b = reference.lower().split(), candidate.lower().split()
    if not a and not b:
        return 1.0
    common = sum(min(a.count(word), b.count(word)) for word in set(a))
    return 2 * common / max(1, len(a) + len(b))


def chart(rows, baselines, output_dir):
    images = list(baselines)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    colors = {"cuda": "#147d86", "vulkan": "#e27645"}
    all_positions, all_labels = [], []
    for index, image in enumerate(images):
        subset = [r for r in rows if r["image"] == image]
        positions = np.arange(len(subset)) + index * (len(subset) + 1)
        axes[0].bar(positions, [r["pipeline_ms"] for r in subset],
                    color=[colors[r["backend"]] for r in subset])
        axes[0].hlines(baselines[image], positions[0] - 0.5, positions[-1] + 0.5,
                       colors="#272d35", linewidth=2)
        axes[1].bar(positions, [r["word_f1"] for r in subset],
                    color=[colors[r["backend"]] for r in subset])
        labels = [f"{r['backend']}\n{r['dtype']}" for r in subset]
        all_positions.extend(positions)
        all_labels.extend(labels)
        axes[0].text(positions.mean(), baselines[image], f"PyTorch CUDA {Path(image).stem}",
                     ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel("Image-to-caption latency (ms)")
    axes[1].set_ylabel("Caption word F1 vs official Python")
    axes[1].set_ylim(0, 1.08)
    for axis in axes:
        axis.set_xticks(all_positions, all_labels)
        axis.tick_params(axis="x", labelsize=8)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    fig.savefig(output_dir / "caption_comparison.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", type=Path,
                        default=[ROOT / "images/demo/demo1.jpg", ROOT / "images/demo/demo2.jpg"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--backends", nargs="+", choices=("cuda", "vulkan"),
                        default=("cuda", "vulkan"))
    parser.add_argument("--dtypes", nargs="+", choices=("f32", "f16", "q8_0"),
                        default=("f32", "f16", "q8_0"))
    parser.add_argument("--output-dir", type=Path,
                        default=OUT / "full_eval/caption-single-image")
    parser.add_argument("--specified-tags", help="comma-separated Tag2Text caption tags")
    parser.add_argument("--beam-batch", type=int, choices=(1, 3), default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gpu = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
               "--format=csv,noheader"]).splitlines()[0]
    rows, baselines, references = [], {}, {}
    for path in args.images:
        image = path.resolve().relative_to(ROOT).as_posix()
        official_args = [sys.executable, "run_inference.py", "--engine", "python",
                         "--model", "tag2text", "--task", "caption", "--image", path]
        if args.specified_tags:
            official_args.extend(("--specified-tags", args.specified_tags))
        official = run(official_args)
        references[image] = field(official, r"^Image Caption:\s*(.*)$")
        timing_args = [sys.executable, "cpp_ggml/scripts/benchmark_python.py",
                       "--model", "tag2text", "--task", "caption", "--image", path,
                       "--pipeline", "--warmup", str(args.warmup), "--repeat", str(args.repeat)]
        if args.specified_tags:
            timing_args.extend(("--specified-tags", args.specified_tags))
        timed = run(timing_args)
        if "backend=python-cuda" not in timed:
            raise RuntimeError("Python CUDA caption baseline unavailable")
        baselines[image] = float(field(timed, r"latency_ms=([0-9.]+)"))
        print(f"{image}: Python CUDA {baselines[image]:.3f} ms; {references[image]}", flush=True)
        for backend in args.backends:
            binary = ROOT / "cpp_ggml" / f"build-{backend}" / "ram-ggml"
            for dtype in args.dtypes:
                model = ROOT / "cpp_ggml/models/gguf" / f"tag2text_swin_14m-{dtype}.gguf"
                command = [binary, "--model", model, "--image", path,
                           "--backend", backend, "--task", "caption", "--pipeline", "1",
                           "--beam-batch", str(args.beam_batch),
                           "--warmup", str(args.warmup), "--repeat", str(args.repeat)]
                if args.specified_tags:
                    command.extend(("--specified-tags", args.specified_tags))
                output = run(command)
                caption = field(output, r"^caption=(.*)$")
                elapsed = float(field(output, r"latency_ms=([0-9.]+)"))
                row = dict(image=image, backend=backend, dtype=dtype, pipeline_ms=elapsed,
                           slowdown=elapsed / baselines[image], exact=caption == references[image],
                           word_f1=word_f1(references[image], caption), caption=caption)
                rows.append(row)
                print(f"{backend} {dtype}: {elapsed:.3f} ms, {row['slowdown']:.3f}x, "
                      f"exact={row['exact']}, F1={row['word_f1']:.3f}", flush=True)
    payload = dict(timestamp=datetime.now(timezone.utc).isoformat(), gpu=gpu,
                   specified_tags=args.specified_tags, beam_batch=args.beam_batch,
                   warmup=args.warmup,
                   repeat=args.repeat, pytorch_cuda_ms=baselines, references=references,
                   measurements=rows)
    (args.output_dir / "caption_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    chart(rows, baselines, args.output_dir)
    lines = ["# Tag2Text caption comparison", "",
             f"Measured on `{gpu}`. Same image-to-caption scope, {args.warmup} warmups and {args.repeat} measured runs "
             "per engine. Models load before timing. The official Python output is the reference; "
             f"GGML uses 3-beam decoding (beam batch {args.beam_batch}) and embedded BERT metadata (the checked-in vocabulary is a fallback).", "",
             "![Caption latency and word agreement](caption_comparison.png)", "",
             "| Image | Engine | Format | Pipeline ms | vs Python CUDA | Exact caption | Word F1 |",
             "| --- | --- | --- | ---: | ---: | --- | ---: |"]
    if args.specified_tags:
        lines[2] += f" Supplied tags: `{args.specified_tags}`."
    for image in baselines:
        lines.append(f"| {Path(image).name} | PyTorch CUDA | checkpoint | {baselines[image]:.3f} | 1.000x | reference | 1.000 |")
        for row in rows:
            if row["image"] == image:
                lines.append(f"| {Path(image).name} | GGML {row['backend'].upper()} | {row['dtype']} | "
                             f"{row['pipeline_ms']:.3f} | {row['slowdown']:.3f}x | "
                             f"{'yes' if row['exact'] else 'no'} | {row['word_f1']:.3f} |")
    lines.extend(["", "## Caption outputs", ""])
    for image, reference in references.items():
        lines.append(f"**{image}**: Python CUDA: {reference}")
        lines.append("")
        for row in rows:
            if row["image"] == image and not row["exact"]:
                lines.append(f"- GGML {row['backend']} {row['dtype']}: {row['caption']}")
        lines.append("")
    lines.extend(["Word F1 is token overlap, not a dataset-level caption metric. "
                  "A caption change may remain semantically valid. This sample is too small "
                  "to claim benchmark-wide caption quality parity.", ""])
    (args.output_dir / "caption_readme.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()

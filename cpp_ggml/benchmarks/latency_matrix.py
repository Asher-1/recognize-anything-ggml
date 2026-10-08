#!/usr/bin/env python3
"""Publish the measured 50-image PyTorch CUDA versus GGML matrix.

The input is a single full_eval run using batch size 1 on the Python side.
Raw run files stay ignored; this script emits a compact, reviewable summary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ("ram", "ram_plus", "tag2text")
BACKENDS = ("cuda", "vulkan")
DTYPES = ("f32", "f16", "q8_0")
DATASETS = ("openimages_common_214", "openimages_rare_200", "imagenet_multi", "hico")


def add_gt_reference(rows: list[dict[str, str]], run_dir: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from full_eval import annotation_metrics, load_matrix, load_rows
    sidecars = Path(__file__).resolve().parent / "label_spaces"
    args = argparse.Namespace(label_space_dir=sidecars)
    references = {}
    for row in rows:
        key = row["dataset"], row["model"]
        if key not in references:
            dataset, model = key
            records = load_rows(run_dir / "manifests" / f"{dataset}.records.jsonl")
            stem = f"datasets__{dataset}__{model}__pytorch"
            predictions = load_rows(run_dir / f"{stem}.jsonl")
            byte_count = (run_dir / f"{stem}.f32").stat().st_size
            classes = byte_count // (4 * len(predictions))
            scores = load_matrix(run_dir / f"{stem}.f32", len(predictions), classes, False)
            references[key] = annotation_metrics(records, model, scores, predictions, dataset, args)
        reference = references[key]
        if reference["annotation_status"] != "ok":
            raise ValueError(f"invalid PyTorch GT label space: {key}")
        for metric in ("mAP", "CP", "CR"):
            row[f"pytorch_{metric}"] = str(reference[metric])
            row[f"delta_{metric}"] = str(float(row[metric]) - reference[metric])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--caption-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--device", default="NVIDIA GeForce RTX 3060 (12 GiB), driver 550.144.03")
    parser.add_argument("--require-acceptance", action="store_true",
                        help="publish only complete passing runs with GPU audit logs")
    return parser.parse_args()


def load_run(path: Path) -> list[dict[str, str]]:
    with (path / "summary.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = {(dataset, model, backend, dtype)
                for dataset in DATASETS for model in MODELS
                for backend in BACKENDS for dtype in DTYPES}
    actual = {(row["dataset"], row["model"], row["backend"], row["dtype"]) for row in rows}
    if len(rows) != len(expected) or actual != expected:
        raise ValueError(f"expected {len(expected)} distinct matrix rows, got {len(rows)}")
    if any(row["pytorch_batch_size"] != "1" or row["annotation_status"] != "ok" for row in rows):
        raise ValueError("the public matrix requires batch_size=1 and valid GT scoring")
    if any(int(row["images"]) != (14 if row["dataset"] == DATASETS[0] else 12) for row in rows):
        raise ValueError("dataset counts do not match the retained 50-image set")
    return rows


def load_caption(path: Path) -> dict:
    payload = json.loads((path / "caption_results.json").read_text(encoding="utf-8"))
    expected = {(backend, dtype) for backend in BACKENDS for dtype in DTYPES}
    rows = payload["measurements"]
    actual = {(row["backend"], row["dtype"]) for row in rows}
    if (payload["device"] != "cuda" or len(payload["images"]) != 50 or
            payload["warmup"] != 1 or payload["repeat"] != 3 or
            len(rows) != len(expected) or actual != expected or
            any(len(row["captions"]) != 50 for row in rows)):
        raise ValueError("caption run must contain the complete 50-image CUDA baseline and six GGML combinations")
    return payload


def audit_run(path: Path, caption: bool = False) -> dict:
    acceptance = json.loads((path / "acceptance.json").read_text())
    if not acceptance["passed"]:
        raise ValueError(f"acceptance failed: {path}")
    if caption:
        logs = [path / f"{backend}-{dtype}.log" for backend in BACKENDS for dtype in DTYPES]
        python_audit = json.loads((path / "pytorch_gpu_audit.json").read_text())
        if python_audit["transient_gpu_pids"] or python_audit["errors"] or not python_audit["samples"]:
            raise ValueError(f"Python caption audit failed: {path}")
    else:
        logs = [path / "logs" / f"datasets__{dataset}__{model}__{suffix}.log"
                for dataset in DATASETS for model in MODELS
                for suffix in ["pytorch"] + [f"ggml__{backend}__{dtype}"
                                             for backend in BACKENDS for dtype in DTYPES]]
    allowed = set()
    samples = 0
    for log in logs:
        text = log.read_text()
        if "--- gpu timing audit ---" not in text:
            raise ValueError(f"missing timing audit: {log}")
        audit = json.loads(text.split("--- gpu timing audit ---")[-1])
        if audit["transient_gpu_pids"] or not audit["samples"]:
            raise ValueError(f"GPU overlap or missing samples: {log}")
        allowed.update(audit["allowed_gpu_pids"])
        samples += len(audit["samples"])
    return {"acceptance": acceptance, "audited_commands": len(logs) + int(caption),
            "process_samples": samples, "allowed_gpu_pids": sorted(allowed),
            "transient_gpu_overlap": False}


def write_metadata(args: argparse.Namespace) -> None:
    import torch
    root = Path(__file__).resolve().parents[2]
    files = ("cpp_ggml/src/main.cpp", "cpp_ggml/include/ram_ggml/inference.hpp",
             "cpp_ggml/scripts/export_gguf.py", "cpp_ggml/benchmarks/full_eval.py",
             "cpp_ggml/benchmarks/python_eval.py", "cpp_ggml/benchmarks/caption_all.py",
             "cpp_ggml/build-cuda/ram-ggml", "cpp_ggml/build-vulkan/ram-ggml",
             "cpp_ggml/third_party/patches/0001-ram-ggml-v0210-integration.patch",
             "cpp_ggml/models/SHA256SUMS")
    source_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}
    model_hashes = {name: digest for digest, name in
                    (line.split() for line in (root / "cpp_ggml/models/SHA256SUMS").read_text().splitlines())}
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                          "--format=csv,noheader"], capture_output=True, text=True, check=True)
    payload = {"tag_run": args.run_dir.name, "caption_run": args.caption_run.name,
               "gpu": gpu.stdout.strip(), "platform": platform.platform(),
               "pytorch_version": torch.__version__, "pytorch_cuda_version": torch.version.cuda,
               "ggml_revision": "8599e0ea3756c4bac4ef813af2241cb1a8bbfb0b",
               "pipeline": {"python_batch_size": 1, "cpp_threads": 8, "warmup": 1, "repeat": 3,
                            "model_loading_timed": False, "json_serialization_timed": False,
                            "cuda_tagging_compute": "FP16 inputs, FP32 accumulation/output",
                            "cuda_caption_compute": "FP32",
                            "gpu_coordination": "idle parent CUDA context during audited continuation; no compute",
                            "allowed_services": "prophet and AVM; desktop graphics may remain active"},
               "tag_audit": audit_run(args.run_dir),
               "caption_audit": audit_run(args.caption_run, caption=True),
               "source_sha256": source_hashes, "gguf_sha256": model_hashes}
    (args.output_dir / "BENCHMARK_METADATA.json").write_text(json.dumps(payload, indent=2) + "\n")


def aggregate(rows: list[dict[str, str]]) -> list[dict[str, float | str | int]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["model"], row["backend"], row["dtype"]].append(row)
    result = []
    for model in MODELS:
        for backend in BACKENDS:
            for dtype in DTYPES:
                group = groups[model, backend, dtype]
                count = sum(int(row["images"]) for row in group)
                weighted = lambda key: sum(int(row["images"]) * float(row[key]) for row in group) / count
                py = weighted("pytorch_latency_ms")
                ggml = weighted("ggml_latency_ms")
                result.append({
                    "model": model, "backend": backend, "dtype": dtype, "images": count,
                    "pytorch_ms": py, "ggml_ms": ggml, "ggml_over_pytorch": ggml / py,
                    "speedup": py / ggml, "cross_f1": weighted("cross_f1"),
                    "minimum_dataset_f1": min(float(row["cross_f1"]) for row in group),
                    "probability_mae": weighted("pytorch_cross_mae"),
                    "failed_datasets": sum(row["ggml_faster"] != "True" for row in group),
                })
    return result


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    columns = ("dataset", "model", "backend", "dtype", "images", "pytorch_latency_ms",
               "ggml_latency_ms", "latency_ratio_ggml_over_pytorch", "ggml_faster",
               "cross_f1", "pytorch_cross_mae", "annotation_status", "mAP", "CP", "CR",
               "pytorch_mAP", "pytorch_CP", "pytorch_CR", "delta_mAP", "delta_CP", "delta_CR")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in columns} for row in rows)


def write_chart(rows: list[dict[str, float | str | int]], path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharey=True)
    for axis, model in zip(axes, MODELS):
        subset = [row for row in rows if row["model"] == model]
        labels = [f"{row['backend'].upper()}\n{row['dtype'].upper()}" for row in subset]
        ratios = [float(row["ggml_over_pytorch"]) for row in subset]
        colors = ["#277f65" if ratio < 1 else "#c24c43" for ratio in ratios]
        bars = axis.bar(range(len(subset)), ratios, color=colors, width=0.7)
        axis.axhline(1, color="#252525", linewidth=1, linestyle="--")
        axis.set_title(model.upper(), fontsize=12, weight="bold")
        axis.set_xticks(range(len(subset)), labels, fontsize=8)
        axis.set_ylim(0, max(1.28, max(ratios) * 1.12))
        axis.grid(axis="y", color="#d8d8d8", linewidth=0.6)
        axis.set_axisbelow(True)
        for bar, ratio in zip(bars, ratios):
            axis.text(bar.get_x() + bar.get_width() / 2, ratio + 0.015, f"{ratio:.2f}x",
                      ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel("GGML latency / PyTorch CUDA latency (lower is faster)")
    figure.suptitle("50 retained GT images | single-image pipeline | 1 warmup + 3 measured passes")
    figure.tight_layout()
    figure.savefig(path, format="svg", bbox_inches="tight")
    plt.close(figure)


def write_accuracy_chart(rows: list[dict[str, float | str | int]], path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharey=True)
    for axis, model in zip(axes, MODELS):
        subset = [row for row in rows if row["model"] == model]
        labels = [f"{row['backend'].upper()}\n{row['dtype'].upper()}" for row in subset]
        minimums = [float(row["minimum_dataset_f1"]) for row in subset]
        colors = ["#277f65" if value >= 0.99 else "#c24c43" for value in minimums]
        bars = axis.bar(range(len(subset)), minimums, color=colors, width=0.7)
        axis.axhline(0.99, color="#252525", linewidth=1, linestyle="--")
        axis.set_title(model.upper(), fontsize=12, weight="bold")
        axis.set_xticks(range(len(subset)), labels, fontsize=8)
        axis.set_ylim(min(0.97, min(minimums) - 0.005), 1.003)
        axis.grid(axis="y", color="#d8d8d8", linewidth=0.6)
        axis.set_axisbelow(True)
        for bar, value in zip(bars, minimums):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 0.0004,
                      f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel("Minimum dataset F1 vs official PyTorch (higher is better)")
    figure.suptitle("Tag parity | minimum across four GT datasets | acceptance F1 >= 0.99")
    figure.tight_layout()
    figure.savefig(path, format="svg", bbox_inches="tight")
    plt.close(figure)


def write_gt_chart(rows: list[dict[str, str]], path: Path) -> None:
    # Tag2Text has no GT annotations in three of the four source label spaces.
    groups = [(dataset, model) for model in ("ram", "ram_plus") for dataset in DATASETS]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for axis, metric in zip(axes, ("mAP", "CP", "CR")):
        py, minimum, maximum = [], [], []
        for dataset, model in groups:
            group = [row for row in rows if row["dataset"] == dataset and row["model"] == model]
            py.append(float(group[0][f"pytorch_{metric}"]))
            values = [float(row[metric]) for row in group]
            minimum.append(min(values))
            maximum.append(max(values))
        x = list(range(len(groups)))
        axis.vlines(x, minimum, maximum, color="#277f65", linewidth=4, label="GGML range (6 variants)")
        axis.scatter(x, py, marker="x", s=40, color="#9a3a21", label="Official PyTorch CUDA", zorder=3)
        axis.set_title(metric)
        axis.set_xticks(x, [f"{model}\n{dataset.replace('openimages_', 'OI ').replace('imagenet_multi', 'ImageNet')}"
                           for dataset, model in groups], rotation=50, ha="right", fontsize=7)
        axis.set_ylim(0, 1.03)
        axis.grid(axis="y", color="#d8d8d8", linewidth=0.6)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Score against GT (higher is better)")
    axes[-1].legend(fontsize=8, loc="upper right")
    figure.suptitle("RAM / RAM++ GT metrics | Python reference and all GGML formats/backends")
    figure.tight_layout()
    figure.savefig(path, format="svg", bbox_inches="tight")
    plt.close(figure)


def write_caption_outputs(payload: dict, directory: Path) -> None:
    baseline = payload["reference_latency_ms"]
    rows = payload["measurements"]
    with (directory / "caption_matrix_6.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("backend", "dtype", "images", "pytorch_cuda_ms", "ggml_ms",
                         "ggml_over_pytorch", "exact_fraction", "word_f1"))
        for row in rows:
            writer.writerow((row["backend"], row["dtype"], len(payload["images"]), baseline,
                             row["latency_ms"], row["latency_ms"] / baseline,
                             row["exact_fraction"], row["word_f1"]))
    labels = [f"{row['backend'].upper()}\n{row['dtype'].upper()}" for row in rows]
    ratios = [row["latency_ms"] / baseline for row in rows]
    colors = ["#277f65" if ratio < 1 else "#c24c43" for ratio in ratios]
    figure, axis = plt.subplots(figsize=(8.5, 4.3))
    bars = axis.bar(range(len(rows)), ratios, color=colors, width=0.65)
    axis.axhline(1, color="#252525", linewidth=1, linestyle="--")
    axis.set_xticks(range(len(rows)), labels)
    axis.set_ylabel("GGML caption latency / PyTorch CUDA latency (lower is faster)")
    axis.set_ylim(0, max(1.3, max(ratios) * 1.13))
    axis.grid(axis="y", color="#d8d8d8", linewidth=0.6)
    axis.set_axisbelow(True)
    for bar, ratio in zip(bars, ratios):
        axis.text(bar.get_x() + bar.get_width() / 2, ratio + 0.015, f"{ratio:.2f}x",
                  ha="center", va="bottom", fontsize=9)
    figure.suptitle("Tag2Text caption | 50 GT images | batch 1 | warmup 1 + repeats 3")
    figure.tight_layout()
    figure.savefig(directory / "caption_latency.svg", format="svg", bbox_inches="tight")
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(8.5, 4.3))
    positions = list(range(len(rows)))
    axis.bar([x - 0.18 for x in positions], [row["exact_fraction"] for row in rows],
             width=0.36, color="#277f65", label="Exact caption")
    axis.bar([x + 0.18 for x in positions], [row["word_f1"] for row in rows],
             width=0.36, color="#3279ad", label="Word F1")
    axis.axhline(0.99, color="#252525", linewidth=1, linestyle="--", label="Acceptance 0.99")
    axis.set_xticks(positions, labels)
    axis.set_ylim(min(0.95, min(row["exact_fraction"] for row in rows) - 0.02), 1.01)
    axis.set_ylabel("Agreement with official PyTorch caption (higher is better)")
    axis.grid(axis="y", color="#d8d8d8", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(fontsize=8, loc="lower right")
    figure.suptitle("Tag2Text caption parity | all 50 GT images | six GPU combinations")
    figure.tight_layout()
    figure.savefig(directory / "caption_accuracy.svg", format="svg", bbox_inches="tight")
    plt.close(figure)


def write_markdown(rows: list[dict[str, str]], aggregated: list[dict[str, float | str | int]],
                   caption: dict,
                   args: argparse.Namespace, path: Path) -> None:
    failures = [row for row in rows if row["ggml_faster"] != "True"]
    overall_failures = [row for row in aggregated if float(row["ggml_over_pytorch"]) >= 1]
    lines = [
        "# PyTorch CUDA and GGML latency matrix", "",
        "Model downloads: [GitHub RAM release](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/RAM) "
        "and [Hugging Face RAM_GGUF](https://huggingface.co/Asher-1/RAM_GGUF/tree/main).", "",
        f"Measured on `{args.device}` from `{args.run_dir.name}`. The retained GT set has 50 images "
        "(OpenImages common 14, rare 12, ImageNet multi 12, HICO 12). "
        "All three model families, both GPU backends and F32/F16/Q8_0 are represented.", "",
        "Both engines keep the model resident. Timed work is image decode, resize/normalization, "
        "host-to-device transfer, tag inference, host readback and threshold selection. "
        "The official PyTorch CUDA baseline uses batch size 1; both engines use one first-image "
        "warmup and three measured passes per image. Numbers below are image-count-weighted "
        "arithmetic means across the four datasets. Model loading and JSONL serialization are excluded. "
        "GGML/PyTorch below 1.00 means GGML is faster. Each timed command passed the "
        "GPU compute-process audit; overlap attempts were discarded. Two explicitly allowed "
        "resident services and desktop graphics remained present. An idle parent CUDA context "
        "coordinates cooperative benchmarks across model-loading gaps. These timings describe "
        "this shared workstation, not an uncontended hardware ceiling. Model, executable and "
        "source hashes, protocol and audit counts are in [BENCHMARK_METADATA.json](BENCHMARK_METADATA.json).", "",
        "![GGML to PyTorch CUDA latency ratio](latency_matrix.svg)", "",
        "| Model | Backend | GGUF | PyTorch CUDA ms | GGML ms | GGML/PyTorch | Tag F1 | Minimum dataset F1 | Probability MAE | Slower datasets |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregated:
        lines.append(f"| {row['model']} | {row['backend']} | {row['dtype']} | "
                     f"{float(row['pytorch_ms']):.2f} | {float(row['ggml_ms']):.2f} | "
                     f"{float(row['ggml_over_pytorch']):.3f}x | {float(row['cross_f1']):.4f} | "
                     f"{float(row['minimum_dataset_f1']):.4f} | "
                     f"{float(row['probability_mae']):.5f} | {row['failed_datasets']}/4 |")
    lines += [
        "", "## Acceptance", "",
        f"- The 50-image aggregate speed target passes for {len(aggregated) - len(overall_failures)}/18 "
        f"model/backend/format combinations. The strict per-dataset speed target passes for "
        f"{len(rows) - len(failures)}/72 rows.",
        ("- All 18 aggregate combinations meet the speed target." if not overall_failures else
         "- The aggregate speed target is **not fully met**. The failed combinations are: "
         + ", ".join(f"`{row['model']} {row['backend']} {row['dtype']}` "
                     f"({float(row['ggml_over_pytorch']):.3f}x)" for row in overall_failures) + "."),
        f"- The per-dataset tag parity target (F1 >= 0.99) passes for "
        f"{sum(float(row['cross_f1']) >= 0.99 for row in rows)}/72 rows; "
        f"the minimum is {min(float(row['cross_f1']) for row in rows):.6f}. "
        "Each model/backend/format's minimum is shown alongside its weighted mean.",
        "- All 72 GT rows have `annotation_status=ok`. Tag F1 compares thresholded PyTorch and GGML "
        "sets; it is not a GT score. The per-dataset GT `mAP`, `CP`, `CR`, and every individual "
        "speed result are in [latency_matrix_72.csv](latency_matrix_72.csv).",
        "- RAM and RAM++ expose tags only. Tag2Text caption decoding is a separate pipeline; "
        "tag latency and F1 do not establish caption parity.", "",
        "![Minimum dataset tag parity](tag_accuracy.svg)", "",
        "![GT metrics: PyTorch versus six GGML variants](gt_metrics.svg)", "",
        "The GT chart covers RAM and RAM++. Its green range contains the six GGML "
        "backend/format values, with the official Python value shown as a cross. The CSV "
        "includes `pytorch_mAP/CP/CR` and GGML-minus-Python differences. Tag2Text's GT "
        "scores on rare/ImageNet/HICO are zero-target diagnostics because those datasets "
        "do not supply Tag2Text GT; they are excluded from this chart.", "",
        "The diagnostic report [ALIGNMENT_DIAGNOSIS.md](ALIGNMENT_DIAGNOSIS.md) explains "
        "the CUDA activation quantization error, selective RAM Q8 policy, FP32 accumulation, "
        "constant caching and Vulkan dispatch/copy reductions. CUDA tagging auto mode uses "
        "FP16 matrix inputs with FP32 accumulation/output, including a resident converted "
        "matrix cache for F32 GGUF weights. `--cuda-compute f32` selects FP32 inputs.", "",
        "## Tag2Text image-to-caption", "",
        f"Measured from `{args.caption_run.name}` on the same 50 images. Official PyTorch CUDA: "
        f"**{caption['reference_latency_ms']:.2f} ms/image**. Both engines keep weights resident, "
        "use batch size 1, one first-image warmup and three measured passes per image. "
        "Timing includes decode, preprocessing, transfer, tag and text generation, and output readback; "
        "model loading and JSONL serialization are excluded. Captions are compared byte-for-byte "
        "and with lexical word F1 against Python, not against human GT captions.", "",
        "![Tag2Text caption latency ratio](caption_latency.svg)", "",
        "![Tag2Text exact-caption and word F1 agreement](caption_accuracy.svg)", "",
        "| Backend | GGUF | PyTorch CUDA ms | GGML ms | GGML/PyTorch | Exact caption | Word F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in caption["measurements"]:
        lines.append(f"| {row['backend']} | {row['dtype']} | "
                     f"{caption['reference_latency_ms']:.2f} | {row['latency_ms']:.2f} | "
                     f"{row['latency_ms'] / caption['reference_latency_ms']:.3f}x | "
                     f"{row['exact_fraction']:.3f} | {row['word_f1']:.3f} |")
    faster_captions = sum(row["latency_ms"] < caption["reference_latency_ms"]
                          for row in caption["measurements"])
    exact_failures = sum(row["exact_fraction"] < 0.99 for row in caption["measurements"])
    exact_status = ("All six combinations meet the requested 0.99 exact-wording target "
                    "on these 50 frames. " if exact_failures == 0 else
                    f"Exact wording is below the requested 0.99 target for "
                    f"**{exact_failures}/6** combinations. ")
    lines += [
        "", f"Caption latency is lower for **{faster_captions}/6** GGML combinations. "
        + exact_status + "Lexical word F1 does not replace exact parity. "
        "The machine-readable table is [caption_matrix_6.csv](caption_matrix_6.csv). "
        "The ignored raw run keeps all 50 Python and GGML sentences for audit.",
        "", "The preview below shows a representative source image beside the Python "
        "sentence and all six GGML sentences. Generate all 50 full-resolution panels "
        "with `visualize_outputs.py`.", "",
        "![Source image and word-level caption comparison](caption_examples.webp)",
        "", "All 50 caption panels are indexed in [visualizations/README.md](visualizations/README.md). "
        "The 150 model/image tagging panels, including explicit extra/missing tags and GT, "
        "are indexed in [visualizations/tags/README.md](visualizations/tags/README.md). "
        "Raw panels remain local and ignored; the previews below are tracked.", "",
        "![Source image, GT and Python versus six GGML tag sets](tag_examples.webp)",
        "", "## Reproduce", "", "```bash",
        "python3 cpp_ggml/benchmarks/full_eval.py --scope datasets --run \\",
        "  --backends cuda vulkan --dtypes f32 f16 q8_0 \\",
        "  --batch-size 1 --warmup 1 --repeat 3 --threads 8 \\",
        "  --label-space-dir cpp_ggml/benchmarks/label_spaces --check-acceptance --wait-gpu-idle",
        "python3 cpp_ggml/benchmarks/caption_all.py --scope datasets \\",
        "  --backends cuda vulkan --dtypes f32 f16 q8_0 --warmup 1 --repeat 3 --check-acceptance --wait-gpu-idle",
        "python3 cpp_ggml/benchmarks/latency_matrix.py \\",
        "  --run-dir cpp_ggml/benchmarks/full_eval/run-YYYYMMDD-HHMMSS \\",
        "  --caption-run cpp_ggml/benchmarks/full_eval/caption-YYYYMMDD-HHMMSS --require-acceptance",
        "python3 cpp_ggml/benchmarks/visualize_outputs.py \\",
        "  --caption-run cpp_ggml/benchmarks/full_eval/caption-YYYYMMDD-HHMMSS \\",
        "  --preview-output cpp_ggml/benchmarks/caption_examples.webp",
        "python3 cpp_ggml/benchmarks/visualize_tags.py \\",
        "  --run-dir cpp_ggml/benchmarks/full_eval/run-YYYYMMDD-HHMMSS \\",
        "  --preview-output cpp_ggml/benchmarks/tag_examples.webp",
        "```", "",
        "Every per-dataset result remains visible in the CSV. "
        "The speed result is hardware- and driver-specific and does not prove a universal advantage. "
        "On a shared GPU add `--wait-gpu-idle` and repeat `--allow-gpu-pid PID` only for "
        "explicitly allowed long-lived services. Every monitored command records process samples; "
        "the tagging harness retries commands with detected transient overlap.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows = load_run(args.run_dir)
    add_gt_reference(rows, args.run_dir)
    caption = load_caption(args.caption_run)
    aggregated = aggregate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.require_acceptance:
        write_metadata(args)
    write_csv(rows, args.output_dir / "latency_matrix_72.csv")
    write_chart(aggregated, args.output_dir / "latency_matrix.svg")
    write_accuracy_chart(aggregated, args.output_dir / "tag_accuracy.svg")
    write_gt_chart(rows, args.output_dir / "gt_metrics.svg")
    write_caption_outputs(caption, args.output_dir)
    write_markdown(rows, aggregated, caption, args, args.output_dir / "LATENCY_MATRIX.md")
    print(f"published {len(aggregated)} tag aggregates, {len(rows)} dataset rows and "
          f"{len(caption['measurements'])} caption rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

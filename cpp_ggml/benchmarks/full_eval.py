#!/usr/bin/env python3
"""Run and report the reproducible PyTorch versus GGML evaluation matrix.

The evaluator keeps the official image order, writes one manifest per data
source, and stores raw per-image JSONL plus float32 matrices under a run
directory.  Use ``--scope datasets --run`` for the retained GT evaluation
subset, or ``--scope images --run`` for the repository smoke set.
Large runs can be split with ``--shard-count/--shard-index`` and resumed with
``--skip-existing``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "cpp_ggml/benchmarks"
sys.path.insert(0, str(ROOT / "cpp_ggml/scripts"))
from model_manifest import MODEL_CHECKPOINTS, gguf_name  # noqa: E402

DATASET_INFO = {
    "openimages_common_214": {
        "image_root": ROOT / "datasets/openimages_common_214/imgs/test",
        "ram_ann": ROOT / "datasets/openimages_common_214/openimages_common_214_ram_annots.txt",
        "tag_ann": ROOT / "datasets/openimages_common_214/openimages_common_214_tag2text_idannots.txt",
        "ram_tags": ROOT / "datasets/openimages_common_214/openimages_common_214_ram_taglist.txt",
        "tag_tags": ROOT / "ram/data/tag2text_ori_tag_list.txt",
    },
    "openimages_rare_200": {
        "image_root": ROOT / "datasets/openimages_rare_200/imgs/test",
        "ram_ann": ROOT / "datasets/openimages_rare_200/openimages_rare_200_ram_annots.txt",
        "ram_tags": ROOT / "datasets/openimages_rare_200/openimages_rare_200_ram_taglist.txt",
    },
    "imagenet_multi": {
        "image_root": ROOT / "datasets/imagenet_multi/imgs",
        "ram_ann": ROOT / "datasets/imagenet_multi/imagenet_multi_1000_annots.txt",
        "ram_tags": ROOT / "datasets/imagenet_multi/imagenet_multi_1000_taglist.txt",
        "index_labels": True,
    },
    "hico": {
        "image_root": ROOT / "datasets/hico/imgs",
        "ram_ann": ROOT / "datasets/hico/hico_600_annots.txt",
        "ram_tags": ROOT / "datasets/hico/hico_600_taglist.txt",
        "index_labels": True,
    },
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".jpe", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("images", "datasets"), default="images")
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASET_INFO), default=tuple(DATASET_INFO))
    parser.add_argument("--models", nargs="+", choices=("ram", "ram_plus", "tag2text"),
                        default=("ram", "ram_plus", "tag2text"))
    parser.add_argument("--backends", nargs="+", choices=("cuda", "vulkan", "cpu"),
                        default=("cuda", "vulkan"))
    parser.add_argument("--dtypes", nargs="+", choices=("f32", "f16", "q8_0"),
                        default=("f32", "f16", "q8_0"))
    parser.add_argument("--output-dir", type=Path, default=BENCH / "full_eval")
    parser.add_argument("--gguf-dir", type=Path, default=ROOT / "cpp_ggml/models/gguf")
    parser.add_argument("--cuda-compute", choices=("auto", "f32", "f16"), default="auto")
    parser.add_argument("--check-acceptance", action="store_true",
                        help="exit with failure when any requested dataset row misses a target")
    parser.add_argument("--min-cross-f1", type=float, default=0.99)
    parser.add_argument("--max-latency-ratio", type=float, default=1.0)
    parser.add_argument("--label-space-dir", type=Path, default=BENCH / "label_spaces",
                        help="optional sidecars: DATASET__MODEL.{tags,thresholds,f32}")
    parser.add_argument("--run", action="store_true", help="run engines after preparing manifests")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--resume-run", type=Path,
                        help="resume a previous run directory, reusing only complete audited outputs")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--python-device", choices=("cuda", "cpu"), default="cuda",
                        help="PyTorch reference device; CUDA is the required comparison baseline")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--wait-gpu-idle", action="store_true",
                        help="wait for transient CUDA jobs and audit every timed command")
    parser.add_argument("--allow-gpu-pid", action="append", type=int, default=[],
                        help="long-lived GPU PID allowed by the timing audit; repeat as needed")
    parser.add_argument("--gpu-retries", type=int, default=3,
                        help="maximum extra attempts after detected GPU overlap")
    parser.add_argument("--wait-process-pid", action="append", type=int, default=[],
                        help="wait for this benchmark coordinator PID to exit, including GPU-idle gaps")
    return parser.parse_args()


def read_annotation(path: Path) -> list[tuple[str, list[str]]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split(",")
        if fields and fields[0]:
            rows.append((fields[0], [item for item in fields[1:] if item]))
    return rows


def decode_labels(fields: list[str], names: list[str], index_labels: bool) -> list[str]:
    """Decode annotation fields into the names used by the dataset tag list.

    OpenImages annotations already contain names. ImageNet and HICO store one
    or more whitespace-separated integer class IDs after the image name.
    Keeping this conversion in the manifest builder makes GT scoring and the
    Python/GGML label-space sidecars use the same canonical names.
    """
    if not index_labels:
        return fields
    decoded = []
    for field in fields:
        for value in field.split():
            try:
                index = int(value)
            except ValueError:
                decoded.append(value)
                continue
            if not 0 <= index < len(names):
                raise ValueError(f"annotation class index {index} is outside tag list of {len(names)}")
            decoded.append(names[index])
    return decoded


def image_path(info: dict[str, Any], identifier: str) -> Path:
    root = info["image_root"]
    if root.name == "test" and identifier.startswith("test/"):
        identifier = identifier.split("/", 1)[1]
    path = root / identifier
    if root.name == "test" and path.suffix == "":
        path = path.with_suffix(".jpg")
    return path


def make_records(scope: str, datasets: list[str]) -> dict[str, list[dict[str, Any]]]:
    if scope == "images":
        paths = sorted(path.resolve() for path in (ROOT / "images").rglob("*")
                       if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        return {"repository_images": [{"image": str(path), "labels": [], "dataset": "repository_images"}
                                      for path in paths]}
    result = {}
    for dataset in datasets:
        info = DATASET_INFO[dataset]
        ram_tags = info["ram_tags"].read_text(encoding="utf-8").splitlines()
        ram_by_index = {str(index): name for index, name in enumerate(ram_tags)}
        rows = read_annotation(info["ram_ann"])
        tag_rows = read_annotation(info["tag_ann"]) if info.get("tag_ann") else []
        tag_names = info["tag_tags"].read_text(encoding="utf-8").splitlines() if info.get("tag_tags") else []
        tag_by_index = {str(index): name for index, name in enumerate(tag_names)}
        records = []
        for index, (identifier, labels) in enumerate(rows):
            path = image_path(info, identifier).resolve()
            labels = decode_labels(labels, ram_tags, bool(info.get("index_labels")))
            tag_labels = []
            if tag_rows and index < len(tag_rows):
                tag_labels = decode_labels(tag_rows[index][1], tag_names, True)
            records.append({
                "image": str(path),
                "dataset": dataset,
                "labels_ram": labels,
                "labels_tag2text": tag_labels,
                "labels": labels,
                "exists": path.is_file(),
                "identifier": identifier,
            })
        result[dataset] = records
    return result


def select_shard(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("invalid shard index/count")
    selected = records[args.shard_index::args.shard_count]
    return selected[:args.limit] if args.limit else selected


def write_manifest(output: Path, records: list[dict[str, Any]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(record["image"] for record in records) + "\n", encoding="utf-8")
    metadata = output.with_suffix(".records.jsonl")
    with metadata.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def command_text(command: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([item]) for item in command)


def run_command(command: list[str], log_path: Path) -> dict[str, Any]:
    begin = time.perf_counter()
    process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    elapsed = time.perf_counter() - begin
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(process.stdout + "\n--- stderr ---\n" + process.stderr, encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"command failed ({process.returncode}): {command_text(command)}\n{process.stderr[-2000:]}")
    return {"wall_seconds": elapsed, "stdout": process.stdout[-1000:]}


def gpu_process_ids() -> set[int]:
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                             "--format=csv,noheader"], capture_output=True,
                            text=True, check=True)
    ids = set()
    for line in result.stdout.splitlines():
        try:
            ids.add(int(line.strip()))
        except ValueError:
            pass
    return ids


def reserve_gpu_slot(allowed_gpu_pids: set[int], wait_process_pids: list[int] = ()) -> Any:
    """Keep a CUDA context visible to cooperative idle-waiting benchmarks."""
    while True:
        pending = [pid for pid in wait_process_pids if Path(f"/proc/{pid}").exists()]
        busy = gpu_process_ids() - allowed_gpu_pids
        if not pending and not busy:
            break
        print(f"waiting to reserve GPU timing slot: coordinators={pending}, "
              f"compute={sorted(busy)}", flush=True)
        time.sleep(5)
    import torch
    allocation = torch.empty(1, device="cuda")
    torch.cuda.synchronize()
    allowed_gpu_pids.add(os.getpid())
    print(f"GPU timing slot reserved by idle CUDA context PID {os.getpid()}", flush=True)
    return allocation


def run_command_monitored(command: list[str], log_path: Path,
                          allowed_gpu_pids: set[int]) -> dict[str, Any]:
    """Run one timed command while recording transient compute processes."""
    while True:
        busy = gpu_process_ids() - allowed_gpu_pids
        if not busy:
            break
        print(f"waiting for transient GPU processes: {sorted(busy)}", flush=True)
        time.sleep(5)
    begin = time.perf_counter()
    samples: list[dict[str, Any]] = []
    monitor_errors: list[str] = []
    stop = threading.Event()
    process = subprocess.Popen(command, cwd=ROOT, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    def monitor() -> None:
        while not stop.is_set():
            try:
                samples.append({"seconds": time.perf_counter() - begin,
                                "pids": sorted(gpu_process_ids())})
            except (OSError, subprocess.SubprocessError) as error:
                monitor_errors.append(str(error))
            stop.wait(0.5)
    worker = threading.Thread(target=monitor, daemon=True)
    worker.start()
    try:
        stdout, stderr = process.communicate()
    except BaseException:
        process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
        raise
    finally:
        stop.set()
        worker.join()
    transient = sorted({pid for sample in samples for pid in sample["pids"]
                        if pid not in allowed_gpu_pids and pid != process.pid})
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(stdout + "\n--- stderr ---\n" + stderr +
                        "\n--- gpu timing audit ---\n" + json.dumps(
                            {"allowed_gpu_pids": sorted(allowed_gpu_pids),
                             "transient_gpu_pids": transient, "samples": samples},
                            indent=2) + "\n", encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"command failed ({process.returncode}): {command_text(command)}\n{stderr[-2000:]}")
    if monitor_errors or not samples:
        raise RuntimeError(f"GPU timing audit failed: {monitor_errors}")
    return {"wall_seconds": time.perf_counter() - begin,
            "transient_gpu_pids": transient, "stdout": stdout[-1000:]}


def output_stem(scope: str, dataset: str, model: str, engine: str, backend: str = "", dtype: str = "") -> str:
    parts = [scope, dataset, model, engine]
    if backend:
        parts.extend([backend, dtype])
    return "__".join(parts)


def label_space(args: argparse.Namespace, dataset: str, model: str) -> dict[str, Path]:
    prefix = args.label_space_dir / f"{dataset}__{model}"
    return {
        "tags": prefix.with_suffix(".tags.txt"),
        "thresholds": prefix.with_suffix(".thresholds.txt"),
        "embedding": prefix.with_suffix(".f32"),
    }


def run_engines(args: argparse.Namespace, scope: str, dataset: str, records: list[dict[str, Any]],
                manifest: Path, run_dir: Path) -> None:
    def complete(json_path: Path, matrix_path: Path, log_path: Path) -> bool:
        if not args.skip_existing or not json_path.is_file() or not matrix_path.is_file():
            return False
        if args.wait_gpu_idle:
            if not log_path.is_file():
                return False
            text = log_path.read_text(encoding="utf-8")
            if "--- gpu timing audit ---" not in text:
                return False
            audit = json.loads(text.split("--- gpu timing audit ---")[-1])
            if audit["transient_gpu_pids"] or not audit["samples"]:
                return False
        return True
    def runner(command: list[str], log_path: Path) -> dict[str, Any]:
        while True:
            pending = [pid for pid in args.wait_process_pid if Path(f"/proc/{pid}").exists()]
            if not pending:
                break
            print(f"waiting for benchmark coordinator processes: {pending}", flush=True)
            time.sleep(5)
        if not args.wait_gpu_idle:
            return run_command(command, log_path)
        for attempt in range(args.gpu_retries + 1):
            result = run_command_monitored(command, log_path, set(args.allow_gpu_pid))
            if not result["transient_gpu_pids"]:
                return result
            contaminated_log = log_path.with_name(
                f"{log_path.stem}.overlap-{time.time_ns()}-{attempt + 1}.log")
            log_path.rename(contaminated_log)
            print(f"discarding timings with transient GPU processes: "
                  f"{result['transient_gpu_pids']}; evidence: {contaminated_log}", flush=True)
        raise RuntimeError(f"GPU overlap persisted after {args.gpu_retries + 1} attempts")
    for model in args.models:
        python_stem = output_stem(scope, dataset, model, "pytorch")
        python_json = run_dir / f"{python_stem}.jsonl"
        python_matrix = run_dir / f"{python_stem}.f32"
        space = label_space(args, dataset, model)
        space_args = []
        if space["tags"].is_file():
            space_args += ["--tag-list", str(space["tags"])]
        if space["thresholds"].is_file():
            space_args += ["--thresholds", str(space["thresholds"])]
        if space["embedding"].is_file():
            space_args += ["--label-embedding", str(space["embedding"])]
        if not complete(python_json, python_matrix, run_dir / "logs" / f"{python_stem}.log"):
            runner([
                sys.executable, str(BENCH / "python_eval.py"), "--model", model,
                "--manifest", str(manifest), "--output-jsonl", str(python_json),
                "--output-matrix", str(python_matrix), "--batch-size", str(args.batch_size),
                "--device", args.python_device, "--warmup", str(args.warmup),
                "--repeat", str(args.repeat),
            ] + space_args, run_dir / "logs" / f"{python_stem}.log")
        for backend in args.backends:
            binary = ROOT / f"cpp_ggml/build-{backend}/ram-ggml"
            if not binary.is_file():
                raise FileNotFoundError(f"missing {backend} build: {binary}")
            for dtype in args.dtypes:
                stem = output_stem(scope, dataset, model, "ggml", backend, dtype)
                json_path = run_dir / f"{stem}.jsonl"
                matrix_path = run_dir / f"{stem}.f32"
                if complete(json_path, matrix_path, run_dir / "logs" / f"{stem}.log"):
                    continue
                model_path = args.gguf_dir / gguf_name(model, dtype)
                runner([
                    str(binary), "--model", str(model_path), "--backend", backend,
                    "--manifest", str(manifest), "--jsonl", str(json_path),
                    "--logits-matrix", str(matrix_path), "--threads", str(args.threads),
                    "--warmup", str(args.warmup), "--repeat", str(args.repeat),
                    "--cuda-compute", args.cuda_compute,
                ] + space_args, run_dir / "logs" / f"{stem}.log")


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_matrix(path: Path, rows: int, classes: int, raw_logits: bool) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    expected = rows * classes
    if values.size != expected:
        raise ValueError(f"{path} has {values.size} floats, expected {expected}")
    values = values.reshape(rows, classes)
    return 1.0 / (1.0 + np.exp(-values)) if raw_logits else values


def set_metrics(expected: list[set[str]], predicted: list[set[str]]) -> dict[str, float]:
    tp = fp = fn = 0
    precisions = []
    recalls = []
    for truth, actual in zip(expected, predicted):
        hit = len(truth & actual)
        tp += hit
        fp += len(actual - truth)
        fn += len(truth - actual)
        precisions.append(hit / len(actual) if actual else (1.0 if not truth else 0.0))
        recalls.append(hit / len(truth) if truth else 1.0)
    return {
        "precision": tp / (tp + fp) if tp + fp else 1.0,
        "recall": tp / (tp + fn) if tp + fn else 1.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
    }


def average_precision(scores: np.ndarray, targets: np.ndarray) -> float:
    order = np.argsort(scores)[::-1]
    positives = targets[order].astype(np.float64)
    count = positives.sum()
    if count == 0:
        return 0.0
    precision = np.cumsum(positives) / np.arange(1, len(positives) + 1)
    return float((precision * positives).sum() / (count + 1e-8))


def annotation_metrics(records: list[dict[str, Any]], model: str, scores: np.ndarray,
                       rows: list[dict[str, Any]], dataset: str,
                       args: argparse.Namespace) -> dict[str, Any]:
    if dataset == "repository_images":
        return {"annotation_status": "not_applicable", "dataset": dataset}
    labels_key = "labels_tag2text" if model == "tag2text" else "labels_ram"
    names = [str(item["name"]) for item in rows[0].get("tags", [])] if rows else []
    space = label_space(args, dataset, model)
    if space["tags"].is_file():
        model_names = space["tags"].read_text(encoding="utf-8").splitlines()
    elif model == "tag2text":
        model_names = (ROOT / "ram/data/tag2text_ori_tag_list.txt").read_text(encoding="utf-8").splitlines()
    else:
        model_names = (ROOT / "ram/data/ram_tag_list.txt").read_text(encoding="utf-8").splitlines()
    if scores.shape[1] != len(model_names):
        return {"annotation_status": "class_count_mismatch", "dataset": dataset}
    expected = [set(record.get(labels_key, [])) for record in records]
    missing = sorted({label for labels in expected for label in labels if label not in model_names})
    if missing:
        return {"annotation_status": "unsupported_label_space", "missing_labels": len(missing),
                "dataset": dataset}
    targets = np.zeros_like(scores, dtype=np.float32)
    index: dict[str, list[int]] = {}
    for position, name in enumerate(model_names):
        index.setdefault(name, []).append(position)
    for row, labels in enumerate(expected):
        for label in labels:
            targets[row, index[label]] = 1.0
    aps = [average_precision(scores[:, column], targets[:, column]) for column in range(scores.shape[1])]
    predicted = [{str(tag["name"]) for tag in row.get("tags", [])} for row in rows]
    prediction_matrix = np.zeros_like(targets, dtype=bool)
    for row, labels in enumerate(predicted):
        for label in labels:
            prediction_matrix[row, index[label]] = True
    truth = targets.astype(bool)
    tp = (prediction_matrix & truth).sum(axis=0)
    fp = (prediction_matrix & ~truth).sum(axis=0)
    fn = (~prediction_matrix & truth).sum(axis=0)
    cp = float(np.mean(tp / (tp + fp + 1e-9)))
    cr = float(np.mean(tp / (tp + fn + 1e-9)))
    metrics = set_metrics(expected, predicted)
    metrics.update({"annotation_status": "ok", "dataset": dataset, "mAP": float(np.mean(aps),),
                    "CP": cp, "CR": cr})
    return metrics


def compare_outputs(records: list[dict[str, Any]], model: str, run_dir: Path,
                    scope: str, dataset: str, backends: list[str], dtypes: list[str],
                    args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    py_stem = output_stem(scope, dataset, model, "pytorch")
    py_json = run_dir / f"{py_stem}.jsonl"
    if not py_json.is_file():
        return rows
    py_rows = load_rows(py_json)
    classes = len(py_rows[0]["tags"]) if False else len(np.fromfile(run_dir / f"{py_stem}.f32", dtype=np.float32)) // len(py_rows)
    py_scores = load_matrix(run_dir / f"{py_stem}.f32", len(py_rows), classes, False)
    py_tags = [{tag["name"] for tag in row.get("tags", [])} for row in py_rows]
    pytorch_latency_ms = float(np.mean([float(item["latency_ms"]) for item in py_rows]))
    pytorch_batch_size = int(py_rows[0].get("batch_size", 1))
    for backend in backends:
        for dtype in dtypes:
            stem = output_stem(scope, dataset, model, "ggml", backend, dtype)
            json_path = run_dir / f"{stem}.jsonl"
            matrix_path = run_dir / f"{stem}.f32"
            if not json_path.is_file() or not matrix_path.is_file():
                continue
            cpp_rows = load_rows(json_path)
            if len(cpp_rows) != len(records):
                raise ValueError(f"row count mismatch: {json_path}")
            cpp_scores = load_matrix(matrix_path, len(cpp_rows), classes, True)
            cpp_tags = [{tag["name"] for tag in row.get("tags", [])} for row in cpp_rows]
            cross = set_metrics(py_tags, cpp_tags)
            annotation = annotation_metrics(records, model, cpp_scores, cpp_rows, dataset, args)
            ggml_latency_ms = float(np.mean([float(item["latency_ms"]) for item in cpp_rows]))
            latency_ratio = ggml_latency_ms / max(pytorch_latency_ms, 1e-9)
            row = {
                "scope": scope, "dataset": dataset, "model": model, "engine": "ggml",
                "backend": backend, "dtype": dtype, "images": len(records),
                "images_per_second": 1000.0 / max(ggml_latency_ms, 1e-9),
                "ggml_latency_ms": ggml_latency_ms,
                "pytorch_latency_ms": pytorch_latency_ms,
                "pytorch_images_per_second": 1000.0 / max(pytorch_latency_ms, 1e-9),
                "pytorch_batch_size": pytorch_batch_size,
                "latency_ratio_ggml_over_pytorch": latency_ratio,
                "ggml_faster": latency_ratio < 1.0,
                "pytorch_cross_mae": float(np.abs(py_scores - cpp_scores).mean()),
                "cross_f1": cross["f1"], "cross_precision": cross["precision"],
                "cross_recall": cross["recall"], **annotation,
            }
            rows.append(row)
    return rows


def write_plots(rows: list[dict[str, Any]], records: list[dict[str, Any]], run_dir: Path) -> None:
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{row['model']}\n{row['backend']}/{row['dtype']}" for row in rows]
    speeds = [row["images_per_second"] for row in rows]
    f1 = [row["cross_f1"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(max(12, len(rows) * 0.55), 5))
    axes[0].bar(range(len(rows)), speeds, color="#1f77b4")
    axes[0].set_title("GGML throughput")
    axes[0].set_ylabel("images/s")
    axes[1].bar(range(len(rows)), f1, color="#2ca02c")
    axes[1].set_title("Tag agreement with PyTorch")
    axes[1].set_ylabel("F1")
    for axis in axes:
        axis.set_xticks(range(len(rows)), labels, rotation=75, ha="right", fontsize=7)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(run_dir / "overview.png", dpi=160)
    plt.close(figure)

    scored = [row for row in rows if row.get("annotation_status") == "ok"]
    if scored:
        labels = [f"{row['model']}\n{row['backend']}/{row['dtype']}" for row in scored]
        figure, axes = plt.subplots(1, 3, figsize=(max(12, len(scored) * 0.55), 5))
        for axis, key in zip(axes, ("mAP", "CP", "CR")):
            axis.bar(range(len(scored)), [row[key] for row in scored], color="#ff7f0e")
            axis.set_title(key)
            axis.set_ylim(0, 1)
            axis.set_xticks(range(len(scored)), labels, rotation=75, ha="right", fontsize=7)
            axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(run_dir / "annotation_metrics.png", dpi=160)
        plt.close(figure)


def write_report(args: argparse.Namespace, inventory: dict[str, Any], rows: list[dict[str, Any]], run_dir: Path) -> None:
    report = run_dir / "REPORT.md"
    with report.open("w", encoding="utf-8") as stream:
        stream.write("# Full GGML evaluation report\n\n")
        stream.write("This report is generated by `cpp_ggml/benchmarks/full_eval.py`. ")
        stream.write("Raw JSONL and float32 matrices remain beside this file for auditability.\n\n")
        stream.write("## Coverage\n\n")
        stream.write("| Source | Images present | Annotation rows | Status |\n|---|---:|---:|---|\n")
        for name, info in inventory.items():
            stream.write(f"| {name} | {info['images_present']} | {info['annotation_rows']} | {info['status']} |\n")
        stream.write("\n## Results\n\n")
        if rows:
            columns = ["dataset", "model", "backend", "dtype", "images", "ggml_latency_ms",
                       "pytorch_latency_ms", "pytorch_batch_size", "latency_ratio_ggml_over_pytorch",
                       "ggml_faster", "images_per_second", "pytorch_images_per_second",
                       "pytorch_cross_mae", "cross_f1", "annotation_status", "mAP", "CP", "CR"]
            stream.write("| " + " | ".join(columns) + " |\n|" + "|".join("---" for _ in columns) + "|\n")
            for row in rows:
                stream.write("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |\n")
        else:
            stream.write("No engine results were requested. Use `--run`.\n")
        stream.write("\n## Metric boundaries\n\n")
        stream.write("`mAP`, `CP`, and `CR` follow the official RAM `batch_inference.py` implementation. ")
        stream.write("Cross-engine F1 compares thresholded tag sets and is not a ground-truth score. ")
        stream.write("A dataset is scored only when its exact tag list is available in GGUF metadata ")
        stream.write("or in a generated label-space sidecar under `--label-space-dir`; otherwise it is ")
        stream.write("marked `unsupported_label_space` while raw predictions remain available.\n")
        stream.write("\n## Reproduction\n\n```bash\n")
        stream.write(f"python3 cpp_ggml/benchmarks/full_eval.py --scope {args.scope} --run \\\n")
        stream.write("  --output-dir cpp_ggml/benchmarks/full_eval\n```\n")


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.threads <= 0 or args.warmup < 0 or args.repeat <= 0:
        raise SystemExit("invalid batch/thread/timing arguments")
    if not 0 <= args.min_cross_f1 <= 1 or args.max_latency_ratio <= 0:
        raise SystemExit("invalid accuracy or latency acceptance target")
    if args.gpu_retries < 0 or any(pid <= 0 for pid in args.allow_gpu_pid + args.wait_process_pid):
        raise SystemExit("invalid GPU retry count or allowed PID")
    if args.wait_gpu_idle and args.python_device != "cuda":
        raise SystemExit("--wait-gpu-idle requires --python-device cuda")
    gpu_reservation = None
    if args.run and args.wait_gpu_idle:
        allowed_gpu_pids = set(args.allow_gpu_pid)
        gpu_reservation = reserve_gpu_slot(allowed_gpu_pids, args.wait_process_pid)
        args.allow_gpu_pid = sorted(allowed_gpu_pids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume_run:
        run_dir = args.resume_run.resolve()
        if not run_dir.is_dir():
            raise SystemExit(f"missing run directory: {run_dir}")
        args.skip_existing = True
    else:
        run_dir = args.output_dir / ("run-" + time.strftime("%Y%m%d-%H%M%S"))
        run_dir.mkdir(parents=True)
    records_by_dataset = make_records(args.scope, args.datasets)
    inventory = {}
    all_rows = []
    for dataset, all_records in records_by_dataset.items():
        records = select_shard(all_records, args)
        manifest = run_dir / "manifests" / f"{dataset}.txt"
        write_manifest(manifest, records)
        inventory[dataset] = {
            "images_present": sum(1 for record in records if record.get("exists", Path(record["image"]).is_file())),
            "annotation_rows": len(all_records),
            "selected_rows": len(records),
            "status": "ready" if all(record.get("exists", True) for record in records) else "missing_images",
        }
        if args.run:
            run_engines(args, args.scope, dataset, records, manifest, run_dir)
            for model in args.models:
                all_rows.extend(compare_outputs(records, model, run_dir, args.scope, dataset,
                                                args.backends, args.dtypes, args))
    (run_dir / "inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    if all_rows:
        with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            columns = sorted({key for row in all_rows for key in row})
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(all_rows)
    write_plots(all_rows, next(iter(records_by_dataset.values()), []), run_dir)
    write_report(args, inventory, all_rows, run_dir)
    if args.check_acceptance:
        expected_rows = len(inventory) * len(args.models) * len(args.backends) * len(args.dtypes)
        failures = [{"dataset": row["dataset"], "model": row["model"],
                     "backend": row["backend"], "dtype": row["dtype"],
                     "cross_f1": row["cross_f1"],
                     "latency_ratio": row["latency_ratio_ggml_over_pytorch"]}
                    for row in all_rows
                    if row["cross_f1"] < args.min_cross_f1 or
                    row["latency_ratio_ggml_over_pytorch"] >= args.max_latency_ratio or
                    row["annotation_status"] not in ("ok", "not_applicable")]
        acceptance = {"passed": len(all_rows) == expected_rows and not failures,
                      "expected_rows": expected_rows, "measured_rows": len(all_rows),
                      "min_cross_f1": args.min_cross_f1,
                      "max_latency_ratio": args.max_latency_ratio, "failures": failures}
        (run_dir / "acceptance.json").write_text(json.dumps(acceptance, indent=2) + "\n")
        print(json.dumps({"acceptance": acceptance}, indent=2))
        if not acceptance["passed"]:
            return 1
    print(json.dumps({"run_dir": str(run_dir), "datasets": inventory, "results": len(all_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

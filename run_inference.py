#!/usr/bin/env python3
"""Run an official Python model or the local C++ GGML executable."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "cpp_ggml/scripts"))
from model_manifest import MODEL_CHECKPOINTS, gguf_name


MODELS = {
    "ram": "inference_ram.py",
    "ram_plus": "inference_ram_plus.py",
    "tag2text": "inference_tag2text.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("cpp", "python"), default="cpp")
    parser.add_argument("--model", choices=MODELS, default="ram")
    parser.add_argument("--task", choices=("tags", "caption"), default="tags")
    parser.add_argument("--image", type=Path, default=ROOT / "images/demo/demo1.jpg")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--backend", choices=("cpu", "cuda", "vulkan"), default="cuda")
    parser.add_argument("--cuda-compute", choices=("auto", "f32", "f16"), default="auto")
    parser.add_argument("--dtype", choices=("f32", "f16", "q8_0"), default="f16")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--build-jobs", type=int,
                        default=min(8, os.cpu_count() or 1),
                        help="parallel C++ build jobs (separate from inference threads)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=30)
    parser.add_argument("--beam-batch", type=int, choices=(1, 3))
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--profile-backends", action="store_true")
    parser.add_argument("--pipeline", action="store_true",
                        help="include image decoding and preprocessing in C++ timing")
    parser.add_argument("--specified-tags", help="comma-separated Tag2Text caption tags")
    parser.add_argument("--tag-list", type=Path, help="label-space tag list sidecar")
    parser.add_argument("--vocab", type=Path, help="BERT vocabulary sidecar for C++ caption inference")
    parser.add_argument("--thresholds", type=Path, help="label-space threshold sidecar")
    parser.add_argument("--label-embedding", type=Path, help="raw float32 label embedding sidecar")
    parser.add_argument("--threshold", type=float, default=-1.0,
                        help="override every label threshold when non-negative")
    parser.add_argument("--threshold-epsilon", type=float, default=0.0,
                        help="C++ boundary tolerance added to label thresholds")
    parser.add_argument("--python-executable", default=sys.executable,
                        help="Python interpreter for the official Python path")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the resolved command without executing it")
    args, extra = parser.parse_known_args()
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if args.build_jobs <= 0:
        parser.error("--build-jobs must be positive")
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("--warmup must be non-negative and --repeat must be positive")
    if args.max_length <= 0:
        parser.error("--max-length must be positive")
    if args.threshold_epsilon < 0:
        parser.error("--threshold-epsilon must be non-negative")
    for name in ("checkpoint", "gguf", "build_dir", "dump_dir", "tag_list", "thresholds",
                 "label_embedding", "vocab"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    image = args.image.resolve()
    if not image.is_file():
        parser.error(f"image does not exist: {image}")

    if args.engine == "python":
        if args.max_length != 30:
            parser.error("the official Python launcher uses a fixed caption max length of 30")
        if args.beam_batch is not None:
            parser.error("--beam-batch applies only to the C++ caption decoder")
        if args.backend != "cuda":
            parser.error("--backend is a C++ GGML backend option; official Python inference uses CUDA when available")
        ignored = []
        if args.gguf:
            ignored.append("--gguf")
        if args.threads != 8:
            ignored.append("--threads")
        if args.build_jobs != min(8, os.cpu_count() or 1):
            ignored.append("--build-jobs")
        if args.cuda_compute != "auto":
            ignored.append("--cuda-compute")
        if args.dtype != "f16":
            ignored.append("--dtype")
        if args.warmup != 3:
            ignored.append("--warmup")
        if args.repeat != 3:
            ignored.append("--repeat")
        if args.pipeline:
            ignored.append("--pipeline")
        if args.dump_dir:
            ignored.append("--dump-dir")
        if args.profile_backends:
            ignored.append("--profile-backends")
        if args.tag_list or args.thresholds or args.label_embedding or args.vocab or args.threshold >= 0 or args.threshold_epsilon:
            ignored.append("label-space/threshold options")
        if ignored:
            parser.error("official Python inference does not support: " + ", ".join(ignored))
        if args.build_dir:
            parser.error("--build-dir applies only to the C++ engine")
        if args.task == "caption" and args.model != "tag2text":
            parser.error("--task caption is only available for the tag2text model")
        script = ("cpp_ggml/scripts/inference_python_tags.py"
                  if args.model == "tag2text" and args.task == "tags" else MODELS[args.model])
        checkpoint = args.checkpoint or ROOT / "cpp_ggml/models/pytorch" / MODEL_CHECKPOINTS[args.model]
        if not checkpoint.is_file():
            parser.error(f"checkpoint does not exist: {checkpoint}")
        command = [args.python_executable, str(ROOT / script), "--image", str(image),
                   "--pretrained", str(checkpoint), *extra]
        if args.specified_tags:
            if args.model != "tag2text" or args.task != "caption":
                parser.error("--specified-tags requires Tag2Text caption")
            command.extend(("--specified-tags", args.specified_tags))
    else:
        if args.task == "caption" and args.model != "tag2text":
            parser.error("--task caption is only available for the tag2text model")
        if extra:
            parser.error(f"unknown C++ arguments: {' '.join(extra)}")
        suffix = "q8_0" if args.dtype == "q8_0" else args.dtype
        gguf = args.gguf or ROOT / "cpp_ggml/models/gguf" / gguf_name(args.model, suffix)
        if not gguf.is_file():
            checkpoint = args.checkpoint or ROOT / "cpp_ggml/models/pytorch" / MODEL_CHECKPOINTS[args.model]
            if not checkpoint.is_file():
                parser.error(f"checkpoint does not exist: {checkpoint}")
            if not args.dry_run:
                subprocess.run([args.python_executable, str(ROOT / "cpp_ggml/scripts/export_gguf.py"),
                            "--model", args.model, "--dtype", args.dtype,
                            "--checkpoint", str(checkpoint), "--output", str(gguf)], check=True)
        build = args.build_dir or ROOT / "cpp_ggml" / f"build-{args.backend}"
        binary = build / "ram-ggml"
        if not (ROOT / "cpp_ggml/third_party/ggml/CMakeLists.txt").is_file():
            subprocess.run(["git", "submodule", "update", "--init",
                            "cpp_ggml/third_party/ggml"], cwd=ROOT, check=True)
        cmake = ["cmake", "-S", str(ROOT / "cpp_ggml"), "-B", str(build),
                 f"-DRAM_GGML_CUDA={'ON' if args.backend == 'cuda' else 'OFF'}",
                 f"-DRAM_GGML_VULKAN={'ON' if args.backend == 'vulkan' else 'OFF'}"]
        if not args.dry_run:
            subprocess.run(cmake, check=True)
            subprocess.run(["cmake", "--build", str(build), "-j", str(args.build_jobs)], check=True)
        if not args.dry_run and not binary.is_file():
            parser.error(f"C++ executable was not created: {binary}")
        command = [str(binary), "--model", str(gguf), "--image", str(image),
                   "--backend", args.backend, "--threads", str(args.threads),
                   "--warmup", str(args.warmup), "--repeat", str(args.repeat),
                   "--cuda-compute", args.cuda_compute, "--task", args.task,
                   "--max-length", str(args.max_length)]
        if args.pipeline:
            command.extend(("--pipeline", "1"))
        if args.dump_dir:
            command.extend(("--dump-dir", str(args.dump_dir)))
        if args.profile_backends:
            command.extend(("--profile-backends", "1"))
        if args.beam_batch is not None:
            if args.task != "caption":
                parser.error("--beam-batch requires --task caption")
            command.extend(("--beam-batch", str(args.beam_batch)))
        if args.specified_tags:
            if args.model != "tag2text" or args.task != "caption":
                parser.error("--specified-tags requires Tag2Text caption")
            command.extend(("--specified-tags", args.specified_tags))
        for option, value in (("--tag-list", args.tag_list), ("--thresholds", args.thresholds),
                              ("--label-embedding", args.label_embedding), ("--vocab", args.vocab)):
            if value:
                command.extend((option, str(value)))
        if args.threshold >= 0:
            command.extend(("--threshold", str(args.threshold)))
        if args.threshold_epsilon:
            command.extend(("--threshold-epsilon", str(args.threshold_epsilon)))
    print("[run_inference] " + " ".join(subprocess.list2cmdline([part]) for part in command))
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render every image's Python/GT/six-GGML tag sets without hiding differences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from visualize_outputs import draw_wrapped, font

ROOT = Path(__file__).resolve().parents[2]


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def names(row: dict) -> set[str]:
    return {str(tag["name"]) for tag in row["tags"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "cpp_ggml/benchmarks/visualizations/tags")
    parser.add_argument("--preview-output", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    heading, body, small = font(23, True), font(18), font(16)
    layout_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    preview = None
    worst_difference = -1
    inventory = []
    for manifest in sorted((args.run_dir / "manifests").glob("*.records.jsonl")):
        dataset = manifest.name.removesuffix(".records.jsonl")
        records = read_rows(manifest)
        for model in ("ram", "ram_plus", "tag2text"):
            stem = f"datasets__{dataset}__{model}"
            reference = read_rows(args.run_dir / f"{stem}__pytorch.jsonl")
            variants = [(backend, dtype,
                         read_rows(args.run_dir / f"{stem}__ggml__{backend}__{dtype}.jsonl"))
                        for backend in ("cuda", "vulkan") for dtype in ("f32", "f16", "q8_0")]
            if len(reference) != len(records) or any(len(rows) != len(records) for _, _, rows in variants):
                raise ValueError(f"incomplete tag outputs: {dataset}/{model}")
            for index, record in enumerate(records):
                image_path = Path(record["image"])
                for row in [reference[index]] + [rows[index] for _, _, rows in variants]:
                    if Path(row["image"]).resolve() != image_path.resolve():
                        raise ValueError(f"image order mismatch: {dataset}/{model}/{index}")
                expected = names(reference[index])
                gt_key = "labels_tag2text" if model == "tag2text" else "labels_ram"
                gt = record.get(gt_key, [])
                rows_to_draw = [("GT", ", ".join(gt) if gt else "No GT in this model's label space", "#303030"),
                                (f"Python CUDA  {reference[index]['latency_ms']:.2f} ms",
                                 ", ".join(sorted(expected)) or "(empty)", "#175c2d")]
                changed = 0
                for backend, dtype, predictions in variants:
                    prediction = predictions[index]
                    actual = names(prediction)
                    extra, missing = sorted(actual - expected), sorted(expected - actual)
                    changed += len(extra) + len(missing)
                    label = f"{backend} / {dtype}  {prediction['latency_ms']:.2f} ms"
                    rows_to_draw.append((label + ("  exact" if actual == expected else "  changed"),
                                         ", ".join(sorted(actual)) or "(empty)", "#175c2d"))
                    if extra:
                        rows_to_draw.append(("GGML only", ", ".join(extra), "#a31919"))
                    if missing:
                        rows_to_draw.append(("Python only", ", ".join(missing), "#a31919"))
                placements = []
                y = 70
                for label, text, color in rows_to_draw:
                    placements.append((y, label, text, color))
                    y = draw_wrapped(layout_draw, (670, y + 23), text, body, color, 1090, 24) + 17
                canvas = Image.new("RGB", (1800, max(570, y + 20)), "white")
                draw = ImageDraw.Draw(canvas)
                draw.text((25, 20), f"{dataset} / {model} / {image_path.name}", font=heading, fill="#111111")
                with Image.open(image_path) as image:
                    source = ImageOps.contain(image.convert("RGB"), (590, 430))
                canvas.paste(source, (30 + (590 - source.width) // 2, 80 + (430 - source.height) // 2))
                draw.rectangle((20, 70, 640, 530), outline="#b5b5b5", width=2)
                for y, label, text, color in placements:
                    draw.text((670, y), label, font=small, fill="#303030")
                    draw_wrapped(draw, (670, y + 23), text, body, color, 1090, 24)
                output = args.output_dir / f"{dataset}__{model}__{index:03d}.png"
                canvas.save(output)
                inventory.append({"image": str(image_path.relative_to(ROOT)), "model": model,
                                  "panel": output.name, "tag_differences": changed})
                if changed > worst_difference:
                    worst_difference = changed
                    preview = canvas
    if not inventory:
        raise ValueError("no dataset manifests found")
    (args.output_dir / "index.json").write_text(json.dumps(inventory, indent=2) + "\n")
    lines = ["# All-image tagging comparisons", "",
             f"Run: `{args.run_dir.name}`. {len(inventory)} panels cover all three models on 50 frames.", "",
             "Each panel shows the source image, available GT, official Python tags and all six GGML outputs. "
             "Extra and missing tags are shown explicitly in red.", ""]
    for item in inventory:
        lines.extend([f"## {item['model']} / {item['image']}", "",
                      f"Total extra/missing tags across six variants: {item['tag_differences']}.", "",
                      f"![{item['model']} tag comparison]({item['panel']})", ""])
    (args.output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    if args.preview_output:
        args.preview_output.parent.mkdir(parents=True, exist_ok=True)
        preview.save(args.preview_output, quality=85, method=6)
    print(json.dumps({"panels": len(inventory), "source_images": len(inventory) // 3,
                      "worst_panel_tag_differences": worst_difference,
                      "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

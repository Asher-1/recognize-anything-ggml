#!/usr/bin/env python3
"""Select and optionally prune a diverse 50-frame GT evaluation subset.

The selector optimizes marginal coverage of rare labels, label-token themes,
label density and image aspect/resolution buckets. It writes the exact source
rows and selection reasons before ``--apply`` removes unselected files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    quota: int
    image_root: Path
    annotation: Path
    tag_list: Path
    indexed_labels: bool
    companion_annotations: tuple[Path, ...] = ()
    archive: Path | None = None


SPECS = (
    DatasetSpec(
        "openimages_common_214", 14,
        DATA / "openimages_common_214/imgs/test",
        DATA / "openimages_common_214/openimages_common_214_ram_annots.txt",
        DATA / "openimages_common_214/openimages_common_214_ram_taglist.txt",
        False,
        (DATA / "openimages_common_214/openimages_common_214_tag2text_idannots.txt",),
    ),
    DatasetSpec(
        "openimages_rare_200", 12,
        DATA / "openimages_rare_200/imgs/test",
        DATA / "openimages_rare_200/openimages_rare_200_ram_annots.txt",
        DATA / "openimages_rare_200/openimages_rare_200_ram_taglist.txt",
        False,
    ),
    DatasetSpec(
        "imagenet_multi", 12,
        DATA / "imagenet_multi/imgs",
        DATA / "imagenet_multi/imagenet_multi_1000_annots.txt",
        DATA / "imagenet_multi/imagenet_multi_1000_taglist.txt",
        True,
        archive=DATA / "ILSVRC2012_img_val.tar",
    ),
    DatasetSpec(
        "hico", 12,
        DATA / "hico/imgs",
        DATA / "hico/hico_600_annots.txt",
        DATA / "hico/hico_600_taglist.txt",
        True,
        archive=DATA / "hico_20150920.tar.gz",
    ),
)


@dataclass
class Row:
    identifier: str
    raw_line: str
    labels: tuple[str, ...]
    image: Path
    size_bytes: int
    width: int
    height: int

    @property
    def aspect(self) -> float:
        return self.width / max(1, self.height)

    @property
    def aspect_bucket(self) -> str:
        if self.aspect < 0.75:
            return "portrait"
        if self.aspect > 1.33:
            return "landscape"
        return "square"

    @property
    def resolution_bucket(self) -> str:
        pixels = self.width * self.height
        if pixels < 320 * 240:
            return "small"
        if pixels > 1600 * 1200:
            return "large"
        return "medium"


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]


def decode_labels(fields: list[str], names: list[str], indexed: bool) -> list[str]:
    if not indexed:
        return [field for field in fields if field]
    result = []
    for field in fields:
        for value in field.split():
            index = int(value)
            if not 0 <= index < len(names):
                raise ValueError(f"class index {index} is outside {len(names)} names")
            result.append(names[index])
    return result


def image_for(spec: DatasetSpec, identifier: str) -> Path:
    if spec.name.startswith("openimages_"):
        return spec.image_root / f"{identifier.removeprefix('test/')}.jpg"
    return spec.image_root / identifier


def image_info(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    with Image.open(path) as image:
        width, height = image.size
    return stat.st_size, width, height


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(spec: DatasetSpec) -> tuple[list[Row], dict[str, str], list[str]]:
    names = read_lines(spec.tag_list)
    rows = []
    for line in read_lines(spec.annotation):
        fields = line.split(",")
        identifier = fields[0]
        labels = tuple(decode_labels(fields[1:], names, spec.indexed_labels))
        if not labels:
            continue
        image = image_for(spec, identifier)
        if not image.is_file():
            continue
        try:
            size_bytes, width, height = image_info(image)
        except Exception:
            continue
        rows.append(Row(identifier, line, labels, image, size_bytes, width, height))

    companion = {}
    for path in spec.companion_annotations:
        for line in read_lines(path):
            fields = line.split(",")
            if fields:
                companion[fields[0]] = line
    return rows, companion, names


def stable_value(seed: int, text: str) -> float:
    digest = hashlib.sha256(f"{seed}:{text}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def choose(rows: list[Row], quota: int, seed: int) -> tuple[list[Row], list[dict[str, object]]]:
    frequencies = Counter(label for row in rows for label in set(row.labels))
    token_frequencies = Counter(
        token.lower()
        for row in rows
        for label in set(row.labels)
        for token in label.replace("_", " ").replace("-", " ").split()
        if len(token) > 2
    )
    total = max(1, len(rows))
    label_weight = {
        label: 1.0 + math.log((total + 1) / (count + 1))
        for label, count in frequencies.items()
    }
    token_weight = {
        token: 0.5 + math.log((total + 1) / (count + 1))
        for token, count in token_frequencies.items()
    }

    selected: list[Row] = []
    selected_labels: set[str] = set()
    selected_tokens: set[str] = set()
    selected_buckets: set[tuple[str, str]] = set()
    reasons: list[dict[str, object]] = []
    remaining = list(rows)
    for ordinal in range(quota):
        if not remaining:
            raise RuntimeError(f"only {len(selected)} usable rows available, need {quota}")

        def score(row: Row) -> tuple[float, float]:
            labels = set(row.labels)
            tokens = {
                token.lower()
                for label in labels
                for token in label.replace("_", " ").replace("-", " ").split()
                if len(token) > 2
            }
            novel_labels = labels - selected_labels
            novel_tokens = tokens - selected_tokens
            buckets = (row.aspect_bucket, row.resolution_bucket)
            rarity = sum(label_weight[label] for label in novel_labels)
            token_gain = 0.2 * sum(token_weight[token] for token in novel_tokens)
            density = min(len(labels), 8) * 0.45
            visual = 1.0 if buckets not in selected_buckets else 0.0
            extreme_aspect = 0.15 if row.aspect < 0.5 or row.aspect > 2.0 else 0.0
            tie = stable_value(seed + ordinal, row.identifier) * 1e-3
            return rarity + token_gain + density + visual + extreme_aspect + tie, tie

        best = max(remaining, key=score)
        remaining.remove(best)
        selected.append(best)
        labels = set(best.labels)
        tokens = {
            token.lower()
            for label in labels
            for token in label.replace("_", " ").replace("-", " ").split()
            if len(token) > 2
        }
        selected_labels.update(labels)
        selected_tokens.update(tokens)
        selected_buckets.add((best.aspect_bucket, best.resolution_bucket))
        reasons.append({
            "rank": ordinal + 1,
            "identifier": best.identifier,
            "labels": list(best.labels),
            "aspect_bucket": best.aspect_bucket,
            "resolution_bucket": best.resolution_bucket,
            "width": best.width,
            "height": best.height,
            "size_bytes": best.size_bytes,
            "sha256": file_sha256(best.image),
            "reason": "rare-label and marginal-coverage greedy selection",
        })
    return selected, reasons


def write_annotation(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prune(spec: DatasetSpec, selected: list[Row], apply: bool) -> dict[str, object]:
    selected_ids = {row.identifier for row in selected}
    original_lines = read_lines(spec.annotation)
    kept_lines = [line for line in original_lines if line.split(",", 1)[0] in selected_ids]
    if len(kept_lines) != len(selected):
        raise RuntimeError(f"{spec.name}: selected rows do not map one-to-one to annotations")

    companion_result = {}
    for path in spec.companion_annotations:
        original = read_lines(path)
        kept = [line for line in original if line.split(",", 1)[0] in selected_ids]
        companion_result[str(path.relative_to(ROOT))] = len(kept)
        if apply:
            write_annotation(path, kept)
    archive_would_remove = bool(spec.archive and spec.archive.is_file())
    if apply:
        write_annotation(spec.annotation, kept_lines)
        keep_paths = {row.image.resolve() for row in selected}
        for path in spec.image_root.iterdir():
            if path.is_file() and path.resolve() not in keep_paths:
                path.unlink()
        if archive_would_remove:
            spec.archive.unlink()

    return {
        "source_annotation_rows": len(original_lines),
        "selected_annotation_rows": len(kept_lines),
        "selected_images": len(selected),
        "companion_annotation_rows": companion_result,
        "archive_removed": bool(archive_would_remove and apply),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="delete unselected images and rewrite GT files")
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--output", type=Path, default=DATA / "selection_manifest.json")
    args = parser.parse_args()

    previous_manifest = {}
    if args.output.is_file():
        try:
            previous_manifest = json.loads(args.output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous_manifest = {}

    manifest = {
        "seed": args.seed,
        "total_selected": 0,
        "source_total_annotation_rows": 0,
        "applied": args.apply,
        "datasets": {},
    }
    reports = []
    for spec in SPECS:
        rows, _companion, names = load_rows(spec)
        selected, reasons = choose(rows, spec.quota, args.seed)
        summary = prune(spec, selected, args.apply)
        previous = previous_manifest.get("datasets", {}).get(spec.name, {})
        if previous.get("source_annotation_rows", 0) > summary["source_annotation_rows"]:
            summary["source_annotation_rows"] = previous["source_annotation_rows"]
        if previous.get("archive_removed"):
            summary["archive_removed"] = True
        summary.update({
            "name": spec.name,
            "quota": spec.quota,
            "label_space_size": len(names),
            "selected_labels": sorted({label for row in selected for label in row.labels}),
            "selection": reasons,
        })
        manifest["datasets"][spec.name] = summary
        manifest["total_selected"] += len(selected)
        manifest["source_total_annotation_rows"] += summary["source_annotation_rows"]
        reports.append((spec, rows, selected, summary))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_path = args.output.with_name("SELECTION_REPORT.md")
    lines = [
        "# Selected GT evaluation subset",
        "",
        f"Seed: `{args.seed}`. Applied: `{args.apply}`. Source GT rows: **{manifest['source_total_annotation_rows']}**. Total selected: **{manifest['total_selected']}**.",
        "",
        "The selector maximizes rare-label marginal coverage and includes image aspect/resolution diversity.",
        "Every retained frame has at least one GT label; companion annotation files are filtered by image ID.",
        "The JSON manifest stores a SHA-256 digest for every retained image.",
        "",
        "| Dataset | Source rows | Selected | Unique selected GT labels |",
        "|---|---:|---:|---:|",
    ]
    for spec, rows, selected, summary in reports:
        lines.append(f"| {spec.name} | {summary['source_annotation_rows']} | {len(selected)} | "
                     f"{len(summary['selected_labels'])} |")
    lines += ["", "## Retained frames", "", "| Dataset | Image | GT labels | Shape | Selection reason |",
              "|---|---|---|---:|---|"]
    for spec, _, selected, _ in reports:
        for row in selected:
            lines.append(f"| {spec.name} | `{row.image.relative_to(ROOT)}` | "
                         f"{', '.join(row.labels)} | {row.width}x{row.height} | "
                         "rare-label and marginal-coverage greedy selection |")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(args.output), "report": str(report_path),
                      "total_selected": manifest["total_selected"], "applied": args.apply}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

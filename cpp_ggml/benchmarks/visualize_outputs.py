#!/usr/bin/env python3
"""Render per-image caption differences as auditable PNG panels."""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[2]


def font(size: int, bold: bool = False):
    candidates = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    try:
        return ImageFont.truetype(candidates[0 if bold else 1], size)
    except OSError:
        return ImageFont.load_default()


def tokens(text: str) -> list[str]:
    return text.split()


def draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str,
                 value_font, fill, width: int, line_height: int) -> int:
    x, y = xy
    line = ""
    for word in text.split():
        candidate = word if not line else line + " " + word
        if draw.textbbox((0, 0), candidate, font=value_font)[2] > width and line:
            draw.text((x, y), line, font=value_font, fill=fill)
            y += line_height
            line = word
        else:
            line = candidate
    if line:
        draw.text((x, y), line, font=value_font, fill=fill)
        y += line_height
    return y


def draw_diff(draw: ImageDraw.ImageDraw, x: int, y: int, reference: str,
              actual: str, value_font, width: int) -> int:
    left = tokens(reference)
    right = tokens(actual)
    matcher = difflib.SequenceMatcher(a=left, b=right)
    parts = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        color = "#175c2d" if tag == "equal" else "#a31919"
        parts.extend((word, color) for word in right[j1:j2])
    cursor = x
    line_y = y
    for word, color in parts:
        prefix = "" if cursor == x else " "
        token_width = draw.textbbox((0, 0), prefix + word, font=value_font)[2]
        if cursor + token_width > x + width and cursor != x:
            cursor = x
            line_y += 27
            prefix = ""
            token_width = draw.textbbox((0, 0), word, font=value_font)[2]
        draw.text((cursor, line_y), prefix + word, font=value_font, fill=color)
        cursor += token_width
    return line_y + 27


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caption-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "cpp_ggml/benchmarks/visualizations")
    parser.add_argument("--preview-output", type=Path,
                        help="optional checked-in two-frame preview image")
    args = parser.parse_args()
    payload = json.loads((args.caption_run / "caption_results.json").read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    heading = font(24, True)
    body = font(19)
    small = font(16)
    panels = []
    inventory = []
    layout_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    layouts = []
    for index, reference in enumerate(payload["references"]):
        y = draw_wrapped(layout_draw, (700, 60), reference, body, "#175c2d", 830, 27) + 24
        placements = []
        for measurement in payload["measurements"]:
            placements.append(y)
            actual = measurement["captions"][index]["caption"]
            y = draw_diff(layout_draw, 880, y, reference, actual, body, 650) + 28
        layouts.append((placements, max(540, y + 20)))
    panel_height = max(height for _, height in layouts)
    for index, image_path in enumerate(payload["images"]):
        source = Image.open(image_path).convert("RGB")
        source.thumbnail((640, 420), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (1600, panel_height), "white")
        draw = ImageDraw.Draw(canvas)
        framed = ImageOps.contain(source, (620, 420))
        canvas.paste(framed, (30 + (620 - framed.width) // 2, 80 + (420 - framed.height) // 2))
        draw.rectangle((20, 70, 660, 510), outline="#b5b5b5", width=2)
        draw.text((30, 22), Path(image_path).name, font=heading, fill="#111111")
        x = 700
        draw.text((x, 25), "Python reference", font=heading, fill="#111111")
        reference = payload["references"][index]
        draw_wrapped(draw, (x, 60), reference, body, "#175c2d", 830, 27)
        for y, measurement in zip(layouts[index][0], payload["measurements"]):
            label = f"{measurement['backend']} / {measurement['dtype']}"
            actual = measurement["captions"][index]["caption"]
            exact = measurement["captions"][index]["exact"]
            draw.text((x, y), label + ("  exact" if exact else "  changed"), font=small,
                      fill="#175c2d" if exact else "#a31919")
            draw_diff(draw, x + 180, y, reference, actual, body, 650)
        output = args.output_dir / f"{index:04d}-{Path(image_path).stem}.png"
        canvas.save(output)
        panels.append(canvas)
        inventory.append({"image": str(Path(image_path).relative_to(ROOT)), "panel": output.name,
                          "changed_variants": sum(not row["captions"][index]["exact"]
                                                  for row in payload["measurements"])})
    columns = 2
    rows = (len(panels) + columns - 1) // columns
    sheet = Image.new("RGB", (1600 * columns, panel_height * rows), "#eeeeee")
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % columns) * 1600, (index // columns) * panel_height))
    sheet.save(args.output_dir / "caption_text_comparison_contact_sheet.png")
    (args.output_dir / "index.json").write_text(json.dumps(inventory, indent=2) + "\n")
    lines = ["# All-image caption comparisons", "",
             f"Run: `{args.caption_run.name}`. All {len(panels)} frames have a Python reference and six GGML sentences.", "",
             "![All frames](caption_text_comparison_contact_sheet.png)", ""]
    for item in inventory:
        lines.extend([f"## {item['image']}", "",
                      f"Changed variants: {item['changed_variants']}.", "",
                      f"![Caption wording comparison]({item['panel']})", ""])
    (args.output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    if args.preview_output:
        changed = max(range(len(panels)), key=lambda index: sum(
            not row["captions"][index]["exact"] for row in payload["measurements"]))
        selected = [0] if changed == 0 else [0, changed]
        preview = Image.new("RGB", (1600, panel_height * len(selected)), "#eeeeee")
        for row, index in enumerate(selected):
            preview.paste(panels[index], (0, row * panel_height))
        args.preview_output.parent.mkdir(parents=True, exist_ok=True)
        preview.save(args.preview_output, quality=82, method=6)
    print(json.dumps({"images": len(panels), "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Create labeled contact sheets for agy review of near-uniform images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def fit_on_white(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, "white")
        white.alpha_composite(rgba)
        image = white.convert("RGB")
    else:
        image = image.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def main() -> int:
    root_default = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root_default)
    parser.add_argument("--candidates", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    candidates_path = args.candidates or root / "reports" / "image_review_candidates.json"
    output_dir = args.output_dir or root / "reports" / "agy_review" / "near_uniform"
    if not candidates_path.is_absolute():
        candidates_path = root / candidates_path
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    report = json.loads(candidates_path.read_text(encoding="utf-8"))
    candidates = report["near_uniform"]
    font = ImageFont.load_default(size=18)
    columns, rows = 5, 5
    view_size = (200, 140)
    tile_size = (420, 180)
    margin = 10
    per_sheet = columns * rows
    manifest: list[dict[str, object]] = []

    for sheet_index in range((len(candidates) + per_sheet - 1) // per_sheet):
        subset = candidates[sheet_index * per_sheet : (sheet_index + 1) * per_sheet]
        sheet = Image.new("RGB", (columns * tile_size[0], rows * tile_size[1]), "#dddddd")
        draw = ImageDraw.Draw(sheet)
        for local_index, candidate in enumerate(subset):
            global_index = sheet_index * per_sheet + local_index
            label = f"NU{global_index:03d}"
            row, column = divmod(local_index, columns)
            x, y = column * tile_size[0], row * tile_size[1]
            path = root / candidate["path"]
            with Image.open(path) as source:
                original = fit_on_white(source, view_size)
                gray = source.convert("L")
                enhanced = ImageOps.autocontrast(gray, cutoff=0).convert("RGB")
                enhanced = fit_on_white(enhanced, view_size)
            sheet.paste(original, (x + 5, y + 32))
            sheet.paste(enhanced, (x + 215, y + 32))
            draw.rectangle((x + 1, y + 1, x + tile_size[0] - 2, y + tile_size[1] - 2), outline="#555555", width=2)
            draw.text((x + 8, y + 7), f"{label}  original | contrast", fill="black", font=font)
            manifest.append(
                {
                    "label": label,
                    "path": candidate["path"],
                    "width": candidate["width"],
                    "height": candidate["height"],
                    "format": candidate["format"],
                    "intensity_range": candidate["intensity_range"],
                    "sheet": f"sheet_{sheet_index + 1:02d}.png",
                }
            )
        sheet.save(output_dir / f"sheet_{sheet_index + 1:02d}.png", format="PNG", optimize=True)

    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "layout": "Each tile shows the original on the left and grayscale autocontrast on the right.",
                "candidate_count": len(manifest),
                "sheet_count": (len(candidates) + per_sheet - 1) // per_sheet,
                "candidates": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Created {len(manifest)} candidates across {(len(candidates) + per_sheet - 1) // per_sheet} sheets in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

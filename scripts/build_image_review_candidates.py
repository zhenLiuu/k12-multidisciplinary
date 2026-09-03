#!/usr/bin/env python3
"""Build actionable near-uniform and dHash-collision image candidate lists."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


IMAGE_NAME_RE = re.compile(r"^[0-9a-f]{64}\.(?:jpg|png|gif|webp|bmp|tif)$")


def batched(values: Iterable[Any], size: int) -> Iterable[list[Any]]:
    iterator = iter(values)
    while batch := list(islice(iterator, size)):
        yield batch


def iter_images(root: Path) -> Iterable[tuple[str, Path]]:
    images_root = root / "images"
    with os.scandir(images_root) as shards:
        for shard in shards:
            if not shard.is_dir(follow_symlinks=False) or not re.fullmatch(r"[0-9a-f]{2}", shard.name):
                continue
            with os.scandir(shard.path) as entries:
                for entry in entries:
                    if entry.is_file(follow_symlinks=False) and IMAGE_NAME_RE.fullmatch(entry.name):
                        yield f"images/{shard.name}/{entry.name}", Path(entry.path)


def inspect(item: tuple[str, Path]) -> dict[str, Any]:
    relative, path = item
    try:
        with Image.open(path) as image:
            width, height = image.size
            image_format = image.format
            if image.mode == "P":
                image = image.convert("RGBA")
            gray = image.convert("L")
            small = gray.resize((9, 8), Image.Resampling.LANCZOS)
            pixels = list(small.get_flattened_data())
            dhash = 0
            for row in range(8):
                for column in range(8):
                    dhash = (dhash << 1) | int(pixels[row * 9 + column] > pixels[row * 9 + column + 1])
            thumb = gray.copy()
            thumb.thumbnail((64, 64), Image.Resampling.LANCZOS)
            low, high = thumb.getextrema()
        return {
            "status": "ok",
            "path": relative,
            "format": image_format,
            "width": width,
            "height": height,
            "dhash64": f"{dhash:016x}",
            "intensity_range": high - low,
            "near_uniform": high - low <= 2,
        }
    except Exception as exc:
        return {"status": "error", "path": relative, "error": f"{type(exc).__name__}: {exc}"}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output = args.output or root / "reports" / "image_review_candidates.json"
    if not output.is_absolute():
        output = root / output

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    near_uniform: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    inspected = 0
    items = iter_images(root)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for batch in batched(items, 4096):
            for result in executor.map(inspect, batch):
                inspected += 1
                if result["status"] != "ok":
                    errors.append(result)
                    continue
                compact = {
                    "path": result["path"],
                    "width": result["width"],
                    "height": result["height"],
                    "format": result["format"],
                    "intensity_range": result["intensity_range"],
                }
                groups[result["dhash64"]].append(compact)
                if result["near_uniform"]:
                    near_uniform.append({"dhash64": result["dhash64"], **compact})
            if inspected % 50_000 < 4096:
                print(f"inspected {inspected:,} images", flush=True)

    collision_groups = [
        {"dhash64": dhash, "images": images}
        for dhash, images in groups.items()
        if len(images) > 1
    ]
    collision_groups.sort(key=lambda group: (-len(group["images"]), group["dhash64"]))
    near_uniform.sort(key=lambda image: image["path"])
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": ".",
        "method": {
            "near_uniform": "grayscale 64x64 thumbnail intensity range <= 2",
            "collision": "identical 64-bit horizontal difference hash on 9x8 grayscale thumbnail",
            "warning": "Both are candidate generators, not automatic deletion rules.",
        },
        "summary": {
            "images_inspected": inspected,
            "inspection_errors": len(errors),
            "near_uniform_images": len(near_uniform),
            "dhash_collision_groups": len(collision_groups),
            "images_in_dhash_collision_groups": sum(len(group["images"]) for group in collision_groups),
        },
        "inspection_errors": errors,
        "near_uniform": near_uniform,
        "dhash_collision_groups": collision_groups,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"Report: {output}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

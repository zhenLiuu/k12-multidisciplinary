#!/usr/bin/env python3
"""Create resumable agy batches for duplicate-answer conflict resolution."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


def compact_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "review_id": item["review_id"],
        "subject": item["subject"],
        "question_type": item["question_type"],
        "question": item["question"],
        "options": item["options"],
        "table": item["table"],
        "sub_questions": item.get("sub_questions", item.get("sub_question", [])),
        "images": item["images"],
        # Omit vote counts and provenance so agy solves independently.
        "candidate_answers": [
            {"candidate_id": candidate["candidate_id"], "values": [v["value"] for v in candidate["variants"]]}
            for candidate in item["candidate_answers"]
        ],
    }


def review_visual(root: Path, item: dict[str, Any], tile_size: tuple[int, int]) -> Image.Image:
    width, height = tile_size
    header = 34
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width - 1, height - 1), outline="black", width=2)
    draw.text((8, 8), item["review_id"], fill="black", font=ImageFont.load_default())
    paths = item["images"]
    if not paths:
        return canvas
    slot_width = max(1, width // len(paths))
    for index, relative in enumerate(paths):
        try:
            with Image.open(root / relative) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail((slot_width - 8, height - header - 8), Image.Resampling.LANCZOS)
                x = index * slot_width + (slot_width - image.width) // 2
                y = header + (height - header - image.height) // 2
                canvas.paste(image, (x, y))
        except Exception as exc:
            draw.text((index * slot_width + 8, header + 8), f"IMAGE ERROR: {exc}", fill="red")
    return canvas


def make_contact_sheets(root: Path, batch_dir: Path, items: list[dict[str, Any]]) -> list[str]:
    cols, rows = 5, 5
    tile_size = (480, 360)
    per_sheet = cols * rows
    outputs: list[str] = []
    for sheet_index in range(math.ceil(len(items) / per_sheet)):
        subset = items[sheet_index * per_sheet : (sheet_index + 1) * per_sheet]
        sheet = Image.new("RGB", (cols * tile_size[0], rows * tile_size[1]), "#dddddd")
        for position, item in enumerate(subset):
            tile = review_visual(root, item, tile_size)
            x = (position % cols) * tile_size[0]
            y = (position // cols) * tile_size[1]
            sheet.paste(tile, (x, y))
        output = batch_dir / f"contact_sheet_{sheet_index + 1:02d}.jpg"
        sheet.save(output, quality=90, optimize=True)
        outputs.append(str(output.relative_to(root)))
    return outputs


def schema_for(count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["scope_confirmation", "results"],
        "properties": {
            "scope_confirmation": {"type": "string"},
            "results": {
                "type": "array", "minItems": count, "maxItems": count,
                "items": {
                    "type": "object",
                    "required": ["review_id", "decision", "matched_candidate_id", "solved_answer", "confidence", "reason"],
                    "properties": {
                        "review_id": {"type": "string"},
                        "decision": {"type": "string", "enum": ["match", "none", "uncertain"]},
                        "matched_candidate_id": {"type": "string"},
                        "solved_answer": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "reason": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            },
        },
        "additionalProperties": False,
    }


def prompt_for(root: Path, batch_path: Path, contact_sheets: list[str], count: int) -> str:
    image_note = ""
    if contact_sheets:
        image_note = (
            "先逐张查看这些联系图：" + "、".join(contact_sheets) +
            "。联系图中的标签等于 review_id。只允许查看这些联系图，不要逐个打开批次 JSON 中的原始 images；"
            "如果联系图不足以可靠辨认解题信息，直接标记 uncertain，不得猜测。"
        )
    return f"""你是 K12 多学科题目答案复核员。工作区根目录是 {root}。
读取 {batch_path.relative_to(root)}，其中恰有 {count} 个答案冲突的完整重复题组。{image_note}

对每一组执行：
1. 仅依据题干、结构化选项、表格、子问题和图片独立解题；不要依据候选答案出现次数，不要多数投票。
2. 解出答案后，再与 candidate_answers 比较。若正确答案与某个候选在语义上等价，decision=match，并填写该 candidate_id；若没有候选匹配，decision=none；若题目条件不足、图片无法辨认或无法可靠求解，decision=uncertain。none/uncertain 的 matched_candidate_id 必须是空字符串。
3. 多选题必须核对完整选项集合；非选择题允许数学等价表达。不得为了保留数据而猜测。
4. 每个 review_id 恰好返回一次，不得遗漏或新增。reason 用中文，最多 60 字，只写关键判断依据。

输出必须严格符合给定 JSON schema。"""


def chunks(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[index:index + size] for index in range(0, len(items), size)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--text-batch-size", type=int, default=125)
    parser.add_argument("--image-batch-size", type=int, default=125)
    args = parser.parse_args()
    root = args.root.resolve()
    review_dir = (args.report_dir or root / "reports" / "duplicate_resolution").resolve()
    manifest = [json.loads(line) for line in (review_dir / "agy_conflict_manifest.jsonl").open(encoding="utf-8")]
    sets = (
        ("text", [item for item in manifest if not item["images"]], args.text_batch_size),
        ("image", [item for item in manifest if item["images"]], args.image_batch_size),
    )
    batch_root = review_dir / "agy_batches"
    batch_root.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, Any]] = []
    for kind, items, size in sets:
        for number, batch in enumerate(chunks(items, size), 1):
            name = f"{kind}_{number:02d}"
            batch_dir = batch_root / name
            batch_dir.mkdir(parents=True, exist_ok=True)
            compact = [compact_item(item) for item in batch]
            input_path = batch_dir / "input.json"
            input_path.write_text(json.dumps(compact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            sheets = make_contact_sheets(root, batch_dir, compact) if kind == "image" else []
            schema_path = batch_dir / "schema.json"
            schema_path.write_text(json.dumps(schema_for(len(batch)), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            prompt_path = batch_dir / "prompt.txt"
            prompt_path.write_text(prompt_for(root, input_path, sheets, len(batch)) + "\n", encoding="utf-8")
            index.append({
                "batch": name, "kind": kind, "count": len(batch),
                "review_ids": [item["review_id"] for item in batch],
                "input": str(input_path.relative_to(root)),
                "schema": str(schema_path.relative_to(root)),
                "prompt": str(prompt_path.relative_to(root)),
                "contact_sheets": sheets,
            })
    index_path = batch_root / "index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"batches": len(index), "text": sum(x["kind"] == "text" for x in index), "image": sum(x["kind"] == "image" for x in index), "index": str(index_path.relative_to(root))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

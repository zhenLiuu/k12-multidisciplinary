#!/usr/bin/env python3
"""Prepare the small semantic-review batch for structure normalization."""

from __future__ import annotations

import argparse
import hashlib
import json
import linecache
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from prepare_structure_fix import normalize_record

SEMANTIC_REASONS = {"unicode_replacement_character", "invalid_subject_ch"}


def read_locator(root: Path, locator: list[Any]) -> dict[str, Any]:
    relative, number = locator
    path = root / relative
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))[number - 1]
    return json.loads(linecache.getline(str(path), number))


def collision_items(root: Path, report: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for collision in report["collision_examples"]:
        first, second = collision["first"], collision["second"]
        raw_a, raw_b = read_locator(root, first), read_locator(root, second)
        split = "test" if first[0].endswith(".json") else "raw"
        a, _, _ = normalize_record(first[0], split, first[1], raw_a)
        b, _, _ = normalize_record(second[0], split, second[1], raw_b)
        if a["answer"] == b["answer"]:
            continue
        review_id = "collision_" + hashlib.sha256(collision["id"].encode()).hexdigest()[:16]
        output.append({
            "review_id": review_id,
            "source": {"file": first[0], "records": [first[1], second[1]], "public_id": collision["id"]},
            "reasons": ["public_id_answer_conflict"],
            "subject": a["subject"], "question_type": a["question_type"],
            "question": a["question"], "raw_question": a["question"], "options": a["options"],
            "original_answer": [raw_a.get("answer"), raw_b.get("answer")],
            "candidate_answers": [a["answer"], b["answer"]],
            "proposed_answer": [],
            "explanation": "\n\n--- candidate 2 ---\n\n".join([a["explanation"], b["explanation"]]),
            "images": [item["path"] for item in a["images"]], "table": a["table"],
            "sub_questions": a["sub_questions"],
        })
    return output


def make_sheets(root: Path, output_dir: Path, items: list[dict[str, Any]]) -> list[str]:
    visual = [item for item in items if item.get("images")]
    if not visual:
        return []
    width, height, cols, rows = 800, 520, 2, 3
    outputs = []
    font = ImageFont.load_default()
    for page in range(math.ceil(len(visual) / (cols * rows))):
        subset = visual[page * cols * rows:(page + 1) * cols * rows]
        sheet = Image.new("RGB", (width * cols, height * rows), "#dddddd")
        for position, item in enumerate(subset):
            tile = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(tile)
            draw.rectangle((0, 0, width - 1, height - 1), outline="black", width=2)
            draw.text((8, 8), item["review_id"], fill="black", font=font)
            paths = item["images"]
            slot = max(1, width // len(paths))
            for image_index, relative in enumerate(paths):
                try:
                    with Image.open(root / relative) as source:
                        image = ImageOps.exif_transpose(source).convert("RGB")
                        image.thumbnail((slot - 12, height - 48), Image.Resampling.LANCZOS)
                        x = image_index * slot + (slot - image.width) // 2
                        y = 38 + (height - 38 - image.height) // 2
                        tile.paste(image, (x, y))
                except Exception as exc:
                    draw.text((image_index * slot + 8, 50), f"IMAGE ERROR: {exc}", fill="red", font=font)
            sheet.paste(tile, ((position % cols) * width, (position // cols) * height))
        path = output_dir / f"contact_sheet_{page + 1:02d}.jpg"
        sheet.save(path, quality=92, optimize=True)
        outputs.append(str(path.relative_to(root)))
    return outputs


def schema(count: int) -> dict[str, Any]:
    option = {
        "type": "object", "required": ["label", "text"],
        "properties": {"label": {"type": "string"}, "text": {"type": "string"}},
        "additionalProperties": False,
    }
    result = {
        "type": "object",
        "required": ["review_id", "decision", "subject", "question", "options", "answer", "confidence", "repairs", "reason"],
        "properties": {
            "review_id": {"type": "string"},
            "decision": {"type": "string", "enum": ["keep", "delete", "uncertain"]},
            "subject": {"type": "string", "enum": ["math", "physics", "biology", "geography", "chemistry"]},
            "question": {"type": "string"}, "options": {"type": "array", "items": option, "maxItems": 8},
            "answer": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "repairs": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"},
        }, "additionalProperties": False,
    }
    return {
        "type": "object", "required": ["scope_confirmation", "results"],
        "properties": {
            "scope_confirmation": {"type": "string"},
            "results": {"type": "array", "minItems": count, "maxItems": count, "items": result},
        }, "additionalProperties": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    report_dir = root / "reports" / "structure_fix"
    output_dir = report_dir / "agy_review"
    output_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for line in (report_dir / "review_manifest.jsonl").open(encoding="utf-8"):
        item = json.loads(line)
        if SEMANTIC_REASONS & set(item["reasons"]):
            items.append(item)
    preparation = json.loads((report_dir / "preparation_report.json").read_text(encoding="utf-8"))
    items.extend(collision_items(root, preparation))
    input_path = output_dir / "input.json"
    input_path.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sheets = make_sheets(root, output_dir, items)
    schema_path = output_dir / "schema.json"
    schema_path.write_text(json.dumps(schema(len(items)), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prompt = f"""你是 K12 数据结构修复复核员。工作区根目录为 {root}。
读取 {input_path.relative_to(root)}，其中恰有 {len(items)} 个待复核项。先查看联系图：{', '.join(sheets) if sheets else '无'}；标签对应 review_id。必要时可打开 input 中列出的单张图片。

任务边界：
1. unicode_replacement_character：只修复 U+FFFD（�）及其直接造成的残缺；优先依据图片、上下文、解析和学科知识。不得润色或改写其他内容。无法可靠还原则 decision=uncertain。
2. invalid_subject_ch：判定五学科之一，仅修正 subject，其他字段原样返回。
3. public_id_answer_conflict：独立解题，在 candidate_answers 中选择语义正确者；若题目自身有误或无法确定则 uncertain。不得多数投票。
4. multiple_choice 必须返回按 A 开始连续编号的 options，answer 为标签数组且全部落在 options 中。non_multiple_choice 的 options 必须为空，answer 保持分问答案数组。
5. 只有所有必要字段都可可靠恢复时 decision=keep；题目信息缺失用 delete，证据不足用 uncertain。不要猜测。
6. 每个 review_id 恰好返回一次。reason 用中文且不超过 80 字；repairs 简述实际修改。

输出严格符合 JSON schema。"""
    prompt_path = output_dir / "prompt.txt"
    prompt_path.write_text(prompt + "\n", encoding="utf-8")
    index = {
        "count": len(items), "review_ids": [item["review_id"] for item in items],
        "input": str(input_path.relative_to(root)), "schema": str(schema_path.relative_to(root)),
        "prompt": str(prompt_path.relative_to(root)), "contact_sheets": sheets,
    }
    (output_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(index, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

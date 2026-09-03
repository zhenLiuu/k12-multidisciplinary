#!/usr/bin/env python3
"""Run one structured agy review over all near-uniform contact sheets."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_response(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise ValueError(f"unexpected response type: {type(value).__name__}")
    text = value.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("model response root must be an object")
    return parsed


def main() -> int:
    root_default = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root_default)
    parser.add_argument("--model", default="gemini-3.7-flash-medium")
    parser.add_argument("--timeout", type=int, default=1500)
    args = parser.parse_args()

    root = args.root.resolve()
    review_dir = root / "reports" / "agy_review" / "near_uniform"
    manifest = json.loads((review_dir / "manifest.json").read_text(encoding="utf-8"))
    labels = [item["label"] for item in manifest["candidates"]]
    sheets = [f"reports/agy_review/near_uniform/sheet_{index:02d}.png" for index in range(1, manifest["sheet_count"] + 1)]

    schema = {
        "type": "object",
        "required": ["scope_confirmation", "reviews"],
        "properties": {
            "scope_confirmation": {"type": "string"},
            "reviews": {
                "type": "array",
                "minItems": len(labels),
                "maxItems": len(labels),
                "items": {
                    "type": "object",
                    "required": ["label", "decision", "reason"],
                    "properties": {
                        "label": {"type": "string"},
                        "decision": {"type": "string", "enum": ["blank", "valid", "uncertain"]},
                        "reason": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            },
        },
        "additionalProperties": False,
    }
    prompt = f"""你是 K12 图像数据的只读质量复核员。只复核近纯色候选，不做 dHash 重复图审核，也不要修改任何文件。

必须使用图像查看工具，以原始分辨率依次打开以下 5 张联系图：
{chr(10).join('- ' + sheet for sheet in sheets)}

每个格子标为 NU000 到 NU124，左侧是原图，右侧是同图的灰度自动对比度增强版。请逐格判断：
- blank：原图及增强图均没有对解题有意义的文字、线条、图形、图表或其他视觉信息，属于空白/无效图片；
- valid：能看到任何可能有意义的文字、线条、几何图形、示意图、图表或题目视觉内容，即使很淡；
- uncertain：联系图分辨率或显示不足以可靠判断。

宁可 uncertain，不要把淡线条或稀疏示意图误判为 blank。必须返回恰好 125 条，NU000–NU124 各一次，按编号升序。reason 使用不超过 20 个中文字。scope_confirmation 必须说明“仅审核了125张近纯色候选，未审核dHash碰撞组”。"""

    command = [
        "agy",
        "--model",
        args.model,
        "--mode",
        "plan",
        "--sandbox",
        "--dangerously-skip-permissions",
        "--disable-slash-commands",
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(schema, ensure_ascii=False),
        "--print-timeout",
        f"{args.timeout}s",
        "-p",
        prompt,
    ]
    started_at = datetime.now(timezone.utc).isoformat()
    completed = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=args.timeout + 60)
    raw_path = review_dir / "agy_cli_output.json"
    raw_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        (review_dir / "agy_cli_stderr.txt").write_text(completed.stderr, encoding="utf-8")
        raise RuntimeError(f"agy failed with exit code {completed.returncode}: {completed.stderr[-1000:]}")

    wrapper = json.loads(completed.stdout)
    parsed = parse_response(wrapper.get("response"))
    reviews = parsed.get("reviews")
    if not isinstance(reviews, list):
        raise ValueError("reviews must be a list")
    returned_labels = [item.get("label") for item in reviews if isinstance(item, dict)]
    missing = sorted(set(labels) - set(returned_labels))
    unexpected = sorted(set(returned_labels) - set(labels))
    duplicates = sorted(label for label in set(returned_labels) if returned_labels.count(label) > 1)
    if len(reviews) != len(labels) or missing or unexpected or duplicates:
        raise ValueError(
            f"invalid label coverage: count={len(reviews)}, missing={missing}, unexpected={unexpected}, duplicates={duplicates}"
        )

    manifest_by_label = {item["label"]: item for item in manifest["candidates"]}
    enriched = []
    counts: dict[str, int] = {"blank": 0, "valid": 0, "uncertain": 0}
    for review in reviews:
        decision = review["decision"]
        counts[decision] += 1
        enriched.append({**manifest_by_label[review["label"]], **review})
    result = {
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "reviewer": {"tool": "agy", "model": args.model, "mode": "plan", "sandbox": True},
        "scope": "125 near-uniform image candidates only; dHash collision groups excluded",
        "scope_confirmation": parsed["scope_confirmation"],
        "summary": counts,
        "agy_metadata": {key: wrapper.get(key) for key in ("conversation_id", "status", "duration_seconds", "num_turns", "usage")},
        "reviews": enriched,
    }
    (review_dir / "agy_review_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    print(f"Results: {review_dir / 'agy_review_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

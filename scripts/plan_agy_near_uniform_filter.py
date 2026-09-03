#!/usr/bin/env python3
"""Create a non-mutating impact plan for agy-confirmed blank images."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SPECS = (
    ("data/processed/raw/all_disciplines_with_idx.jsonl", "jsonl", "images", "raw"),
    ("data/processed/raw/math_non_mc.jsonl", "jsonl", "images", "raw"),
    ("data/processed/raw/merge_multiple_choice.jsonl", "jsonl", "images", "raw"),
    ("data/processed/test/final_data_v8.2.json", "json", "image", "test"),
)


def iter_records(path: Path, fmt: str) -> Iterable[tuple[int, dict[str, Any]]]:
    if fmt == "jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                yield number, json.loads(line)
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        yield from enumerate(data, 1)


def image_paths(record: dict[str, Any], field: str, role: str) -> list[str]:
    images = record.get(field, [])
    if not isinstance(images, list):
        return []
    if role == "test":
        return [value for value in images if isinstance(value, str)]
    return [item["path"] for item in images if isinstance(item, dict) and isinstance(item.get("path"), str)]


def main() -> int:
    root_default = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root_default)
    parser.add_argument("--review", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    review_path = args.review or root / "reports" / "agy_review" / "near_uniform" / "agy_review_results.json"
    output = args.output or root / "reports" / "agy_review" / "near_uniform" / "filter_impact_plan.json"
    if not review_path.is_absolute():
        review_path = root / review_path
    if not output.is_absolute():
        output = root / output

    review = json.loads(review_path.read_text(encoding="utf-8"))
    blank = {item["path"] for item in review["reviews"] if item["decision"] == "blank"}
    valid = {item["path"] for item in review["reviews"] if item["decision"] == "valid"}
    uncertain = {item["path"] for item in review["reviews"] if item["decision"] == "uncertain"}
    remaining_paths_if_drop_records: set[str] = set()
    remaining_paths_if_strip_blank: set[str] = set()
    affected_paths: set[str] = set()
    affected_records: list[dict[str, Any]] = []
    impacts = Counter()
    blank_only = 0
    with_other_images = 0

    for relative, fmt, field, role in SPECS:
        for number, record in iter_records(root / relative, fmt):
            paths = image_paths(record, field, role)
            hits = sorted(blank.intersection(paths))
            if hits:
                impacts[relative] += 1
                affected_paths.update(paths)
                retained_paths = [path for path in paths if path not in blank]
                remaining_paths_if_strip_blank.update(retained_paths)
                if retained_paths:
                    with_other_images += 1
                    recommended_action = "remove_blank_references_and_retain_record"
                else:
                    blank_only += 1
                    recommended_action = "remove_record"
                affected_records.append(
                    {
                        "file": relative,
                        "record": number,
                        "id": record.get("id", record.get("idx", record.get("index"))),
                        "blank_images": hits,
                        "all_record_images": paths,
                        "remaining_images_after_blank_removal": retained_paths,
                        "recommended_action": recommended_action,
                    }
                )
            else:
                remaining_paths_if_drop_records.update(paths)
                remaining_paths_if_strip_blank.update(paths)

    whole_record_orphans = sorted(affected_paths - remaining_paths_if_drop_records)
    whole_record_bytes = sum((root / path).stat().st_size for path in whole_record_orphans if (root / path).is_file())
    recommended_orphans = sorted(affected_paths - remaining_paths_if_strip_blank)
    recommended_bytes = sum((root / path).stat().st_size for path in recommended_orphans if (root / path).is_file())
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": ".",
        "mutated_data": False,
        "review_result": review_path.relative_to(root).as_posix(),
        "reviewer": review["reviewer"],
        "recommended_policy_if_approved": "Remove blank image references; retain a record when other valid images remain, otherwise remove the record; then remove only images with no remaining references.",
        "summary": {
            "agy_blank_images": len(blank),
            "agy_valid_images": len(valid),
            "agy_uncertain_images": len(uncertain),
            "affected_records": len(affected_records),
            "affected_records_by_file": dict(impacts),
            "recommended_records_retained_after_reference_removal": with_other_images,
            "recommended_records_removed": blank_only,
            "recommended_orphan_images_removed": len(recommended_orphans),
            "recommended_orphan_bytes_removed": recommended_bytes,
            "whole_record_policy_records_removed": len(affected_records),
            "whole_record_policy_orphan_images": len(whole_record_orphans),
            "whole_record_policy_orphan_bytes": whole_record_bytes,
        },
        "affected_records": affected_records,
        "recommended_orphan_images": recommended_orphans,
        "whole_record_policy_orphan_images": whole_record_orphans,
        "retained_valid_images": sorted(valid),
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"Plan: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

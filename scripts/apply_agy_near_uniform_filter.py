#!/usr/bin/env python3
"""Apply the approved minimal-impact agy near-uniform filter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGET = "data/processed/raw/all_disciplines_with_idx.jsonl"


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    root_default = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root_default)
    parser.add_argument("--review", type=Path, default=None)
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    review_path = args.review or root / "reports" / "agy_review" / "near_uniform" / "agy_review_results.json"
    plan_path = args.plan or root / "reports" / "agy_review" / "near_uniform" / "filter_impact_plan.json"
    report_path = args.report or root / "reports" / "agy_review" / "near_uniform" / "filter_application_report.json"
    for name in ("review_path", "plan_path", "report_path"):
        value = locals()[name]
        if not value.is_absolute():
            locals()[name] = root / value
    if report_path.exists():
        raise FileExistsError(f"refusing to apply twice; report already exists: {report_path}")

    review = json.loads(review_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    blank = {item["path"] for item in review["reviews"] if item["decision"] == "blank"}
    valid = {item["path"] for item in review["reviews"] if item["decision"] == "valid"}
    uncertain = {item["path"] for item in review["reviews"] if item["decision"] == "uncertain"}

    expected = plan["summary"]
    if review["reviewer"] != plan["reviewer"]:
        raise ValueError("reviewer metadata differs between review and plan")
    if len(blank) != 122 or len(valid) != 3 or uncertain:
        raise ValueError("review decision counts differ from approved plan")
    if expected["recommended_records_retained_after_reference_removal"] != 101:
        raise ValueError("unexpected retained-record count in plan")
    if expected["recommended_records_removed"] != 40:
        raise ValueError("unexpected removed-record count in plan")
    if expected["recommended_orphan_images_removed"] != 122:
        raise ValueError("unexpected orphan-image count in plan")

    target = root / TARGET
    temporary = target.with_name(target.name + ".agy-filter.tmp")
    before_sha256 = hash_file(target)
    input_records = 0
    output_records = 0
    records_modified = 0
    records_removed = 0
    blank_references_removed = 0
    affected_ids: list[dict[str, Any]] = []
    remaining_references: Counter[str] = Counter()

    try:
        with target.open("r", encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as output:
            for line_number, line in enumerate(source, 1):
                input_records += 1
                record = json.loads(line)
                images = record.get("images")
                if not isinstance(images, list):
                    raise ValueError(f"record {line_number}: images is not a list")
                old_paths = [item.get("path") for item in images if isinstance(item, dict)]
                hits = [path for path in old_paths if path in blank]
                if hits:
                    new_images = [item for item in images if not (isinstance(item, dict) and item.get("path") in blank)]
                    blank_references_removed += len(hits)
                    action = "retain_after_removing_blank_references" if new_images else "remove_record"
                    affected_ids.append(
                        {
                            "source_line": line_number,
                            "idx": record.get("idx"),
                            "action": action,
                            "removed_blank_images": hits,
                        }
                    )
                    if not new_images:
                        records_removed += 1
                        continue
                    record["images"] = new_images
                    records_modified += 1
                for item in record["images"]:
                    if isinstance(item, dict) and isinstance(item.get("path"), str):
                        remaining_references[item["path"]] += 1
                output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                output_records += 1
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise

    if records_modified != 101 or records_removed != 40 or blank_references_removed != 147:
        temporary.unlink()
        raise RuntimeError(
            f"actual impact differs from plan: modified={records_modified}, removed={records_removed}, "
            f"blank_refs={blank_references_removed}"
        )
    if blank.intersection(remaining_references):
        temporary.unlink()
        raise RuntimeError("blank image references remain in transformed file")

    after_sha256 = hash_file(temporary)
    os.replace(temporary, target)

    deleted_images = []
    deleted_bytes = 0
    for relative in sorted(blank):
        path = root / relative
        if relative in remaining_references:
            raise RuntimeError(f"refusing to delete referenced image: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"expected blank image is missing: {relative}")
        size = path.stat().st_size
        path.unlink()
        deleted_images.append(relative)
        deleted_bytes += size

    result = {
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": ".",
        "review_result": review_path.relative_to(root).as_posix(),
        "impact_plan": plan_path.relative_to(root).as_posix(),
        "reviewer": review["reviewer"],
        "policy": "Remove agy-confirmed blank image references; retain records with other images; remove records left with no images; delete unreferenced blank files.",
        "summary": {
            "input_records": input_records,
            "output_records": output_records,
            "records_modified": records_modified,
            "records_removed": records_removed,
            "blank_references_removed": blank_references_removed,
            "blank_image_files_removed": len(deleted_images),
            "blank_image_bytes_removed": deleted_bytes,
        },
        "target": {
            "path": TARGET,
            "before_sha256": before_sha256,
            "after_sha256": after_sha256,
            "after_bytes": target.stat().st_size,
        },
        "affected_records": affected_ids,
        "deleted_images": deleted_images,
        "retained_valid_images": sorted(valid),
    }
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

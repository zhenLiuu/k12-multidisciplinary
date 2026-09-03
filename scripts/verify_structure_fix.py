#!/usr/bin/env python3
"""Verify the unified public schema, identifiers, text and image references."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prepare_structure_fix import FILES
from structure_common import ANSWER_PROMPT_RE, OCR_NOISE, PUBLIC_FIELDS, QUESTION_PREFIXES, iter_records, public_id

CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
ID_RE = re.compile(r"k12_[0-9a-f]{64}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    errors = Counter()
    stats = Counter()
    distributions = {"question_type": Counter(), "subject": Counter(), "language": Counter(), "split": Counter()}
    ids: set[str] = set()
    referenced: set[str] = set()
    examples = []

    def fail(code: str, relative: str, number: int, detail: str = "") -> None:
        errors[code] += 1
        if len(examples) < 100:
            examples.append({"code": code, "file": relative, "record": number, "detail": detail[:300]})

    for relative, fmt, expected_split in FILES:
        for number, record in iter_records(root / relative, fmt):
            stats["records"] += 1
            if tuple(record) != PUBLIC_FIELDS:
                fail("schema_fields_or_order", relative, number, repr(list(record)))
                continue
            if not isinstance(record["id"], str) or not ID_RE.fullmatch(record["id"]):
                fail("invalid_id_format", relative, number)
            elif record["id"] in ids:
                fail("duplicate_id", relative, number, record["id"])
            else:
                ids.add(record["id"])
            try:
                if public_id(record) != record["id"]:
                    fail("non_reproducible_id", relative, number)
            except Exception as exc:
                fail("id_recompute_error", relative, number, str(exc))
            if not isinstance(record["source_id"], str) or not record["source_id"]:
                fail("invalid_source_id", relative, number)
            if record["source_file"] not in {"all_disciplines_with_idx", "math_non_mc", "merge_multiple_choice", "final_data_v8.2"}:
                fail("invalid_source_file", relative, number)
            if record["split"] != expected_split:
                fail("invalid_split", relative, number)
            if record["question_type"] not in {"multiple_choice", "non_multiple_choice"}:
                fail("invalid_question_type", relative, number)
            if record["subject"] not in {"math", "physics", "biology", "geography", "chemistry"}:
                fail("invalid_subject", relative, number)
            if record["language"] not in {"zh", "en"}:
                fail("invalid_language", relative, number)
            for key in distributions:
                distributions[key][record[key]] += 1
            if not isinstance(record["question"], str) or not record["question"]:
                fail("empty_question", relative, number)
            if any(record["question"].startswith(prefix) for prefix in QUESTION_PREFIXES) or ANSWER_PROMPT_RE.search(record["question"]):
                fail("question_template_remains", relative, number)
            if not isinstance(record["options"], list):
                fail("options_not_list", relative, number)
                options = []
            else:
                options = record["options"]
            expected_labels = [chr(ord("A") + index) for index in range(len(options))]
            labels = []
            for option in options:
                if not isinstance(option, dict) or tuple(option) != ("label", "text"):
                    fail("invalid_option_object", relative, number)
                    continue
                labels.append(option["label"])
                if not isinstance(option["text"], str) or not option["text"]:
                    fail("empty_option_text", relative, number)
            if labels != expected_labels:
                fail("nonsequential_option_labels", relative, number, repr(labels))
            if not isinstance(record["answer"], list) or not record["answer"] or not all(isinstance(x, str) and x for x in record["answer"]):
                fail("invalid_answer", relative, number)
            elif record["question_type"] == "multiple_choice":
                if not options:
                    fail("multiple_choice_without_options", relative, number)
                if any(answer not in set(labels) for answer in record["answer"]):
                    fail("answer_outside_options", relative, number, repr(record["answer"]))
            elif options:
                fail("non_multiple_choice_with_options", relative, number)
            if not isinstance(record["explanation"], str):
                fail("explanation_not_string", relative, number)
            if not isinstance(record["images"], list):
                fail("images_not_list", relative, number)
            else:
                for image in record["images"]:
                    if not isinstance(image, dict) or tuple(image) != ("path", "caption", "part_of"):
                        fail("invalid_image_object", relative, number)
                        continue
                    path = image["path"]
                    if not isinstance(path, str) or not path.startswith("images/") or Path(path).is_absolute() or ".." in Path(path).parts:
                        fail("invalid_image_path", relative, number, repr(path))
                    else:
                        referenced.add(path)
            if not isinstance(record["table"], list) or not isinstance(record["sub_questions"], list) or not isinstance(record["metadata"], dict):
                fail("invalid_collection_type", relative, number)
            for value in strings(record):
                if CONTROL_RE.search(value):
                    fail("control_character", relative, number)
                    break
                if "�" in value:
                    fail("unicode_replacement", relative, number)
                    break
                if value != value.strip():
                    fail("boundary_whitespace", relative, number)
                    break
                if any(token in value for token in OCR_NOISE):
                    fail("ocr_template_noise", relative, number)
                    break

    actual_images: set[str] = set()
    with os.scandir(root / "images") as shards:
        for shard in shards:
            if not shard.is_dir(follow_symlinks=False):
                continue
            with os.scandir(shard.path) as entries:
                for entry in entries:
                    if entry.is_file(follow_symlinks=False):
                        actual_images.add(f"images/{shard.name}/{entry.name}")
    orphans = sorted(actual_images - referenced)
    missing = sorted(referenced - actual_images)
    errors["orphan_images"] += len(orphans)
    errors["referenced_images_missing"] += len(missing)
    report = {
        "verified_at": datetime.now(timezone.utc).isoformat(), "valid": not any(errors.values()),
        "records": stats["records"], "unique_ids": len(ids), "referenced_images": len(referenced),
        "actual_images": len(actual_images), "errors": dict(errors),
        "distributions": {key: dict(value.most_common()) for key, value in distributions.items()},
        "files": {relative: {"sha256": sha256(root / relative), "bytes": (root / relative).stat().st_size} for relative, _, _ in FILES},
        "orphan_examples": orphans[:100], "missing_examples": missing[:100], "error_examples": examples,
    }
    output = root / "reports" / "structure_fix" / "verification_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

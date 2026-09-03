#!/usr/bin/env python3
"""Apply the audited public-schema migration atomically to processed data."""

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

from prepare_structure_fix import FILES, normalize_record
from structure_common import PUBLIC_FIELDS, clean_text, iter_records, public_id

SEMANTIC_REASONS = {"unicode_replacement_character", "invalid_subject_ch", "invalid_subject"}
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
OCR_TOKENS = ("试题答案练习册答案在线课程", "题目详情", "查看答案", "点击查看", "菁优网")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def review_id(relative: str, number: int) -> str:
    return hashlib.sha256(f"{relative}:{number}".encode()).hexdigest()[:20]


def repaired_text(original: str, proposed: Any) -> str:
    """Accept only a U+FFFD repair; preserve every unaffected source string."""
    if "�" not in original:
        return original
    revised = clean_text(proposed)
    if original.count("\n") and not revised.count("\n") and revised.count("\\n") == original.count("\n"):
        revised = revised.replace("\\n", "\n")
    return revised


def apply_agy(record: dict[str, Any], result: dict[str, Any]) -> dict[str, Any] | None:
    if result.get("decision") != "keep":
        return None
    subject = result.get("subject")
    raw_options = result.get("options")
    if subject not in {"math", "physics", "biology", "geography", "chemistry"}:
        return None
    if not isinstance(raw_options, list) or len(raw_options) != len(record["options"]):
        return None
    question = repaired_text(record["question"], result.get("question"))
    options = []
    for index, (original, proposed) in enumerate(zip(record["options"], raw_options)):
        if not isinstance(proposed, dict):
            return None
        label = chr(ord("A") + index)
        if clean_text(proposed.get("label")).upper() != label:
            return None
        text = repaired_text(original["text"], proposed.get("text"))
        if not text:
            return None
        options.append({"label": label, "text": text})
    # Review is not allowed to rewrite an already parseable answer.
    answer = record["answer"]
    record.update(subject=subject, question=question, options=options, answer=answer)
    if "�" in json.dumps(record, ensure_ascii=False):
        return None
    record["id"] = public_id(record)
    return record


def collision_choices(root: Path, preparation: dict[str, Any], agy_input: list[dict[str, Any]], agy_results: dict[str, dict[str, Any]]) -> dict[str, tuple[str, int] | None]:
    conflict_by_id = {
        item["source"]["public_id"]: item for item in agy_input
        if "public_id_answer_conflict" in item.get("reasons", [])
    }
    choices: dict[str, tuple[str, int] | None] = {}
    for collision in preparation["collision_examples"]:
        public = collision["id"]
        locators = [tuple(collision["first"]), tuple(collision["second"])]
        item = conflict_by_id.get(public)
        if item is None:
            choices[public] = locators[0]
            continue
        result = agy_results.get(item["review_id"])
        if not result or result.get("decision") != "keep":
            choices[public] = None
            continue
        desired = [clean_text(value) for value in result.get("answer", [])]
        selected = None
        for relative, number in locators:
            source = root / relative
            if source.suffix == ".json":
                original = json.loads(source.read_text(encoding="utf-8"))[number - 1]
            else:
                import linecache
                original = json.loads(linecache.getline(str(source), number))
            normalized, _, _ = normalize_record(relative, "test" if source.suffix == ".json" else "raw", number, original)
            if normalized["answer"] == desired:
                selected = (relative, number)
                break
        choices[public] = selected
    return choices


def output_record(handle: Any, record: dict[str, Any], fmt: str, first: bool) -> bool:
    payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    if fmt == "jsonl":
        handle.write(payload + "\n")
        return False
    if not first:
        handle.write(",\n")
    handle.write("  " + payload)
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--apply", action="store_true", help="Required: atomically replace processed files")
    args = parser.parse_args()
    if not args.apply:
        raise SystemExit("Refusing to modify data without --apply")
    root = args.root.resolve()
    report_dir = root / "reports" / "structure_fix"
    agy_dir = report_dir / "agy_review"
    agy_report = json.loads((agy_dir / "validated_results.json").read_text(encoding="utf-8"))
    if not agy_report.get("validation", {}).get("valid"):
        raise SystemExit("agy review is absent or invalid")
    agy_results = {item["review_id"]: item for item in agy_report["results"]}
    agy_input = json.loads((agy_dir / "input.json").read_text(encoding="utf-8"))
    preparation = json.loads((report_dir / "preparation_report.json").read_text(encoding="utf-8"))
    collision_keep = collision_choices(root, preparation, agy_input, agy_results)

    before = {relative: {"sha256": sha256(root / relative), "bytes": (root / relative).stat().st_size} for relative, _, _ in FILES}
    stats = Counter()
    by_file: dict[str, dict[str, int]] = {}
    parser_stats = Counter()
    deletion_path = report_dir / "deletions.jsonl"
    id_map_path = report_dir / "id_map.jsonl"
    temp_paths: list[tuple[Path, Path]] = []
    seen_ids: set[str] = set()

    with deletion_path.open("w", encoding="utf-8") as deletion_log, id_map_path.open("w", encoding="utf-8") as id_map:
        for relative, fmt, split in FILES:
            source = root / relative
            temporary = source.with_name(source.name + ".structure.tmp")
            if temporary.exists():
                temporary.unlink()
            local = Counter()
            with temporary.open("w", encoding="utf-8") as output:
                first = True
                if fmt == "json":
                    output.write("[\n")
                for number, original in iter_records(source, fmt):
                    stats["input_records"] += 1
                    local["input_records"] += 1
                    serialized = json.dumps(original, ensure_ascii=False)
                    stats["source_records_with_control_chars"] += bool(CONTROL_RE.search(serialized))
                    stats["source_records_with_ocr_noise"] += any(token in serialized for token in OCR_TOKENS)
                    normalized, reasons, strategy = normalize_record(relative, split, number, original)
                    parser_stats[strategy] += 1
                    locator = (relative, number)
                    semantic = bool(set(reasons) & SEMANTIC_REASONS)
                    if semantic:
                        result = agy_results.get(review_id(relative, number))
                        normalized = apply_agy(normalized, result) if result else None
                        if normalized is None:
                            reasons.append("agy_not_kept_or_invalid")
                    elif reasons:
                        normalized = None
                    if normalized is not None and normalized["id"] in collision_keep:
                        chosen = collision_keep[normalized["id"]]
                        if chosen != locator:
                            reasons.append("public_id_collision_not_selected")
                            normalized = None
                    if normalized is not None and normalized["id"] in seen_ids:
                        reasons.append("post_normalization_id_collision")
                        normalized = None
                    if normalized is None:
                        stats["deleted_records"] += 1
                        local["deleted_records"] += 1
                        deletion_log.write(json.dumps({
                            "file": relative, "record": number,
                            "source_id": clean_text(original.get("id") if split == "test" else original.get("idx")),
                            "reasons": sorted(set(reasons)), "parser_strategy": strategy,
                        }, ensure_ascii=False) + "\n")
                        continue
                    seen_ids.add(normalized["id"])
                    first = output_record(output, normalized, fmt, first)
                    stats["kept_records"] += 1
                    local["kept_records"] += 1
                    id_map.write(json.dumps({
                        "file": relative, "record": number, "source_id": normalized["source_id"], "id": normalized["id"]
                    }, ensure_ascii=False) + "\n")
                if fmt == "json":
                    output.write("\n]\n")
            temp_paths.append((temporary, source))
            by_file[relative] = dict(local)

    # All outputs were written and closed successfully; replacement is now atomic per file.
    for temporary, source in temp_paths:
        os.replace(temporary, source)
    after = {relative: {"sha256": sha256(root / relative), "bytes": (root / relative).stat().st_size} for relative, _, _ in FILES}
    report = {
        "applied_at": datetime.now(timezone.utc).isoformat(), "schema_fields": list(PUBLIC_FIELDS),
        "answer_convention": "list[str]", "counts": dict(stats), "files": by_file,
        "parser_strategies": dict(parser_stats.most_common()), "input": before, "output": after,
        "agy_model": agy_report.get("model"), "agy_usage": agy_report.get("usage"),
        "agy_decisions": dict(Counter(item["decision"] for item in agy_report["results"])),
        "deletions": str(deletion_path.relative_to(root)), "id_map": str(id_map_path.relative_to(root)),
    }
    output = report_dir / "application_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

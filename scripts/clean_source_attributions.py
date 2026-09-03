#!/usr/bin/env python3
"""Remove high-confidence source-attribution blocks from question prefixes.

The cleaner is deliberately boundary anchored. It never removes bracketed text
from the body of a question, where words such as ``学校`` are often semantic.
Dry-run is the default; pass ``--apply`` to atomically replace processed files.
"""

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
from structure_common import iter_records, public_id


INSTITUTION_RE = re.compile(
    r"(?:学校|中学|小学|附中|附小|校区|学院|大学|教育集团|教研室)"
)
YEAR_RE = re.compile(r"(?:19\d{2}|20\d{2}|\d{2}年)")
STRONG_SOURCE_RE = re.compile(
    r"(?:模拟|联考|统考|质检|调研|月考|期中|期末|一模|二模|三模|诊断|"
    r"检测|抽样|竞赛|高考|中考|会考|适应性|热身|联测|联评)"
)
LEADING_NUMBER_RE = re.compile(r"^\s*(?:第?\s*\d{1,3}\s*(?:题|[.．、:：])\s*)?")
BRACKETS = (
    ("【", "】", "square"),
    ("[", "]", "square"),
    ("［", "］", "square"),
    ("〔", "〕", "square"),
    ("〖", "〗", "square"),
    ("（", "）", "round"),
    ("(", ")", "round"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_like(body: str, bracket_kind: str) -> bool:
    """Return true only for a defensible source label, not semantic content."""
    has_institution = bool(INSTITUTION_RE.search(body))
    has_signal = bool(YEAR_RE.search(body) or STRONG_SOURCE_RE.search(body))
    if bracket_kind == "square":
        return has_institution or has_signal
    return has_institution and has_signal


def strip_leading_attributions(question: str) -> tuple[str, list[str]]:
    """Strip one or more source blocks from the beginning of a question."""
    current = question
    removed: list[str] = []
    while True:
        lead = LEADING_NUMBER_RE.match(current)
        assert lead is not None
        start = lead.end()
        matched = False
        for opening, closing, kind in BRACKETS:
            if not current.startswith(opening, start):
                continue
            end = current.find(closing, start + len(opening))
            if end < 0 or end - start > 182:
                continue
            body = current[start + len(opening):end].strip()
            if not body or not _source_like(body, kind):
                continue
            removed.append(current[start:end + len(closing)])
            current = current[end + len(closing):].lstrip()
            matched = True
            break
        if not matched:
            break
    return current.strip(), removed


def write_record(handle: Any, record: dict[str, Any], fmt: str, first: bool) -> bool:
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
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    report_dir = root / "reports" / "source_attribution_cleanup"
    report_dir.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    file_counts: dict[str, Counter[str]] = {}
    changes: list[dict[str, Any]] = []
    before: dict[str, dict[str, Any]] = {}
    after: dict[str, dict[str, Any]] = {}
    temporary_paths: list[tuple[Path, Path]] = []
    seen_ids: dict[str, tuple[str, int]] = {}
    collisions: list[dict[str, Any]] = []

    # A cleaned ID can collide with an unchanged record that occurs later in
    # the stream. Reserve every original ID before considering replacements.
    original_ids: dict[str, tuple[str, int]] = {}
    for relative, fmt, _ in FILES:
        for number, record in iter_records(root / relative, fmt):
            if record["id"] in original_ids:
                raise ValueError(f"pre-existing duplicate public id: {record['id']}")
            original_ids[record["id"]] = (relative, number)

    for relative, fmt, _ in FILES:
        source = root / relative
        before[relative] = {
            "sha256": sha256(source),
            "bytes": source.stat().st_size,
        }
        local = Counter()
        temporary = source.with_name(source.name + ".source-clean.tmp")
        output = None
        first = True
        if args.apply:
            if temporary.exists():
                temporary.unlink()
            output = temporary.open("w", encoding="utf-8")
            if fmt == "json":
                output.write("[\n")
        try:
            for number, record in iter_records(source, fmt):
                counts["records"] += 1
                local["records"] += 1
                original_question = record["question"]
                question, removed = strip_leading_attributions(original_question)
                old_id = record["id"]
                if removed:
                    if not question:
                        raise ValueError(f"cleaning emptied question at {relative}:{number}")
                    record["question"] = question
                    record["id"] = public_id(record)
                    current_location = (relative, number)
                    collision_location = seen_ids.get(record["id"])
                    if collision_location is None:
                        reserved = original_ids.get(record["id"])
                        if reserved != current_location:
                            collision_location = reserved
                    if collision_location is not None:
                        collisions.append({
                            "file": relative,
                            "record": number,
                            "source_id": record["source_id"],
                            "candidate_id": record["id"],
                            "collides_with": list(collision_location),
                            "removed_candidate": removed,
                        })
                        counts["skipped_id_collisions"] += 1
                        local["skipped_id_collisions"] += 1
                        record["question"] = original_question
                        record["id"] = old_id
                    else:
                        counts["changed_records"] += 1
                        counts["removed_blocks"] += len(removed)
                        local["changed_records"] += 1
                        local["removed_blocks"] += len(removed)
                        changes.append({
                            "file": relative,
                            "record": number,
                            "source_id": record["source_id"],
                            "old_id": old_id,
                            "new_id": record["id"],
                            "removed": removed,
                            "old_question": original_question,
                            "new_question": question,
                        })
                if record["id"] in seen_ids:
                    raise ValueError(f"pre-existing duplicate public id: {record['id']}")
                seen_ids[record["id"]] = (relative, number)
                if output is not None:
                    first = write_record(output, record, fmt, first)
            if output is not None and fmt == "json":
                output.write("\n]\n")
        finally:
            if output is not None:
                output.close()
        file_counts[relative] = local
        if args.apply:
            temporary_paths.append((temporary, source))

    # Store complete old/new text in the line-oriented audit file. Keeping only
    # bounded examples in the summary makes it suitable for a release manifest.
    change_path = report_dir / "changes.jsonl"
    if args.apply:
        with change_path.open("w", encoding="utf-8") as handle:
            for item in changes:
                handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
        for temporary, source in temporary_paths:
            os.replace(temporary, source)
        after = {
            relative: {"sha256": sha256(root / relative), "bytes": (root / relative).stat().st_size}
            for relative, _, _ in FILES
        }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "applied": args.apply,
        "rule": {
            "scope": "question prefix only",
            "square_brackets": "institution keyword OR exam/source signal",
            "round_brackets": "institution keyword AND exam/source signal",
            "leading_question_number": "removed only when directly attached to a removed source block",
            "middle_and_suffix_blocks": "preserved",
        },
        "counts": dict(counts),
        "files": {key: dict(value) for key, value in file_counts.items()},
        "before": before,
        "after": after,
        "changes": str(change_path.relative_to(root)) if args.apply else None,
        "skipped_collisions": collisions,
        "examples": changes[:100],
    }
    report_path = report_dir / ("application_report.json" if args.apply else "dry_run_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

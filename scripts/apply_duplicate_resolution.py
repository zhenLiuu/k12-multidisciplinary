#!/usr/bin/env python3
"""Atomically deduplicate processed data using deterministic or agy decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prepare_duplicate_resolution import FILES, canonical_answer, group_id, is_multiple_choice, iter_records, source_id


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quality_score(record: dict[str, Any]) -> tuple[int, int, int, int]:
    explanation = record.get("explanation")
    explanation_text = explanation.strip() if isinstance(explanation, str) else ""
    populated = sum(value not in (None, "", [], {}) for value in record.values())
    return (
        int(bool(explanation_text)),
        min(len(explanation_text), 2000),
        populated,
        int(bool(source_id(record))),
    )


def load_agy_decisions(root: Path, report_dir: Path, manifest_path: Path) -> tuple[dict[str, str | None], list[str]]:
    by_review = {json.loads(line)["review_id"]: json.loads(line) for line in manifest_path.open(encoding="utf-8")}
    batch_root = report_dir / "agy_batches"
    index = json.loads((batch_root / "index.json").read_text(encoding="utf-8"))
    decisions: dict[str, str | None] = {}
    missing: list[str] = []
    for batch in index:
        result_path = batch_root / batch["batch"] / "validated_results.json"
        if not result_path.exists():
            missing.extend(batch["review_ids"])
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if not payload.get("validation", {}).get("valid"):
            missing.extend(batch["review_ids"])
            continue
        for result in payload["results"]:
            review_id = result["review_id"]
            manifest_item = by_review[review_id]
            if result["decision"] != "match":
                decisions[manifest_item["group_id"]] = None
                continue
            candidate = next(
                item for item in manifest_item["candidate_answers"]
                if item["candidate_id"] == result["matched_candidate_id"]
            )
            decisions[manifest_item["group_id"]] = candidate["answer_key"]
    return decisions, missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--consistent-only", action="store_true", help="Resolve only same-answer groups; leave conflicts unchanged")
    args = parser.parse_args()
    root = args.root.resolve()
    report_dir = (args.report_dir or root / "reports" / "duplicate_resolution").resolve()
    connection = sqlite3.connect(report_dir / "duplicate_index.sqlite3")
    rows = connection.execute(
        "SELECT group_id, COUNT(*), COUNT(DISTINCT answer_key), MIN(answer_key) FROM occurrences GROUP BY group_id HAVING COUNT(*) > 1"
    ).fetchall()
    consistent_answers = {gid: answer for gid, _count, answers, answer in rows if answers == 1}
    conflict_groups = {gid for gid, _count, answers, _answer in rows if answers > 1}
    agy_decisions: dict[str, str | None] = {}
    missing_reviews: list[str] = []
    if not args.consistent_only:
        agy_decisions, missing_reviews = load_agy_decisions(root, report_dir, report_dir / "agy_conflict_manifest.jsonl")
        if missing_reviews:
            raise SystemExit(f"Refusing partial conflict application: {len(missing_reviews)} agy reviews are missing or invalid")

    targets = set(consistent_answers) | set(agy_decisions)
    # Select the most complete representative among records eligible to survive.
    best: dict[str, tuple[tuple[int, int, int, int], str, int]] = {}
    for relative, fmt, _split in FILES:
        for number, record in iter_records(root / relative, fmt):
            gid = group_id(record)
            if gid not in targets:
                continue
            wanted = consistent_answers.get(gid, agy_decisions.get(gid))
            if wanted is None:
                continue
            answer_key = canonical_answer(record.get("answer"), is_multiple_choice(record))
            if answer_key != wanted:
                continue
            score = quality_score(record)
            current = best.get(gid)
            if current is None or score > current[0]:
                best[gid] = (score, relative, number)

    missing_representatives = [gid for gid in targets if (consistent_answers.get(gid, agy_decisions.get(gid)) is not None and gid not in best)]
    if missing_representatives:
        raise SystemExit(f"Refusing application: {len(missing_representatives)} groups have no eligible representative")

    audit_path = report_dir / ("consistent_dedup_removed.jsonl" if args.consistent_only else "final_dedup_removed.jsonl")
    file_reports = []
    total_removed = 0
    reason_counts: Counter[str] = Counter()
    with audit_path.open("w", encoding="utf-8") as audit:
        for relative, fmt, _split in FILES:
            path = root / relative
            before_hash = sha256(path)
            input_count = output_count = removed = 0
            temp = path.with_name(path.name + ".dedup.tmp")
            if fmt == "jsonl":
                output = temp.open("w", encoding="utf-8")
            else:
                output = None
                kept_json: list[dict[str, Any]] = []
            try:
                for number, record in iter_records(path, fmt):
                    input_count += 1
                    gid = group_id(record)
                    remove_reason = ""
                    if gid in consistent_answers:
                        _score, keep_file, keep_number = best[gid]
                        if (relative, number) != (keep_file, keep_number):
                            remove_reason = "consistent_duplicate"
                    elif gid in agy_decisions:
                        wanted = agy_decisions[gid]
                        if wanted is None:
                            remove_reason = "agy_no_matching_answer_or_uncertain"
                        else:
                            _score, keep_file, keep_number = best[gid]
                            if (relative, number) != (keep_file, keep_number):
                                remove_reason = "agy_resolved_duplicate"
                    elif gid in conflict_groups and args.consistent_only:
                        pass
                    if remove_reason:
                        removed += 1
                        total_removed += 1
                        reason_counts[remove_reason] += 1
                        audit.write(json.dumps({
                            "file": relative, "record": number, "source_id": source_id(record),
                            "group_id": gid, "answer": record.get("answer"), "reason": remove_reason,
                        }, ensure_ascii=False, separators=(",", ":")) + "\n")
                        continue
                    output_count += 1
                    if fmt == "jsonl":
                        output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    else:
                        kept_json.append(record)
                if fmt == "json":
                    temp.write_text(json.dumps(kept_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            finally:
                if output is not None:
                    output.close()
            os.replace(temp, path)
            file_reports.append({
                "file": relative, "input_records": input_count, "output_records": output_count,
                "removed": removed, "before_sha256": before_hash, "after_sha256": sha256(path),
            })

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "consistent_only" if args.consistent_only else "complete",
        "selection_rule": "highest tuple: non-empty explanation, explanation length capped at 2000, populated-field count, source-id presence; stable first occurrence breaks ties",
        "identity_definition": "See preparation_summary.json",
        "consistent_groups_resolved": len(consistent_answers),
        "conflict_groups_total": len(conflict_groups),
        "conflict_groups_resolved": len(agy_decisions),
        "conflict_groups_left_unchanged": len(conflict_groups - set(agy_decisions)),
        "removed_records": total_removed,
        "removal_reasons": dict(reason_counts),
        "files": file_reports,
        "removed_records_audit": str(audit_path.relative_to(root)),
    }
    report_path = report_dir / ("consistent_dedup_application.json" if args.consistent_only else "final_dedup_application.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    connection.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

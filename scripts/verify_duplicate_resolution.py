#!/usr/bin/env python3
"""Verify every originally duplicated full-input group after resolution."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from apply_duplicate_resolution import load_agy_decisions
from prepare_duplicate_resolution import FILES, canonical_answer, group_id, is_multiple_choice, iter_records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    report_dir = (args.report_dir or root / "reports" / "duplicate_resolution").resolve()
    connection = sqlite3.connect(report_dir / "duplicate_index.sqlite3")
    rows = connection.execute(
        "SELECT group_id, COUNT(DISTINCT answer_key), MIN(answer_key) FROM occurrences GROUP BY group_id HAVING COUNT(*) > 1"
    ).fetchall()
    consistent = {gid: answer for gid, answer_count, answer in rows if answer_count == 1}
    conflicts = {gid for gid, answer_count, _answer in rows if answer_count > 1}
    agy, missing = load_agy_decisions(root, report_dir, report_dir / "agy_conflict_manifest.jsonl")
    if missing or set(agy) != conflicts:
        raise SystemExit(f"agy coverage failure: missing={len(missing)}, mapped={len(agy)}, expected={len(conflicts)}")

    tracked = set(consistent) | conflicts
    counts: Counter[str] = Counter()
    answers: dict[str, set[str]] = defaultdict(set)
    records = 0
    file_counts: Counter[str] = Counter()
    for relative, fmt, _split in FILES:
        for _number, record in iter_records(root / relative, fmt):
            records += 1
            file_counts[relative] += 1
            gid = group_id(record)
            if gid in tracked:
                counts[gid] += 1
                answers[gid].add(canonical_answer(record.get("answer"), is_multiple_choice(record)))

    failures = []
    for gid, expected_answer in consistent.items():
        if counts[gid] != 1 or answers[gid] != {expected_answer}:
            failures.append({"group_id": gid, "kind": "consistent", "count": counts[gid], "answers": sorted(answers[gid])})
    for gid, expected_answer in agy.items():
        expected_count = 0 if expected_answer is None else 1
        expected_answers = set() if expected_answer is None else {expected_answer}
        if counts[gid] != expected_count or answers[gid] != expected_answers:
            failures.append({"group_id": gid, "kind": "agy", "count": counts[gid], "answers": sorted(answers[gid]), "expected_answer": expected_answer})

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "records": records,
        "files": dict(file_counts),
        "original_duplicate_groups_checked": len(tracked),
        "consistent_groups_singleton": sum(counts[gid] == 1 for gid in consistent),
        "agy_match_groups_singleton": sum(expected is not None and counts[gid] == 1 for gid, expected in agy.items()),
        "agy_none_or_uncertain_groups_absent": sum(expected is None and counts[gid] == 0 for gid, expected in agy.items()),
        "remaining_duplicate_groups_from_original_index": sum(counts[gid] > 1 for gid in tracked),
        "verification_failures": len(failures),
        "failure_examples": failures[:20],
    }
    output = report_dir / "final_dedup_verification.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    connection.close()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

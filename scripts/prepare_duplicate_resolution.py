#!/usr/bin/env python3
"""Build an auditable exact-input duplicate index and agy review manifest.

Only records with the same normalized question, ordered packaged image paths,
structured options, table, and sub-question are grouped. The immutable source
copies are never read or changed; this script operates on data/processed only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


FILES = (
    ("data/processed/raw/all_disciplines_with_idx.jsonl", "jsonl", "raw"),
    ("data/processed/raw/math_non_mc.jsonl", "jsonl", "raw"),
    ("data/processed/raw/merge_multiple_choice.jsonl", "jsonl", "raw"),
    ("data/processed/test/final_data_v8.2.json", "json", "test"),
)
CHOICE_SEPARATORS_RE = re.compile(r"[\s,，、;/|+&\[\](){}<>.。:：]+")


def normalized_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    value = unicodedata.normalize("NFKC", value).strip().lower()
    return " ".join(value.split())


def canonical_value(value: Any) -> Any:
    if isinstance(value, str):
        return normalized_text(value)
    if isinstance(value, list):
        return [canonical_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): canonical_value(value[key]) for key in sorted(value)}
    return value


def image_paths(record: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for field in ("images", "image"):
        value = record.get(field)
        if not isinstance(value, list):
            continue
        for item in value:
            path = item.get("path") if isinstance(item, dict) else item
            if isinstance(path, list):
                result.extend(str(part) for part in path)
            elif isinstance(path, str):
                result.append(path)
    return result


def input_payload(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": normalized_text(record.get("question", "")),
        "images": image_paths(record),
        "options": canonical_value(record.get("options", [])),
        "table": canonical_value(record.get("table", [])),
        "sub_questions": canonical_value(
            record.get("sub_questions", record.get("sub_question", []))
        ),
    }


def group_id(record: dict[str, Any]) -> str:
    raw = json.dumps(input_payload(record), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def is_multiple_choice(record: dict[str, Any]) -> bool:
    value = normalized_text(record.get("question_type", record.get("task_type", "")))
    return value in {"multiple_choice", "multiple choice", "choice"}


def canonical_answer(value: Any, multiple_choice: bool) -> str:
    if multiple_choice:
        candidate = "".join(str(item) for item in value) if isinstance(value, list) else str(value)
        candidate = unicodedata.normalize("NFKC", candidate).upper().strip()
        compact = CHOICE_SEPARATORS_RE.sub("", candidate)
        if compact and re.fullmatch(r"[A-H]+", compact):
            return "choice:" + "".join(sorted(set(compact)))
    normalized = canonical_value(value)
    return "value:" + json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def source_id(record: dict[str, Any]) -> str:
    for key in ("source_id", "id", "idx", "index"):
        if key in record:
            return str(record[key])
    return ""


def iter_records(path: Path, fmt: str) -> Iterable[tuple[int, dict[str, Any]]]:
    if fmt == "jsonl":
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if line.strip():
                    yield number, json.loads(line)
        return
    with path.open(encoding="utf-8") as handle:
        root = json.load(handle)
    for number, record in enumerate(root, 1):
        yield number, record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = (args.output_dir or root / "reports" / "duplicate_resolution").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "duplicate_index.sqlite3"
    if db_path.exists():
        db_path.unlink()
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE occurrences (
            group_id TEXT NOT NULL,
            file TEXT NOT NULL,
            record INTEGER NOT NULL,
            split TEXT NOT NULL,
            answer_key TEXT NOT NULL,
            answer_json TEXT NOT NULL,
            source_id TEXT NOT NULL,
            subject TEXT NOT NULL,
            has_images INTEGER NOT NULL,
            PRIMARY KEY (file, record)
        );
        CREATE INDEX occurrences_group_idx ON occurrences(group_id);
        CREATE INDEX occurrences_group_answer_idx ON occurrences(group_id, answer_key);
        """
    )

    records_scanned = 0
    file_counts: Counter[str] = Counter()
    for relative, fmt, split in FILES:
        batch: list[tuple[Any, ...]] = []
        for number, record in iter_records(root / relative, fmt):
            answer = record.get("answer")
            batch.append(
                (
                    group_id(record), relative, number, split,
                    canonical_answer(answer, is_multiple_choice(record)),
                    json.dumps(answer, ensure_ascii=False, separators=(",", ":")),
                    source_id(record), str(record.get("subject", "")),
                    int(bool(image_paths(record))),
                )
            )
            records_scanned += 1
            file_counts[relative] += 1
            if len(batch) >= 10_000:
                connection.executemany("INSERT INTO occurrences VALUES (?,?,?,?,?,?,?,?,?)", batch)
                connection.commit()
                batch.clear()
        if batch:
            connection.executemany("INSERT INTO occurrences VALUES (?,?,?,?,?,?,?,?,?)", batch)
            connection.commit()

    duplicate_rows = connection.execute(
        """
        SELECT group_id, COUNT(*) AS n, COUNT(DISTINCT answer_key) AS answers,
               MAX(has_images) AS has_images, COUNT(DISTINCT split) AS splits
        FROM occurrences GROUP BY group_id HAVING n > 1
        """
    ).fetchall()
    conflict_group_ids = {row[0] for row in duplicate_rows if row[2] > 1}

    conflict_examples: dict[str, dict[str, Any]] = {}
    for relative, fmt, _split in FILES:
        for _number, record in iter_records(root / relative, fmt):
            gid = group_id(record)
            if gid in conflict_group_ids and gid not in conflict_examples:
                conflict_examples[gid] = {
                    "question": record.get("question", ""),
                    "options": record.get("options", []),
                    "table": record.get("table", []),
                    "sub_questions": record.get(
                        "sub_questions", record.get("sub_question", [])
                    ),
                    "images": image_paths(record),
                    "subject": record.get("subject", ""),
                    "question_type": record.get("question_type", record.get("task_type", "")),
                }

    manifest_path = output_dir / "agy_conflict_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for ordinal, gid in enumerate(sorted(conflict_group_ids), 1):
            rows = connection.execute(
                """
                SELECT file, record, split, answer_key, answer_json, source_id, subject
                FROM occurrences WHERE group_id = ? ORDER BY file, record
                """, (gid,),
            ).fetchall()
            by_answer: dict[str, dict[str, Any]] = {}
            for file, number, split, answer_key, answer_json, rid, subject in rows:
                entry = by_answer.setdefault(
                    answer_key,
                    {"variants": Counter(), "occurrences": [], "subjects": Counter()},
                )
                entry["variants"][answer_json] += 1
                entry["subjects"][subject] += 1
                entry["occurrences"].append(
                    {"file": file, "record": number, "split": split, "source_id": rid}
                )
            candidates = []
            for candidate_number, answer_key in enumerate(sorted(by_answer), 1):
                entry = by_answer[answer_key]
                candidates.append(
                    {
                        "candidate_id": f"A{candidate_number}",
                        "answer_key": answer_key,
                        "variants": [
                            {"value": json.loads(value), "count": count}
                            for value, count in entry["variants"].most_common()
                        ],
                        "occurrence_count": len(entry["occurrences"]),
                        "subjects": dict(entry["subjects"]),
                        "occurrences": entry["occurrences"],
                    }
                )
            item = {
                "review_id": f"dup-{ordinal:05d}", "group_id": gid,
                **conflict_examples[gid], "candidate_answers": candidates,
                "record_count": len(rows),
            }
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")

    duplicate_groups = len(duplicate_rows)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root), "records_scanned": records_scanned,
        "files": dict(file_counts),
        "identity_definition": {
            "question": "NFKC + lowercase + trim + collapse whitespace",
            "images": "ordered packaged image paths (content-addressed)",
            "structured_context": ["options", "table", "sub_questions"],
            "excluded": ["answer", "explanation", "id", "subject", "source metadata"],
        },
        "answer_equivalence": {
            "multiple_choice": "choice labels A-H; separators/order ignored and labels deduplicated",
            "other": "recursive NFKC + lowercase + trim + collapse whitespace; no semantic inference",
        },
        "duplicate_groups": duplicate_groups,
        "records_in_duplicate_groups": sum(row[1] for row in duplicate_rows),
        "duplicate_records_removable_if_one_kept": sum(row[1] - 1 for row in duplicate_rows),
        "consistent_answer_groups": sum(1 for row in duplicate_rows if row[2] == 1),
        "conflicting_answer_groups": len(conflict_group_ids),
        "records_in_conflicting_groups": sum(row[1] for row in duplicate_rows if row[2] > 1),
        "groups_with_images": sum(1 for row in duplicate_rows if row[3]),
        "conflicting_groups_with_images": sum(1 for row in duplicate_rows if row[2] > 1 and row[3]),
        "cross_split_duplicate_groups": sum(1 for row in duplicate_rows if row[4] > 1),
        "outputs": {
            "sqlite_index": str(db_path.relative_to(root)),
            "agy_manifest": str(manifest_path.relative_to(root)),
        },
    }
    summary_path = output_dir / "preparation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    connection.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

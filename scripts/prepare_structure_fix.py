#!/usr/bin/env python3
"""Plan schema normalization and isolate only records requiring semantic review."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from structure_common import (
    PUBLIC_FIELDS, SOURCE_KEYS, clean_nested, clean_text, iter_records, normalize_explanation,
    normalize_images, normalize_mc_answer, normalize_open_answer,
    normalize_structured_options, parse_embedded_options, public_id,
    strip_question_template,
)


FILES = (
    ("data/processed/raw/all_disciplines_with_idx.jsonl", "jsonl", "raw"),
    ("data/processed/raw/math_non_mc.jsonl", "jsonl", "raw"),
    ("data/processed/raw/merge_multiple_choice.jsonl", "jsonl", "raw"),
    ("data/processed/test/final_data_v8.2.json", "json", "test"),
)
SAFE_OPTION_STRATEGIES = {"strong_line", "ordered_line", "hybrid_line", "punct_any", "paren_any"}
KNOWN_SOURCE_FIELDS = {
    "idx", "index", "id", "question_type", "task_type", "question", "options",
    "answer", "explanation", "images", "image", "subject", "category_id",
    "language", "table", "sub_question", "block_ids", "completeness",
    "answer_validation", "llm_validation", "error", "cot", "resource", "tag",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_id(record: dict[str, Any], split: str) -> str:
    value = record.get("id") if split == "test" else record.get("idx")
    if value is None and split == "test":
        value = record.get("index")
    return clean_text(value)


def metadata(record: dict[str, Any], split: str, explanation_parts: list[str] | None) -> dict[str, Any]:
    output: dict[str, Any] = {}
    if split == "test" and record.get("index") is not None:
        output["source_index"] = record["index"]
    source_type = record.get("task_type", record.get("question_type"))
    if source_type not in (None, "multiple_choice", "non_multiple_choice"):
        output["source_question_type"] = clean_text(source_type)
    if explanation_parts is not None:
        output["source_explanation_parts"] = explanation_parts
    for key in ("error", "cot", "resource", "tag", "block_ids", "completeness", "answer_validation", "llm_validation"):
        value = record.get(key)
        if value not in (None, "", [], {}):
            output[key] = clean_nested(value)
    for key in sorted(set(record) - KNOWN_SOURCE_FIELDS):
        value = record[key]
        if value not in (None, "", [], {}):
            output[key] = clean_nested(value)
    return output


def normalize_record(relative: str, split: str, number: int, record: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str], str]:
    reasons: list[str] = []
    sid = source_id(record, split)
    source_file = SOURCE_KEYS[relative]
    raw_type = clean_text(record.get("question_type", record.get("task_type", "")))
    question_type = "non_multiple_choice" if raw_type == "open_end" else raw_type
    question = clean_text(record.get("question", ""))
    options: list[dict[str, str]] = []
    parser_strategy = "not_applicable"

    if question_type == "multiple_choice":
        if relative.endswith("all_disciplines_with_idx.jsonl"):
            parsed = parse_embedded_options(question, record.get("answer"))
            if parsed is None:
                reasons.append("unparseable_embedded_options")
                question = strip_question_template(question)
                parser_strategy = "failed"
            else:
                question, options = parsed.question, parsed.options
                parser_strategy = parsed.strategy
                if parser_strategy not in SAFE_OPTION_STRATEGIES:
                    reasons.append("ambiguous_option_boundaries")
        else:
            normalized = normalize_structured_options(record.get("options"))
            if normalized is None:
                reasons.append("missing_or_empty_structured_options")
            else:
                options = normalized
                parser_strategy = "structured_positional_relabel"
    elif question_type == "non_multiple_choice":
        question = strip_question_template(question)
    else:
        reasons.append("invalid_question_type")

    if not sid:
        reasons.append("missing_source_id")
    if not question:
        reasons.append("empty_question")
    if "\ufffd" in json.dumps(record, ensure_ascii=False):
        reasons.append("unicode_replacement_character")

    if split == "test":
        categories = record.get("category_id")
        subject = clean_text(categories[0]) if isinstance(categories, list) and len(categories) == 1 else ""
        language = clean_text(record.get("language", ""))
    else:
        subject = clean_text(record.get("subject", ""))
        language = "zh"
    if subject == "ch":
        reasons.append("invalid_subject_ch")
    if subject not in {"math", "physics", "biology", "geography", "chemistry"}:
        reasons.append("invalid_subject")
    if not language:
        reasons.append("missing_language")

    if question_type == "multiple_choice" and options:
        answer = normalize_mc_answer(record.get("answer"), options)
        if answer is None:
            reasons.append("unresolved_multiple_choice_answer")
            answer = []
    else:
        answer = normalize_open_answer(record.get("answer"))
        if answer is None:
            reasons.append("empty_answer")
            answer = []

    explanation, explanation_parts = normalize_explanation(record.get("explanation"))
    sub_questions = record.get("sub_question")
    if sub_questions is None:
        sub_questions = []
    elif not isinstance(sub_questions, list):
        sub_questions = [sub_questions]
    sub_questions = clean_nested(sub_questions)
    table = record.get("table")
    if not isinstance(table, list):
        table = [] if table in (None, "") else [table]
    table = clean_nested(table)
    images = normalize_images(record.get("image") if split == "test" else record.get("images"))
    normalized_record = {
        "id": "",
        "source_id": sid,
        "source_file": source_file,
        "split": split,
        "question_type": question_type,
        "subject": subject,
        "language": language,
        "question": question,
        "options": options,
        "answer": answer,
        "explanation": explanation,
        "images": images,
        "table": table,
        "sub_questions": sub_questions,
        "metadata": metadata(record, split, explanation_parts),
    }
    normalized_record["id"] = public_id(normalized_record)
    assert tuple(normalized_record) == PUBLIC_FIELDS
    return normalized_record, sorted(set(reasons)), parser_strategy


def compact_review(relative: str, split: str, number: int, original: dict[str, Any], normalized: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    return {
        "review_id": hashlib.sha256(f"{relative}:{number}".encode()).hexdigest()[:20],
        "source": {"file": relative, "record": number, "source_id": normalized["source_id"]},
        "reasons": reasons,
        "subject": normalized["subject"],
        "question_type": normalized["question_type"],
        "question": normalized["question"],
        "raw_question": strip_question_template(original.get("question", "")),
        "options": normalized["options"],
        "original_answer": original.get("answer"),
        "proposed_answer": normalized["answer"],
        "explanation": normalized["explanation"],
        "images": [item["path"] for item in normalized["images"]],
        "table": normalized["table"],
        "sub_questions": normalized["sub_questions"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    report_dir = root / "reports" / "structure_fix"
    report_dir.mkdir(parents=True, exist_ok=True)
    review_path = report_dir / "review_manifest.jsonl"
    counts = Counter()
    parser_counts = Counter()
    reason_counts = Counter()
    file_reports: dict[str, Any] = {}
    public_ids: dict[str, tuple[str, int]] = {}
    collisions: list[dict[str, Any]] = []
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    with review_path.open("w", encoding="utf-8") as review_handle:
        for relative, fmt, split in FILES:
            path = root / relative
            local = Counter()
            for number, record in iter_records(path, fmt):
                counts["input_records"] += 1
                local["input_records"] += 1
                normalized, reasons, strategy = normalize_record(relative, split, number, record)
                assert normalized is not None
                parser_counts[strategy] += 1
                if reasons:
                    counts["review_records"] += 1
                    local["review_records"] += 1
                    for reason in reasons:
                        reason_counts[reason] += 1
                        if len(samples[reason]) < 5:
                            samples[reason].append({
                                "file": relative, "record": number, "source_id": normalized["source_id"],
                                "question": normalized["question"][:500], "answer": record.get("answer"),
                            })
                    review_handle.write(json.dumps(compact_review(relative, split, number, record, normalized, reasons), ensure_ascii=False) + "\n")
                else:
                    counts["automatic_records"] += 1
                    local["automatic_records"] += 1
                previous = public_ids.setdefault(normalized["id"], (relative, number))
                if previous != (relative, number):
                    collisions.append({"id": normalized["id"], "first": previous, "second": (relative, number)})
            file_reports[relative] = dict(local)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "read_only_plan",
        "schema_fields": list(PUBLIC_FIELDS),
        "answer_convention": "list[str]",
        "input_sha256": {relative: sha256(root / relative) for relative, _, _ in FILES},
        "counts": dict(counts),
        "files": file_reports,
        "parser_strategies": dict(parser_counts.most_common()),
        "review_reasons": dict(reason_counts.most_common()),
        "public_id_collisions": len(collisions),
        "collision_examples": collisions[:20],
        "examples": dict(samples),
        "review_manifest": str(review_path.relative_to(root)),
    }
    output = report_dir / "preparation_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "counts": report["counts"], "parser_strategies": report["parser_strategies"],
        "review_reasons": report["review_reasons"], "public_id_collisions": len(collisions),
        "report": str(output.relative_to(root)),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

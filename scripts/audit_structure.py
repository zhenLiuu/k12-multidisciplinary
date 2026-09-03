#!/usr/bin/env python3
"""Profile current processed schemas and structure-cleaning candidates."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prepare_duplicate_resolution import FILES, image_paths, iter_records, normalized_text, source_id


OPTION_LINE_RE = re.compile(r"(?m)^[ \t]*([A-H])[ \t]*[.．、:：]\s*")
OPTION_ANY_RE = re.compile(r"(?<![A-Za-z])([A-H])[ \t]*[.．、:：]\s*")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
REPLACEMENT = "\ufffd"
ANSWER_PROMPT_RE = re.compile(r"\s*[。．.]?\s*你的答案是\s*[：:]\s*$")
OCR_NOISE = (
    "试题答案练习册答案在线课程",
    "菁优网",
    "题目详情",
    "查看答案",
    "点击查看",
)


def empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": {},
        "global": {
            "records": 0,
            "field_presence": Counter(),
            "field_types": defaultdict(Counter),
            "empty_fields": Counter(),
            "question_types": Counter(),
            "answer_types": Counter(),
            "subjects": Counter(),
            "languages": Counter(),
            "source_id_duplicates": 0,
            "source_id_empty": 0,
            "records_with_control_chars": 0,
            "records_with_replacement_chars": 0,
            "records_with_boundary_whitespace": 0,
            "records_with_ocr_noise": 0,
        },
        "multiple_choice": {
            "records": 0,
            "structured_options_nonempty": 0,
            "structured_options_empty": 0,
            "embedded_line_labels": Counter(),
            "embedded_anywhere_labels": Counter(),
            "has_answer_prompt_suffix": 0,
            "parse_candidate_four_plus_line_labels": 0,
            "parse_candidate_four_plus_anywhere_labels": 0,
            "examples_without_four_line_labels": [],
        },
        "examples": {
            "subject_ch": [],
            "control_chars": [],
            "replacement_chars": [],
            "ocr_noise": [],
            "option_anomalies": [],
        },
    }
    seen_ids: set[tuple[str, str]] = set()
    for relative, fmt, split in FILES:
        file_report = {
            "records": 0, "schema_signatures": Counter(), "field_presence": Counter(),
            "field_types": defaultdict(Counter), "empty_fields": Counter(),
        }
        for number, record in iter_records(root / relative, fmt):
            file_report["records"] += 1
            report["global"]["records"] += 1
            file_report["schema_signatures"][",".join(sorted(record))] += 1
            rid = source_id(record)
            if not rid:
                report["global"]["source_id_empty"] += 1
            else:
                key = (relative, rid)
                if key in seen_ids:
                    report["global"]["source_id_duplicates"] += 1
                seen_ids.add(key)
            record_control = record_replacement = record_boundary = record_noise = False
            for field, value in record.items():
                file_report["field_presence"][field] += 1
                file_report["field_types"][field][type(value).__name__] += 1
                report["global"]["field_presence"][field] += 1
                report["global"]["field_types"][field][type(value).__name__] += 1
                if empty(value):
                    file_report["empty_fields"][field] += 1
                    report["global"]["empty_fields"][field] += 1
                if isinstance(value, str):
                    record_control |= bool(CONTROL_RE.search(value))
                    record_replacement |= REPLACEMENT in value
                    record_boundary |= value != value.strip()
                    record_noise |= any(token in value for token in OCR_NOISE)
            if record_control:
                report["global"]["records_with_control_chars"] += 1
                if len(report["examples"]["control_chars"]) < 10:
                    report["examples"]["control_chars"].append({"file": relative, "record": number, "source_id": rid})
            if record_replacement:
                report["global"]["records_with_replacement_chars"] += 1
                if len(report["examples"]["replacement_chars"]) < 10:
                    report["examples"]["replacement_chars"].append({"file": relative, "record": number, "source_id": rid})
            if record_boundary:
                report["global"]["records_with_boundary_whitespace"] += 1
            if record_noise:
                report["global"]["records_with_ocr_noise"] += 1
                if len(report["examples"]["ocr_noise"]) < 10:
                    report["examples"]["ocr_noise"].append({"file": relative, "record": number, "source_id": rid})

            question_type = normalized_text(record.get("question_type", record.get("task_type", "")))
            report["global"]["question_types"][question_type] += 1
            report["global"]["answer_types"][type(record.get("answer")).__name__] += 1
            subject = record.get("subject")
            if subject is None and isinstance(record.get("category_id"), list) and record["category_id"]:
                subject = record["category_id"][0]
            report["global"]["subjects"][str(subject)] += 1
            report["global"]["languages"][str(record.get("language", "<missing>"))] += 1
            if subject == "ch" and len(report["examples"]["subject_ch"]) < 10:
                report["examples"]["subject_ch"].append({"file": relative, "record": number, "source_id": rid, "question": record.get("question", "")})

            if question_type == "multiple_choice":
                section = report["multiple_choice"]
                section["records"] += 1
                options = record.get("options")
                if isinstance(options, list) and options:
                    section["structured_options_nonempty"] += 1
                    labels = [str(x.get("op_idx", "")) for x in options if isinstance(x, dict)]
                    if len(labels) != len(set(labels)) or any(not re.fullmatch(r"[A-H]", label, re.I) for label in labels):
                        if len(report["examples"]["option_anomalies"]) < 20:
                            report["examples"]["option_anomalies"].append({"file": relative, "record": number, "source_id": rid, "labels": labels})
                else:
                    section["structured_options_empty"] += 1
                    question = str(record.get("question", ""))
                    line_labels = OPTION_LINE_RE.findall(question)
                    any_labels = OPTION_ANY_RE.findall(question)
                    section["embedded_line_labels"]["".join(line_labels)] += 1
                    section["embedded_anywhere_labels"]["".join(any_labels)] += 1
                    section["has_answer_prompt_suffix"] += bool(ANSWER_PROMPT_RE.search(question))
                    section["parse_candidate_four_plus_line_labels"] += len(line_labels) >= 4
                    section["parse_candidate_four_plus_anywhere_labels"] += len(any_labels) >= 4
                    if len(line_labels) < 4 and len(section["examples_without_four_line_labels"]) < 30:
                        section["examples_without_four_line_labels"].append({
                            "file": relative, "record": number, "source_id": rid,
                            "line_labels": line_labels, "any_labels": any_labels,
                            "question": question[:1000],
                        })
        report["files"][relative] = {
            "records": file_report["records"],
            "schema_signatures": dict(file_report["schema_signatures"].most_common()),
            "field_presence": dict(file_report["field_presence"].most_common()),
            "field_types": {key: dict(value) for key, value in sorted(file_report["field_types"].items())},
            "empty_fields": dict(file_report["empty_fields"].most_common()),
        }

    global_report = report["global"]
    for key in ("field_presence", "empty_fields", "question_types", "answer_types", "subjects", "languages"):
        global_report[key] = dict(global_report[key].most_common())
    global_report["field_types"] = {key: dict(value) for key, value in sorted(global_report["field_types"].items())}
    for key in ("embedded_line_labels", "embedded_anywhere_labels"):
        report["multiple_choice"][key] = dict(report["multiple_choice"][key].most_common())
    output = root / "reports" / "structure_audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"global": report["global"], "multiple_choice": {k: v for k, v in report["multiple_choice"].items() if not k.startswith("examples")}}, ensure_ascii=False, indent=2))
    print(f"Report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

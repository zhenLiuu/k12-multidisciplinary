#!/usr/bin/env python3
"""Validate the K12 dataset release candidate with Python's standard library only.

The checker is intentionally read-only. It validates syntax and schema, summarizes
the corpus, detects exact duplicates and split leakage, and audits image references.
It never rewrites or filters source records.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any, Iterable


DECLARED_SUBJECTS = {"math", "physics", "biology", "geography", "chemistry"}
IMAGE_TOKEN_RE = re.compile(r"<img_?\d+>", re.IGNORECASE)
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
ANSWER_LABEL_RE = re.compile(r"^[\s\[\](){}]*([A-H](?:[\s,，、;/|+&]+[A-H])*)[\s\[\](){}.。]*$", re.IGNORECASE)


@dataclass(frozen=True)
class DatasetSpec:
    path: str
    split: str
    fmt: str
    id_fields: tuple[str, ...]
    required: dict[str, tuple[type, ...]]


SPECS = (
    DatasetSpec(
        path="data/raw/all_disciplines_with_idx.jsonl",
        split="raw",
        fmt="jsonl",
        id_fields=("idx",),
        required={
            "idx": (str,),
            "question_type": (str,),
            "question": (str,),
            "answer": (str,),
            "images": (list,),
            "subject": (str,),
        },
    ),
    DatasetSpec(
        path="data/raw/math_non_mc.jsonl",
        split="raw",
        fmt="jsonl",
        id_fields=("idx",),
        required={
            "idx": (str,),
            "question_type": (str,),
            "question": (str,),
            "answer": (str,),
            "images": (list,),
            "subject": (str,),
            "options": (list,),
        },
    ),
    DatasetSpec(
        path="data/raw/merge_multiple_choice.jsonl",
        split="raw",
        fmt="jsonl",
        id_fields=("idx",),
        required={
            "idx": (str,),
            "question_type": (str,),
            "question": (str,),
            "answer": (str,),
            "images": (list,),
            "subject": (str,),
            "options": (list,),
        },
    ),
    DatasetSpec(
        path="data/test/final_data_v8.2.json",
        split="test",
        fmt="json",
        id_fields=("id", "index"),
        required={
            "index": (int,),
            "id": (str,),
            "question": (str,),
            "answer": (list,),
            "task_type": (str,),
            "image": (list,),
            "category_id": (list,),
            "language": (str,),
        },
    ),
)


def type_name(types: tuple[type, ...]) -> str:
    return " or ".join(t.__name__ for t in types)


def compact(value: Any, limit: int = 180) -> str:
    text = repr(value).replace("\n", "\\n")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def normalized_text(value: Any) -> str:
    if isinstance(value, list):
        value = " ".join(str(x) for x in value)
    elif not isinstance(value, str):
        value = str(value)
    value = unicodedata.normalize("NFKC", value).strip().lower()
    return " ".join(value.split())


def answer_fingerprint(value: Any) -> str:
    return normalized_text(value)


def is_empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


class Audit:
    def __init__(self, root: Path, max_examples: int, check_image_files: bool, image_workers: int) -> None:
        self.root = root
        self.max_examples = max_examples
        self.check_image_files = check_image_files
        self.image_workers = image_workers
        self.issue_counts: Counter[str] = Counter()
        self.issue_severity: dict[str, str] = {}
        self.issue_descriptions: dict[str, str] = {}
        self.issue_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.file_issue_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.file_reports: list[dict[str, Any]] = []
        self.global_subjects: Counter[str] = Counter()
        self.global_question_types: Counter[str] = Counter()
        self.global_languages: Counter[str] = Counter()
        self.global_resources: Counter[str] = Counter()
        self.global_task_types: Counter[str] = Counter()
        self.total_records = 0
        self.records_with_images = 0
        self.total_image_references = 0
        self.absolute_image_references = 0
        self.embedded_image_references = 0
        self.unique_ids: dict[tuple[str, str], tuple[str, int, str]] = {}
        self.question_hashes: dict[str, tuple[str, int, str, str]] = {}
        self.exact_duplicate_questions = 0
        self.conflicting_duplicate_answers = 0
        self.raw_test_exact_overlaps = 0
        self.image_paths: set[str] = set()

    def issue(
        self,
        code: str,
        severity: str,
        description: str,
        file: str,
        record: int | None,
        detail: str,
    ) -> None:
        self.issue_counts[code] += 1
        self.file_issue_counts[file][code] += 1
        self.issue_severity[code] = severity
        self.issue_descriptions[code] = description
        if len(self.issue_examples[code]) < self.max_examples:
            self.issue_examples[code].append(
                {"file": file, "record": record, "detail": detail}
            )

    def audit_file(self, spec: DatasetSpec) -> None:
        path = self.root / spec.path
        report: dict[str, Any] = {
            "path": spec.path,
            "split": spec.split,
            "format": spec.fmt,
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": None,
            "records": 0,
            "valid_records": 0,
            "records_with_images": 0,
            "image_references": 0,
            "subjects": Counter(),
            "question_types": Counter(),
            "languages": Counter(),
            "resources": Counter(),
            "task_types": Counter(),
            "schema_signatures": Counter(),
            "field_types": defaultdict(Counter),
            "empty_fields": Counter(),
        }
        self.file_reports.append(report)
        if not path.is_file():
            self.issue("file_missing", "error", "Configured dataset file is missing.", spec.path, None, str(path))
            return

        if spec.fmt == "jsonl":
            records, digest = self.read_jsonl(path, spec)
        else:
            records, digest = self.read_json_array(path, spec)
        report["sha256"] = digest

        for record_number, obj in records:
            report["records"] += 1
            if not isinstance(obj, dict):
                self.issue(
                    "record_not_object",
                    "error",
                    "Every record must be a JSON object.",
                    spec.path,
                    record_number,
                    f"got {type(obj).__name__}",
                )
                continue
            report["valid_records"] += 1
            self.total_records += 1
            self.audit_record(spec, report, record_number, obj)

        for key in ("subjects", "question_types", "languages", "resources", "task_types", "schema_signatures", "empty_fields"):
            report[key] = dict(report[key].most_common())
        report["field_types"] = {
            field: dict(counts.most_common()) for field, counts in sorted(report["field_types"].items())
        }

    def read_jsonl(self, path: Path, spec: DatasetSpec) -> tuple[Iterable[tuple[int, Any]], str]:
        digest = hashlib.sha256()

        def iterator() -> Iterable[tuple[int, Any]]:
            try:
                with path.open("rb") as handle:
                    for line_number, raw in enumerate(handle, 1):
                        digest.update(raw)
                        if not raw.strip():
                            self.issue("blank_jsonl_line", "warning", "JSONL contains a blank line.", spec.path, line_number, "blank line")
                            continue
                        try:
                            text = raw.decode("utf-8")
                        except UnicodeDecodeError as exc:
                            self.issue("invalid_utf8", "error", "File contains invalid UTF-8.", spec.path, line_number, str(exc))
                            continue
                        try:
                            yield line_number, json.loads(text)
                        except json.JSONDecodeError as exc:
                            self.issue("invalid_json", "error", "A JSONL line is not valid JSON.", spec.path, line_number, str(exc))
            except OSError as exc:
                self.issue("file_read_error", "error", "Dataset file could not be read.", spec.path, None, str(exc))

        # The digest is finalized only after consuming this generator.
        stream = iterator()

        def digesting_iterator() -> Iterable[tuple[int, Any]]:
            yield from stream

        return digesting_iterator(), "__DEFERRED__"

    def read_json_array(self, path: Path, spec: DatasetSpec) -> tuple[Iterable[tuple[int, Any]], str]:
        try:
            raw = path.read_bytes()
        except OSError as exc:
            self.issue("file_read_error", "error", "Dataset file could not be read.", spec.path, None, str(exc))
            return (), ""
        digest = hashlib.sha256(raw).hexdigest()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            self.issue("invalid_utf8", "error", "File contains invalid UTF-8.", spec.path, None, str(exc))
            return (), digest
        try:
            root = json.loads(text)
        except json.JSONDecodeError as exc:
            self.issue("invalid_json", "error", "JSON file is not valid JSON.", spec.path, None, str(exc))
            return (), digest
        if not isinstance(root, list):
            self.issue("json_root_not_array", "error", "The test JSON root must be an array.", spec.path, None, f"got {type(root).__name__}")
            return (), digest
        return enumerate(root, 1), digest

    def audit_record(
        self,
        spec: DatasetSpec,
        report: dict[str, Any],
        record_number: int,
        obj: dict[str, Any],
    ) -> None:
        signature = ",".join(sorted(obj))
        report["schema_signatures"][signature] += 1
        for field, value in obj.items():
            report["field_types"][field][type(value).__name__] += 1
            if is_empty(value):
                report["empty_fields"][field] += 1

        for field, expected_types in spec.required.items():
            if field not in obj:
                self.issue("missing_required_field", "error", "A required field is absent.", spec.path, record_number, field)
                continue
            value = obj[field]
            if not isinstance(value, expected_types) or isinstance(value, bool) and int in expected_types:
                self.issue(
                    "wrong_field_type",
                    "error",
                    "A required field has the wrong JSON type.",
                    spec.path,
                    record_number,
                    f"{field}: expected {type_name(expected_types)}, got {type(value).__name__}",
                )
            if field in {"idx", "id", "question", "answer", "subject", "language", "task_type"} and is_empty(value):
                self.issue("empty_required_value", "error", "A semantically required value is empty.", spec.path, record_number, field)

        self.audit_ids(spec, record_number, obj)
        self.audit_question(spec, record_number, obj)
        self.audit_categories(spec, report, record_number, obj)
        self.audit_options(spec, record_number, obj)
        self.audit_images(spec, report, record_number, obj)
        self.audit_text_quality(spec, record_number, obj)

    def audit_ids(self, spec: DatasetSpec, record_number: int, obj: dict[str, Any]) -> None:
        for field in spec.id_fields:
            if field not in obj or is_empty(obj[field]):
                continue
            key = (field, normalized_text(obj[field]))
            previous = self.unique_ids.get(key)
            if previous:
                previous_file, previous_record, previous_split = previous
                code = "duplicate_id_within_file" if previous_file == spec.path else "duplicate_id_across_files"
                self.issue(
                    code,
                    "warning",
                    "An identifier is reused; public identifiers should be globally unique.",
                    spec.path,
                    record_number,
                    f"{field}={compact(obj[field])}; first seen at {previous_file}:{previous_record} ({previous_split})",
                )
            else:
                self.unique_ids[key] = (spec.path, record_number, spec.split)

    def audit_question(self, spec: DatasetSpec, record_number: int, obj: dict[str, Any]) -> None:
        question = obj.get("question")
        if not isinstance(question, str) or not question.strip():
            return
        normalized = normalized_text(question)
        question_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        current_answer = answer_fingerprint(obj.get("answer", ""))
        previous = self.question_hashes.get(question_hash)
        if previous:
            previous_file, previous_record, previous_split, previous_answer = previous
            self.exact_duplicate_questions += 1
            self.issue(
                "exact_duplicate_question",
                "warning",
                "A normalized question is repeated.",
                spec.path,
                record_number,
                f"first seen at {previous_file}:{previous_record}",
            )
            if previous_answer != current_answer:
                self.conflicting_duplicate_answers += 1
                self.issue(
                    "duplicate_question_conflicting_answer",
                    "warning",
                    "The same normalized question has different answers.",
                    spec.path,
                    record_number,
                    f"first seen at {previous_file}:{previous_record}",
                )
            if previous_split != spec.split and {previous_split, spec.split} == {"raw", "test"}:
                self.raw_test_exact_overlaps += 1
                self.issue(
                    "raw_test_exact_overlap",
                    "error",
                    "An exact normalized question occurs in both raw and test splits.",
                    spec.path,
                    record_number,
                    f"raw/test match at {previous_file}:{previous_record}",
                )
        else:
            self.question_hashes[question_hash] = (spec.path, record_number, spec.split, current_answer)

    def audit_categories(
        self,
        spec: DatasetSpec,
        report: dict[str, Any],
        record_number: int,
        obj: dict[str, Any],
    ) -> None:
        subject = obj.get("subject")
        if isinstance(subject, str) and subject.strip():
            subject = subject.strip()
            report["subjects"][subject] += 1
            self.global_subjects[subject] += 1
            if subject not in DECLARED_SUBJECTS:
                self.issue("subject_outside_declared_scope", "warning", "Subject is outside the five declared disciplines.", spec.path, record_number, compact(subject))
        categories = obj.get("category_id")
        if isinstance(categories, list):
            for category in categories:
                if isinstance(category, str) and category.strip():
                    report["subjects"][category.strip()] += 1
                    self.global_subjects[category.strip()] += 1
                    if category.strip() not in DECLARED_SUBJECTS:
                        self.issue("subject_outside_declared_scope", "warning", "Category is outside the five declared disciplines.", spec.path, record_number, compact(category))
                else:
                    self.issue("invalid_category", "warning", "category_id entries should be non-empty strings.", spec.path, record_number, compact(category))

        for field, report_key, global_counter in (
            ("question_type", "question_types", self.global_question_types),
            ("language", "languages", self.global_languages),
            ("resource", "resources", self.global_resources),
            ("task_type", "task_types", self.global_task_types),
        ):
            value = obj.get(field)
            if isinstance(value, str) and value.strip():
                value = value.strip()
                report[report_key][value] += 1
                global_counter[value] += 1

    def audit_options(self, spec: DatasetSpec, record_number: int, obj: dict[str, Any]) -> None:
        question_type = obj.get("question_type")
        task_type = obj.get("task_type")
        is_mc = question_type == "multiple_choice" or task_type in {"multiple_choice", "multi_choice"}
        options = obj.get("options")
        if is_mc and not isinstance(options, list):
            self.issue("mc_options_not_structured", "warning", "Multiple-choice options are not stored in a structured options list.", spec.path, record_number, "options field absent or not a list")
            return
        if is_mc and isinstance(options, list) and not options:
            self.issue("mc_options_empty", "warning", "Multiple-choice record has an empty options list.", spec.path, record_number, "options=[]")
            return
        if question_type == "non_multiple_choice" and isinstance(options, list) and options:
            self.issue("non_mc_has_options", "warning", "Non-multiple-choice record unexpectedly has options.", spec.path, record_number, f"{len(options)} options")
        if not isinstance(options, list) or not options:
            return

        labels: list[str] = []
        for option in options:
            if not isinstance(option, dict):
                self.issue("invalid_option", "error", "Each option must be an object.", spec.path, record_number, compact(option))
                continue
            label = option.get("op_idx")
            text = option.get("op_text")
            if not isinstance(label, str) or not label.strip() or not isinstance(text, str) or not text.strip():
                self.issue("invalid_option", "error", "Each option needs non-empty op_idx and op_text strings.", spec.path, record_number, compact(option))
                continue
            labels.append(label.strip().upper())
        if len(labels) != len(set(labels)):
            self.issue("duplicate_option_label", "error", "Option labels within a question are duplicated.", spec.path, record_number, compact(labels))

        answer = obj.get("answer")
        if isinstance(answer, str):
            match = ANSWER_LABEL_RE.fullmatch(answer.strip())
            if match:
                answer_labels = set(re.findall(r"[A-H]", match.group(1).upper()))
                unknown = answer_labels - set(labels)
                if unknown:
                    self.issue("answer_not_in_options", "error", "Answer label does not exist in the structured options.", spec.path, record_number, f"answer={compact(answer)}, labels={labels}")

    def audit_images(
        self,
        spec: DatasetSpec,
        report: dict[str, Any],
        record_number: int,
        obj: dict[str, Any],
    ) -> None:
        field = "images" if "images" in obj else "image" if "image" in obj else None
        if field is None:
            return
        images = obj[field]
        if not isinstance(images, list):
            return
        if images:
            report["records_with_images"] += 1
            self.records_with_images += 1

        def add_path(path_value: Any) -> None:
            if not isinstance(path_value, str) or not path_value.strip():
                self.issue("invalid_image_entry", "error", "Image path must be a non-empty string.", spec.path, record_number, compact(path_value))
                return
            path = path_value.strip()
            report["image_references"] += 1
            self.total_image_references += 1
            self.image_paths.add(path)
            if os.path.isabs(path):
                self.absolute_image_references += 1
                self.issue("absolute_image_path", "error", "Absolute image paths are not portable for a public release.", spec.path, record_number, path)
            if ".." in Path(path).parts:
                self.issue("image_path_traversal", "error", "Image path contains a parent-directory traversal.", spec.path, record_number, path)

        def add_embedded_image(encoded: str) -> bool:
            payload = encoded.strip()
            if payload.startswith("data:image/") and "," in payload:
                payload = payload.split(",", 1)[1]
            if len(payload) < 100 or re.fullmatch(r"[A-Za-z0-9+/=\r\n]+", payload) is None:
                return False
            try:
                decoded = base64.b64decode(payload, validate=True)
            except (binascii.Error, ValueError):
                self.issue("invalid_base64_image", "error", "Embedded image is not valid base64.", spec.path, record_number, f"length={len(payload)}")
                return True
            known_magic = (
                decoded.startswith(b"\xff\xd8\xff")
                or decoded.startswith(b"\x89PNG\r\n\x1a\n")
                or decoded.startswith((b"GIF87a", b"GIF89a"))
                or decoded.startswith(b"RIFF") and decoded[8:12] == b"WEBP"
            )
            if not known_magic:
                self.issue("unknown_embedded_image_format", "warning", "Embedded image has an unrecognized file signature.", spec.path, record_number, f"decoded bytes={len(decoded)}")
            report["image_references"] += 1
            self.total_image_references += 1
            self.embedded_image_references += 1
            return True

        for image in images:
            if isinstance(image, dict):
                path = image.get("path")
                if isinstance(path, list):
                    self.issue("image_path_is_list", "warning", "Image objects use both string and list path schemas; normalize before release.", spec.path, record_number, f"list length={len(path)}")
                    for path_item in path:
                        add_path(path_item)
                else:
                    add_path(path)
            elif isinstance(image, str):
                if not add_embedded_image(image):
                    add_path(image)
            else:
                self.issue("invalid_image_entry", "error", "Image entry must be an object or path string.", spec.path, record_number, compact(image))

        combined_text = " ".join(str(obj.get(key, "")) for key in ("question", "explanation"))
        token_count = len(IMAGE_TOKEN_RE.findall(combined_text))
        if token_count and not images:
            self.issue("unresolved_image_token", "warning", "Text contains image placeholders but the image list is empty.", spec.path, record_number, f"{token_count} image token(s)")

    def audit_text_quality(self, spec: DatasetSpec, record_number: int, obj: dict[str, Any]) -> None:
        for field in ("question", "explanation"):
            value = obj.get(field)
            if not isinstance(value, str):
                continue
            if "\ufffd" in value:
                self.issue("unicode_replacement_character", "warning", "Text contains the Unicode replacement character, suggesting decoding loss.", spec.path, record_number, field)
            if CONTROL_CHAR_RE.search(value):
                self.issue("control_character", "warning", "Text contains an unexpected control character.", spec.path, record_number, field)
            if value and value != value.strip():
                self.issue("surrounding_whitespace", "info", "Text has leading or trailing whitespace.", spec.path, record_number, field)

    def audit_image_files(self) -> dict[str, Any]:
        result = {"enabled": self.check_image_files, "unique_paths": len(self.image_paths), "existing": None, "missing": None}
        if not self.check_image_files:
            return result
        existing = 0
        missing = 0
        def check_one(path_text: str) -> tuple[str, bool]:
            path = Path(path_text)
            resolved = path if path.is_absolute() else self.root / path
            try:
                return path_text, resolved.is_file()
            except OSError:
                return path_text, False

        # Process bounded batches so hundreds of thousands of futures are not
        # retained in memory at once. Threads help because remote stat calls are
        # I/O-bound; all issue aggregation remains on the main thread.
        path_iter = iter(sorted(self.image_paths))
        with ThreadPoolExecutor(max_workers=self.image_workers) as executor:
            while batch := list(islice(path_iter, 4096)):
                for path_text, is_file in executor.map(check_one, batch):
                    if is_file:
                        existing += 1
                    else:
                        missing += 1
                        self.issue("image_file_missing", "error", "Referenced image file does not exist at audit time.", "(image inventory)", None, path_text)
        result["existing"] = existing
        result["missing"] = missing
        return result

    def finalize_deferred_hashes(self) -> None:
        # JSONL hashes are recalculated here because the record generator must be
        # consumed before its incremental digest is complete. This keeps parsing
        # streaming and memory bounded, at the cost of one sequential hash pass.
        for report in self.file_reports:
            if report["format"] != "jsonl" or report["sha256"] != "__DEFERRED__":
                continue
            path = self.root / report["path"]
            digest = hashlib.sha256()
            try:
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                report["sha256"] = digest.hexdigest()
            except OSError:
                report["sha256"] = None

    def result(self, image_inventory: dict[str, Any]) -> dict[str, Any]:
        issues = []
        for code, count in sorted(
            self.issue_counts.items(),
            key=lambda item: ({"error": 0, "warning": 1, "info": 2}.get(self.issue_severity[item[0]], 3), -item[1], item[0]),
        ):
            issues.append(
                {
                    "code": code,
                    "severity": self.issue_severity[code],
                    "count": count,
                    "description": self.issue_descriptions[code],
                    "examples": self.issue_examples[code],
                }
            )
        severity_counts = Counter()
        for issue in issues:
            severity_counts[issue["severity"]] += issue["count"]
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_root": ".",
            "summary": {
                "files_configured": len(SPECS),
                "files_present": sum(1 for r in self.file_reports if r["exists"]),
                "records": self.total_records,
                "records_with_images": self.records_with_images,
                "records_with_images_ratio": round(self.records_with_images / self.total_records, 6) if self.total_records else 0,
                "image_references": self.total_image_references,
                "absolute_image_references": self.absolute_image_references,
                "embedded_image_references": self.embedded_image_references,
                "exact_duplicate_questions": self.exact_duplicate_questions,
                "duplicate_questions_with_conflicting_answers": self.conflicting_duplicate_answers,
                "raw_test_exact_overlaps": self.raw_test_exact_overlaps,
                "issue_occurrences_by_severity": dict(severity_counts),
            },
            "distributions": {
                "subjects": dict(self.global_subjects.most_common()),
                "question_types": dict(self.global_question_types.most_common()),
                "languages": dict(self.global_languages.most_common()),
                "resources": dict(self.global_resources.most_common()),
                "task_types": dict(self.global_task_types.most_common()),
            },
            "files": self.file_reports,
            "issues_by_file": {
                file: dict(counts.most_common())
                for file, counts in sorted(self.file_issue_counts.items())
            },
            "image_inventory": image_inventory,
            "issues": issues,
            "manual_release_checks": [
                "Confirm redistribution rights and licenses for every upstream source and every image.",
                "Document provenance at record level; a resource/source field is currently not universal.",
                "Run privacy and sensitive-content review, including names, contact details, faces, and document metadata.",
                "Detect near-duplicates and paraphrases across raw/test splits; this checker only detects normalized exact matches.",
                "Sample-check answer correctness, explanation correctness, OCR quality, and subject/grade labels.",
                "Define a stable public schema and migrate absolute image paths to packaged relative paths.",
                "Decide whether explanations/CoT may be redistributed and whether they belong in the public release.",
                "Add dataset license, source licenses, citation, versioning, changelog, and removal/contact policy.",
            ],
        }


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root, help="Dataset root (default: parent of scripts/).")
    parser.add_argument("--output", type=Path, default=None, help="JSON report path (default: ROOT/reports/data_check_report.json).")
    parser.add_argument("--max-examples", type=int, default=5, help="Maximum examples retained per issue type.")
    parser.add_argument("--check-image-files", action="store_true", help="Stat every unique referenced image path; may be slow on remote storage.")
    parser.add_argument("--image-workers", type=int, default=32, help="Concurrent workers for --check-image-files (default: 32).")
    parser.add_argument("--fail-on", choices=("never", "error", "warning"), default="never", help="Choose when the command exits non-zero.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output = args.output or root / "reports" / "data_check_report.json"
    if not output.is_absolute():
        output = root / output
    audit = Audit(root, max(0, args.max_examples), args.check_image_files, max(1, args.image_workers))
    for spec in SPECS:
        audit.audit_file(spec)
    audit.finalize_deferred_hashes()
    image_inventory = audit.audit_image_files()
    result = audit.result(image_inventory)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = result["summary"]
    print(f"Checked {summary['records']:,} records across {summary['files_present']}/{summary['files_configured']} files.")
    print(f"Records with images: {summary['records_with_images']:,} ({summary['records_with_images_ratio']:.2%})")
    print(f"Issue occurrences: {summary['issue_occurrences_by_severity']}")
    print(f"Report: {output}")

    severities = result["summary"]["issue_occurrences_by_severity"]
    if args.fail_on == "error" and severities.get("error", 0):
        return 1
    if args.fail_on == "warning" and (severities.get("error", 0) or severities.get("warning", 0)):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

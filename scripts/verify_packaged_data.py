#!/usr/bin/env python3
"""Verify processed image references and content-addressed packaged files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any, Iterable


IMAGE_TOKEN_RE = re.compile(r"<img_?\d+>", re.IGNORECASE)
HASH_NAME_RE = re.compile(r"^[0-9a-f]{64}\.(?:jpg|png|gif|webp|bmp|tif)$")
SPECS = (
    ("data/processed/raw/all_disciplines_with_idx.jsonl", "jsonl", "images", "raw"),
    ("data/processed/raw/math_non_mc.jsonl", "jsonl", "images", "raw"),
    ("data/processed/raw/merge_multiple_choice.jsonl", "jsonl", "images", "raw"),
    ("data/processed/test/final_data_v8.2.json", "json", "images", "raw"),
)


def batched(values: Iterable[Any], size: int) -> Iterable[list[Any]]:
    iterator = iter(values)
    while batch := list(islice(iterator, size)):
        yield batch


def iter_records(path: Path, fmt: str) -> Iterable[tuple[int, Any]]:
    if fmt == "jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                yield number, json.loads(line)
    else:
        root = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(root, list):
            raise ValueError(f"{path}: JSON root must be an array")
        yield from enumerate(root, 1)


class Verifier:
    def __init__(self, root: Path, workers: int, max_examples: int, content_hashes: bool) -> None:
        self.root = root
        self.workers = workers
        self.max_examples = max_examples
        self.content_hashes = content_hashes
        self.references: set[str] = set()
        self.actual: dict[str, Path] = {}
        self.issues: Counter[str] = Counter()
        self.examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.file_results: list[dict[str, Any]] = []

    def issue(self, code: str, detail: dict[str, Any]) -> None:
        self.issues[code] += 1
        if len(self.examples[code]) < self.max_examples:
            self.examples[code].append(detail)

    def validate_path(self, value: Any, file: str, record: int) -> None:
        if not isinstance(value, str) or not value:
            self.issue("invalid_packaged_path", {"file": file, "record": record, "value_type": type(value).__name__})
            return
        if os.path.isabs(value):
            self.issue("absolute_path_remains", {"file": file, "record": record, "path": value})
            return
        path = Path(value)
        if len(path.parts) != 3 or path.parts[0] != "images" or path.parts[1] != path.name[:2] or not HASH_NAME_RE.fullmatch(path.name):
            self.issue("noncanonical_packaged_path", {"file": file, "record": record, "path": value})
            return
        self.references.add(path.as_posix())

    def validate_processed(self) -> None:
        for relative, fmt, image_field, role in SPECS:
            path = self.root / relative
            records = 0
            image_records = 0
            image_references = 0
            if not path.is_file():
                self.issue("processed_file_missing", {"file": relative})
                continue
            try:
                iterator = iter_records(path, fmt)
                for number, record in iterator:
                    records += 1
                    if not isinstance(record, dict):
                        self.issue("processed_record_not_object", {"file": relative, "record": number})
                        continue
                    images = record.get(image_field)
                    if not isinstance(images, list):
                        self.issue("processed_image_field_not_list", {"file": relative, "record": number})
                        continue
                    if images:
                        image_records += 1
                    text = " ".join(str(record.get(field, "")) for field in ("question", "explanation"))
                    if IMAGE_TOKEN_RE.search(text) and not images:
                        self.issue("unresolved_image_token", {"file": relative, "record": number})
                    for image in images:
                        image_references += 1
                        if role == "raw":
                            if not isinstance(image, dict) or set(["path"]) - set(image):
                                self.issue("invalid_processed_image_object", {"file": relative, "record": number})
                                continue
                            if isinstance(image.get("path"), list):
                                self.issue("image_path_list_remains", {"file": relative, "record": number})
                                continue
                            self.validate_path(image.get("path"), relative, number)
                        else:
                            self.validate_path(image, relative, number)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                self.issue("processed_parse_error", {"file": relative, "error": f"{type(exc).__name__}: {exc}"})
            self.file_results.append(
                {
                    "path": relative,
                    "records": records,
                    "records_with_images": image_records,
                    "image_references": image_references,
                    "bytes": path.stat().st_size,
                }
            )

    def inventory_files(self) -> None:
        root = self.root / "images"
        if not root.is_dir():
            self.issue("images_directory_missing", {"path": "images"})
            return
        with os.scandir(root) as shard_entries:
            for shard in shard_entries:
                relative_shard = f"images/{shard.name}"
                if shard.name.startswith("."):
                    if shard.is_dir(follow_symlinks=False):
                        with os.scandir(shard.path) as hidden_entries:
                            if next(hidden_entries, None) is not None:
                                self.issue("nonempty_staging_directory", {"path": relative_shard})
                    continue
                if not shard.is_dir(follow_symlinks=False) or not re.fullmatch(r"[0-9a-f]{2}", shard.name):
                    self.issue("unexpected_images_entry", {"path": relative_shard})
                    continue
                with os.scandir(shard.path) as image_entries:
                    for entry in image_entries:
                        relative = f"images/{shard.name}/{entry.name}"
                        if not entry.is_file(follow_symlinks=False):
                            self.issue("unexpected_images_entry", {"path": relative})
                            continue
                        if not HASH_NAME_RE.fullmatch(entry.name) or entry.name[:2] != shard.name:
                            self.issue("noncanonical_image_filename", {"path": relative})
                            continue
                        self.actual[relative] = Path(entry.path)

    @staticmethod
    def hash_one(item: tuple[str, Path]) -> tuple[str, str | None, str | None]:
        relative, path = item
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return relative, digest.hexdigest(), None
        except OSError as exc:
            return relative, None, f"{type(exc).__name__}: {exc}"

    def verify_hashes(self) -> None:
        if not self.content_hashes:
            return
        completed = 0
        items = self.actual.items()
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            for batch in batched(items, 4096):
                for relative, digest, error in executor.map(self.hash_one, batch):
                    completed += 1
                    if error:
                        self.issue("packaged_image_read_error", {"path": relative, "error": error})
                    elif digest != Path(relative).stem:
                        self.issue("packaged_image_hash_mismatch", {"path": relative, "actual_sha256": digest})
                if completed % 50_000 < 4096:
                    print(f"  verified hashes: {completed:,}/{len(self.actual):,}", flush=True)

    def run(self) -> dict[str, Any]:
        print("Validating processed records...", flush=True)
        self.validate_processed()
        print("Inventorying packaged images...", flush=True)
        self.inventory_files()
        missing = self.references - set(self.actual)
        orphan = set(self.actual) - self.references
        for path in sorted(missing):
            self.issue("referenced_image_missing", {"path": path})
        for path in sorted(orphan):
            self.issue("orphan_packaged_image", {"path": path})
        print(f"Verifying {len(self.actual):,} content hashes...", flush=True)
        self.verify_hashes()
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_root": ".",
            "content_hash_verification_enabled": self.content_hashes,
            "summary": {
                "processed_records": sum(item["records"] for item in self.file_results),
                "processed_records_with_images": sum(item["records_with_images"] for item in self.file_results),
                "processed_image_references": sum(item["image_references"] for item in self.file_results),
                "unique_referenced_images": len(self.references),
                "packaged_image_files": len(self.actual),
                "missing_referenced_images": len(missing),
                "orphan_packaged_images": len(orphan),
                "issue_occurrences": sum(self.issues.values()),
            },
            "processed_files": self.file_results,
            "issues": dict(self.issues.most_common()),
            "issue_examples": self.examples,
        }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--skip-content-hashes", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    report_path = args.report or root / "reports" / "packaged_data_verification.json"
    if not report_path.is_absolute():
        report_path = root / report_path
    verifier = Verifier(root, max(1, args.workers), max(0, args.max_examples), not args.skip_content_hashes)
    report = verifier.run()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"Report: {report_path}", flush=True)
    return 1 if report["summary"]["issue_occurrences"] else 0


if __name__ == "__main__":
    sys.exit(main())

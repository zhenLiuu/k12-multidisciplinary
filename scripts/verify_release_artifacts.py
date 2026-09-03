#!/usr/bin/env python3
"""Verify checksums, Parquet round-trip fidelity and image tar/index closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

from build_release import RECORD_SCHEMA
from prepare_structure_fix import FILES
from structure_common import iter_records


METADATA_FIELDS = (
    "source_index",
    "source_question_type",
    "cot",
    "resource",
    "tag",
    "source_explanation_parts",
    "error",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_rows(root: Path, split: str) -> Iterable[dict[str, Any]]:
    for relative, fmt, expected_split in FILES:
        if expected_split != split:
            continue
        for _, record in iter_records(root / relative, fmt):
            normalized = dict(record)
            normalized["metadata"] = {key: record["metadata"].get(key) for key in METADATA_FIELDS}
            yield normalized


def parquet_rows(paths: list[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        file = pq.ParquetFile(path)
        for batch in file.iter_batches(batch_size=10_000):
            yield from batch.to_pylist()


def compare_round_trip(root: Path, release: Path, split: str, errors: Counter[str]) -> int:
    paths = sorted((release / "data").glob(f"{split}-*.parquet"))
    source = source_rows(root, split)
    packaged = parquet_rows(paths)
    count = 0
    while True:
        try:
            left = next(source)
        except StopIteration:
            left = None
        try:
            right = next(packaged)
        except StopIteration:
            right = None
        if left is None or right is None:
            if left is not None or right is not None:
                errors[f"{split}_round_trip_length"] += 1
            break
        count += 1
        if left != right:
            errors[f"{split}_round_trip_content"] += 1
            if errors[f"{split}_round_trip_content"] <= 3:
                print(f"  round-trip mismatch: {split} record {count}", flush=True)
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--release", type=Path, default=Path("dist/v0.1.0/huggingface"))
    args = parser.parse_args()
    root = args.root.resolve()
    release = args.release if args.release.is_absolute() else root / args.release
    errors: Counter[str] = Counter()

    print("Verifying SHA-256 manifest...", flush=True)
    checksum_entries = []
    for line in (release / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = release / relative
        if not path.is_file():
            errors["missing_artifact"] += 1
            continue
        if sha256(path) != digest:
            errors["checksum_mismatch"] += 1
        checksum_entries.append(relative)

    parquet_files = sorted((release / "data").glob("*.parquet"))
    parquet_rows_by_split: Counter[str] = Counter()
    for path in parquet_files:
        file = pq.ParquetFile(path)
        if file.schema_arrow != RECORD_SCHEMA:
            errors["parquet_schema_mismatch"] += 1
        split = path.name.split("-", 1)[0]
        parquet_rows_by_split[split] += file.metadata.num_rows

    print("Verifying Parquet round-trip fidelity...", flush=True)
    round_trip = {
        split: compare_round_trip(root, release, split, errors)
        for split in ("raw", "test")
    }

    print("Verifying image index and tar members...", flush=True)
    index = pq.read_table(release / "images/index.parquet").to_pylist()
    index_by_path = {item["path"]: item for item in index}
    if len(index_by_path) != len(index):
        errors["duplicate_image_index_path"] += len(index) - len(index_by_path)
    tar_members = set()
    tar_count = 0
    for path in sorted((release / "images").glob("*.tar")):
        relative_shard = f"images/{path.name}"
        with tarfile.open(path, "r:") as archive:
            for member in archive:
                tar_count += 1
                if not member.isfile() or member.name.startswith("/") or ".." in Path(member.name).parts:
                    errors["unsafe_tar_member"] += 1
                    continue
                if member.name in tar_members:
                    errors["duplicate_tar_member"] += 1
                tar_members.add(member.name)
                row = index_by_path.get(member.name)
                if row is None:
                    errors["tar_member_missing_from_index"] += 1
                    continue
                if row["shard"] != relative_shard:
                    errors["image_index_wrong_shard"] += 1
                if row["bytes"] != member.size:
                    errors["image_index_wrong_size"] += 1
                if row["sha256"] != Path(member.name).stem:
                    errors["image_index_wrong_sha256"] += 1
    errors["index_paths_missing_from_tar"] += len(set(index_by_path) - tar_members)
    errors["tar_paths_missing_from_index"] += len(tar_members - set(index_by_path))

    expected = {"raw": 735650, "test": 1051}
    for split, count in expected.items():
        if parquet_rows_by_split[split] != count:
            errors[f"{split}_record_count"] += 1
        if round_trip[split] != count:
            errors[f"{split}_round_trip_count"] += 1
    if len(index) != 416634 or tar_count != 416634:
        errors["image_count"] += 1

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "valid": not any(errors.values()),
        "release": str(release.relative_to(root)),
        "checksum_entries": len(checksum_entries),
        "parquet_files": len(parquet_files),
        "parquet_records": dict(parquet_rows_by_split),
        "round_trip_records": round_trip,
        "image_index_records": len(index),
        "tar_members": tar_count,
        "errors": dict(errors),
    }
    output = root / "reports/release_artifact_verification.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    github = root / "dist/v0.1.0/github"
    if github.is_dir():
        shutil.copyfile(output, github / "reports/release_artifact_verification.json")
        shutil.copyfile(release / "SHA256SUMS", github / "SHA256SUMS.huggingface")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

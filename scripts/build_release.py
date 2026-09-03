#!/usr/bin/env python3
"""Build deterministic Hugging Face release artifacts from the frozen snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from prepare_structure_fix import FILES
from structure_common import iter_records


ROW_GROUP_ROWS = 50_000
PARQUET_TARGET = 256 * 1024 * 1024
PARQUET_HARD_MAX = 512 * 1024 * 1024
TAR_TARGET = 1024 * 1024 * 1024
TAR_HARD_MAX = 1280 * 1024 * 1024
TAR_RECORD = 10_240

RECORD_SCHEMA = pa.schema([
    pa.field("id", pa.string(), nullable=False),
    pa.field("source_id", pa.string(), nullable=False),
    pa.field("source_file", pa.string(), nullable=False),
    pa.field("split", pa.string(), nullable=False),
    pa.field("question_type", pa.string(), nullable=False),
    pa.field("subject", pa.string(), nullable=False),
    pa.field("language", pa.string(), nullable=False),
    pa.field("question", pa.string(), nullable=False),
    pa.field("options", pa.list_(pa.struct([
        pa.field("label", pa.string(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
    ])), nullable=False),
    pa.field("answer", pa.list_(pa.string()), nullable=False),
    pa.field("explanation", pa.string(), nullable=False),
    pa.field("images", pa.list_(pa.struct([
        pa.field("path", pa.string(), nullable=False),
        pa.field("caption", pa.string(), nullable=False),
        pa.field("part_of", pa.string(), nullable=False),
    ])), nullable=False),
    pa.field("table", pa.list_(pa.struct([
        pa.field("caption", pa.string()),
        pa.field("path", pa.list_(pa.list_(pa.string()))),
        pa.field("part_of", pa.string(), nullable=False),
    ])), nullable=False),
    pa.field("sub_questions", pa.list_(pa.string()), nullable=False),
    pa.field("metadata", pa.struct([
        pa.field("source_index", pa.int64()),
        pa.field("source_question_type", pa.string()),
        pa.field("cot", pa.bool_()),
        pa.field("resource", pa.string()),
        pa.field("tag", pa.list_(pa.string())),
        pa.field("source_explanation_parts", pa.list_(pa.string())),
        pa.field("error", pa.string()),
    ]), nullable=False),
], metadata={b"dataset_version": b"0.1.0", b"schema_version": b"1.0.0"})

IMAGE_INDEX_SCHEMA = pa.schema([
    pa.field("path", pa.string(), nullable=False),
    pa.field("shard", pa.string(), nullable=False),
    pa.field("bytes", pa.int64(), nullable=False),
    pa.field("sha256", pa.string(), nullable=False),
])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def batches(values: Iterable[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def split_records(root: Path, split: str) -> Iterable[dict[str, Any]]:
    for relative, fmt, expected_split in FILES:
        if expected_split != split:
            continue
        for _, record in iter_records(root / relative, fmt):
            yield record


def compressed_row_group_size(group: pq.RowGroupMetaData) -> int:
    return sum(group.column(index).total_compressed_size for index in range(group.num_columns))


def plan_row_groups(metadata: pq.FileMetaData, target: int) -> list[list[int]]:
    groups: list[list[int]] = []
    current: list[int] = []
    current_size = 0
    for index in range(metadata.num_row_groups):
        size = compressed_row_group_size(metadata.row_group(index))
        if current and current_size + size > target:
            groups.append(current)
            current = []
            current_size = 0
        current.append(index)
        current_size += size
    if current:
        groups.append(current)
    return groups


def parquet_writer(path: Path) -> pq.ParquetWriter:
    return pq.ParquetWriter(
        path,
        RECORD_SCHEMA,
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
        data_page_version="2.0",
    )


def write_parquet_split(root: Path, output: Path, split: str) -> list[dict[str, Any]]:
    staging = output / f".{split}.staging.parquet"
    writer = parquet_writer(staging)
    count = 0
    try:
        for batch in batches(split_records(root, split), ROW_GROUP_ROWS):
            writer.write_table(pa.Table.from_pylist(batch, schema=RECORD_SCHEMA), row_group_size=len(batch))
            count += len(batch)
            print(f"  {split} staging: {count:,} records", flush=True)
    finally:
        writer.close()

    source = pq.ParquetFile(staging)
    plan = plan_row_groups(source.metadata, PARQUET_TARGET)
    width = 5
    artifacts = []
    for shard_index, row_groups in enumerate(plan):
        name = f"{split}-{shard_index:0{width}d}-of-{len(plan):0{width}d}.parquet"
        path = output / name
        writer = parquet_writer(path)
        shard_rows = 0
        try:
            for row_group in row_groups:
                table = source.read_row_group(row_group)
                writer.write_table(table, row_group_size=table.num_rows)
                shard_rows += table.num_rows
        finally:
            writer.close()
        size = path.stat().st_size
        if size > PARQUET_HARD_MAX:
            raise RuntimeError(f"Parquet hard limit exceeded: {path} ({size})")
        artifacts.append({"path": f"data/{name}", "kind": "parquet", "split": split, "records": shard_rows})
        print(f"  wrote {name}: {shard_rows:,} records, {size:,} bytes", flush=True)
    staging.unlink()
    if sum(item["records"] for item in artifacts) != count:
        raise RuntimeError(f"Parquet count mismatch for {split}")
    return artifacts


def tar_final_size(payload: int) -> int:
    return math.ceil((payload + 1024) / TAR_RECORD) * TAR_RECORD


def inventory_images(root: Path) -> list[tuple[str, Path, int]]:
    output = []
    for shard in sorted((root / "images").iterdir(), key=lambda item: item.name):
        if not shard.is_dir() or shard.name.startswith("."):
            continue
        for path in sorted(shard.iterdir(), key=lambda item: item.name):
            if path.is_file():
                output.append((path.relative_to(root).as_posix(), path, path.stat().st_size))
    return output


def plan_tar_shards(images: list[tuple[str, Path, int]]) -> list[list[tuple[str, Path, int]]]:
    shards = []
    current = []
    payload = 0
    for image in images:
        entry = 512 + math.ceil(image[2] / 512) * 512
        if current and tar_final_size(payload + entry) > TAR_TARGET:
            shards.append(current)
            current = []
            payload = 0
        current.append(image)
        payload += entry
    if current:
        shards.append(current)
    return shards


def write_image_shards(root: Path, output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    images = inventory_images(root)
    plan = plan_tar_shards(images)
    width = 5
    artifacts = []
    index_rows = []
    for shard_index, members in enumerate(plan):
        name = f"images-{shard_index:0{width}d}-of-{len(plan):0{width}d}.tar"
        path = output / name
        with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as archive:
            for relative, source, size in members:
                info = tarfile.TarInfo(relative)
                info.size = size
                info.mtime = 0
                info.mode = 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with source.open("rb") as handle:
                    archive.addfile(info, handle)
                index_rows.append({
                    "path": relative,
                    "shard": f"images/{name}",
                    "bytes": size,
                    "sha256": Path(relative).stem,
                })
        size = path.stat().st_size
        if size > TAR_HARD_MAX:
            raise RuntimeError(f"tar hard limit exceeded: {path} ({size})")
        artifacts.append({"path": f"images/{name}", "kind": "image_tar", "members": len(members)})
        print(f"  wrote {name}: {len(members):,} images, {size:,} bytes", flush=True)
    return artifacts, index_rows


def populate_file_metadata(base: Path, artifacts: list[dict[str, Any]]) -> None:
    for item in artifacts:
        path = base / item["path"]
        item["bytes"] = path.stat().st_size
        item["sha256"] = sha256(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("dist/v0.1.0/huggingface"))
    args = parser.parse_args()
    root = args.root.resolve()
    destination = args.output if args.output.is_absolute() else root / args.output
    if destination.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {destination}")
    output = destination.with_name(f".{destination.name}.building")
    if output.exists():
        raise SystemExit(f"Stale build directory exists; inspect or remove it first: {output}")
    (output / "data").mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(parents=True, exist_ok=True)

    print("Building Parquet splits...", flush=True)
    artifacts = []
    artifacts.extend(write_parquet_split(root, output / "data", "raw"))
    artifacts.extend(write_parquet_split(root, output / "data", "test"))

    print("Building image tar shards...", flush=True)
    image_artifacts, index_rows = write_image_shards(root, output / "images")
    artifacts.extend(image_artifacts)
    index_path = output / "images" / "index.parquet"
    pq.write_table(
        pa.Table.from_pylist(index_rows, schema=IMAGE_INDEX_SCHEMA),
        index_path,
        compression="zstd",
        compression_level=9,
        version="2.6",
        data_page_version="2.0",
    )
    artifacts.append({"path": "images/index.parquet", "kind": "image_index", "records": len(index_rows)})

    shutil.copyfile(root / "release/v0.1.0/HF_DATASET_CARD.md", output / "README.md")
    shutil.copyfile(root / "release/v0.1.0/record.schema.json", output / "record.schema.json")
    shutil.copyfile(root / "release/v0.1.0/release_spec.json", output / "release_spec.json")
    artifacts.extend([
        {"path": "README.md", "kind": "documentation"},
        {"path": "record.schema.json", "kind": "schema"},
        {"path": "release_spec.json", "kind": "release_spec"},
    ])
    populate_file_metadata(output, artifacts)
    artifacts.sort(key=lambda item: item["path"])
    manifest = {
        "dataset_version": "0.1.0",
        "schema_version": "1.0.0",
        "records": {"raw": 735650, "test": 1051, "total": 736701},
        "images": len(index_rows),
        "artifacts": artifacts,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checksummed = artifacts + [{
        "path": "manifest.json",
        "sha256": sha256(manifest_path),
        "bytes": manifest_path.stat().st_size,
    }]
    (output / "SHA256SUMS").write_text(
        "".join(f"{item['sha256']}  {item['path']}\n" for item in sorted(checksummed, key=lambda item: item["path"])),
        encoding="utf-8",
    )
    output.rename(destination)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output": str(destination.relative_to(root)),
        "artifacts": len(artifacts),
        "bytes": sum(item["bytes"] for item in artifacts),
        "parquet_shards": sum(item["kind"] == "parquet" for item in artifacts),
        "image_tar_shards": len(image_artifacts),
        "image_members": len(index_rows),
        "manifest": str((destination / "manifest.json").relative_to(root)),
    }
    report_path = root / "reports/release_build_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

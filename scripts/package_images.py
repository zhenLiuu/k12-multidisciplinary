#!/usr/bin/env python3
"""Package external/embedded images and build image-complete processed data.

The source files under data/raw, data/test, and data/train are never modified.
Valid images are verified with Pillow, deduplicated by SHA-256, and stored at
images/<sha-prefix>/<sha256>.<ext>. Records depending on a missing, malformed,
or undecodable image are omitted from data/processed, with counts in a report.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import io
import json
import os
import re
import sys
import threading
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


IMAGE_TOKEN_RE = re.compile(r"<img_?\d+>", re.IGNORECASE)
FORMAT_EXTENSIONS = {
    "JPEG": "jpg",
    "MPO": "jpg",
    "PNG": "png",
    "GIF": "gif",
    "WEBP": "webp",
    "BMP": "bmp",
    "TIFF": "tif",
}


@dataclass(frozen=True)
class DataSpec:
    input_path: str
    output_path: str
    fmt: str
    image_field: str
    source_base: str
    role: str


SPECS = (
    DataSpec(
        "data/raw/all_disciplines_with_idx.jsonl",
        "data/processed/raw/all_disciplines_with_idx.jsonl",
        "jsonl",
        "images",
        "../open_source_mllm",
        "raw",
    ),
    DataSpec(
        "data/raw/math_non_mc.jsonl",
        "data/processed/raw/math_non_mc.jsonl",
        "jsonl",
        "images",
        "../open_source_mllm",
        "raw",
    ),
    DataSpec(
        "data/raw/merge_multiple_choice.jsonl",
        "data/processed/raw/merge_multiple_choice.jsonl",
        "jsonl",
        "images",
        "../0309_data",
        "raw",
    ),
    DataSpec(
        "data/test/final_data_v8.2.json",
        "data/processed/test/final_data_v8.2.json",
        "json",
        "image",
        "../0309_data",
        "test",
    ),
)


def batched(values: Iterable[Any], size: int) -> Iterable[list[Any]]:
    iterator = iter(values)
    while batch := list(islice(iterator, size)):
        yield batch


def extract_base64(value: str) -> bytes | None:
    payload = value.strip()
    if payload.startswith("data:image/") and "," in payload:
        payload = payload.split(",", 1)[1]
    if len(payload) < 100 or re.fullmatch(r"[A-Za-z0-9+/=\r\n]+", payload) is None:
        return None
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        return None


def iter_records(path: Path, fmt: str) -> Iterable[tuple[int, Any]]:
    if fmt == "jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                try:
                    yield number, json.loads(line)
                except json.JSONDecodeError as exc:
                    yield number, {"__packaging_error__": f"invalid_json: {exc}"}
    else:
        root = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(root, list):
            raise ValueError(f"{path}: JSON root must be an array")
        yield from enumerate(root, 1)


class Packager:
    def __init__(self, root: Path, workers: int, max_examples: int) -> None:
        self.root = root
        self.workers = workers
        self.max_examples = max_examples
        self.external_refs: dict[tuple[str, str], tuple[DataSpec, str]] = {}
        self.embedded_payloads: dict[str, bytes] = {}
        self.assets: dict[tuple[str, str], dict[str, Any]] = {}
        self.embedded_assets: dict[str, dict[str, Any]] = {}
        self.content_assets: dict[str, dict[str, Any]] = {}
        self.collection_errors: Counter[str] = Counter()
        self.collection_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.filtered: dict[str, Counter[str]] = defaultdict(Counter)
        self.filtered_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.file_results: list[dict[str, Any]] = []
        self.write_locks = [threading.Lock() for _ in range(256)]
        self.images_root = root / "images"

    def example(self, target: dict[str, list[dict[str, Any]]], code: str, value: dict[str, Any]) -> None:
        if len(target[code]) < self.max_examples:
            target[code].append(value)

    def ref_key(self, spec: DataSpec, path: str) -> tuple[str, str]:
        return ("ABS" if os.path.isabs(path) else spec.input_path, path)

    def collect(self) -> None:
        print("Collecting image references...", flush=True)
        for spec in SPECS:
            path = self.root / spec.input_path
            for number, record in iter_records(path, spec.fmt):
                if not isinstance(record, dict) or "__packaging_error__" in record:
                    continue
                value = record.get(spec.image_field, [])
                if not isinstance(value, list):
                    self.collection_errors["image_field_not_list"] += 1
                    self.example(self.collection_examples, "image_field_not_list", {"file": spec.input_path, "record": number})
                    continue
                if spec.role == "test":
                    for encoded in value:
                        if not isinstance(encoded, str):
                            self.collection_errors["invalid_embedded_image"] += 1
                            self.example(self.collection_examples, "invalid_embedded_image", {"file": spec.input_path, "record": number})
                            continue
                        payload = extract_base64(encoded)
                        if payload is None:
                            self.collection_errors["invalid_embedded_image"] += 1
                            self.example(self.collection_examples, "invalid_embedded_image", {"file": spec.input_path, "record": number})
                            continue
                        digest = hashlib.sha256(payload).hexdigest()
                        self.embedded_payloads.setdefault(digest, payload)
                    continue
                for item in value:
                    if not isinstance(item, dict):
                        self.collection_errors["invalid_image_object"] += 1
                        self.example(self.collection_examples, "invalid_image_object", {"file": spec.input_path, "record": number})
                        continue
                    path_value = item.get("path")
                    paths = path_value if isinstance(path_value, list) else [path_value]
                    for candidate in paths:
                        if not isinstance(candidate, str) or not candidate.strip():
                            self.collection_errors["invalid_image_path"] += 1
                            self.example(self.collection_examples, "invalid_image_path", {"file": spec.input_path, "record": number})
                            continue
                        candidate = candidate.strip()
                        key = self.ref_key(spec, candidate)
                        self.external_refs.setdefault(key, (spec, candidate))
        print(
            f"Collected {len(self.external_refs):,} unique external references and "
            f"{len(self.embedded_payloads):,} unique embedded payloads.",
            flush=True,
        )

    def resolve(self, spec: DataSpec, reference: str) -> Path | None:
        path = Path(reference)
        if path.is_absolute():
            candidates = (path,)
        else:
            candidates = (
                (self.root / path),
                (self.root / spec.source_base / path),
                (self.root.parent / path),
            )
        for candidate in candidates:
            try:
                if candidate.is_file():
                    return candidate.resolve()
            except OSError:
                continue
        return None

    def inspect_payload(self, payload: bytes, digest: str) -> dict[str, Any]:
        try:
            with Image.open(io.BytesIO(payload)) as image:
                image_format = image.format
                width, height = image.size
                frames = getattr(image, "n_frames", 1)
                image.verify()
            if image_format not in FORMAT_EXTENSIONS:
                raise ValueError(f"unsupported image format: {image_format}")
            with Image.open(io.BytesIO(payload)) as image:
                if image.mode == "P":
                    image = image.convert("RGBA")
                gray = image.convert("L")
                small = gray.resize((9, 8), Image.Resampling.LANCZOS)
                pixels = list(small.get_flattened_data())
                bits = [pixels[row * 9 + column] > pixels[row * 9 + column + 1] for row in range(8) for column in range(8)]
                dhash = 0
                for bit in bits:
                    dhash = (dhash << 1) | int(bit)
                audit_thumb = gray.copy()
                audit_thumb.thumbnail((64, 64), Image.Resampling.LANCZOS)
                low, high = audit_thumb.getextrema()
                near_uniform = high - low <= 2
            extension = FORMAT_EXTENSIONS[image_format]
            relative = Path("images") / digest[:2] / f"{digest}.{extension}"
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            lock = self.write_locks[int(digest[:2], 16)]
            with lock:
                if not destination.exists() or destination.stat().st_size != len(payload):
                    temporary = destination.with_name(f".{digest}.tmp")
                    temporary.write_bytes(payload)
                    os.replace(temporary, destination)
            return {
                "status": "ok",
                "path": relative.as_posix(),
                "sha256": digest,
                "bytes": len(payload),
                "format": image_format,
                "width": width,
                "height": height,
                "frames": frames,
                "dhash64": f"{dhash:016x}",
                "near_uniform": near_uniform,
                "large_image": width * height > 100_000_000,
            }
        except Exception as exc:
            return {"status": "audit_failed", "error": f"{type(exc).__name__}: {exc}"}

    def stage_stream(self, source: Path) -> dict[str, Any]:
        try:
            payload = source.read_bytes()
        except OSError as exc:
            return {"status": "read_failed", "error": f"{type(exc).__name__}: {exc}"}
        digest = hashlib.sha256(payload).hexdigest()
        return self.inspect_payload(payload, digest)

    def stage_bytes(self, payload: bytes) -> dict[str, Any]:
        digest = hashlib.sha256(payload).hexdigest()
        return self.inspect_payload(payload, digest)

    def process_external(self, item: tuple[tuple[str, str], tuple[DataSpec, str]]) -> tuple[tuple[str, str], dict[str, Any]]:
        key, (spec, reference) = item
        resolved = self.resolve(spec, reference)
        if resolved is None:
            return key, {"status": "missing"}
        result = self.stage_stream(resolved)
        result["source_bytes"] = result.get("bytes")
        return key, result

    def package(self) -> None:
        self.images_root.mkdir(parents=True, exist_ok=True)
        warnings.filterwarnings("error", category=Image.DecompressionBombWarning)
        Image.MAX_IMAGE_PIXELS = 100_000_000

        print(f"Packaging {len(self.external_refs):,} external image references...", flush=True)
        completed = 0
        items = self.external_refs.items()
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            for batch in batched(items, 2048):
                for key, result in executor.map(self.process_external, batch):
                    self.assets[key] = result
                    completed += 1
                if completed % 10_000 < 2048:
                    print(f"  external assets: {completed:,}/{len(self.external_refs):,}", flush=True)

        print(f"Packaging {len(self.embedded_payloads):,} embedded images...", flush=True)
        for digest, payload in self.embedded_payloads.items():
            result = self.stage_bytes(payload)
            self.embedded_assets[digest] = result

        for result in list(self.assets.values()) + list(self.embedded_assets.values()):
            if result.get("status") == "ok":
                self.content_assets.setdefault(result["sha256"], result)

    def transform_raw_images(self, spec: DataSpec, record: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        images = record.get(spec.image_field, [])
        if not isinstance(images, list):
            return None, "image_field_not_list"
        if not images:
            return record, None
        output: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        for item in images:
            if not isinstance(item, dict):
                return None, "invalid_image_object"
            value = item.get("path")
            paths = value if isinstance(value, list) else [value]
            for reference in paths:
                if not isinstance(reference, str) or not reference.strip():
                    return None, "invalid_image_path"
                result = self.assets.get(self.ref_key(spec, reference.strip()))
                if result is None:
                    return None, "unmapped_image"
                if result.get("status") != "ok":
                    return None, result.get("status", "image_failed")
                packaged_path = result["path"]
                if packaged_path in seen_paths:
                    continue
                seen_paths.add(packaged_path)
                new_item = {key: value for key, value in item.items() if key != "path"}
                new_item["path"] = packaged_path
                output.append(new_item)
        record[spec.image_field] = output
        return record, None

    def transform_test_images(self, spec: DataSpec, record: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        images = record.get(spec.image_field, [])
        if not isinstance(images, list):
            return None, "image_field_not_list"
        output: list[str] = []
        seen_paths: set[str] = set()
        for encoded in images:
            if not isinstance(encoded, str):
                return None, "invalid_embedded_image"
            payload = extract_base64(encoded)
            if payload is None:
                return None, "invalid_embedded_image"
            digest = hashlib.sha256(payload).hexdigest()
            result = self.embedded_assets.get(digest)
            if result is None or result.get("status") != "ok":
                return None, result.get("status", "embedded_image_failed") if result else "unmapped_embedded_image"
            if result["path"] not in seen_paths:
                output.append(result["path"])
                seen_paths.add(result["path"])
        record[spec.image_field] = output
        return record, None

    def transform(self, spec: DataSpec, number: int, record: Any) -> tuple[dict[str, Any] | None, str | None]:
        if not isinstance(record, dict):
            return None, "record_not_object"
        if "__packaging_error__" in record:
            return None, "invalid_json"
        text = " ".join(str(record.get(field, "")) for field in ("question", "explanation"))
        images = record.get(spec.image_field, [])
        if IMAGE_TOKEN_RE.search(text) and (not isinstance(images, list) or not images):
            return None, "unresolved_image_token"
        if spec.role == "test":
            return self.transform_test_images(spec, record)
        return self.transform_raw_images(spec, record)

    def write_processed(self) -> None:
        print("Writing processed datasets...", flush=True)
        for spec in SPECS:
            source = self.root / spec.input_path
            destination = self.root / spec.output_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".tmp")
            input_records = 0
            output_records = 0
            file_filtered: Counter[str] = Counter()

            if spec.fmt == "jsonl":
                with temporary.open("w", encoding="utf-8") as output:
                    for number, record in iter_records(source, spec.fmt):
                        input_records += 1
                        transformed, reason = self.transform(spec, number, record)
                        if reason:
                            file_filtered[reason] += 1
                            self.filtered[spec.input_path][reason] += 1
                            self.example(self.filtered_examples, reason, {"file": spec.input_path, "record": number})
                            continue
                        output.write(json.dumps(transformed, ensure_ascii=False, separators=(",", ":")) + "\n")
                        output_records += 1
            else:
                output_data: list[dict[str, Any]] = []
                for number, record in iter_records(source, spec.fmt):
                    input_records += 1
                    transformed, reason = self.transform(spec, number, record)
                    if reason:
                        file_filtered[reason] += 1
                        self.filtered[spec.input_path][reason] += 1
                        self.example(self.filtered_examples, reason, {"file": spec.input_path, "record": number})
                        continue
                    output_data.append(transformed)
                    output_records += 1
                temporary.write_text(json.dumps(output_data, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
            os.replace(temporary, destination)
            self.file_results.append(
                {
                    "input": spec.input_path,
                    "output": spec.output_path,
                    "input_records_or_lines": input_records,
                    "output_records": output_records,
                    "removed_records": input_records - output_records,
                    "removed_by_reason": dict(file_filtered.most_common()),
                    "output_bytes": destination.stat().st_size,
                    "output_sha256": self.hash_file(destination),
                }
            )
            print(f"  {spec.input_path}: {input_records:,} -> {output_records:,}", flush=True)

    @staticmethod
    def hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def report(self) -> dict[str, Any]:
        external_status = Counter(result.get("status", "unknown") for result in self.assets.values())
        embedded_status = Counter(result.get("status", "unknown") for result in self.embedded_assets.values())
        formats = Counter(asset["format"] for asset in self.content_assets.values())
        dhashes: dict[str, set[str]] = defaultdict(set)
        for digest, asset in self.content_assets.items():
            dhashes[asset["dhash64"]].add(digest)
        perceptual_groups = [members for members in dhashes.values() if len(members) > 1]
        ok_reference_results = [
            result for result in list(self.assets.values()) + list(self.embedded_assets.values())
            if result.get("status") == "ok"
        ]
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_root": ".",
            "policy": {
                "source_files_modified": False,
                "record_removal": "Remove a record if any declared image is missing, malformed, unreadable, or fails decode/verification; also remove records with an image token but an empty image list.",
                "image_naming": "images/<first-two-sha256-chars>/<sha256>.<detected-extension>",
                "exact_deduplication": "SHA-256 of image bytes",
                "perceptual_audit": "64-bit difference hash; collisions are review candidates, not automatic removals",
            },
            "inventory": {
                "unique_external_references": len(self.external_refs),
                "unique_embedded_payloads": len(self.embedded_payloads),
                "external_status": dict(external_status.most_common()),
                "embedded_status": dict(embedded_status.most_common()),
                "valid_reference_assets": len(ok_reference_results),
                "unique_packaged_content_files": len(self.content_assets),
                "exact_duplicate_reference_assets": len(ok_reference_results) - len(self.content_assets),
                "packaged_unique_bytes": sum(asset["bytes"] for asset in self.content_assets.values()),
                "formats": dict(formats.most_common()),
                "near_uniform_unique_images": sum(1 for asset in self.content_assets.values() if asset["near_uniform"]),
                "large_unique_images_over_100mp": sum(1 for asset in self.content_assets.values() if asset["large_image"]),
                "perceptual_hash_collision_groups": len(perceptual_groups),
                "unique_images_in_perceptual_collision_groups": sum(len(group) for group in perceptual_groups),
            },
            "processed_files": self.file_results,
            "collection_errors": dict(self.collection_errors.most_common()),
            "collection_error_examples": self.collection_examples,
            "filtered_examples_by_reason": self.filtered_examples,
            "manual_follow_up": [
                "Review near-uniform images before deciding whether they are blank; sparse diagrams may be valid.",
                "Review perceptual-hash collision groups with a stronger image matcher before cross-split removal.",
                "Inspect watermarks and copyright marks; this script does not use OCR/logo recognition.",
                "Audit semantic question-image correspondence by sampling; successful decode does not prove the image is the correct one.",
            ],
        }
        return report


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    report_path = args.report or root / "reports" / "image_packaging_report.json"
    if not report_path.is_absolute():
        report_path = root / report_path
    packager = Packager(root, max(1, args.workers), max(0, args.max_examples))
    packager.collect()
    packager.package()
    packager.write_processed()
    report = packager.report()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    inventory = report["inventory"]
    print(
        f"Packaged {inventory['unique_packaged_content_files']:,} unique valid images; "
        f"processed report: {report_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

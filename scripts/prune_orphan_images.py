#!/usr/bin/env python3
"""Remove only unreferenced content-addressed images from the processed package."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from verify_packaged_data import Verifier


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report", type=Path, help="Report path relative to root")
    args = parser.parse_args()
    root = args.root.resolve()
    verifier = Verifier(root, workers=1, max_examples=20, content_hashes=False)
    verifier.validate_processed()
    verifier.inventory_files()
    missing = verifier.references - set(verifier.actual)
    if missing or verifier.issues:
        raise SystemExit(f"Refusing prune: missing={len(missing)}, validation_issues={dict(verifier.issues)}")
    orphan = sorted(set(verifier.actual) - verifier.references)
    files = []
    for relative in orphan:
        path = verifier.actual[relative]
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest(path)})
    if args.apply:
        for item in files:
            path = (root / item["path"]).resolve()
            if root not in path.parents or not path.is_file():
                raise SystemExit(f"Unsafe or missing target: {path}")
            path.unlink()
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "applied" if args.apply else "plan",
        "orphan_images": len(files),
        "bytes": sum(item["bytes"] for item in files),
        "files": files,
        "source_data_modified": False,
    }
    output = root / (args.report or Path("reports/duplicate_resolution/orphan_image_prune.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

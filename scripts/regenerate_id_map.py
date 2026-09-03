#!/usr/bin/env python3
"""Rebuild the public source-to-ID map from the current processed snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from prepare_structure_fix import FILES
from structure_common import iter_records


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("reports/structure_fix/id_map.jsonl"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/final_schema_dedup/id_map_regeneration.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    report_path = args.report if args.report.is_absolute() else root / args.report
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")

    records = 0
    files: dict[str, int] = {}
    seen_ids: set[str] = set()
    duplicate_ids = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for relative, fmt, _ in FILES:
            count = 0
            for number, record in iter_records(root / relative, fmt):
                public_id = record.get("id")
                if public_id in seen_ids:
                    duplicate_ids += 1
                else:
                    seen_ids.add(public_id)
                handle.write(
                    json.dumps(
                        {
                            "file": relative,
                            "record": number,
                            "source_id": record.get("source_id"),
                            "id": public_id,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                count += 1
                records += 1
            files[relative] = count

    if duplicate_ids:
        temporary.unlink()
        raise SystemExit(f"Refusing to publish ID map: duplicate_ids={duplicate_ids}")
    os.replace(temporary, output)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "records": records,
        "unique_ids": len(seen_ids),
        "duplicate_ids": duplicate_ids,
        "files": files,
        "output": str(output.relative_to(root)),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

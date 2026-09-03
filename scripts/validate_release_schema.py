#!/usr/bin/env python3
"""Validate every processed record against the frozen release JSON Schema."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from jsonschema import Draft202012Validator

from prepare_structure_fix import FILES
from structure_common import iter_records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--version", default="v0.1.0")
    parser.add_argument("--max-examples", type=int, default=100)
    args = parser.parse_args()
    root = args.root.resolve()
    schema_path = root / "release" / args.version / "record.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    counts = Counter()
    files = Counter()
    examples = []

    for relative, fmt, _ in FILES:
        for number, record in iter_records(root / relative, fmt):
            counts["records"] += 1
            files[relative] += 1
            errors = sorted(validator.iter_errors(record), key=lambda item: list(item.path))
            if errors:
                counts["invalid_records"] += 1
                if len(examples) < args.max_examples:
                    examples.append({
                        "file": relative,
                        "record": number,
                        "errors": [
                            {
                                "path": "/".join(str(part) for part in error.path),
                                "message": error.message,
                            }
                            for error in errors[:20]
                        ],
                    })

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema": str(schema_path.relative_to(root)),
        "valid": counts["invalid_records"] == 0,
        "records": counts["records"],
        "invalid_records": counts["invalid_records"],
        "files": dict(files),
        "error_examples": examples,
    }
    output = root / "reports" / "release_schema_validation.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

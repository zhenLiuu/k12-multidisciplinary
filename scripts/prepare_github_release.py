#!/usr/bin/env python3
"""Create a small, self-contained GitHub repository bundle."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPORTS = (
    "final_freeze_check.json",
    "release_schema_validation.json",
    "release_build_report.json",
    "release_artifact_verification.json",
)


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("dist/v0.1.0/github"))
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    copy(root / "release/v0.1.0/GITHUB_README.md", output / "README.md")
    copy(root / "release/v0.1.0/GITHUB_GITIGNORE", output / ".gitignore")
    for name in ("CHANGELOG.md", "PUBLISHING.md", "requirements.txt"):
        copy(root / name, output / name)

    for source in sorted((root / "scripts").glob("*.py")):
        copy(source, output / "scripts" / source.name)
    for name in ("record.schema.json", "release_spec.json"):
        copy(root / "release/v0.1.0" / name, output / "release/v0.1.0" / name)
    for name in REPORTS:
        source = root / "reports" / name
        if not source.is_file():
            raise SystemExit(f"Required release report is missing: {source}")
        copy(source, output / "reports" / name)
    copy(root / "dist/v0.1.0/huggingface/SHA256SUMS", output / "SHA256SUMS.huggingface")

    files = sum(1 for path in output.rglob("*") if path.is_file())
    size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    print(f"GitHub bundle: {output}")
    print(f"Files: {files}; bytes: {size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

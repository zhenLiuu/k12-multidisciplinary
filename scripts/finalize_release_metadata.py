#!/usr/bin/env python3
"""Finalize repository identifiers and refresh release-document checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


REPO_ID = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_placeholders(path: Path, replacements: dict[str, str]) -> dict[str, int]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    counts = {old: text.count(old) for old in replacements}
    for old, new in replacements.items():
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")
    return counts


def refresh_hf_manifests(release: Path) -> None:
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    readme = release / "README.md"
    for artifact in manifest["artifacts"]:
        if artifact["path"] == "README.md":
            artifact["bytes"] = readme.stat().st_size
            artifact["sha256"] = sha256(readme)
            break
    else:
        raise RuntimeError("README.md is missing from manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    entries = list(manifest["artifacts"])
    entries.append({
        "path": "manifest.json",
        "bytes": manifest_path.stat().st_size,
        "sha256": sha256(manifest_path),
    })
    (release / "SHA256SUMS").write_text(
        "".join(
            f"{item['sha256']}  {item['path']}\n"
            for item in sorted(entries, key=lambda item: item["path"])
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--hf-repo", required=True, help="namespace/dataset")
    parser.add_argument("--github-repo", required=True, help="owner/repository")
    args = parser.parse_args()
    for label, value in (("HF", args.hf_repo), ("GitHub", args.github_repo)):
        if not REPO_ID.fullmatch(value):
            raise SystemExit(f"Invalid {label} repository identifier: {value}")

    root = args.root.resolve()
    hf_release = root / "dist/v0.1.0/huggingface"
    github_release = root / "dist/v0.1.0/github"
    if not (hf_release / "manifest.json").is_file():
        raise SystemExit(f"HF release is not built: {hf_release}")

    replacements = {
        "YOUR_NAMESPACE/YOUR_DATASET": args.hf_repo,
        "YOUR_NAMESPACE/YOUR_REPOSITORY": args.github_repo,
    }
    targets = [
        root / "release/v0.1.0/HF_DATASET_CARD.md",
        root / "release/v0.1.0/GITHUB_README.md",
        root / "PUBLISHING.md",
        hf_release / "README.md",
        github_release / "README.md",
        github_release / "PUBLISHING.md",
    ]
    totals = {key: 0 for key in replacements}
    for target in targets:
        for key, count in replace_placeholders(target, replacements).items():
            totals[key] += count
    if not all(totals.values()):
        raise SystemExit(f"Expected placeholders were not found: {totals}")

    refresh_hf_manifests(hf_release)
    if github_release.is_dir():
        shutil.copyfile(hf_release / "SHA256SUMS", github_release / "SHA256SUMS.huggingface")
    print({"hf_repo": args.hf_repo, "github_repo": args.github_repo, "replacements": totals})
    print("Run scripts/verify_release_artifacts.py again before upload.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

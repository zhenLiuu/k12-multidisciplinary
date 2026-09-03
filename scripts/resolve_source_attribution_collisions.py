#!/usr/bin/env python3
"""Resolve source-prefix cleanup collisions after the final deduplication pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from apply_duplicate_resolution import quality_score
from clean_source_attributions import strip_leading_attributions, write_record
from prepare_structure_fix import FILES
from structure_common import iter_records, public_id


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    report_dir = root / "reports" / "source_attribution_cleanup"
    previous = json.loads((report_dir / "application_report.json").read_text(encoding="utf-8"))
    targets = [item["removed_candidate"][0] for item in previous["skipped_collisions"]]

    records: dict[tuple[str, int], dict[str, Any]] = {}
    formats: dict[str, str] = {}
    by_id: dict[str, tuple[str, int]] = {}
    before: dict[str, dict[str, Any]] = {}
    for relative, fmt, _ in FILES:
        formats[relative] = fmt
        path = root / relative
        before[relative] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        for number, record in iter_records(path, fmt):
            locator = (relative, number)
            records[locator] = record
            if record["id"] in by_id:
                raise SystemExit(f"pre-existing duplicate ID: {record['id']}")
            by_id[record["id"]] = locator

    removals: set[tuple[str, int]] = set()
    replacements: dict[tuple[str, int], dict[str, Any]] = {}
    decisions = []
    for prefix in targets:
        matches = []
        cleaned_by_location: dict[tuple[str, int], str] = {}
        for loc, record in records.items():
            candidate_question, candidate_removed = strip_leading_attributions(record["question"])
            if candidate_removed == [prefix]:
                matches.append(loc)
                cleaned_by_location[loc] = candidate_question
        if not matches:
            decisions.append({"prefix": prefix, "status": "already_absent"})
            continue
        if len(matches) != 1:
            raise SystemExit(f"expected one current match for {prefix!r}, got {len(matches)}")
        tagged_loc = matches[0]
        tagged = records[tagged_loc]
        cleaned_question = cleaned_by_location[tagged_loc]
        _, removed = strip_leading_attributions(tagged["question"])
        if removed != [prefix] or not cleaned_question:
            raise SystemExit(f"unexpected cleaner result for {tagged_loc}: {removed!r}")
        cleaned = deepcopy(tagged)
        cleaned["question"] = cleaned_question
        cleaned["id"] = public_id(cleaned)
        untagged_loc = by_id.get(cleaned["id"])
        if untagged_loc is None or untagged_loc == tagged_loc:
            raise SystemExit(f"expected an existing collision for {tagged_loc}")
        untagged = records[untagged_loc]
        if untagged["question"] != cleaned_question:
            raise SystemExit(f"candidate question mismatch for {tagged_loc} and {untagged_loc}")
        excluded = {"id", "question", "explanation", "metadata"}
        differences = [key for key in tagged if key not in excluded and tagged[key] != untagged[key]]
        if differences:
            raise SystemExit(f"unsafe semantic differences for {tagged_loc}: {differences}")
        tagged_score = quality_score(cleaned)
        untagged_score = quality_score(untagged)
        if tagged_score > untagged_score:
            kept_loc, removed_loc = tagged_loc, untagged_loc
            replacements[tagged_loc] = cleaned
            action = "clean_tagged_keep_tagged"
        else:
            kept_loc, removed_loc = untagged_loc, tagged_loc
            action = "keep_existing_untagged"
        removals.add(removed_loc)
        decisions.append({
            "prefix": prefix,
            "status": "resolved",
            "action": action,
            "tagged": {"file": tagged_loc[0], "record": tagged_loc[1], "id": tagged["id"], "score": tagged_score},
            "untagged": {"file": untagged_loc[0], "record": untagged_loc[1], "id": untagged["id"], "score": untagged_score},
            "kept": {"file": kept_loc[0], "record": kept_loc[1]},
            "removed": {"file": removed_loc[0], "record": removed_loc[1]},
            "answer": tagged["answer"],
        })

    affected = sorted({loc[0] for loc in removals | set(replacements)})
    if args.apply:
        for relative in affected:
            path = root / relative
            fmt = formats[relative]
            temporary = path.with_name(path.name + ".source-collision.tmp")
            with temporary.open("w", encoding="utf-8") as output:
                first = True
                if fmt == "json":
                    output.write("[\n")
                for number, record in iter_records(path, fmt):
                    locator = (relative, number)
                    if locator in removals:
                        continue
                    first = write_record(output, replacements.get(locator, record), fmt, first)
                if fmt == "json":
                    output.write("\n]\n")
            os.replace(temporary, path)

    after = {
        relative: {"sha256": sha256(root / relative), "bytes": (root / relative).stat().st_size}
        for relative, _, _ in FILES
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "applied": args.apply,
        "selection_rule": "same deterministic quality_score used by final exact deduplication",
        "safety_rule": "question matches after prefix removal and every field except id/question/explanation/metadata is identical",
        "targeted_prefixes": len(targets),
        "already_absent": sum(item["status"] == "already_absent" for item in decisions),
        "resolved_pairs": sum(item["status"] == "resolved" for item in decisions),
        "removed_records": len(removals),
        "changed_kept_records": len(replacements),
        "decisions": decisions,
        "before": before,
        "after": after,
    }
    name = "final_collision_resolution.json" if args.apply else "final_collision_resolution_preview.json"
    (report_dir / name).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

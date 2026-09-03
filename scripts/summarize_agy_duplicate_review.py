#!/usr/bin/env python3
"""Validate coverage and summarize all agy duplicate-conflict decisions."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    report_dir = (args.report_dir or root / "reports" / "duplicate_resolution").resolve()
    manifest_items = [json.loads(line) for line in (report_dir / "agy_conflict_manifest.jsonl").open(encoding="utf-8")]
    manifest = {item["review_id"]: item for item in manifest_items}
    batch_root = report_dir / "agy_batches"
    batches = json.loads((batch_root / "index.json").read_text(encoding="utf-8"))
    decisions: dict[str, dict[str, Any]] = {}
    usage: Counter[str] = Counter()
    by_kind: dict[str, Counter[str]] = defaultdict(Counter)
    by_subject: dict[str, Counter[str]] = defaultdict(Counter)
    batch_summaries = []
    for batch in batches:
        path = batch_root / batch["batch"] / "validated_results.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("validation", {}).get("valid"):
            raise SystemExit(f"Invalid batch: {batch['batch']}")
        batch_counts: Counter[str] = Counter()
        for key, value in (payload.get("usage") or {}).items():
            if isinstance(value, int):
                usage[key] += value
        for result in payload["results"]:
            review_id = result["review_id"]
            if review_id in decisions:
                raise SystemExit(f"Duplicate result: {review_id}")
            if review_id not in manifest:
                raise SystemExit(f"Unexpected result: {review_id}")
            decisions[review_id] = result
            decision = result["decision"]
            batch_counts[decision] += 1
            by_kind[batch["kind"]][decision] += 1
            by_subject[str(manifest[review_id].get("subject", ""))][decision] += 1
        batch_summaries.append({
            "batch": batch["batch"], "kind": batch["kind"], "count": batch["count"],
            "decisions": dict(batch_counts), "usage": payload.get("usage"),
            "source_status": payload.get("source_status", "SUCCESS"),
        })
    if set(decisions) != set(manifest):
        missing = sorted(set(manifest) - set(decisions))
        raise SystemExit(f"Coverage mismatch: missing {len(missing)}")

    combined_path = report_dir / "agy_conflict_decisions.jsonl"
    confidences = []
    total_counts: Counter[str] = Counter()
    with combined_path.open("w", encoding="utf-8") as handle:
        for review_id in sorted(decisions):
            result = decisions[review_id]
            item = manifest[review_id]
            total_counts[result["decision"]] += 1
            confidences.append(float(result["confidence"]))
            handle.write(json.dumps({
                "review_id": review_id, "group_id": item["group_id"],
                "subject": item.get("subject", ""), "has_images": bool(item.get("images")),
                **result,
            }, ensure_ascii=False, separators=(",", ":")) + "\n")

    sorted_conf = sorted(confidences)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "gemini-3.7-flash-medium",
        "groups_reviewed": len(decisions),
        "coverage_complete": True,
        "decision_rule": "match keeps one best record carrying the matched existing answer; none/uncertain deletes the whole group",
        "decisions": dict(total_counts),
        "by_kind": {key: dict(value) for key, value in sorted(by_kind.items())},
        "by_subject": {key: dict(value) for key, value in sorted(by_subject.items())},
        "confidence": {
            "mean": statistics.fmean(confidences),
            "min": min(confidences),
            "p50": statistics.median(confidences),
            "p10": sorted_conf[int(0.10 * (len(sorted_conf) - 1))],
            "below_0_8": sum(value < 0.8 for value in confidences),
        },
        "usage_for_accepted_batches": dict(usage),
        "batches": batch_summaries,
        "combined_decisions": str(combined_path.relative_to(root)),
    }
    output = report_dir / "agy_review_summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

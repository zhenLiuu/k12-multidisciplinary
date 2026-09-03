#!/usr/bin/env python3
"""Run and validate the bounded agy structure review."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def structured(raw: dict[str, Any]) -> dict[str, Any] | None:
    value = raw.get("structured_output")
    if isinstance(value, dict):
        return value
    response = raw.get("response")
    if isinstance(response, str):
        try:
            value = json.loads(response)
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None
    return None


def validate(raw: dict[str, Any], index: dict[str, Any]) -> tuple[bool, str, list[dict[str, Any]]]:
    value = structured(raw)
    recoverable = raw.get("status") == "ERROR" and "exceeded the output token limit" in str(raw.get("error", "")) and value
    if raw.get("status") != "SUCCESS" and not recoverable:
        return False, f"agy status={raw.get('status')!r}: {raw.get('error')}", []
    results = value.get("results") if isinstance(value, dict) else None
    if not isinstance(results, list):
        return False, "missing results", []
    expected = index["review_ids"]
    actual = [item.get("review_id") for item in results if isinstance(item, dict)]
    if len(results) != len(expected) or set(actual) != set(expected) or len(set(actual)) != len(actual):
        return False, "review IDs/count mismatch", results
    required = {"review_id", "decision", "subject", "question", "options", "answer", "confidence", "repairs", "reason"}
    for item in results:
        if set(item) != required:
            return False, f"field mismatch for {item.get('review_id')}", results
        if item["decision"] not in {"keep", "delete", "uncertain"}:
            return False, f"bad decision for {item['review_id']}", results
        if item["subject"] not in {"math", "physics", "biology", "geography", "chemistry"}:
            return False, f"bad subject for {item['review_id']}", results
        if not isinstance(item["question"], str) or not isinstance(item["options"], list) or not isinstance(item["answer"], list):
            return False, f"bad content types for {item['review_id']}", results
        if not isinstance(item["repairs"], list) or not isinstance(item["reason"], str):
            return False, f"bad audit fields for {item['review_id']}", results
        confidence = item["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            return False, f"bad confidence for {item['review_id']}", results
        if item["decision"] == "keep" and "�" in json.dumps(item, ensure_ascii=False):
            return False, f"replacement character remains for {item['review_id']}", results
    return True, "ok", results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model", default="gemini-3.7-flash-medium")
    args = parser.parse_args()
    root = args.root.resolve()
    directory = root / "reports" / "structure_fix" / "agy_review"
    index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
    prompt = (root / index["prompt"]).read_text(encoding="utf-8")
    command = [
        "agy", "--log-file", str(directory / "agy_debug.log"), "--model", args.model,
        "--effort", "medium", "--mode", "plan", "--sandbox", "--dangerously-skip-permissions",
        "--disable-slash-commands", "--output-format", "json", "--json-schema", str(root / index["schema"]),
        "--print-timeout", "45m", "-p", prompt,
    ]
    started = datetime.now(timezone.utc).isoformat()
    completed = subprocess.run(command, cwd=root, text=True, capture_output=True)
    (directory / "agy_cli_output.json").write_text(completed.stdout, encoding="utf-8")
    (directory / "agy_stderr.log").write_text(completed.stderr, encoding="utf-8")
    try:
        raw = json.loads(completed.stdout)
        valid, detail, results = validate(raw, index)
    except Exception as exc:
        raw, valid, detail, results = {}, False, f"parse error: {exc}", []
    report = {
        "model": args.model, "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
        "returncode": completed.returncode, "validation": {"valid": valid, "detail": detail},
        "conversation_id": raw.get("conversation_id"), "usage": raw.get("usage"), "results": results,
    }
    (directory / "validated_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("model", "returncode", "validation", "usage")}, ensure_ascii=False, indent=2))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

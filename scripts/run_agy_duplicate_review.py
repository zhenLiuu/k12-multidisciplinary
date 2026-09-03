#!/usr/bin/env python3
"""Run and validate resumable agy duplicate-conflict review batches."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def structured_output(raw: dict[str, Any]) -> dict[str, Any] | None:
    structured = raw.get("structured_output")
    if isinstance(structured, dict) and isinstance(structured.get("results"), list):
        return structured
    response = raw.get("response")
    if isinstance(response, str) and response:
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict) and isinstance(parsed.get("results"), list):
            return parsed
    return None


def validate(raw: dict[str, Any], batch: dict[str, Any]) -> tuple[bool, str]:
    structured = structured_output(raw)
    recoverable_limit_error = (
        raw.get("status") == "ERROR"
        and "exceeded the output token limit" in str(raw.get("error", ""))
        and structured is not None
    )
    if raw.get("status") != "SUCCESS" and not recoverable_limit_error:
        return False, f"agy status={raw.get('status')!r}"
    if structured is None:
        return False, "missing structured_output.results"
    results = structured["results"]
    expected = batch["review_ids"]
    actual = [item.get("review_id") for item in results if isinstance(item, dict)]
    if len(results) != len(expected) or len(actual) != len(expected):
        return False, f"result count {len(results)} != {len(expected)}"
    if set(actual) != set(expected) or len(set(actual)) != len(actual):
        return False, "review IDs are missing, duplicated, or unexpected"
    input_items = json.loads((Path(batch["_root"]) / batch["input"]).read_text(encoding="utf-8"))
    candidate_ids = {
        item["review_id"]: {candidate["candidate_id"] for candidate in item["candidate_answers"]}
        for item in input_items
    }
    required = {"review_id", "decision", "matched_candidate_id", "solved_answer", "confidence", "reason"}
    for result in results:
        if not isinstance(result, dict) or set(result) != required:
            return False, "result fields do not match schema"
        if not all(isinstance(result[key], str) for key in ("review_id", "decision", "matched_candidate_id", "solved_answer", "reason")):
            return False, f"invalid string field types for {result.get('review_id')}"
        confidence = result.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            return False, f"invalid confidence for {result.get('review_id')}"
        decision = result.get("decision")
        matched = result.get("matched_candidate_id")
        if decision not in {"match", "none", "uncertain"}:
            return False, f"invalid decision for {result['review_id']}"
        if decision == "match" and matched not in candidate_ids[result["review_id"]]:
            return False, f"invalid candidate {matched!r} for {result['review_id']}"
        if decision != "match" and matched != "":
            return False, f"non-match has candidate for {result['review_id']}"
    return True, "ok"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--model", default="gemini-3.7-flash-medium")
    parser.add_argument("--batch", action="append", help="Run only named batch; repeatable")
    parser.add_argument("--kind", choices=("text", "image"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--revalidate-existing", action="store_true", help="Validate saved raw output without calling agy")
    args = parser.parse_args()
    root = args.root.resolve()
    report_dir = (args.report_dir or root / "reports" / "duplicate_resolution").resolve()
    batch_root = report_dir / "agy_batches"
    batches = json.loads((batch_root / "index.json").read_text(encoding="utf-8"))
    selected = [b for b in batches if (not args.batch or b["batch"] in args.batch) and (not args.kind or b["kind"] == args.kind)]
    failures = 0
    for batch in selected:
        batch_dir = batch_root / batch["batch"]
        raw_path = batch_dir / "agy_cli_output.json"
        validated_path = batch_dir / "validated_results.json"
        if args.revalidate_existing:
            if not raw_path.exists():
                print(f"FAIL {batch['batch']}: no saved raw output", flush=True)
                failures += 1
                continue
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            valid, detail = validate(raw, dict(batch, _root=str(root)))
            structured = structured_output(raw) or {}
            result = {
                "batch": batch["batch"], "model": args.model,
                "revalidated_at": datetime.now(timezone.utc).isoformat(),
                "source_status": raw.get("status"), "source_error": raw.get("error"),
                "validation": {"valid": valid, "detail": detail},
                "conversation_id": raw.get("conversation_id"), "usage": raw.get("usage"),
                "results": structured.get("results", []),
            }
            validated_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"REVALIDATE {batch['batch']}: valid={valid} detail={detail}", flush=True)
            failures += int(not valid)
            continue
        if validated_path.exists() and not args.force:
            existing = json.loads(validated_path.read_text(encoding="utf-8"))
            if existing.get("validation", {}).get("valid"):
                print(f"SKIP {batch['batch']}: already valid", flush=True)
                continue
        prompt = (root / batch["prompt"]).read_text(encoding="utf-8")
        prior_files = [raw_path, batch_dir / "agy_stderr.log", batch_dir / "agy_debug.log"]
        if any(path.exists() for path in prior_files):
            attempts = batch_dir / "failed_attempts"
            attempts.mkdir(exist_ok=True)
            attempt_number = len(list(attempts.glob("agy_cli_output_*.json"))) + 1
            if raw_path.exists():
                shutil.copy2(raw_path, attempts / f"agy_cli_output_{attempt_number:02d}.json")
            stderr_path = batch_dir / "agy_stderr.log"
            if stderr_path.exists():
                shutil.copy2(stderr_path, attempts / f"agy_stderr_{attempt_number:02d}.log")
            debug_path = batch_dir / "agy_debug.log"
            if debug_path.exists():
                shutil.copy2(debug_path, attempts / f"agy_debug_{attempt_number:02d}.log")
        command = [
            "agy", "--log-file", str(batch_dir / "agy_debug.log"),
            "--model", args.model, "--effort", "medium", "--mode", "plan", "--sandbox",
            "--dangerously-skip-permissions", "--disable-slash-commands", "--output-format", "json",
            "--json-schema", str(root / batch["schema"]), "--print-timeout", "45m", "-p", prompt,
        ]
        print(f"RUN {batch['batch']} ({batch['count']} groups)", flush=True)
        started = datetime.now(timezone.utc).isoformat()
        completed = subprocess.run(command, cwd=root, text=True, capture_output=True)
        raw_path.write_text(completed.stdout, encoding="utf-8")
        (batch_dir / "agy_stderr.log").write_text(completed.stderr, encoding="utf-8")
        try:
            raw = json.loads(completed.stdout)
            batch_for_validation = dict(batch, _root=str(root))
            valid, detail = validate(raw, batch_for_validation)
        except Exception as exc:
            raw = {}
            valid, detail = False, f"output parse/validation error: {exc}"
        result = {
            "batch": batch["batch"], "model": args.model, "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(), "returncode": completed.returncode,
            "validation": {"valid": valid, "detail": detail},
            "conversation_id": raw.get("conversation_id"), "usage": raw.get("usage"),
            "results": (structured_output(raw) or {}).get("results", []),
        }
        validated_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"DONE {batch['batch']}: valid={valid} detail={detail}", flush=True)
        if not valid:
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

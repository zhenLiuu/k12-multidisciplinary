#!/usr/bin/env python3
"""Shared, deterministic normalization for the public K12 schema."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PUBLIC_FIELDS = (
    "id", "source_id", "source_file", "split", "question_type", "subject",
    "language", "question", "options", "answer", "explanation", "images",
    "table", "sub_questions", "metadata",
)
SOURCE_KEYS = {
    "data/processed/raw/all_disciplines_with_idx.jsonl": "all_disciplines_with_idx",
    "data/processed/raw/math_non_mc.jsonl": "math_non_mc",
    "data/processed/raw/merge_multiple_choice.jsonl": "merge_multiple_choice",
    "data/processed/test/final_data_v8.2.json": "final_data_v8.2",
}
QUESTION_PREFIXES = (
    "根据图示，回答下面的问题。\n问题是：",
    "根据问题描述，回答下面的问题。\n问题是：",
)
ANSWER_PROMPT_RE = re.compile(r"\s*[。．.]?\s*你的答案是\s*[：:]\s*$")
STRONG_LINE_RE = re.compile(r"(?m)^[ \t]*([A-Ha-h])[ \t]*[.．、:：]\s*")
STRONG_ANY_RE = re.compile(r"(?<![A-Za-z0-9])([A-Ha-h])[ \t]*[.．、:：]\s*")
PUNCT_ANY_RE = re.compile(r"([A-Ha-h])[ \t]*[.．、:：]\s*")
PAREN_ANY_RE = re.compile(r"[（(]\s*([A-Ha-h])\s*[）)]\s*")
BARE_ANY_RE = re.compile(r"([A-H])(?=[^\sA-Za-z.．、:：,，;；）)])")
SPLIT_LINE_RE = re.compile(r"(?m)^[ \t]*([A-Ha-h])[ \t]*$")
LOOSE_LINE_RE = re.compile(
    r"(?m)^[ \t]*([A-Ha-h])(?:[ \t]*[.．、:：)）]|[ \t]+)(?=\S)"
)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
OCR_NOISE = (
    "试题答案练习册答案在线课程",
    "题目详情",
    "查看答案",
    "点击查看",
    "菁优网",
)
ANSWER_LETTER_RE = re.compile(r"[A-H]", re.I)
ANSWER_HINT_RE = re.compile(r"(?:故选|答案(?:为|是)?|应选|选择)\s*([A-H](?:\s*[,，、/和及]?\s*[A-H])*)", re.I)
CHECKED_LETTER_RE = re.compile(r"([A-H])\s*[（(]\s*[√✓]\s*[）)]", re.I)


@dataclass(frozen=True)
class Marker:
    label: str
    start: int
    end: int


@dataclass(frozen=True)
class ParsedOptions:
    question: str
    options: list[dict[str, str]]
    strategy: str


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", str(value))
    text = text.replace("\x0b", "\n").replace("\x0c", "\n")
    text = CONTROL_RE.sub("", text)
    for token in OCR_NOISE:
        text = text.replace(token, "")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def clean_nested(value: Any) -> Any:
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, list):
        return [clean_nested(item) for item in value]
    if isinstance(value, dict):
        return {str(key): clean_nested(item) for key, item in value.items()}
    return value


def strip_question_template(question: str) -> str:
    text = clean_text(question)
    for prefix in QUESTION_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    if text.startswith("【题目】"):
        text = text[len("【题目】"):]
    return ANSWER_PROMPT_RE.sub("", text).strip()


def _markers(pattern: re.Pattern[str], text: str) -> list[Marker]:
    return [Marker(match.group(1).upper(), match.start(), match.end()) for match in pattern.finditer(text)]


def _runs(markers: list[Marker], first: str = "A", minimum: int = 2) -> Iterable[list[Marker]]:
    for index, marker in enumerate(markers):
        if marker.label != first:
            continue
        run = [marker]
        cursor = index + 1
        while cursor < len(markers) and ord(markers[cursor].label) == ord(run[-1].label) + 1:
            run.append(markers[cursor])
            cursor += 1
        if len(run) >= minimum:
            yield run


def _candidate(text: str, markers: list[Marker], strategy: str) -> ParsedOptions | None:
    if len(markers) < 2:
        return None
    expected = [chr(ord("A") + index) for index in range(len(markers))]
    if [marker.label for marker in markers] != expected:
        return None
    stem = text[:markers[0].start].strip()
    option_texts = []
    for index, marker in enumerate(markers):
        end = markers[index + 1].start if index + 1 < len(markers) else len(text)
        option_texts.append(text[marker.end:end].strip())
    if not stem or any(not option for option in option_texts):
        return None
    return ParsedOptions(
        question=stem,
        options=[{"label": label, "text": option} for label, option in zip(expected, option_texts)],
        strategy=strategy,
    )


def _ordered_sequences(markers: list[Marker]) -> Iterable[list[Marker]]:
    """Recover A..N while skipping OCR fragments such as A、B inside option text."""
    for final_index, final in enumerate(markers):
        count = ord(final.label) - ord("A") + 1
        if count < 2:
            continue
        selected = [final]
        cursor = final_index - 1
        for codepoint in range(ord(final.label) - 1, ord("A") - 1, -1):
            wanted = chr(codepoint)
            while cursor >= 0 and markers[cursor].label != wanted:
                cursor -= 1
            if cursor < 0:
                break
            selected.append(markers[cursor])
            cursor -= 1
        if len(selected) == count:
            yield list(reversed(selected))


def parse_embedded_options(question: str, answer_value: Any = None) -> ParsedOptions | None:
    """Parse only layouts with a defensible ordered A..N marker sequence."""
    text = strip_question_template(question)
    strong_line = _markers(STRONG_LINE_RE, text)
    strong_any = _markers(STRONG_ANY_RE, text)
    punct_any = _markers(PUNCT_ANY_RE, text)
    paren_any = _markers(PAREN_ANY_RE, text)
    bare_any = _markers(BARE_ANY_RE, text)
    split_line = _markers(SPLIT_LINE_RE, text)
    loose_line = _markers(LOOSE_LINE_RE, text)
    candidates: list[tuple[tuple[int, int, int, int], ParsedOptions]] = []

    strategy_priority = {
        "strong_line": 12, "ordered_line": 11, "hybrid_line": 10,
        "loose_line": 9, "ordered_loose": 8, "split_line": 7,
        "punct_any": 6, "paren_any": 5, "strong_any": 4,
        "ordered_punct": 3, "ordered_paren": 2, "bare_any": 1,
        "ordered_any": 1, "ordered_bare": 0,
    }
    answer_labels = _answer_letters(clean_text(answer_value)) if answer_value is not None else None

    def add(markers: list[Marker], strategy: str) -> None:
        parsed = _candidate(text, markers, strategy)
        if parsed is None:
            return
        count = len(markers)
        allowed = {chr(ord("A") + index) for index in range(count)}
        answer_compatible = int(answer_labels is not None and set(answer_labels) <= allowed)
        score = (answer_compatible, strategy_priority[strategy], count, markers[-1].start)
        candidates.append((score, parsed))

    for run in _runs(strong_line):
        add(run, "strong_line")
    for run in _ordered_sequences(strong_line):
        add(run, "ordered_line")

    # Common OCR layout: A is appended to the stem line, while B..D start lines.
    for tail in _runs(strong_line, first="B", minimum=2):
        prior_a = [marker for marker in strong_any if marker.label == "A" and marker.start < tail[0].start]
        if prior_a:
            add([prior_a[-1], *tail], "hybrid_line")

    for run in _runs(loose_line):
        add(run, "loose_line")
    for run in _ordered_sequences(loose_line):
        add(run, "ordered_loose")
    for run in _runs(split_line):
        add(run, "split_line")
    for run in _runs(punct_any):
        add(run, "punct_any")
    for run in _ordered_sequences(punct_any):
        add(run, "ordered_punct")
    for run in _runs(paren_any):
        add(run, "paren_any")
    for run in _ordered_sequences(paren_any):
        add(run, "ordered_paren")
    for run in _runs(strong_any, minimum=3):
        add(run, "strong_any")
    for run in _ordered_sequences(strong_any):
        add(run, "ordered_any")
    for run in _runs(bare_any, minimum=3):
        add(run, "bare_any")
    for run in _ordered_sequences(bare_any):
        add(run, "ordered_bare")

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def normalize_structured_options(value: Any) -> list[dict[str, str]] | None:
    if not isinstance(value, list) or not value or len(value) > 8:
        return None
    output: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            return None
        text = clean_text(item.get("text", item.get("op_text", "")))
        if not text:
            return None
        output.append({"label": chr(ord("A") + index), "text": text})
    return output


def _answer_letters(text: str) -> list[str] | None:
    compact = clean_text(text).upper()
    if not compact:
        return None
    direct = re.sub(r"[\s,，、;；/\\|+&和或及答案：:（）()\[\]。.]", "", compact)
    if direct and re.fullmatch(r"[A-H]+", direct):
        return list(dict.fromkeys(direct))
    hint = ANSWER_HINT_RE.search(compact)
    if hint:
        letters = ANSWER_LETTER_RE.findall(hint.group(1))
        if letters:
            return list(dict.fromkeys(letter.upper() for letter in letters))
    checked = CHECKED_LETTER_RE.findall(compact)
    if checked:
        return list(dict.fromkeys(letter.upper() for letter in checked))
    return None


def _match_answer_text(text: str, options: list[dict[str, str]]) -> list[str] | None:
    def comparable(value: str) -> str:
        value = clean_text(value).casefold()
        return re.sub(r"[\s`$\\{}_^.,，。:：;；'\"（）()\[\]]", "", value)

    answer = comparable(text)
    if not answer:
        return None
    matches = [option["label"] for option in options if comparable(option["text"]) == answer]
    return matches if len(matches) == 1 else None


def normalize_mc_answer(value: Any, options: list[dict[str, str]]) -> list[str] | None:
    if isinstance(value, list):
        raw_parts = [clean_text(item) for item in value]
    else:
        raw_parts = [clean_text(value)]
    labels: list[str] = []
    for part in raw_parts:
        parsed = _answer_letters(part) or _match_answer_text(part, options)
        if parsed is None:
            return None
        labels.extend(parsed)
    labels = list(dict.fromkeys(labels))
    allowed = {option["label"] for option in options}
    if not labels or any(label not in allowed for label in labels):
        return None
    return sorted(labels, key=lambda label: ord(label))


def normalize_open_answer(value: Any) -> list[str] | None:
    values = value if isinstance(value, list) else [value]
    output = [clean_text(item) for item in values]
    return output if output and all(output) else None


def normalize_images(value: Any) -> list[dict[str, str]]:
    if value is None:
        return []
    output: list[dict[str, str]] = []
    for item in value if isinstance(value, list) else [value]:
        if isinstance(item, str):
            path, caption, part_of = clean_text(item), "", "question"
        elif isinstance(item, dict):
            path = clean_text(item.get("path", ""))
            caption = clean_text(item.get("caption", ""))
            part_of = clean_text(item.get("part_of", "question")) or "question"
        else:
            continue
        if path:
            output.append({"path": path, "caption": caption, "part_of": part_of})
    return output


def normalize_explanation(value: Any) -> tuple[str, list[str] | None]:
    if isinstance(value, list):
        parts = [clean_text(item) for item in value if clean_text(item)]
        return "\n\n".join(parts), parts
    return clean_text(value), None


def public_id(record: dict[str, Any]) -> str:
    identity = {
        "source_file": record["source_file"],
        "source_id": record["source_id"],
        "question_type": record["question_type"],
        "question": record["question"],
        "options": record["options"],
        "images": [item["path"] for item in record["images"]],
        "table": record["table"],
        "sub_questions": record["sub_questions"],
    }
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "k12_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def iter_records(path: Path, fmt: str) -> Iterable[tuple[int, dict[str, Any]]]:
    if fmt == "json":
        for number, record in enumerate(json.loads(path.read_text(encoding="utf-8")), 1):
            yield number, record
        return
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                yield number, json.loads(line)

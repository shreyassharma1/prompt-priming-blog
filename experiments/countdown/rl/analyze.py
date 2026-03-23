"""Summarize exported Countdown inference results."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def split_into_sentences(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []

    sentences: list[str] = []
    for block in stripped.splitlines():
        line = block.strip()
        if not line:
            continue
        parts = [part.strip() for part in SENTENCE_RE.split(line) if part.strip()]
        if parts:
            sentences.extend(parts)
        else:
            sentences.append(line)
    return sentences


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Countdown inference JSON output")
    parser.add_argument("path", help="Path to an inference output JSON file")
    args = parser.parse_args()

    path = Path(args.path)
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} does not contain a non-empty list of results")

    n = len(rows)
    accuracy = sum(1 for row in rows if row.get("is_correct")) / n
    mean_score = sum(float(row.get("score", 0.0)) for row in rows) / n
    mean_sentence_count = sum(
        len(split_into_sentences(str(row.get("response_text", "") or ""))) for row in rows
    ) / n
    mean_chars = sum(len(str(row.get("response_text", "") or "")) for row in rows) / n

    print(f"file: {path}")
    print(f"samples: {n}")
    print(f"accuracy: {accuracy:.2%}")
    print(f"mean_score: {mean_score:.3f}")
    print(f"mean_trace_sentences: {mean_sentence_count:.2f}")
    print(f"mean_trace_chars: {mean_chars:.1f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slime.utils.toolcall_health import (
    TOOLCALL_FORMAT_DANGLING_CLOSING,
    TOOLCALL_FORMAT_NATIVE,
    TOOLCALL_FORMAT_TEXT_ONLY_JSON,
    classify_openai_message_toolcall_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate parser health on a tool-call regression dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="local-liveweb-bench")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=256)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _call_model(args: argparse.Namespace, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
    url = args.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": args.model,
        "messages": messages,
        "tools": tools,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {args.api_key}"},
        json=payload,
        timeout=args.timeout,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(Path(args.dataset))
    if args.limit > 0:
        rows = rows[: args.limit]

    counts: Counter[str] = Counter()
    exact_tool_match = 0
    details_path = output_dir / "details.jsonl"
    start = time.time()

    with details_path.open("w", encoding="utf-8") as detail_f:
        for row in rows:
            prompt_messages = row["messages"][:-1]
            expected = row["messages"][-1]
            expected_tool_name = ""
            expected_tool_calls = expected.get("tool_calls") or []
            if expected_tool_calls:
                expected_tool_name = str((expected_tool_calls[0].get("function") or {}).get("name") or "")

            raw = _call_model(args, prompt_messages, row.get("tools") or [])
            message = (((raw.get("choices") or [{}])[0].get("message")) or {})
            output_class = classify_openai_message_toolcall_output(message)
            counts[output_class] += 1

            tool_calls = message.get("tool_calls") or []
            actual_tool_name = ""
            if tool_calls:
                actual_tool_name = str((tool_calls[0].get("function") or {}).get("name") or "")
                if actual_tool_name == expected_tool_name:
                    exact_tool_match += 1

            detail = {
                "id": row.get("id"),
                "expected_tool_name": expected_tool_name,
                "actual_tool_name": actual_tool_name,
                "output_class": output_class,
                "native_tool_calls": bool(tool_calls),
                "content_preview": (message.get("content") or "")[:500] if isinstance(message.get("content"), str) else None,
                "raw_path": None,
            }
            detail_f.write(json.dumps(detail, ensure_ascii=False) + "\n")

    total = len(rows)
    summary = {
        "total": total,
        "duration_seconds": round(time.time() - start, 3),
        "parser_success_rate": (counts[TOOLCALL_FORMAT_NATIVE] / total) if total else 0.0,
        "native_tool_calls_rate": (counts[TOOLCALL_FORMAT_NATIVE] / total) if total else 0.0,
        "dangling_tool_call_rate": (counts[TOOLCALL_FORMAT_DANGLING_CLOSING] / total) if total else 0.0,
        "text_only_tool_call_rate": (counts[TOOLCALL_FORMAT_TEXT_ONLY_JSON] / total) if total else 0.0,
        "exact_tool_name_match_rate": (exact_tool_match / total) if total else 0.0,
        "class_counts": dict(sorted(counts.items())),
        "details_path": str(details_path),
    }
    (output_dir / "format_eval.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer

from slime.utils.toolcall_health import rendered_qwen_toolcall_is_complete


ALLOWED_ACTIONS = ("goto", "click", "click_role", "type", "stop")

CANONICAL_TOOLS = {
    "goto": {
        "type": "function",
        "function": {
            "name": "goto",
            "description": "Navigate to a URL",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to navigate to"},
                },
                "required": ["url"],
            },
        },
    },
    "click": {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Click an element by CSS selector",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "CSS selector of the element to click",
                    },
                },
                "required": ["selector"],
            },
        },
    },
    "click_role": {
        "type": "function",
        "function": {
            "name": "click_role",
            "description": "Click an element by accessibility role and name (more stable than CSS selectors)",
            "parameters": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "description": "Accessibility role (e.g., button, link, tab)"},
                    "name": {"type": "string", "description": "Accessible name of the element"},
                    "exact": {
                        "type": "boolean",
                        "description": "Require exact name match",
                        "default": False,
                    },
                },
                "required": ["role", "name"],
            },
        },
    },
    "type": {
        "type": "function",
        "function": {
            "name": "type",
            "description": "Type text into an input field",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector of the input field"},
                    "text": {"type": "string", "description": "Text to type"},
                    "press_enter": {
                        "type": "boolean",
                        "description": "Press Enter after typing",
                        "default": False,
                    },
                },
                "required": ["selector", "text"],
            },
        },
    },
    "stop": {
        "type": "function",
        "function": {
            "name": "stop",
            "description": "Complete the task and submit final answers",
            "parameters": {
                "type": "object",
                "properties": {
                    "answers": {
                        "type": "object",
                        "description": 'Final answers as key-value pairs (e.g., {"answer1": "value1"})',
                    },
                },
                "required": ["answers"],
            },
        },
    },
}

CANONICAL_ARGUMENTS = {
    "goto": '{"url": "https://example.com"}',
    "click": '{"selector": "#submit"}',
    "click_role": '{"role": "button", "name": "Search", "exact": true}',
    "type": '{"selector": "#query", "text": "hello world", "press_enter": false}',
    "stop": '{"answers": {"answer1": "done"}}',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Qwen tool-call regression dataset from LiveWeb SFT data.")
    parser.add_argument("--input", required=True, help="Path to the task SFT dataset JSONL.")
    parser.add_argument("--output", required=True, help="Output regression JSONL.")
    parser.add_argument("--summary-output", default="", help="Optional summary path.")
    parser.add_argument("--hf-model-dir", required=True, help="Model/tokenizer dir used for template validation.")
    parser.add_argument("--per-action", type=int, default=48, help="Max real-prefix samples per action.")
    parser.add_argument("--minimal-per-action", type=int, default=16, help="Max synthetic minimal samples per action.")
    return parser.parse_args()


def _stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _message_copy(message: dict[str, Any]) -> dict[str, Any]:
    copied = dict(message)
    if "tool_calls" in copied and copied["tool_calls"] is not None:
        copied["tool_calls"] = list(copied["tool_calls"])
    return copied


def _render_is_valid(tokenizer, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> bool:
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
    )
    return rendered_qwen_toolcall_is_complete(rendered)


def _minimal_messages(action: str, tool_call: dict[str, Any]) -> list[dict[str, Any]]:
    arguments = json.loads((tool_call.get("function") or {}).get("arguments") or "{}")
    if action == "goto":
        user_prompt = f'Use exactly one goto tool call to open {arguments["url"]}.'
    elif action == "click":
        user_prompt = f'Use exactly one click tool call to click the element matching selector {arguments["selector"]}.'
    elif action == "click_role":
        exact = " with exact=true" if arguments.get("exact") else ""
        user_prompt = (
            f'Use exactly one click_role tool call to click the {arguments["role"]} named "{arguments["name"]}"{exact}.'
        )
    elif action == "type":
        user_prompt = (
            f'Use exactly one type tool call to type "{arguments["text"]}" into selector {arguments["selector"]}.'
        )
    elif action == "stop":
        user_prompt = (
            "Use exactly one stop tool call to submit these final answers: "
            + json.dumps(arguments["answers"], ensure_ascii=False, sort_keys=True)
        )
    else:
        raise ValueError(f"Unsupported action: {action}")
    return [
        {
            "role": "system",
            "content": "You are a web automation agent. Reply only with a single tool call.",
        },
        {
            "role": "user",
            "content": user_prompt,
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call],
        },
    ]


def _canonical_minimal_row(action: str, idx: int) -> dict[str, Any]:
    tool_call = {
        "id": f"canonical_call_{idx}",
        "type": "function",
        "function": {
            "name": action,
            "arguments": CANONICAL_ARGUMENTS[action],
        },
    }
    return {
        "id": f"minimal:{action}:canonical:{idx}",
        "source_kind": "minimal_synthetic",
        "action_type": action,
        "messages": _minimal_messages(action, tool_call),
        "tools": [CANONICAL_TOOLS[action]],
        "metadata": {
            "source": "minimal_synthetic",
            "source_id": None,
            "action": action,
        },
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = (
        Path(args.summary_output).resolve()
        if args.summary_output
        else output_path.with_suffix(output_path.suffix + ".summary.json")
    )

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_dir, trust_remote_code=True)
    rows = _read_jsonl(input_path)

    real_counts: Counter[str] = Counter()
    minimal_counts: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    seen_hashes: set[str] = set()
    examples_by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        messages = row.get("messages") or []
        tools = row.get("tools") or []
        if not isinstance(messages, list) or not isinstance(tools, list):
            dropped["invalid_row"] += 1
            continue

        for idx, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            tool_calls = message.get("tool_calls") or []
            if len(tool_calls) != 1:
                continue
            tool_call = tool_calls[0]
            function = tool_call.get("function") or {}
            action = str(function.get("name") or "")
            if action not in ALLOWED_ACTIONS:
                continue
            if real_counts[action] >= args.per_action:
                continue

            prefix_messages = [_message_copy(m) for m in messages[: idx + 1]]
            candidate = {
                "id": f"real:{action}:{row.get('id')}:{idx}",
                "source_kind": "real_prefix",
                "action_type": action,
                "messages": prefix_messages,
                "tools": tools,
                "metadata": {
                    "source": "real_prefix",
                    "source_id": row.get("id"),
                    "action": action,
                    "task_metadata": row.get("metadata") or {},
                },
            }
            row_hash = _stable_hash({"messages": candidate["messages"], "tools": candidate["tools"]})
            if row_hash in seen_hashes:
                dropped["duplicate"] += 1
                continue
            if not _render_is_valid(tokenizer, candidate["messages"], tools):
                dropped["template_invalid"] += 1
                continue
            seen_hashes.add(row_hash)
            examples_by_action[action].append(candidate)
            real_counts[action] += 1

    for action in ALLOWED_ACTIONS:
        source_examples = examples_by_action.get(action) or []

        for idx in range(args.minimal_per_action):
            if minimal_counts[action] >= args.minimal_per_action:
                break
            minimal_row = _canonical_minimal_row(action, idx)
            row_hash = _stable_hash({"messages": minimal_row["messages"], "tools": minimal_row["tools"]})
            if row_hash in seen_hashes:
                continue
            if not _render_is_valid(tokenizer, minimal_row["messages"], minimal_row["tools"]):
                dropped["template_invalid"] += 1
                continue
            seen_hashes.add(row_hash)
            source_examples.insert(minimal_counts[action], minimal_row)
            minimal_counts[action] += 1

    written_rows = []
    for action in ALLOWED_ACTIONS:
        action_rows = examples_by_action.get(action) or []
        written_rows.extend([row for row in action_rows if row.get("source_kind") == "minimal_synthetic"])
    for action in ALLOWED_ACTIONS:
        action_rows = examples_by_action.get(action) or []
        written_rows.extend([row for row in action_rows if row.get("source_kind") == "real_prefix"])

    with output_path.open("w", encoding="utf-8") as f:
        for row in written_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "total_rows": len(written_rows),
        "real_counts": dict(sorted(real_counts.items())),
        "minimal_counts": dict(sorted(minimal_counts.items())),
        "dropped": dict(sorted(dropped.items())),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

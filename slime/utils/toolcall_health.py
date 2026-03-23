from __future__ import annotations

import json
from typing import Any


TOOLCALL_FORMAT_NATIVE = "native_tool_calls"
TOOLCALL_FORMAT_XML_WRAPPED = "xml_tool_call_wrapped"
TOOLCALL_FORMAT_DANGLING_CLOSING = "dangling_closing_tool_call"
TOOLCALL_FORMAT_TEXT_ONLY_JSON = "text_only_json_like_tool_call"
TOOLCALL_FORMAT_NO_TOOLCALL = "no_tool_call"


def _stringify_arguments(arguments: Any) -> str:
    if arguments is None:
        return ""
    if isinstance(arguments, str):
        return arguments
    try:
        return json.dumps(arguments, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(arguments)


def extract_assistant_response_text(conversation: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in conversation:
        if message.get("role") != "assistant":
            continue
        if isinstance(message.get("content"), str) and message["content"]:
            parts.append(message["content"])
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function", {})
            parts.append(
                json.dumps(
                    {
                        "name": function.get("name"),
                        "arguments": function.get("arguments"),
                    },
                    ensure_ascii=False,
                )
            )
    return "\n".join(parts)


def preserve_structured_toolcall_message(message: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(message)
    if normalized.get("role") != "assistant":
        return normalized

    if not isinstance(normalized.get("content"), str) or not normalized.get("content"):
        normalized["content"] = None

    tool_calls = normalized.get("tool_calls")
    if tool_calls:
        normalized["tool_calls"] = list(tool_calls)
    else:
        normalized.pop("tool_calls", None)
    return normalized


def preserve_structured_conversation(conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [preserve_structured_toolcall_message(message) for message in conversation]


def looks_like_text_toolcall_json(text: str) -> bool:
    stripped = text.strip()
    if not stripped.startswith("{") or '"name"' not in stripped or '"arguments"' not in stripped:
        return False
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return False
    return isinstance(payload, dict) and "name" in payload and "arguments" in payload


def classify_toolcall_output(*, content: Any, tool_calls: Any) -> str:
    if tool_calls:
        return TOOLCALL_FORMAT_NATIVE

    if not isinstance(content, str) or not content.strip():
        return TOOLCALL_FORMAT_NO_TOOLCALL

    stripped = content.strip()
    has_open = "<tool_call>" in stripped
    has_close = "</tool_call>" in stripped

    if has_open and has_close:
        return TOOLCALL_FORMAT_XML_WRAPPED
    if has_close and not has_open:
        return TOOLCALL_FORMAT_DANGLING_CLOSING
    if looks_like_text_toolcall_json(stripped):
        return TOOLCALL_FORMAT_TEXT_ONLY_JSON
    return TOOLCALL_FORMAT_NO_TOOLCALL


def classify_openai_message_toolcall_output(message: dict[str, Any]) -> str:
    return classify_toolcall_output(
        content=message.get("content"),
        tool_calls=message.get("tool_calls"),
    )


def is_native_toolcall_message(message: dict[str, Any]) -> bool:
    return classify_openai_message_toolcall_output(message) == TOOLCALL_FORMAT_NATIVE


def rendered_qwen_toolcall_is_complete(rendered_text: str) -> bool:
    return rendered_text.count("<tool_call>") == rendered_text.count("</tool_call>") and "<tool_call>" in rendered_text


def message_toolcall_signature(message: dict[str, Any]) -> tuple[str, str]:
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        return "", ""
    function = tool_calls[0].get("function", {})
    return str(function.get("name") or ""), _stringify_arguments(function.get("arguments"))

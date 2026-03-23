from slime.utils.toolcall_health import (
    TOOLCALL_FORMAT_DANGLING_CLOSING,
    TOOLCALL_FORMAT_NATIVE,
    TOOLCALL_FORMAT_TEXT_ONLY_JSON,
    classify_openai_message_toolcall_output,
    preserve_structured_conversation,
)


def test_classify_native_tool_calls():
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "goto", "arguments": "{}"}}],
    }
    assert classify_openai_message_toolcall_output(message) == TOOLCALL_FORMAT_NATIVE


def test_classify_dangling_tool_call():
    message = {
        "role": "assistant",
        "content": '{"name":"goto","arguments":{"url":"https://example.com"}}\n</tool_call>',
        "tool_calls": None,
    }
    assert classify_openai_message_toolcall_output(message) == TOOLCALL_FORMAT_DANGLING_CLOSING


def test_preserve_structured_conversation_keeps_tool_calls():
    conversation = [
        {"role": "system", "content": "sys"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "goto", "arguments": "{}"}}],
        },
    ]
    normalized = preserve_structured_conversation(conversation)
    assert normalized[1]["content"] is None
    assert normalized[1]["tool_calls"][0]["function"]["name"] == "goto"


def test_classify_text_only_json_tool_call():
    message = {
        "role": "assistant",
        "content": '{"name":"stop","arguments":{"answers":{"answer1":"ok"}}}',
        "tool_calls": [],
    }
    assert classify_openai_message_toolcall_output(message) == TOOLCALL_FORMAT_TEXT_ONLY_JSON

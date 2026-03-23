from slime.env_adapters.training_utils import normalize_conversation_for_training as adapter_normalize
from slime.rollout.liveweb_online.common import (
    _cleanup_interceptor_value,
    normalize_conversation_for_training as rollout_normalize,
)


def _conversation():
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "goto", "arguments": '{"url":"https://example.com"}'},
                }
            ],
        },
    ]


def test_adapter_training_normalization_preserves_tool_calls():
    normalized = adapter_normalize(_conversation())
    assert normalized[-1]["content"] is None
    assert normalized[-1]["tool_calls"][0]["function"]["name"] == "goto"


def test_rollout_training_normalization_preserves_tool_calls():
    normalized = rollout_normalize(_conversation())
    assert normalized[-1]["content"] is None
    assert normalized[-1]["tool_calls"][0]["function"]["name"] == "goto"


def test_cleanup_interceptor_value_handles_tuple_wrapper():
    class _FakeInterceptor:
        def __init__(self):
            self.cleaned = False

        def cleanup(self):
            self.cleaned = True

    interceptor = _FakeInterceptor()
    _cleanup_interceptor_value(("session", interceptor))
    assert interceptor.cleaned is True

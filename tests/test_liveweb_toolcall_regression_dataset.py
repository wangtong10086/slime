import json
import subprocess
import sys
from pathlib import Path


def test_build_toolcall_regression_dataset(tmp_path: Path):
    input_path = tmp_path / "train.jsonl"
    output_path = tmp_path / "regression.jsonl"
    row = {
        "id": "row1",
        "messages": [
            {"role": "system", "content": "You are a web automation agent."},
            {"role": "user", "content": "Open the page."},
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
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "goto",
                    "description": "Navigate",
                    "parameters": {
                        "type": "object",
                        "properties": {"url": {"type": "string"}},
                        "required": ["url"],
                    },
                },
            }
        ],
        "metadata": {},
    }
    input_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "/home/xmyf/slime/scripts/build_liveweb_toolcall_regression_dataset.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--hf-model-dir",
            "/home/xmyf/Qwen3-32B",
            "--per-action",
            "1",
            "--minimal-per-action",
            "1",
        ],
        check=True,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 2
    assert all((row["messages"][-1]["tool_calls"][0]["function"]["name"] == "goto") for row in rows)
    assert all(row["messages"][-1].get("content") is None for row in rows)
    assert rows[0]["source_kind"] == "minimal_synthetic"
    assert rows[0]["action_type"] == "goto"
    assert rows[1]["source_kind"] == "real_prefix"

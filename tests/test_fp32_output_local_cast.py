import sys
import types

import torch

sys.modules.setdefault("wandb", types.SimpleNamespace())

from slime.utils import ppo_utils


def test_calculate_log_probs_and_entropy_accepts_bf16_logits(monkeypatch):
    calls = {"log_prob_dtype": None, "entropy_dtype": None}

    def _fake_compute_log_probs(logits, tokens, process_group):
        calls["log_prob_dtype"] = logits.dtype
        return torch.zeros((logits.size(0), 1), dtype=torch.float32)

    def _fake_compute_entropy_from_logits(logits, process_group):
        calls["entropy_dtype"] = logits.dtype
        return torch.zeros((logits.size(0),), dtype=torch.float32)

    monkeypatch.setattr(ppo_utils, "compute_log_probs", _fake_compute_log_probs)
    monkeypatch.setattr(ppo_utils, "compute_entropy_from_logits", _fake_compute_entropy_from_logits)

    logits = torch.randn(4, 8, dtype=torch.bfloat16)
    tokens = torch.randint(0, 10, (4,), dtype=torch.int64)

    log_prob, entropy = ppo_utils.calculate_log_probs_and_entropy(
        logits,
        tokens,
        tp_group=None,
        with_entropy=True,
        chunk_size=2,
    )

    assert calls["log_prob_dtype"] == torch.float32
    assert calls["entropy_dtype"] == torch.float32
    assert log_prob.dtype == torch.float32
    assert entropy.dtype == torch.float32

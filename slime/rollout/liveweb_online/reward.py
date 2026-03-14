from __future__ import annotations

from typing import Any

from slime.utils.types import Sample

from .common import compute_reward_from_result


async def reward_func(args, sample_or_samples, **kwargs):
    def _one(sample: Sample) -> float | None:
        result = sample.metadata.get("liveweb_result", {}) if sample.metadata else {}
        reward, _ = compute_reward_from_result(result)
        return reward

    if isinstance(sample_or_samples, list):
        return [_one(sample) for sample in sample_or_samples]
    return _one(sample_or_samples)


def post_process_rewards(args, samples: list[Sample], **kwargs):
    raw_rewards = [float(sample.metadata.get("raw_reward", sample.reward or 0.0)) for sample in samples]
    rewards = []
    for sample in samples:
        reward = sample.reward if sample.reward is not None else 0.0
        reward = min(1.0, max(-0.10, float(reward)))
        rewards.append(reward)
    return raw_rewards, rewards

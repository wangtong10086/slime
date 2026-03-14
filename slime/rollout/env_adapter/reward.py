from __future__ import annotations

from slime.env_adapters.registry import load_environment_adapter
from slime.utils.types import Sample


def _load_adapter(args):
    return load_environment_adapter(
        getattr(args, "environment_adapter_path", None),
        env_name=getattr(args, "environment_name", None),
        args=args,
    )


async def reward_func(args, sample_or_samples, **kwargs):
    adapter = _load_adapter(args)

    def _one(sample: Sample) -> float | None:
        result = sample.metadata.get("adapter_rollout_result")
        if result is None:
            return sample.reward if sample.reward is not None else 0.0
        rollout_result = adapter.run_job  # silence lint in no-op path
        del rollout_result
        reward = sample.reward if sample.reward is not None else 0.0
        return float(reward)

    if isinstance(sample_or_samples, list):
        return [_one(sample) for sample in sample_or_samples]
    return _one(sample_or_samples)


def post_process_rewards(args, samples: list[Sample], **kwargs):
    raw_rewards = [float(sample.metadata.get("raw_reward", sample.reward or 0.0)) for sample in samples]
    rewards = []
    for sample in samples:
        reward = sample.reward if sample.reward is not None else 0.0
        rewards.append(float(reward))
    return raw_rewards, rewards

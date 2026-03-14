from __future__ import annotations

from pathlib import Path

import torch

from slime.env_adapters.registry import load_environment_adapter
from slime.rollout.data_source import DataSource
from slime.utils.types import Sample


class AdapterDataSource(DataSource):
    def __init__(self, args):
        self.args = args
        self.adapter = load_environment_adapter(
            getattr(args, "environment_adapter_path", None),
            env_name=getattr(args, "environment_name", None),
            args=args,
        )
        self.phase = self.adapter.get_train_phase(args)
        self.sample_group_index = 0
        self.sample_index = 0

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        tasks = self.adapter.sample_tasks(split="train", count=num_samples, phase=self.phase)
        jobs = self.adapter.expand_jobs(tasks=tasks, n_samples_per_task=self.args.n_samples_per_prompt, mode="train")
        grouped: dict[str, list[Sample]] = {}
        group_order: list[str] = []
        for job in jobs:
            if job.group_id not in grouped:
                grouped[job.group_id] = []
                group_order.append(job.group_id)
            sample = Sample(
                group_index=self.sample_group_index + len(group_order) - 1,
                index=self.sample_index,
                prompt=job.prompt_hint or job.job_id,
                metadata={"job_spec": job.to_dict(), "env_name": self.adapter.name},
                session_id=job.affinity_key,
            )
            self.sample_index += 1
            grouped[job.group_id].append(sample)
        samples = [grouped[group_id] for group_id in group_order]
        self.sample_group_index += len(samples)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        return None

    def save(self, rollout_id):
        path = Path(self.args.save) / "rollout" / f"env_adapter_data_source_{rollout_id}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "phase": self.phase,
                "sample_group_index": self.sample_group_index,
                "sample_index": self.sample_index,
                "adapter_state": self.adapter.export_state(),
            },
            path,
        )

    def load(self, rollout_id=None):
        if self.args.load is None or rollout_id is None:
            return
        path = Path(self.args.load) / "rollout" / f"env_adapter_data_source_{rollout_id}.pt"
        if not path.exists():
            return
        state = torch.load(path, weights_only=False)
        self.phase = state.get("phase", self.phase)
        self.sample_group_index = state.get("sample_group_index", self.sample_group_index)
        self.sample_index = state.get("sample_index", self.sample_index)
        self.adapter.load_state(state.get("adapter_state") or {})

    def __len__(self) -> int:
        return 10**12

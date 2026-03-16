from __future__ import annotations

import os
from pathlib import Path

import torch

from slime.rollout.data_source import DataSource
from slime.utils.types import Sample

from .common import build_eval_jobs, build_prompt_job, derive_llm_seed, derive_route_key, get_phase_name


class LiveWebOnlineDataSource(DataSource):
    def __init__(self, args):
        self.args = args
        self.phase = get_phase_name(evaluation=False)
        self.parent_seed_cursor = int(os.getenv("LIVEWEB_TRAIN_BASE_SEED", "100000"))
        self.sample_group_index = 0
        self.sample_index = 0
        self.curated_warmup_pool = []
        self.curated_warmup_cursor = 0
        use_curated_warmup_pool = os.getenv("LIVEWEB_USE_CURATED_WARMUP_POOL", "1") == "1"
        if self.phase == "warmup" and use_curated_warmup_pool:
            num_prompts = int(os.getenv("LIVEWEB_WARMUP_POOL_SIZE", os.getenv("LIVEWEB_QUICK_EVAL_PROMPTS", "32")))
            base_seed = int(os.getenv("LIVEWEB_WARMUP_POOL_BASE_SEED", "900000"))
            self.curated_warmup_pool = build_eval_jobs(
                rollout_id=0,
                dataset_name="warmup_pool",
                num_prompts=num_prompts,
                phase="warmup",
                base_seed=base_seed,
            )

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        groups: list[list[Sample]] = []
        for _ in range(num_samples):
            if self.curated_warmup_pool:
                base_job = self.curated_warmup_pool[self.curated_warmup_cursor % len(self.curated_warmup_pool)]
                self.curated_warmup_cursor += 1
                parent_seed = base_job.parent_seed
            else:
                parent_seed = self.parent_seed_cursor
                self.parent_seed_cursor += 1
            group: list[Sample] = []
            for sample_offset in range(self.args.n_samples_per_prompt):
                if self.curated_warmup_pool:
                    llm_seed = derive_llm_seed(parent_seed + self.sample_group_index * 1000, sample_offset)
                    job = type(base_job)(
                        parent_seed=base_job.parent_seed,
                        task_id=base_job.task_id,
                        task_seed=base_job.task_seed,
                        llm_seed=llm_seed,
                        subtask_index=base_job.subtask_index,
                        num_subtasks=base_job.num_subtasks,
                        templates=list(base_job.templates),
                        task_name=base_job.task_name,
                        plugin_name=base_job.plugin_name,
                        plugin_names=list(base_job.plugin_names),
                        combo_index=base_job.combo_index,
                        combo_key=base_job.combo_key,
                        phase=base_job.phase,
                        route_key=derive_route_key(
                            f"{self.phase}:curated",
                            base_job.parent_seed,
                            base_job.subtask_index,
                            self.sample_group_index * self.args.n_samples_per_prompt + sample_offset,
                        ),
                    )
                else:
                    job = build_prompt_job(
                        parent_seed=parent_seed,
                        group_index=self.sample_group_index,
                        sample_index=sample_offset,
                        phase=self.phase,
                    )
                sample = Sample(
                    group_index=self.sample_group_index,
                    index=self.sample_index,
                    prompt=f"{job.task_name}:{job.parent_seed}",
                    metadata=job.to_metadata(),
                    session_id=job.route_key,
                )
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            groups.append(group)
        return groups

    def add_samples(self, samples: list[list[Sample]]):
        return None

    def save(self, rollout_id):
        path = Path(self.args.save) / "rollout" / f"liveweb_online_data_source_{rollout_id}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "phase": self.phase,
                "parent_seed_cursor": self.parent_seed_cursor,
                "sample_group_index": self.sample_group_index,
                "sample_index": self.sample_index,
                "curated_warmup_cursor": self.curated_warmup_cursor,
            },
            path,
        )

    def load(self, rollout_id=None):
        if self.args.load is None or rollout_id is None:
            return
        path = Path(self.args.load) / "rollout" / f"liveweb_online_data_source_{rollout_id}.pt"
        if not path.exists():
            return
        state = torch.load(path, weights_only=False)
        self.phase = state.get("phase", self.phase)
        self.parent_seed_cursor = state.get("parent_seed_cursor", self.parent_seed_cursor)
        self.sample_group_index = state.get("sample_group_index", self.sample_group_index)
        self.sample_index = state.get("sample_index", self.sample_index)
        self.curated_warmup_cursor = state.get("curated_warmup_cursor", self.curated_warmup_cursor)

    def __len__(self) -> int:
        return 10**12

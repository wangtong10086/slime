from __future__ import annotations

import json
import os
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_TASK_MIX_PATH = Path("/home/xmyf/slime/scripts/configs/liveweb_online_task_mix.json")
DEFAULT_LIVEWEB_ARENA_DIR = Path("/home/xmyf/liveweb-arena")


def ensure_liveweb_import_path() -> Path:
    import sys

    arena_dir = Path(os.getenv("LIVEWEB_ARENA_DIR", str(DEFAULT_LIVEWEB_ARENA_DIR))).resolve()
    if str(arena_dir) not in sys.path:
        sys.path.insert(0, str(arena_dir))
    return arena_dir


def load_task_mix_config() -> dict[str, Any]:
    path = Path(os.getenv("LIVEWEB_TASK_MIX_CONFIG", str(DEFAULT_TASK_MIX_PATH)))
    return json.loads(path.read_text())


def stable_plugin_allowlist() -> list[str]:
    return load_task_mix_config()["stable_plugins"]


def full_plugin_allowlist() -> list[str]:
    return load_task_mix_config()["full_plugins"]


def phase_plugin_weights(phase: str) -> dict[str, float]:
    cfg = load_task_mix_config()["phases"][phase]
    stable = cfg.get("stable_weight", 1.0)
    full_weight = cfg.get("full_weight", 0.0)

    weights: dict[str, float] = {}
    stable_plugins = stable_plugin_allowlist()
    full_plugins = full_plugin_allowlist()
    stable_share = stable / max(1, len(stable_plugins))
    full_share = full_weight / max(1, len(full_plugins))
    for plugin in stable_plugins:
        weights[plugin] = weights.get(plugin, 0.0) + stable_share
    for plugin in full_plugins:
        weights[plugin] = weights.get(plugin, 0.0) + full_share
    return weights


def parse_plugin_csv_env(name: str, default: str = "") -> set[str]:
    raw = os.getenv(name, default)
    return {item.strip() for item in raw.split(",") if item.strip()}


def registry_symbols():
    import importlib.util

    arena_dir = ensure_liveweb_import_path()
    module_path = arena_dir / "liveweb_arena" / "core" / "task_registry.py"
    spec = importlib.util.spec_from_file_location("liveweb_arena_task_registry", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load TaskRegistry from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TaskRegistry, module.parse_task_id


@dataclass(frozen=True)
class TaskComboCandidate:
    combo_index: int
    template_ids: tuple[int, ...]
    templates: tuple[tuple[str, str], ...]
    plugin_names: tuple[str, ...]
    combo_key: str


def build_combo_candidates(*, excluded_plugins: set[str], min_unique_plugins: int) -> list[TaskComboCandidate]:
    TaskRegistry, _ = registry_symbols()
    candidates: list[TaskComboCandidate] = []
    for combo_index, template_ids in enumerate(TaskRegistry._combinations):
        templates = tuple(TaskRegistry.TEMPLATES[tid] for tid in template_ids)
        plugin_names = tuple(sorted({plugin for plugin, _ in templates}))
        if len(plugin_names) < min_unique_plugins:
            continue
        if excluded_plugins and any(plugin in excluded_plugins for plugin in plugin_names):
            continue
        candidates.append(
            TaskComboCandidate(
                combo_index=combo_index,
                template_ids=tuple(template_ids),
                templates=templates,
                plugin_names=plugin_names,
                combo_key="+".join(plugin_names),
            )
        )
    if not candidates:
        raise RuntimeError(
            "No LIVEWEB task combinations available after applying exclusion and min plugin filters"
        )
    return candidates


def _weighted_choice(rng: random.Random, items: list[TaskComboCandidate], weights: list[float]) -> TaskComboCandidate:
    total = sum(weights)
    if total <= 0:
        return rng.choice(items)
    x = rng.random() * total
    cumulative = 0.0
    for item, weight in zip(items, weights, strict=True):
        cumulative += weight
        if x <= cumulative:
            return item
    return items[-1]


class LiveWebDynamicSampler:
    def __init__(
        self,
        *,
        excluded_plugins: set[str] | None = None,
        min_unique_plugins: int = 2,
        window_size: int = 64,
        dynamic_mix: tuple[float, float, float] = (0.5, 0.3, 0.2),
    ):
        self.excluded_plugins = excluded_plugins or set()
        self.min_unique_plugins = min_unique_plugins
        self.window_size = window_size
        self.dynamic_mix = dynamic_mix
        self.candidates = build_combo_candidates(
            excluded_plugins=self.excluded_plugins,
            min_unique_plugins=self.min_unique_plugins,
        )
        self._history: dict[str, deque[dict[str, float]]] = defaultdict(lambda: deque(maxlen=self.window_size))

    def export_state(self) -> dict[str, Any]:
        return {
            "excluded_plugins": sorted(self.excluded_plugins),
            "min_unique_plugins": self.min_unique_plugins,
            "window_size": self.window_size,
            "dynamic_mix": list(self.dynamic_mix),
            "history": {key: list(value) for key, value in self._history.items()},
        }

    def load_state(self, state: dict[str, Any]) -> None:
        history = state.get("history") or {}
        self._history.clear()
        for key, records in history.items():
            self._history[key] = deque((dict(record) for record in records), maxlen=self.window_size)

    def _base_weight(self, candidate: TaskComboCandidate, phase: str) -> float:
        weights = phase_plugin_weights(phase)
        plugin_weights = [weights.get(plugin, 0.0) for plugin in candidate.plugin_names]
        if not plugin_weights:
            return 1.0
        mean_weight = sum(plugin_weights) / len(plugin_weights)
        return max(mean_weight, 0.01)

    def summarize_combo(self, combo_key: str) -> dict[str, float]:
        records = list(self._history.get(combo_key, ()))
        if not records:
            return {
                "mean_score": 0.0,
                "success_rate": 0.0,
                "reward_variance": 0.0,
                "env_error_rate": 0.0,
                "accepted_rate": 0.0,
                "zero_std_rate": 0.0,
                "count": 0.0,
            }

        scores = [float(record["mean_score"]) for record in records]
        mean_score = sum(scores) / len(scores)
        reward_variance = 0.0
        if len(scores) > 1:
            centered = [(score - mean_score) ** 2 for score in scores]
            reward_variance = sum(centered) / len(centered)

        return {
            "mean_score": mean_score,
            "success_rate": sum(float(record["success_rate"]) for record in records) / len(records),
            "reward_variance": reward_variance,
            "env_error_rate": sum(float(record["env_error_rate"]) for record in records) / len(records),
            "accepted_rate": sum(float(record["accepted"]) for record in records) / len(records),
            "zero_std_rate": sum(float(record["zero_std"]) for record in records) / len(records),
            "count": float(len(records)),
        }

    def compute_dynamic_weight(self, candidate: TaskComboCandidate, *, phase: str) -> float:
        base = self._base_weight(candidate, phase)
        summary = self.summarize_combo(candidate.combo_key)
        if summary["count"] <= 0:
            return base

        utility_weight = min(2.5, max(0.25, 0.5 + (summary["reward_variance"] * 4.0) + summary["accepted_rate"]))

        success_rate = summary["success_rate"]
        if success_rate >= 0.8:
            difficulty_weight = 0.5
        elif success_rate <= 0.05 and summary["mean_score"] <= 0.05:
            difficulty_weight = 0.6
        else:
            difficulty_weight = 1.25 - abs(success_rate - 0.3) * 2.0
            difficulty_weight = min(1.5, max(0.5, difficulty_weight))

        noise_penalty = max(0.25, 1.0 - summary["env_error_rate"])
        zero_std_penalty = max(0.25, 1.0 - (summary["zero_std_rate"] * 0.75))
        return base * utility_weight * difficulty_weight * noise_penalty * zero_std_penalty

    def sample(self, *, seed: int, phase: str, evaluation: bool = False) -> dict[str, Any]:
        TaskRegistry, parse_task_id = registry_symbols()
        rng = random.Random(seed)
        if evaluation:
            strategy = "base"
        else:
            dynamic_ratio, base_ratio, _uniform_ratio = self.dynamic_mix
            draw = rng.random()
            if draw < dynamic_ratio:
                strategy = "dynamic"
            elif draw < dynamic_ratio + base_ratio:
                strategy = "base"
            else:
                strategy = "uniform"

        if strategy == "uniform":
            candidate = rng.choice(self.candidates)
            final_weight = 1.0
        elif strategy == "base":
            weights = [self._base_weight(candidate, phase) for candidate in self.candidates]
            candidate = _weighted_choice(rng, self.candidates, weights)
            final_weight = self._base_weight(candidate, phase)
        else:
            dynamic_weights = [self.compute_dynamic_weight(candidate, phase=phase) for candidate in self.candidates]
            floor = max(0.01, (sum(dynamic_weights) / max(1, len(dynamic_weights))) * 0.25)
            dynamic_weights = [max(floor, weight) for weight in dynamic_weights]
            candidate = _weighted_choice(rng, self.candidates, dynamic_weights)
            final_weight = self.compute_dynamic_weight(candidate, phase=phase)

        variation_seed = rng.randrange(TaskRegistry.TASK_IDS_PER_COMBO)
        task_id = candidate.combo_index * TaskRegistry.TASK_IDS_PER_COMBO + variation_seed + 1
        config = parse_task_id(task_id)
        return {
            "task_id": task_id,
            "task_seed": int(config["variation_seed"]),
            "combo_index": candidate.combo_index,
            "combo_key": candidate.combo_key,
            "plugin_names": list(candidate.plugin_names),
            "templates": list(config["templates"]),
            "num_subtasks": int(config["num_tasks"]),
            "sampling_strategy": strategy,
            "base_weight": self._base_weight(candidate, phase),
            "final_weight": final_weight,
        }

    def record_group_feedback(self, feedback: list[dict[str, Any]]) -> None:
        for item in feedback:
            combo_key = str(item.get("combo_key") or "")
            if not combo_key:
                continue
            self._history[combo_key].append(
                {
                    "mean_score": float(item.get("mean_score", 0.0)),
                    "success_rate": float(item.get("success_rate", 0.0)),
                    "env_error_rate": float(item.get("env_error_rate", 0.0)),
                    "accepted": 1.0 if item.get("accepted") else 0.0,
                    "zero_std": 1.0 if item.get("zero_std") else 0.0,
                }
            )

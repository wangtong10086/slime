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
FAILURE_BUCKET_NAMES = ("normal", "wrong_domain_loop", "premature_stop", "near_miss")

PHASE_ALIAS_FALLBACK = {
    "warmup": "bootstrap",
    "main": "main_warm",
}


def ensure_liveweb_import_path() -> Path:
    import sys

    arena_dir = Path(os.getenv("LIVEWEB_ARENA_DIR", str(DEFAULT_LIVEWEB_ARENA_DIR))).resolve()
    if str(arena_dir) not in sys.path:
        sys.path.insert(0, str(arena_dir))
    return arena_dir


def load_task_mix_config() -> dict[str, Any]:
    path = Path(os.getenv("LIVEWEB_TASK_MIX_CONFIG", str(DEFAULT_TASK_MIX_PATH)))
    return json.loads(path.read_text())


def canonicalize_phase_name(phase: str) -> str:
    config = load_task_mix_config()
    aliases = dict(PHASE_ALIAS_FALLBACK)
    aliases.update(config.get("phase_aliases") or {})
    return aliases.get(phase, phase)


def _phase_config(phase: str) -> dict[str, Any]:
    config = load_task_mix_config()
    canonical = canonicalize_phase_name(phase)
    phase_cfg = (config.get("phases") or {}).get(canonical)
    if phase_cfg is None:
        raise KeyError(f"Unknown LIVEWEB phase: {phase} (canonical={canonical})")
    return phase_cfg


def stable_plugin_allowlist() -> list[str]:
    return load_task_mix_config()["stable_plugins"]


def full_plugin_allowlist() -> list[str]:
    return load_task_mix_config()["full_plugins"]


def phase_plugin_weights(phase: str) -> dict[str, float]:
    cfg = _phase_config(phase)
    explicit = cfg.get("plugin_weights")
    if explicit:
        total = sum(max(0.0, float(value)) for value in explicit.values())
        if total > 0:
            return {
                str(plugin): max(0.0, float(value)) / total
                for plugin, value in explicit.items()
                if float(value) > 0
            }
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


def _parse_bucket_ratio_env() -> dict[str, float]:
    ratios = {
        "normal": float(os.getenv("LIVEWEB_FAILURE_BUCKET_NORMAL_RATIO", "" ) or 0.0),
        "wrong_domain_loop": float(os.getenv("LIVEWEB_FAILURE_BUCKET_WRONG_DOMAIN_RATIO", "" ) or 0.0),
        "premature_stop": float(os.getenv("LIVEWEB_FAILURE_BUCKET_PREMATURE_STOP_RATIO", "" ) or 0.0),
        "near_miss": float(os.getenv("LIVEWEB_FAILURE_BUCKET_NEAR_MISS_RATIO", "" ) or 0.0),
    }
    if not any(value > 0 for value in ratios.values()):
        phase = os.getenv("LIVEWEB_TASK_MIX_PHASE", "bootstrap")
        phase_cfg = _phase_config(phase)
        ratios = {
            key: float((phase_cfg.get("bucket_ratios") or {}).get(key, 0.0))
            for key in FAILURE_BUCKET_NAMES
        }
    total = sum(max(0.0, value) for value in ratios.values())
    if total <= 0:
        return {"normal": 1.0, "wrong_domain_loop": 0.0, "premature_stop": 0.0, "near_miss": 0.0}
    return {key: max(0.0, value) / total for key, value in ratios.items()}


def _num_task_bounds() -> tuple[int, int]:
    phase_cfg = _phase_config(os.getenv("LIVEWEB_TASK_MIX_PHASE", "bootstrap"))
    num_task_weights = phase_cfg.get("num_task_weights") or {}
    configured_tasks = sorted(int(key) for key, value in num_task_weights.items() if float(value) > 0)
    min_tasks = int(os.getenv("LIVEWEB_MIN_NUM_TASKS", str(configured_tasks[0] if configured_tasks else 1)))
    max_tasks = int(os.getenv("LIVEWEB_MAX_NUM_TASKS", str(configured_tasks[-1] if configured_tasks else 4)))
    if min_tasks > max_tasks:
        raise ValueError(f"Invalid LIVEWEB num-task bounds: min={min_tasks} max={max_tasks}")
    return min_tasks, max_tasks


def phase_num_task_weights(phase: str) -> dict[int, float]:
    phase_cfg = _phase_config(phase)
    weights = {
        int(key): max(0.0, float(value))
        for key, value in (phase_cfg.get("num_task_weights") or {}).items()
    }
    total = sum(weights.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in weights.items() if value > 0}


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
    min_tasks, max_tasks = _num_task_bounds()
    candidates: list[TaskComboCandidate] = []
    for combo_index, template_ids in enumerate(TaskRegistry._combinations):
        templates = tuple(TaskRegistry.TEMPLATES[tid] for tid in template_ids)
        plugin_names = tuple(sorted({plugin for plugin, _ in templates}))
        if not (min_tasks <= len(template_ids) <= max_tasks):
            continue
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
        self._candidate_by_combo_index = {candidate.combo_index: candidate for candidate in self.candidates}
        self._history: dict[str, deque[dict[str, float]]] = defaultdict(lambda: deque(maxlen=self.window_size))
        self._failure_bucket_mix = _parse_bucket_ratio_env()
        self._bucket_window = int(os.getenv("LIVEWEB_FAILURE_BUCKET_WINDOW", "256"))
        self._bucket_pools: dict[str, deque[int]] = {
            name: deque(maxlen=self._bucket_window) for name in FAILURE_BUCKET_NAMES if name != "normal"
        }
        self._bucket_seen_task_ids: dict[str, set[int]] = {
            name: set() for name in FAILURE_BUCKET_NAMES if name != "normal"
        }
        self._bootstrap_failure_buckets()
        self._current_phase = canonicalize_phase_name(os.getenv("LIVEWEB_TASK_MIX_PHASE", "bootstrap"))

    def export_state(self) -> dict[str, Any]:
        return {
            "excluded_plugins": sorted(self.excluded_plugins),
            "min_unique_plugins": self.min_unique_plugins,
            "window_size": self.window_size,
            "dynamic_mix": list(self.dynamic_mix),
            "current_phase": self._current_phase,
            "history": {key: list(value) for key, value in self._history.items()},
            "failure_bucket_mix": dict(self._failure_bucket_mix),
            "failure_bucket_pools": {key: list(value) for key, value in self._bucket_pools.items()},
        }

    def load_state(self, state: dict[str, Any]) -> None:
        history = state.get("history") or {}
        self._current_phase = canonicalize_phase_name(str(state.get("current_phase", self._current_phase)))
        self._history.clear()
        for key, records in history.items():
            self._history[key] = deque((dict(record) for record in records), maxlen=self.window_size)
        bucket_pools = state.get("failure_bucket_pools") or {}
        for bucket_name, records in bucket_pools.items():
            if bucket_name not in self._bucket_pools:
                continue
            normalized = [int(task_id) for task_id in records]
            self._bucket_pools[bucket_name] = deque(normalized, maxlen=self._bucket_window)
            self._bucket_seen_task_ids[bucket_name] = set(normalized)

    def _register_bucket_task(self, bucket_name: str, task_id: int) -> None:
        if bucket_name not in self._bucket_pools:
            return
        if task_id in self._bucket_seen_task_ids[bucket_name]:
            return
        self._bucket_pools[bucket_name].append(task_id)
        self._bucket_seen_task_ids[bucket_name].add(task_id)

    def _bootstrap_failure_buckets(self) -> None:
        raw_dirs = os.getenv("LIVEWEB_FAILURE_BUCKET_RESULTS_DIRS", "").strip()
        if not raw_dirs:
            return
        for raw_dir in [item.strip() for item in raw_dirs.split(":") if item.strip()]:
            path = Path(raw_dir)
            if not path.exists():
                continue
            for json_path in path.rglob("task_*.json"):
                try:
                    payload = json.loads(json_path.read_text())
                except Exception:
                    continue
                extra = payload.get("extra") or {}
                task_id = extra.get("task_id")
                bucket_name = extra.get("rl_failure_bucket")
                if bucket_name in self._bucket_pools and task_id is not None:
                    self._register_bucket_task(str(bucket_name), int(task_id))

    def _selection_from_task_id(self, task_id: int) -> dict[str, Any]:
        TaskRegistry, parse_task_id = registry_symbols()
        config = parse_task_id(int(task_id))
        combo_index = (int(task_id) - 1) // TaskRegistry.TASK_IDS_PER_COMBO
        candidate = self._candidate_by_combo_index.get(combo_index)
        if candidate is None:
            raise RuntimeError(f"Unable to map task_id={task_id} to candidate combo index {combo_index}")
        return {
            "task_id": int(task_id),
            "task_seed": int(config["variation_seed"]),
            "combo_index": candidate.combo_index,
            "combo_key": candidate.combo_key,
            "plugin_names": list(candidate.plugin_names),
            "templates": list(config["templates"]),
            "num_subtasks": int(config["num_tasks"]),
        }

    def _choose_failure_bucket(self, rng: random.Random) -> str:
        active = {"normal": self._failure_bucket_mix.get("normal", 1.0)}
        for bucket_name, pool in self._bucket_pools.items():
            if pool:
                active[bucket_name] = self._failure_bucket_mix.get(bucket_name, 0.0)
        total = sum(max(0.0, value) for value in active.values())
        if total <= 0:
            return "normal"
        draw = rng.random() * total
        cumulative = 0.0
        for bucket_name, value in active.items():
            cumulative += max(0.0, value)
            if draw <= cumulative:
                return bucket_name
        return "normal"

    def _base_weight(self, candidate: TaskComboCandidate, phase: str) -> float:
        canonical_phase = canonicalize_phase_name(phase)
        weights = phase_plugin_weights(canonical_phase)
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
                "mean_progress_score": 0.0,
                "near_miss_rate": 0.0,
                "format_failure_rate": 0.0,
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
            "mean_progress_score": sum(float(record["mean_progress_score"]) for record in records) / len(records),
            "near_miss_rate": sum(float(record["near_miss_rate"]) for record in records) / len(records),
            "format_failure_rate": sum(float(record["format_failure_rate"]) for record in records) / len(records),
            "count": float(len(records)),
        }

    def compute_dynamic_weight(self, candidate: TaskComboCandidate, *, phase: str) -> float:
        canonical_phase = canonicalize_phase_name(phase)
        base = self._base_weight(candidate, canonical_phase)
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

        if canonical_phase == "bootstrap":
            penalty_floor = 0.10
            noise_penalty = max(penalty_floor, 1.0 - (summary["env_error_rate"] * 1.5))
            zero_std_penalty = max(penalty_floor, 1.0 - summary["zero_std_rate"])
        else:
            penalty_floor = 0.25
            noise_penalty = max(penalty_floor, 1.0 - summary["env_error_rate"])
            zero_std_penalty = max(penalty_floor, 1.0 - (summary["zero_std_rate"] * 0.75))

        site_penalty = 1.0
        plugin_set = set(candidate.plugin_names)
        if canonical_phase in {"main_warm", "online_align"}:
            if "taostats" in plugin_set:
                site_penalty *= max(0.20, 1.0 - (summary["env_error_rate"] * 1.6))
            if "stooq" in plugin_set:
                site_penalty *= max(0.30, 1.0 - (summary["env_error_rate"] * 1.25))
            if "coingecko" in plugin_set:
                site_penalty *= max(0.40, 1.0 - (summary["env_error_rate"] * 1.0))
        progress_bonus = 1.0
        format_penalty = 1.0
        if os.getenv("LIVEWEB_ENABLE_PROGRESS_AWARE_SAMPLER", "0") == "1":
            progress_bonus = min(1.5, max(0.85, 0.85 + summary["mean_progress_score"]))
            format_penalty = min(1.0, max(0.75, 1.0 - (0.5 * summary["format_failure_rate"])))
        return (
            base
            * utility_weight
            * difficulty_weight
            * noise_penalty
            * zero_std_penalty
            * site_penalty
            * progress_bonus
            * format_penalty
        )

    def sample(self, *, seed: int, phase: str, evaluation: bool = False) -> dict[str, Any]:
        TaskRegistry, parse_task_id = registry_symbols()
        canonical_phase = canonicalize_phase_name(phase)
        self._current_phase = canonical_phase
        min_tasks, max_tasks = _num_task_bounds()
        rng = random.Random(seed)
        if evaluation:
            strategy = "base"
            bucket_name = "normal"
        else:
            bucket_name = self._choose_failure_bucket(rng)
            dynamic_ratio, base_ratio, _uniform_ratio = self.dynamic_mix
            draw = rng.random()
            if draw < dynamic_ratio:
                strategy = "dynamic"
            elif draw < dynamic_ratio + base_ratio:
                strategy = "base"
            else:
                strategy = "uniform"

        if bucket_name != "normal":
            bucket_tasks = list(self._bucket_pools.get(bucket_name) or [])
            rng.shuffle(bucket_tasks)
            for bucket_task_id in bucket_tasks:
                try:
                    selection = self._selection_from_task_id(bucket_task_id)
                except Exception:
                    continue
                if min_tasks <= int(selection["num_subtasks"]) <= max_tasks:
                    return {
                        **selection,
                        "sampling_strategy": f"bucket:{bucket_name}",
                        "failure_bucket": bucket_name,
                        "base_weight": 1.0,
                        "final_weight": 1.0,
                    }
            bucket_name = "normal"

        if strategy == "uniform":
            candidate = rng.choice(self.candidates)
            final_weight = 1.0
        elif strategy == "base":
            weights = [self._base_weight(candidate, canonical_phase) for candidate in self.candidates]
            candidate = _weighted_choice(rng, self.candidates, weights)
            final_weight = self._base_weight(candidate, canonical_phase)
        else:
            dynamic_weights = [self.compute_dynamic_weight(candidate, phase=canonical_phase) for candidate in self.candidates]
            floor = max(0.01, (sum(dynamic_weights) / max(1, len(dynamic_weights))) * 0.25)
            dynamic_weights = [max(floor, weight) for weight in dynamic_weights]
            candidate = _weighted_choice(rng, self.candidates, dynamic_weights)
            final_weight = self.compute_dynamic_weight(candidate, phase=canonical_phase)

        task_weights = phase_num_task_weights(canonical_phase)
        allowed_task_counts = [task_count for task_count in sorted(task_weights) if min_tasks <= task_count <= max_tasks]
        if not allowed_task_counts:
            allowed_task_counts = list(range(min_tasks, max_tasks + 1))
        if task_weights:
            task_weight_values = [task_weights.get(task_count, 0.0) for task_count in allowed_task_counts]
            total_task_weight = sum(task_weight_values)
            if total_task_weight > 0:
                x = rng.random() * total_task_weight
                cumulative = 0.0
                target_num_tasks = allowed_task_counts[-1]
                for task_count, weight in zip(allowed_task_counts, task_weight_values, strict=True):
                    cumulative += weight
                    if x <= cumulative:
                        target_num_tasks = task_count
                        break
            else:
                target_num_tasks = rng.choice(allowed_task_counts)
        else:
            target_num_tasks = rng.choice(allowed_task_counts)

        config = None
        task_id = None
        for _ in range(TaskRegistry.TASK_IDS_PER_COMBO * 2):
            variation_seed = rng.randrange(TaskRegistry.TASK_IDS_PER_COMBO)
            task_id = candidate.combo_index * TaskRegistry.TASK_IDS_PER_COMBO + variation_seed + 1
            config = parse_task_id(task_id)
            if int(config["num_tasks"]) == target_num_tasks and min_tasks <= int(config["num_tasks"]) <= max_tasks:
                break
        if config is None or task_id is None or int(config["num_tasks"]) != target_num_tasks:
            for _ in range(TaskRegistry.TASK_IDS_PER_COMBO):
                variation_seed = rng.randrange(TaskRegistry.TASK_IDS_PER_COMBO)
                task_id = candidate.combo_index * TaskRegistry.TASK_IDS_PER_COMBO + variation_seed + 1
                config = parse_task_id(task_id)
                if min_tasks <= int(config["num_tasks"]) <= max_tasks:
                    break
        if config is None or task_id is None or not (min_tasks <= int(config["num_tasks"]) <= max_tasks):
            raise RuntimeError(
                f"Unable to sample LIVEWEB task with num_tasks in [{min_tasks}, {max_tasks}] from combo {candidate.combo_key}"
            )
        return {
            "task_id": task_id,
            "task_seed": int(config["variation_seed"]),
            "combo_index": candidate.combo_index,
            "combo_key": candidate.combo_key,
            "plugin_names": list(candidate.plugin_names),
            "templates": list(config["templates"]),
            "num_subtasks": int(config["num_tasks"]),
            "sampling_strategy": strategy,
            "failure_bucket": bucket_name,
            "base_weight": self._base_weight(candidate, canonical_phase),
            "final_weight": final_weight,
            "phase": canonical_phase,
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
                    "mean_progress_score": float(item.get("mean_progress_score", 0.0)),
                    "near_miss_rate": float(item.get("near_miss_rate", 0.0)),
                    "format_failure_rate": float(item.get("format_failure_rate", 0.0)),
                }
            )
            for task_record in item.get("task_records") or []:
                bucket_name = str(task_record.get("rl_failure_bucket") or "")
                task_id = task_record.get("task_id")
                if bucket_name in self._bucket_pools and task_id is not None:
                    self._register_bucket_task(bucket_name, int(task_id))

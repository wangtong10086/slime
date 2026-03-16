import dataclasses
import os
from typing import Any


@dataclasses.dataclass
class StepPlan:
    retained_indices: list[int]
    step_boundaries: list[int]
    step_token_counts: list[int]
    step_num_samples: list[int]
    step_long_sample_counts: list[int]
    dynamic_global_batch_size: int
    requested_tokens: int
    retained_tokens: int
    trimmed_tokens: int
    oversize_samples_dropped: int
    underfilled_steps: int
    long_samples_trimmed: int
    density_restricted_steps: int


def compute_sample_token_cost(sample: Any) -> int:
    total_length = getattr(sample, "total_length", None)
    if isinstance(total_length, int) and total_length > 0:
        return total_length

    if isinstance(sample, dict):
        total_length = sample.get("total_length")
        if isinstance(total_length, int) and total_length > 0:
            return total_length
        tokens = sample.get("tokens")
        if tokens is not None:
            return len(tokens)

    tokens = getattr(sample, "tokens", None)
    if tokens is not None:
        return len(tokens)

    raise ValueError(f"Cannot infer token cost from sample of type {type(sample)!r}")


def choose_dynamic_global_batch_size(
    *,
    num_samples: int,
    dp_size: int,
    original_gbs: int,
    configured_cap: int = 0,
    configured_min: int = 0,
) -> int:
    """Choose a per-step global batch size that maximizes sample use."""
    max_gbs = (num_samples // dp_size) * dp_size
    if configured_cap > 0:
        max_gbs = min(max_gbs, max(dp_size, (configured_cap // dp_size) * dp_size))

    if configured_min <= 0:
        configured_min = max(dp_size, original_gbs // 2)
    min_gbs = max(dp_size, (configured_min // dp_size) * dp_size)
    min_gbs = min(min_gbs, max_gbs) if max_gbs > 0 else dp_size

    if num_samples < min_gbs:
        dynamic_gbs = max(dp_size, (num_samples // dp_size) * dp_size)
    else:
        dynamic_gbs = max_gbs
        best_remainder = num_samples % dynamic_gbs if dynamic_gbs > 0 else num_samples
        for candidate in range(max_gbs, min_gbs - dp_size, -dp_size):
            if candidate <= 0:
                continue
            remainder = num_samples % candidate
            if remainder < best_remainder:
                dynamic_gbs = candidate
                best_remainder = remainder
                if remainder == 0:
                    break

    return dynamic_gbs if dynamic_gbs > 0 else dp_size


def resolve_max_samples_per_rollout(global_batch_size: int) -> int | None:
    """Return the configured upper bound on trainable samples per rollout."""
    configured = int(os.environ.get("TRAIN_MAX_SAMPLES_PER_ROLLOUT", "0") or "0")
    if configured <= 0:
        return None

    capped = (configured // global_batch_size) * global_batch_size
    if capped <= 0:
        return global_batch_size
    return capped


def compute_train_trim_length(
    num_samples: int, global_batch_size: int, max_samples_per_rollout: int | None = None
) -> int:
    """Keep the largest trainable sample count as multiples of ``global_batch_size``."""
    if num_samples < global_batch_size:
        return 0

    trim_len = (num_samples // global_batch_size) * global_batch_size
    if max_samples_per_rollout is not None:
        trim_len = min(trim_len, max_samples_per_rollout)
    return trim_len


def plan_train_steps_by_token_budget(
    samples: list[Any],
    *,
    max_samples_per_step: int,
    min_samples_per_step: int,
    underfilled_min_samples: int,
    step_token_budget: int,
    max_samples_per_rollout: int | None = None,
    packing_strategy: str = "greedy_desc",
    long_sample_threshold: int | None = None,
    max_long_samples_per_step: int | None = None,
    max_single_sample_tokens_per_step: int | None = None,
) -> StepPlan:
    if packing_strategy != "greedy_desc":
        raise ValueError(f"Unsupported packing strategy: {packing_strategy}")
    if step_token_budget <= 0:
        raise ValueError(f"step_token_budget must be positive, got {step_token_budget}")
    if max_samples_per_step <= 0:
        raise ValueError(f"max_samples_per_step must be positive, got {max_samples_per_step}")
    if long_sample_threshold is None or long_sample_threshold <= 0:
        long_sample_threshold = max(1, int(step_token_budget * 0.35))
    if max_long_samples_per_step is None or max_long_samples_per_step <= 0:
        max_long_samples_per_step = max_samples_per_step
    if max_single_sample_tokens_per_step is None or max_single_sample_tokens_per_step <= 0:
        max_single_sample_tokens_per_step = step_token_budget

    sample_costs = [compute_sample_token_cost(sample) for sample in samples]
    requested_tokens = sum(sample_costs)

    kept_items: list[tuple[int, int]] = []
    trimmed_tokens = 0
    oversize_samples_dropped = 0
    long_samples_trimmed = 0
    for index, cost in enumerate(sample_costs):
        is_long = cost >= long_sample_threshold
        if cost > step_token_budget or cost > max_single_sample_tokens_per_step:
            oversize_samples_dropped += 1
            trimmed_tokens += cost
            if is_long:
                long_samples_trimmed += 1
            continue
        kept_items.append((index, cost))

    kept_items.sort(key=lambda item: (-item[1], item[0]))
    bins: list[dict[str, list[int] | int]] = []
    density_restricted_steps = 0
    for index, cost in kept_items:
        is_long = cost >= long_sample_threshold
        placed = False
        feasible_bins: list[tuple[int, int]] = []
        density_restricted = False
        for bin_idx, current in enumerate(bins):
            current_tokens = int(current["tokens"])
            current_indices = current["indices"]
            assert isinstance(current_indices, list)
            if len(current_indices) >= max_samples_per_step:
                continue
            if current_tokens + cost > step_token_budget:
                continue
            current_long_count = int(current["long_samples"])
            if is_long and current_long_count + 1 > max_long_samples_per_step:
                density_restricted = True
                continue
            feasible_bins.append((step_token_budget - (current_tokens + cost), bin_idx))
        if feasible_bins:
            _, chosen_idx = min(feasible_bins, key=lambda item: item[0])
            current = bins[chosen_idx]
            current_indices = current["indices"]
            assert isinstance(current_indices, list)
            current_indices.append(index)
            current["tokens"] = int(current["tokens"]) + cost
            if is_long:
                current["long_samples"] = int(current["long_samples"]) + 1
            placed = True
        if density_restricted and not placed:
            density_restricted_steps += 1
        if not placed:
            bins.append({"indices": [index], "tokens": cost, "long_samples": 1 if is_long else 0})

    if max_samples_per_rollout is not None and max_samples_per_rollout > 0:
        capped_bins: list[dict[str, list[int] | int]] = []
        retained_so_far = 0
        for current in bins:
            current_indices = current["indices"]
            assert isinstance(current_indices, list)
            current_count = len(current_indices)
            if retained_so_far + current_count > max_samples_per_rollout:
                trimmed_tokens += int(current["tokens"])
                long_samples_trimmed += int(current["long_samples"])
                continue
            capped_bins.append(current)
            retained_so_far += current_count
        bins = capped_bins

    kept_bins: list[dict[str, list[int] | int]] = []
    underfilled_steps = 0
    for bin_index, current in enumerate(bins):
        current_indices = current["indices"]
        assert isinstance(current_indices, list)
        current_count = len(current_indices)
        if current_count < underfilled_min_samples and len(bins) > 1:
            trimmed_tokens += int(current["tokens"])
            long_samples_trimmed += int(current["long_samples"])
            continue
        if current_count < min_samples_per_step:
            underfilled_steps += 1
        kept_bins.append(current)

    retained_indices: list[int] = []
    step_boundaries = [0]
    step_token_counts: list[int] = []
    step_num_samples: list[int] = []
    step_long_sample_counts: list[int] = []
    retained_tokens = 0
    for current in kept_bins:
        current_indices = current["indices"]
        assert isinstance(current_indices, list)
        retained_indices.extend(current_indices)
        retained_tokens += int(current["tokens"])
        step_token_counts.append(int(current["tokens"]))
        step_num_samples.append(len(current_indices))
        step_long_sample_counts.append(int(current["long_samples"]))
        step_boundaries.append(len(retained_indices))

    trimmed_tokens = requested_tokens - retained_tokens
    dynamic_global_batch_size = max(step_num_samples, default=0)

    return StepPlan(
        retained_indices=retained_indices,
        step_boundaries=step_boundaries,
        step_token_counts=step_token_counts,
        step_num_samples=step_num_samples,
        step_long_sample_counts=step_long_sample_counts,
        dynamic_global_batch_size=dynamic_global_batch_size,
        requested_tokens=requested_tokens,
        retained_tokens=retained_tokens,
        trimmed_tokens=trimmed_tokens,
        oversize_samples_dropped=oversize_samples_dropped,
        underfilled_steps=underfilled_steps,
        long_samples_trimmed=long_samples_trimmed,
        density_restricted_steps=density_restricted_steps,
    )

import os


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

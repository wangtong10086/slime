import os


def parse_mem_fraction_overrides(spec: str | None) -> dict[int, float]:
    """Parse per-GPU mem-fraction overrides from ``GPU_ID:FRACTION`` pairs."""
    if not spec:
        return {}

    overrides: dict[int, float] = {}
    for raw_entry in spec.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        gpu_id_text, sep, fraction_text = entry.partition(":")
        if not sep:
            raise ValueError(
                "SGLANG_MEM_FRACTION_STATIC_BY_GPU_ID entries must use GPU_ID:FRACTION, "
                f"got {raw_entry!r}"
            )
        gpu_id = int(gpu_id_text.strip())
        fraction = float(fraction_text.strip())
        if fraction <= 0 or fraction > 1:
            raise ValueError(f"mem fraction override must be in (0, 1], got {fraction!r} for GPU {gpu_id}")
        overrides[gpu_id] = fraction
    return overrides


def resolve_engine_mem_fraction(
    *,
    default_fraction: float | None,
    base_gpu_id: int,
    num_gpus_per_engine: int,
    gpu_id_step: int = 1,
    overrides_spec: str | None = None,
) -> float | None:
    """Resolve the mem-fraction budget for one rollout engine.

    If any GPU covered by the engine has an override, we use the minimum value
    across the matching GPUs and the default fraction. Using the minimum keeps
    the whole TP group within the most conservative budget.
    """
    overrides = parse_mem_fraction_overrides(
        overrides_spec
        if overrides_spec is not None
        else os.environ.get("SGLANG_MEM_FRACTION_STATIC_BY_GPU_ID")
    )
    if not overrides:
        return default_fraction

    engine_gpu_ids = [base_gpu_id + idx * gpu_id_step for idx in range(num_gpus_per_engine)]
    matched = [overrides[gpu_id] for gpu_id in engine_gpu_ids if gpu_id in overrides]
    if not matched:
        return default_fraction

    if default_fraction is None:
        return min(matched)
    return min([default_fraction, *matched])

import math
from argparse import Namespace


def next_train_log_step(args: Namespace) -> int:
    """Return a run-local train step for logging."""

    step = int(getattr(args, "_liveweb_train_log_step", 0))
    setattr(args, "_liveweb_train_log_step", step + 1)
    return step


def normalize_grad_norm_for_logging(raw_grad_norm: float, clip_grad: float | None) -> tuple[float, float]:
    """Return `(effective_grad_norm, raw_grad_norm)` for dashboard logging."""

    raw = float(raw_grad_norm)
    if not math.isfinite(raw):
        return raw, raw
    if clip_grad is None or clip_grad <= 0:
        return raw, raw
    return min(raw, float(clip_grad)), raw

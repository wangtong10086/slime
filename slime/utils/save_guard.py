from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class SavePlan:
    mode: str
    is_final: bool
    write_rollout_state: bool
    hf_export: bool = False


def read_meminfo_kib() -> dict[str, int]:
    meminfo: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            parts = value.strip().split()
            if parts:
                meminfo[key] = int(parts[0])
    except Exception:
        return {}
    return meminfo


def collect_save_memory_snapshot(limit: int = 8) -> dict[str, Any]:
    meminfo = read_meminfo_kib()
    total_kib = meminfo.get("MemTotal", 0)
    available_kib = meminfo.get("MemAvailable", 0)
    free_kib = meminfo.get("MemFree", 0)
    swap_total_kib = meminfo.get("SwapTotal", 0)
    swap_free_kib = meminfo.get("SwapFree", 0)
    used_ratio = 1.0 - (available_kib / total_kib) if total_kib else 0.0
    top_processes = _top_process_rss_gb(limit=limit)
    return {
        "save/host_mem_total_gb": total_kib / 1024 / 1024,
        "save/host_mem_available_gb": available_kib / 1024 / 1024,
        "save/host_mem_free_gb": free_kib / 1024 / 1024,
        "save/host_mem_used_ratio": used_ratio,
        "save/swap_used_gb": max(0, swap_total_kib - swap_free_kib) / 1024 / 1024,
        "save/ray_memory_threshold": float(os.environ.get("RAY_MEMORY_USAGE_THRESHOLD", "0.99")),
        "save/top_process_rss_gb": top_processes,
    }


def should_skip_save_due_to_memory_guard(
    snapshot: dict[str, Any],
    *,
    min_available_gb: float,
    max_used_ratio: float,
) -> bool:
    available_gb = float(snapshot.get("save/host_mem_available_gb", 0.0) or 0.0)
    used_ratio = float(snapshot.get("save/host_mem_used_ratio", 0.0) or 0.0)
    return available_gb < min_available_gb or used_ratio > max_used_ratio


def resolve_hf_export_policy(*, rollout_id: int, num_rollout: int, force_sync: bool) -> bool:
    if force_sync or rollout_id == num_rollout - 1:
        return True
    if os.environ.get("LIVEWEB_ENABLE_PERIODIC_HF_EXPORT", "0") != "1":
        return False
    interval = max(1, int(os.environ.get("LIVEWEB_HF_EXPORT_INTERVAL", "50")))
    return (rollout_id + 1) % interval == 0


def resolve_save_plan(
    *,
    rollout_id: int,
    num_rollout: int,
    archive_interval: int | None,
    full_interval: int | None,
    final_save_mode: str = "full",
) -> SavePlan | None:
    is_final = rollout_id == num_rollout - 1
    if is_final and final_save_mode == "full":
        return SavePlan(mode="full", is_final=True, write_rollout_state=True)
    if full_interval is not None and full_interval > 0 and (rollout_id + 1) % full_interval == 0:
        return SavePlan(mode="full", is_final=is_final, write_rollout_state=True)
    if archive_interval is not None and archive_interval > 0 and (rollout_id + 1) % archive_interval == 0:
        return SavePlan(mode="archive", is_final=is_final, write_rollout_state=False)
    return None


def resolve_save_mode(
    *,
    rollout_id: int,
    num_rollout: int,
    archive_interval: int | None,
    full_interval: int | None,
    final_save_mode: str = "full",
) -> str | None:
    plan = resolve_save_plan(
        rollout_id=rollout_id,
        num_rollout=num_rollout,
        archive_interval=archive_interval,
        full_interval=full_interval,
        final_save_mode=final_save_mode,
    )
    return plan.mode if plan is not None else None


def wait_for_save_headroom(
    *,
    snapshot_provider: Callable[[], dict[str, Any]],
    status_provider: Callable[[], dict[str, Any]],
    min_available_gb: float,
    max_used_ratio: float,
    timeout_s: float,
    poll_interval_s: float = 5.0,
    require_zero_rollout_actors: bool = False,
) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    deadline = time.time() + timeout_s
    last_snapshot = snapshot_provider()
    last_status = status_provider()

    while True:
        available_gb = float(last_snapshot.get("save/host_mem_available_gb", 0.0) or 0.0)
        used_ratio = float(last_snapshot.get("save/host_mem_used_ratio", 0.0) or 0.0)
        live_rollout_actor_count = int(last_status.get("live_rollout_actor_count", 0) or 0)

        memory_ready = available_gb >= min_available_gb and used_ratio <= max_used_ratio
        rollout_ready = (not require_zero_rollout_actors) or live_rollout_actor_count == 0
        if memory_ready and rollout_ready:
            return True, last_snapshot, last_status

        if time.time() >= deadline:
            return False, last_snapshot, last_status

        time.sleep(poll_interval_s)
        last_snapshot = snapshot_provider()
        last_status = status_provider()


def _top_process_rss_gb(limit: int = 8) -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid,rss,comm", "--sort=-rss"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []

    lines = result.stdout.strip().splitlines()[1 : limit + 1]
    top: list[dict[str, Any]] = []
    for line in lines:
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, rss_kib, command = parts
        try:
            rss_gb = int(rss_kib) / 1024 / 1024
        except ValueError:
            continue
        top.append({"pid": int(pid), "rss_gb": round(rss_gb, 2), "command": command})
    return top

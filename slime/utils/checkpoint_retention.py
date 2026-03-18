from __future__ import annotations

import logging
import shutil
from pathlib import Path
import re

logger = logging.getLogger(__name__)

ITER_PREFIX = "iter_"
STAGING_PREFIX = ".tmp_iter_"
INCOMPLETE_SUFFIX = ".incomplete"


def parse_iteration_from_name(name: str) -> int | None:
    if not name.startswith(ITER_PREFIX):
        return None
    suffix = name[len(ITER_PREFIX) :]
    if not suffix.isdigit():
        return None
    return int(suffix)


def read_latest_complete_iteration(checkpoint_root: str | Path) -> int | None:
    path = Path(checkpoint_root) / "latest_checkpointed_iteration.txt"
    if not path.is_file():
        return None
    raw = path.read_text().strip()
    if not raw or not raw.isdigit():
        return None
    return int(raw)


def list_training_checkpoint_dirs(checkpoint_root: str | Path) -> list[tuple[int, Path]]:
    root = Path(checkpoint_root)
    checkpoints: list[tuple[int, Path]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        iteration = parse_iteration_from_name(child.name)
        if iteration is None:
            continue
        checkpoints.append((iteration, child))
    return sorted(checkpoints)


def infer_hf_export_path(save_hf_template: str | None, iteration: int) -> Path | None:
    if not save_hf_template:
        return None
    return Path(save_hf_template.format(rollout_id=iteration))


def build_staging_checkpoint_root(checkpoint_root: str | Path, iteration: int) -> Path:
    return Path(checkpoint_root) / f"{STAGING_PREFIX}{iteration:07d}{INCOMPLETE_SUFFIX}"


def cleanup_incomplete_checkpoint_roots(checkpoint_root: str | Path) -> list[Path]:
    root = Path(checkpoint_root)
    removed: list[Path] = []
    if not root.exists():
        return removed
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if child.name.startswith(STAGING_PREFIX) and child.name.endswith(INCOMPLETE_SUFFIX):
            logger.warning("Removing stale staged checkpoint root %s", child)
            shutil.rmtree(child, ignore_errors=True)
            removed.append(child)
            continue
        if child.name.startswith(ITER_PREFIX) and child.name.endswith(INCOMPLETE_SUFFIX):
            logger.warning("Removing stale incomplete checkpoint directory %s", child)
            shutil.rmtree(child, ignore_errors=True)
            removed.append(child)
    return removed


def commit_staged_checkpoint(
    staging_root: str | Path,
    checkpoint_root: str | Path,
    *,
    iteration: int,
    write_latest_marker: bool,
) -> Path:
    staging_root = Path(staging_root)
    checkpoint_root = Path(checkpoint_root)
    staged_iter_dir = staging_root / f"{ITER_PREFIX}{iteration:07d}"
    if not staged_iter_dir.is_dir():
        raise FileNotFoundError(f"Staged checkpoint directory missing: {staged_iter_dir}")

    final_iter_dir = checkpoint_root / staged_iter_dir.name
    incomplete_final_iter_dir = checkpoint_root / f"{staged_iter_dir.name}{INCOMPLETE_SUFFIX}"
    if final_iter_dir.exists():
        shutil.rmtree(final_iter_dir, ignore_errors=True)
    if incomplete_final_iter_dir.exists():
        shutil.rmtree(incomplete_final_iter_dir, ignore_errors=True)

    checkpoint_root.mkdir(parents=True, exist_ok=True)
    staged_iter_dir.rename(final_iter_dir)

    rollout_dir = staging_root / "rollout"
    if rollout_dir.is_dir():
        final_rollout_dir = checkpoint_root / "rollout"
        final_rollout_dir.mkdir(parents=True, exist_ok=True)
        for child in rollout_dir.iterdir():
            destination = final_rollout_dir / child.name
            if destination.exists():
                if destination.is_dir():
                    shutil.rmtree(destination, ignore_errors=True)
                else:
                    destination.unlink()
            child.rename(destination)

    if write_latest_marker:
        marker_tmp = checkpoint_root / "latest_checkpointed_iteration.txt.tmp"
        marker_tmp.write_text(str(iteration))
        marker_tmp.replace(checkpoint_root / "latest_checkpointed_iteration.txt")

    shutil.rmtree(staging_root, ignore_errors=True)
    return final_iter_dir


def prune_rollout_state(checkpoint_root: str | Path, *, keep_iterations: set[int]) -> list[Path]:
    rollout_root = Path(checkpoint_root) / "rollout"
    removed: list[Path] = []
    if not rollout_root.is_dir():
        return removed

    for child in rollout_root.iterdir():
        if not child.is_file():
            continue
        match = re.search(r"_(\d+)\.pt$", child.name)
        if match is None:
            continue
        iteration = int(match.group(1))
        if iteration in keep_iterations:
            continue
        logger.info("Pruning old rollout state %s", child)
        child.unlink(missing_ok=True)
        removed.append(child)
    return removed


def prune_training_checkpoints(
    checkpoint_root: str | Path,
    *,
    keep_latest_complete: int = 1,
) -> dict[str, list[Path]]:
    root = Path(checkpoint_root)
    removed_stale_roots = cleanup_incomplete_checkpoint_roots(root)
    latest_complete = read_latest_complete_iteration(root)
    checkpoint_dirs = list_training_checkpoint_dirs(root)

    removed_incomplete: list[Path] = []
    removed_old_full: list[Path] = []

    complete_dirs: list[tuple[int, Path]] = []
    for iteration, path in checkpoint_dirs:
        if latest_complete is not None and iteration > latest_complete:
            logger.warning("Removing incomplete checkpoint directory %s", path)
            shutil.rmtree(path, ignore_errors=True)
            removed_incomplete.append(path)
            continue
        complete_dirs.append((iteration, path))

    if keep_latest_complete < 0:
        keep_latest_complete = 0

    keep_set = {
        iteration
        for iteration, _ in sorted(complete_dirs, reverse=True)[:keep_latest_complete]
    }

    for iteration, path in complete_dirs:
        if iteration in keep_set:
            continue
        logger.info("Pruning old full checkpoint %s", path)
        shutil.rmtree(path, ignore_errors=True)
        removed_old_full.append(path)

    removed_rollout_state = prune_rollout_state(root, keep_iterations=keep_set)

    return {
        "removed_stale_roots": removed_stale_roots,
        "removed_incomplete": removed_incomplete,
        "removed_old_full": removed_old_full,
        "removed_rollout_state": removed_rollout_state,
    }


def prune_archive_checkpoints(
    checkpoint_root: str | Path,
    *,
    keep_last_n: int = 3,
) -> dict[str, list[Path]]:
    root = Path(checkpoint_root)
    removed_stale_roots = cleanup_incomplete_checkpoint_roots(root)
    checkpoint_dirs = list_training_checkpoint_dirs(root)
    removed_old_archive: list[Path] = []

    if keep_last_n < 0:
        keep_last_n = 0

    keep_set = {
        iteration
        for iteration, _ in sorted(checkpoint_dirs, reverse=True)[:keep_last_n]
    }

    for iteration, path in checkpoint_dirs:
        if iteration in keep_set:
            continue
        logger.info("Pruning old archive checkpoint %s", path)
        shutil.rmtree(path, ignore_errors=True)
        removed_old_archive.append(path)

    return {
        "removed_stale_roots": removed_stale_roots,
        "removed_old_archive": removed_old_archive,
    }

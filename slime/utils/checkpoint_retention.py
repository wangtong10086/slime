from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

ITER_PREFIX = "iter_"


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


def prune_training_checkpoints(
    checkpoint_root: str | Path,
    *,
    save_hf_template: str | None = None,
    keep_latest_complete: int = 1,
) -> dict[str, list[Path]]:
    root = Path(checkpoint_root)
    latest_complete = read_latest_complete_iteration(root)
    checkpoint_dirs = list_training_checkpoint_dirs(root)

    removed_incomplete: list[Path] = []
    removed_old_full: list[Path] = []
    skipped_without_hf: list[Path] = []

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
        expected_hf = infer_hf_export_path(save_hf_template, iteration)
        if expected_hf is not None and expected_hf.exists() and any(expected_hf.iterdir()):
            logger.info("Pruning full checkpoint %s because HF export exists at %s", path, expected_hf)
            shutil.rmtree(path, ignore_errors=True)
            removed_old_full.append(path)
        else:
            logger.warning(
                "Keeping full checkpoint %s because HF export is missing at %s",
                path,
                expected_hf,
            )
            skipped_without_hf.append(path)

    return {
        "removed_incomplete": removed_incomplete,
        "removed_old_full": removed_old_full,
        "skipped_without_hf": skipped_without_hf,
    }

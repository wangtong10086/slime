from __future__ import annotations

from pathlib import Path


def _normalize_megatron_checkpoint_dir(path_str: str) -> Path:
    path = Path(path_str).resolve()
    latest_file = path / "latest_checkpointed_iteration.txt"
    if latest_file.is_file():
        return path

    if path.name.startswith("iter_"):
        parent = path.parent
        parent_latest = parent / "latest_checkpointed_iteration.txt"
        if parent_latest.is_file():
            return parent.resolve()

    return path


def resolve_liveweb_run_config(
    *,
    run_root: str,
    run_checkpoint_dir: str,
    run_mode: str,
    resume_mode: str,
    resume_checkpoint_dir: str = "",
    explicit_load_checkpoint_dir: str = "",
) -> dict[str, str]:
    run_root_path = Path(run_root).resolve()
    run_checkpoint_path = Path(run_checkpoint_dir).resolve()
    run_mode = (run_mode or "fresh").strip().lower()
    resume_mode = (resume_mode or "checkpoint").strip().lower()
    explicit_load_checkpoint_dir = explicit_load_checkpoint_dir.strip()
    resume_checkpoint_dir = resume_checkpoint_dir.strip()

    if run_mode not in {"fresh", "resume"}:
        raise ValueError(f"Unsupported LIVEWEB_RUN_MODE={run_mode!r}")
    if resume_mode not in {"checkpoint", "hf_only"}:
        raise ValueError(f"Unsupported LIVEWEB_RESUME_MODE={resume_mode!r}")

    if run_mode == "fresh":
        if resume_checkpoint_dir:
            raise ValueError("RESUME_CHECKPOINT_DIR must be empty when LIVEWEB_RUN_MODE=fresh")
        effective_load = (
            _normalize_megatron_checkpoint_dir(explicit_load_checkpoint_dir)
            if explicit_load_checkpoint_dir
            else run_checkpoint_path
        )
        return {
            "effective_load_checkpoint_dir": str(effective_load),
            "start_rollout_id_override": "",
            "hf_only_load_dir": "",
            "resume_checkpoint_dir": "",
        }

    if not resume_checkpoint_dir:
        raise ValueError("RESUME_CHECKPOINT_DIR is required when LIVEWEB_RUN_MODE=resume")

    resume_checkpoint_path = _normalize_megatron_checkpoint_dir(resume_checkpoint_dir)
    if run_root_path == resume_checkpoint_path.parent or run_checkpoint_path == resume_checkpoint_path:
        raise ValueError("RUN_ROOT must be different from the source checkpoint run when resuming")

    if resume_mode == "checkpoint":
        return {
            "effective_load_checkpoint_dir": str(resume_checkpoint_path),
            "start_rollout_id_override": "",
            "hf_only_load_dir": "",
            "resume_checkpoint_dir": str(resume_checkpoint_path),
        }

    latest_file = resume_checkpoint_path / "latest_checkpointed_iteration.txt"
    if not latest_file.is_file():
        raise ValueError(
            f"latest_checkpointed_iteration.txt not found in resume checkpoint dir: {resume_checkpoint_path}"
        )
    iteration = int(latest_file.read_text().strip())
    hf_dir = resume_checkpoint_path.parent / "hf_exports" / f"iter_{iteration:07d}"
    if not hf_dir.is_dir():
        raise ValueError(f"HF export directory not found for iteration {iteration}: {hf_dir}")
    return {
        "effective_load_checkpoint_dir": str(hf_dir.resolve()),
        "start_rollout_id_override": str(iteration + 1),
        "hf_only_load_dir": str(hf_dir.resolve()),
        "resume_checkpoint_dir": str(resume_checkpoint_path),
    }


def validate_resume_rollout_range(
    *,
    start_rollout_id: int | None,
    num_rollout: int | None,
    resume_checkpoint_dir: str = "",
) -> None:
    if (
        start_rollout_id is not None
        and num_rollout is not None
        and num_rollout > 0
        and start_rollout_id >= num_rollout
    ):
        raise ValueError(
            "Resume rollout range is empty: "
            f"start_rollout_id={start_rollout_id}, "
            f"num_rollout={num_rollout}, "
            f"resume_checkpoint_dir={resume_checkpoint_dir or '<unset>'}"
        )

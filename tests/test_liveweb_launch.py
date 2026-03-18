from pathlib import Path

import pytest

from slime.utils.liveweb_launch import resolve_liveweb_run_config


def test_resolve_liveweb_run_config_fresh_uses_run_checkpoint_dir(tmp_path):
    run_root = tmp_path / "run"
    checkpoint_dir = run_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    resolved = resolve_liveweb_run_config(
        run_root=str(run_root),
        run_checkpoint_dir=str(checkpoint_dir),
        run_mode="fresh",
        resume_mode="checkpoint",
    )

    assert resolved["effective_load_checkpoint_dir"] == str(checkpoint_dir.resolve())
    assert resolved["start_rollout_id_override"] == ""


def test_resolve_liveweb_run_config_resume_requires_new_run_dir(tmp_path):
    run_root = tmp_path / "run"
    checkpoint_dir = run_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="RUN_ROOT must be different"):
        resolve_liveweb_run_config(
            run_root=str(run_root),
            run_checkpoint_dir=str(checkpoint_dir),
            run_mode="resume",
            resume_mode="checkpoint",
            resume_checkpoint_dir=str(checkpoint_dir),
        )


def test_resolve_liveweb_run_config_hf_only_derives_latest_export(tmp_path):
    source_run = tmp_path / "source"
    resume_checkpoint_dir = source_run / "checkpoints"
    resume_checkpoint_dir.mkdir(parents=True)
    (resume_checkpoint_dir / "latest_checkpointed_iteration.txt").write_text("7")
    hf_export_dir = source_run / "hf_exports" / "iter_0000007"
    hf_export_dir.mkdir(parents=True)

    new_run = tmp_path / "new-run"
    new_checkpoint_dir = new_run / "checkpoints"
    new_checkpoint_dir.mkdir(parents=True)

    resolved = resolve_liveweb_run_config(
        run_root=str(new_run),
        run_checkpoint_dir=str(new_checkpoint_dir),
        run_mode="resume",
        resume_mode="hf_only",
        resume_checkpoint_dir=str(resume_checkpoint_dir),
    )

    assert resolved["effective_load_checkpoint_dir"] == str(hf_export_dir.resolve())
    assert resolved["hf_only_load_dir"] == str(hf_export_dir.resolve())
    assert resolved["start_rollout_id_override"] == "8"

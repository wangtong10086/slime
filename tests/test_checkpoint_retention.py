from pathlib import Path

from slime.utils.checkpoint_retention import (
    build_staging_checkpoint_root,
    cleanup_incomplete_checkpoint_roots,
    commit_staged_checkpoint,
    parse_iteration_from_name,
    prune_archive_checkpoints,
    prune_training_checkpoints,
    read_latest_complete_iteration,
)


def test_parse_iteration():
    assert parse_iteration_from_name("iter_0000009") == 9
    assert parse_iteration_from_name("hf_iter_0000009") is None


def test_prune_keeps_latest_complete_and_removes_incomplete(tmp_path: Path):
    ckpt_root = tmp_path / "checkpoints"
    ckpt_root.mkdir()

    for name in ["iter_0000009", "iter_0000019", "iter_0000029", "iter_0000039"]:
        path = ckpt_root / name
        path.mkdir()
        (path / "dummy").write_text("x")

    (ckpt_root / "latest_checkpointed_iteration.txt").write_text("29")

    result = prune_training_checkpoints(
        ckpt_root,
        keep_latest_complete=1,
    )

    assert read_latest_complete_iteration(ckpt_root) == 29
    assert not (ckpt_root / "iter_0000039").exists()
    assert not (ckpt_root / "iter_0000009").exists()
    assert not (ckpt_root / "iter_0000019").exists()
    assert (ckpt_root / "iter_0000029").exists()
    assert [p.name for p in result["removed_incomplete"]] == ["iter_0000039"]
    assert sorted(p.name for p in result["removed_old_full"]) == ["iter_0000009", "iter_0000019"]


def test_prune_removes_old_full_even_without_hf_export(tmp_path: Path):
    ckpt_root = tmp_path / "checkpoints"
    ckpt_root.mkdir()

    for name in ["iter_0000009", "iter_0000019"]:
        path = ckpt_root / name
        path.mkdir()
        (path / "dummy").write_text("x")

    (ckpt_root / "latest_checkpointed_iteration.txt").write_text("19")

    result = prune_training_checkpoints(
        ckpt_root,
        keep_latest_complete=1,
    )

    assert not (ckpt_root / "iter_0000009").exists()
    assert (ckpt_root / "iter_0000019").exists()
    assert [p.name for p in result["removed_old_full"]] == ["iter_0000009"]


def test_prune_archive_keeps_last_n(tmp_path: Path):
    archive_root = tmp_path / "checkpoints_archive"
    archive_root.mkdir()

    for name in ["iter_0000009", "iter_0000019", "iter_0000029", "iter_0000039"]:
        path = archive_root / name
        path.mkdir()
        (path / "dummy").write_text("x")

    result = prune_archive_checkpoints(archive_root, keep_last_n=2)

    assert not (archive_root / "iter_0000009").exists()
    assert not (archive_root / "iter_0000019").exists()
    assert (archive_root / "iter_0000029").exists()
    assert (archive_root / "iter_0000039").exists()
    assert sorted(p.name for p in result["removed_old_archive"]) == ["iter_0000009", "iter_0000019"]


def test_commit_staged_checkpoint_writes_latest_marker(tmp_path: Path):
    checkpoint_root = tmp_path / "checkpoints_full"
    checkpoint_root.mkdir()
    staging_root = build_staging_checkpoint_root(checkpoint_root, 19)
    staged_iter = staging_root / "iter_0000019"
    staged_iter.mkdir(parents=True)
    (staged_iter / "dummy").write_text("x")
    (staging_root / "latest_checkpointed_iteration.txt").write_text("19")
    rollout_root = staging_root / "rollout"
    rollout_root.mkdir()
    (rollout_root / "env_adapter_data_source_19.pt").write_text("y")

    final_iter = commit_staged_checkpoint(staging_root, checkpoint_root, iteration=19, write_latest_marker=True)

    assert final_iter == checkpoint_root / "iter_0000019"
    assert final_iter.exists()
    assert (checkpoint_root / "latest_checkpointed_iteration.txt").read_text().strip() == "19"
    assert (checkpoint_root / "rollout" / "env_adapter_data_source_19.pt").exists()
    assert not staging_root.exists()


def test_cleanup_incomplete_checkpoint_roots(tmp_path: Path):
    checkpoint_root = tmp_path / "checkpoints_archive"
    checkpoint_root.mkdir()
    stale_hidden = checkpoint_root / ".tmp_iter_0000009.incomplete"
    stale_hidden.mkdir()
    stale_iter = checkpoint_root / "iter_0000011.incomplete"
    stale_iter.mkdir()

    removed = cleanup_incomplete_checkpoint_roots(checkpoint_root)

    assert sorted(path.name for path in removed) == [".tmp_iter_0000009.incomplete", "iter_0000011.incomplete"]
    assert not stale_hidden.exists()
    assert not stale_iter.exists()

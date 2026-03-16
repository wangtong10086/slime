from pathlib import Path

from slime.utils.checkpoint_retention import (
    infer_hf_export_path,
    parse_iteration_from_name,
    prune_training_checkpoints,
    read_latest_complete_iteration,
)


def test_parse_iteration_and_hf_path():
    assert parse_iteration_from_name("iter_0000009") == 9
    assert parse_iteration_from_name("hf_iter_0000009") is None
    assert infer_hf_export_path("/tmp/hf_iter_{rollout_id:07d}", 19) == Path("/tmp/hf_iter_0000019")


def test_prune_keeps_latest_complete_and_removes_incomplete(tmp_path: Path):
    ckpt_root = tmp_path / "checkpoints"
    ckpt_root.mkdir()

    for name in ["iter_0000009", "iter_0000019", "iter_0000029", "iter_0000039"]:
        path = ckpt_root / name
        path.mkdir()
        (path / "dummy").write_text("x")

    (ckpt_root / "latest_checkpointed_iteration.txt").write_text("29")

    hf9 = tmp_path / "hf_iter_0000009"
    hf9.mkdir()
    (hf9 / "config.json").write_text("{}")
    hf19 = tmp_path / "hf_iter_0000019"
    hf19.mkdir()
    (hf19 / "config.json").write_text("{}")

    result = prune_training_checkpoints(
        ckpt_root,
        save_hf_template=str(tmp_path / "hf_iter_{rollout_id:07d}"),
        keep_latest_complete=1,
    )

    assert read_latest_complete_iteration(ckpt_root) == 29
    assert not (ckpt_root / "iter_0000039").exists()
    assert not (ckpt_root / "iter_0000009").exists()
    assert not (ckpt_root / "iter_0000019").exists()
    assert (ckpt_root / "iter_0000029").exists()
    assert [p.name for p in result["removed_incomplete"]] == ["iter_0000039"]
    assert sorted(p.name for p in result["removed_old_full"]) == ["iter_0000009", "iter_0000019"]


def test_prune_keeps_old_full_if_hf_export_missing(tmp_path: Path):
    ckpt_root = tmp_path / "checkpoints"
    ckpt_root.mkdir()

    for name in ["iter_0000009", "iter_0000019"]:
        path = ckpt_root / name
        path.mkdir()
        (path / "dummy").write_text("x")

    (ckpt_root / "latest_checkpointed_iteration.txt").write_text("19")

    result = prune_training_checkpoints(
        ckpt_root,
        save_hf_template=str(tmp_path / "hf_iter_{rollout_id:07d}"),
        keep_latest_complete=1,
    )

    assert (ckpt_root / "iter_0000009").exists()
    assert (ckpt_root / "iter_0000019").exists()
    assert result["removed_old_full"] == []
    assert [p.name for p in result["skipped_without_hf"]] == ["iter_0000009"]

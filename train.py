import os
import logging

import ray

from slime.ray.placement_group import (
    attach_rollout_manager_to_training_models,
    create_placement_groups,
    create_rollout_manager,
    create_training_models,
    log_startup_memory_snapshot,
)
from slime.utils.arguments import parse_args
from slime.utils.checkpoint_retention import (
    build_staging_checkpoint_root,
    cleanup_incomplete_checkpoint_roots,
    commit_staged_checkpoint,
    prune_archive_checkpoints,
    prune_training_checkpoints,
)
from slime.utils.logging_utils import configure_logger, init_tracking, finish_tracking
from slime.utils.misc import should_run_periodic_action
from slime.utils.save_guard import (
    SavePlan,
    collect_save_memory_snapshot,
    resolve_hf_export_policy,
    resolve_save_plan,
    should_skip_save_due_to_memory_guard,
    wait_for_save_headroom,
)

logger = logging.getLogger(__name__)


def _log_save_snapshot(stage: str) -> dict:
    snapshot = collect_save_memory_snapshot()
    logger.info(
        "%s: save/host_mem_available_gb=%.2f, save/host_mem_used_ratio=%.4f, save/swap_used_gb=%.2f, save/ray_memory_threshold=%.3f, save/top_process_rss_gb=%s",
        stage,
        snapshot["save/host_mem_available_gb"],
        snapshot["save/host_mem_used_ratio"],
        snapshot["save/swap_used_gb"],
        snapshot["save/ray_memory_threshold"],
        snapshot["save/top_process_rss_gb"],
    )
    return snapshot


def _free_rollout_data_ref(rollout_data_ref) -> None:
    if rollout_data_ref is None:
        return
    try:
        from ray._private.internal_api import free as ray_free

        ray_free([rollout_data_ref], local_only=False)
    except Exception:
        pass


def _log_save_status(stage: str, status: dict | None) -> None:
    status = status or {}
    logger.info(
        "%s: save/live_rollout_actor_count=%s, save/live_sglang_actor_count=%s, save/num_servers=%s, save/save_quiesced=%s",
        stage,
        status.get("live_rollout_actor_count"),
        status.get("live_sglang_actor_count"),
        status.get("num_servers"),
        status.get("save_quiesced"),
    )


def train(args):
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    staged_bringup = os.getenv("LIVEWEB_STAGED_BRINGUP", "0") == "1"
    if staged_bringup and args.num_rollout is None:
        logger.warning("LIVEWEB_STAGED_BRINGUP=1 requires explicit num_rollout; falling back to legacy bring-up")
        staged_bringup = False

    num_rollout_per_epoch = None
    rollout_manager = None
    if staged_bringup:
        logger.info("startup/restore_phase_begin")
        log_startup_memory_snapshot("startup/restore_phase_begin")
        actor_model, critic_model = create_training_models(args, pgs, rollout_manager=None)
        log_startup_memory_snapshot("startup/restore_phase_end", ready_count=args.actor_num_nodes * args.actor_num_gpus_per_node)
        logger.info("startup/restore_phase_end")

        logger.info("startup/rollout_phase_begin")
        log_startup_memory_snapshot("startup/rollout_phase_begin")
        rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])
        attach_rollout_manager_to_training_models(args, actor_model, critic_model, rollout_manager)
        log_startup_memory_snapshot("startup/rollout_phase_end")
        logger.info("startup/rollout_phase_end")
    else:
        # create the rollout manager, with sglang engines inside.
        # need to initialize rollout manager first to calculate num_rollout
        rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

        # create the actor and critic models
        actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())

    # always update weight first so that sglang has the loaded weights from training.
    if not args.critic_train_only:
        actor_model.update_weights()

        if args.check_weight_update_equal:
            ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def offload_train(rollout_id):
        if args.offload_train:
            if args.use_critic:
                critic_model.offload()
                if rollout_id >= args.num_critic_only_steps and not args.critic_train_only:
                    actor_model.offload()
            else:
                actor_model.offload()
        else:
            if args.critic_train_only:
                critic_model.clear_memory()
            else:
                actor_model.clear_memory()

    skip_save = os.getenv("LIVEWEB_SKIP_SAVE", "0") == "1"
    archive_save_min_host_mem_gb = float(os.getenv("LIVEWEB_ARCHIVE_SAVE_MIN_HOST_MEM_GB", os.getenv("LIVEWEB_LIGHT_SAVE_MIN_HOST_MEM_GB", os.getenv("LIVEWEB_SAVE_MIN_HOST_MEM_GB", "40"))))
    archive_save_max_host_mem_ratio = float(os.getenv("LIVEWEB_ARCHIVE_SAVE_MAX_HOST_MEM_RATIO", os.getenv("LIVEWEB_LIGHT_SAVE_MAX_HOST_MEM_RATIO", os.getenv("LIVEWEB_SAVE_MAX_HOST_MEM_RATIO", "0.96"))))
    full_save_min_host_mem_gb = float(os.getenv("LIVEWEB_FULL_SAVE_MIN_HOST_MEM_GB", "220"))
    full_save_max_host_mem_ratio = float(os.getenv("LIVEWEB_FULL_SAVE_MAX_HOST_MEM_RATIO", "0.78"))
    full_checkpoint_interval = int(os.getenv("LIVEWEB_FULL_CHECKPOINT_INTERVAL", "25"))
    archive_keep_last_n = int(os.getenv("LIVEWEB_ARCHIVE_KEEP_LAST_N", "2"))
    enable_final_hf_export = os.getenv("LIVEWEB_ENABLE_FINAL_HF_EXPORT", "0") == "1"
    final_save_mode = os.getenv("LIVEWEB_FINAL_SAVE_MODE", "full")
    archive_checkpoint_dir = os.getenv("RUN_ARCHIVE_CHECKPOINT_DIR", os.path.join(args.save, "..", "checkpoints_archive"))
    full_save_wait_timeout_s = float(os.getenv("LIVEWEB_FULL_SAVE_WAIT_TIMEOUT_SECONDS", "180"))
    final_full_save_wait_timeout_s = float(os.getenv("LIVEWEB_FINAL_FULL_SAVE_WAIT_TIMEOUT_SECONDS", "600"))
    save_headroom_poll_interval_s = float(os.getenv("LIVEWEB_SAVE_HEADROOM_POLL_INTERVAL_SECONDS", "5"))
    rollout_quiesced = False

    removed_stale_full = cleanup_incomplete_checkpoint_roots(args.save)
    removed_stale_archive = cleanup_incomplete_checkpoint_roots(archive_checkpoint_dir)
    if removed_stale_full or removed_stale_archive:
        logger.warning(
            "save/startup_cleanup removed stale staged checkpoints: full=%s archive=%s",
            [p.name for p in removed_stale_full],
            [p.name for p in removed_stale_archive],
        )

    def save(rollout_id, *, save_plan: SavePlan):
        nonlocal rollout_quiesced
        if skip_save:
            if getattr(args, "rank", 0) == 0:
                print(
                    f"[train] skipping checkpoint save at rollout_id={rollout_id} because LIVEWEB_SKIP_SAVE=1",
                    flush=True,
                )
            return
        save_mode = save_plan.mode
        checkpoint_root = args.save if save_mode == "full" else archive_checkpoint_dir
        staging_root = build_staging_checkpoint_root(checkpoint_root, rollout_id)
        logger.info(
            "save/plan_begin rollout_id=%s save_mode=%s save/is_final=%s save_dir=%s staging_dir=%s",
            rollout_id,
            save_mode,
            save_plan.is_final,
            checkpoint_root,
            staging_root,
        )
        if staging_root.exists():
            cleanup_incomplete_checkpoint_roots(checkpoint_root)
        _log_save_snapshot("save/phase_begin")
        logger.info("save/phase_begin rollout_id=%s save_mode=%s save/is_final=%s", rollout_id, save_mode, save_plan.is_final)
        checkpoint_saved = False
        save_status: dict | None = None
        try:
            if rollout_manager is not None:
                logger.info("save/quiesce_begin rollout_id=%s save_mode=%s", rollout_id, save_mode)
                if save_mode == "full":
                    save_status = ray.get(rollout_manager.begin_full_save_quiesce.remote())
                else:
                    save_status = ray.get(rollout_manager.begin_save_quiesce.remote())
                rollout_quiesced = True
                _log_save_snapshot("save/quiesce_end")
                _log_save_status("save/quiesce_end", save_status)
                logger.info("save/quiesce_end rollout_id=%s save_mode=%s", rollout_id, save_mode)

            if save_mode == "full":
                logger.info("save/headroom_wait_begin rollout_id=%s save_mode=%s", rollout_id, save_mode)
                ready, snapshot, save_status = wait_for_save_headroom(
                    snapshot_provider=collect_save_memory_snapshot,
                    status_provider=(
                        (lambda: ray.get(rollout_manager.get_save_status.remote()))
                        if rollout_manager is not None
                        else (lambda: {"live_rollout_actor_count": 0, "live_sglang_actor_count": 0, "num_servers": 0})
                    ),
                    min_available_gb=full_save_min_host_mem_gb,
                    max_used_ratio=full_save_max_host_mem_ratio,
                    timeout_s=final_full_save_wait_timeout_s if save_plan.is_final else full_save_wait_timeout_s,
                    poll_interval_s=save_headroom_poll_interval_s,
                    require_zero_rollout_actors=True,
                )
                _log_save_snapshot("save/headroom_wait_end")
                _log_save_status("save/headroom_wait_end", save_status)
                logger.info(
                    "save/headroom_wait_end rollout_id=%s save_mode=%s ready=%s",
                    rollout_id,
                    save_mode,
                    ready,
                )
                if not ready:
                    if save_plan.is_final:
                        raise RuntimeError(
                            "final_full_save_precondition_failed: rollout actors or host memory did not reach save barrier"
                        )
                    logger.warning(
                        "save/skipped_due_to_memory_guard rollout_id=%s save_mode=%s available_gb=%.2f used_ratio=%.4f live_rollout_actor_count=%s",
                        rollout_id,
                        save_mode,
                        snapshot["save/host_mem_available_gb"],
                        snapshot["save/host_mem_used_ratio"],
                        save_status.get("live_rollout_actor_count"),
                    )
                    return
            else:
                snapshot = collect_save_memory_snapshot()
                if should_skip_save_due_to_memory_guard(
                    snapshot,
                    min_available_gb=archive_save_min_host_mem_gb,
                    max_used_ratio=archive_save_max_host_mem_ratio,
                ):
                    logger.warning(
                        "save/skipped_due_to_memory_guard rollout_id=%s save_mode=%s available_gb=%.2f used_ratio=%.4f",
                        rollout_id,
                        save_mode,
                        snapshot["save/host_mem_available_gb"],
                        snapshot["save/host_mem_used_ratio"],
                    )
                    return

            should_export_hf = (
                save_mode == "full"
                and enable_final_hf_export
                and save_plan.is_final
                and resolve_hf_export_policy(rollout_id=rollout_id, num_rollout=args.num_rollout, force_sync=True)
            )
            logger.info(
                "save/checkpoint_begin rollout_id=%s save_mode=%s hf_export=%s save_dir=%s",
                rollout_id,
                save_mode,
                should_export_hf,
                staging_root,
            )
            if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps and not args.critic_train_only):
                logger.info("save/actor_checkpoint_begin rollout_id=%s save_mode=%s", rollout_id, save_mode)
                actor_model.prepare_for_save(save_mode=save_mode)
                actor_model.save_model(
                    rollout_id,
                    force_sync=save_plan.is_final,
                    hf_export=should_export_hf,
                    save_mode=save_mode,
                    save_dir=str(staging_root),
                )
                logger.info("save/actor_checkpoint_end rollout_id=%s save_mode=%s", rollout_id, save_mode)
            if args.use_critic:
                logger.info("save/critic_checkpoint_begin rollout_id=%s save_mode=%s", rollout_id, save_mode)
                critic_model.prepare_for_save(save_mode=save_mode)
                critic_model.save_model(
                    rollout_id,
                    force_sync=save_plan.is_final,
                    hf_export=False,
                    save_mode=save_mode,
                    save_dir=str(staging_root),
                )
                logger.info("save/critic_checkpoint_end rollout_id=%s save_mode=%s", rollout_id, save_mode)
            if save_plan.write_rollout_state and args.rollout_global_dataset:
                logger.info("save/rollout_state_begin rollout_id=%s save_mode=%s", rollout_id, save_mode)
                ray.get(rollout_manager.save.remote(rollout_id, save_root=str(staging_root)))
                logger.info("save/rollout_state_end rollout_id=%s save_mode=%s", rollout_id, save_mode)

            logger.info("save/commit_begin rollout_id=%s save_mode=%s", rollout_id, save_mode)
            commit_staged_checkpoint(
                staging_root,
                checkpoint_root,
                iteration=rollout_id,
                write_latest_marker=save_mode == "full",
            )
            logger.info("save/commit_end rollout_id=%s save_mode=%s", rollout_id, save_mode)
            logger.info(
                "save/checkpoint_end rollout_id=%s save_mode=%s hf_export=%s",
                rollout_id,
                save_mode,
                should_export_hf,
            )
            checkpoint_saved = True
        finally:
            if rollout_manager is not None and rollout_quiesced:
                if save_mode == "full" and not save_plan.is_final:
                    save_status = ray.get(rollout_manager.end_full_save_quiesce.remote())
                elif save_mode == "archive":
                    save_status = ray.get(rollout_manager.end_save_quiesce.remote())
                else:
                    save_status = None
                rollout_quiesced = False
                _log_save_snapshot("save/quiesce_released")
                _log_save_status("save/quiesce_released", save_status)
            logger.info("save/phase_end rollout_id=%s save_mode=%s checkpoint_saved=%s", rollout_id, save_mode, checkpoint_saved)
            _log_save_snapshot("save/phase_end")

        if checkpoint_saved and save_mode == "full":
            retention = prune_training_checkpoints(
                args.save,
                keep_latest_complete=1,
            )
            if retention["removed_stale_roots"] or retention["removed_incomplete"] or retention["removed_old_full"] or retention["removed_rollout_state"]:
                print(
                    "[train] checkpoint retention pruned "
                    f"stale_roots={len(retention['removed_stale_roots'])} "
                    f"incomplete={len(retention['removed_incomplete'])} "
                    f"old_full={len(retention['removed_old_full'])} "
                    f"rollout_state={len(retention['removed_rollout_state'])}",
                    flush=True,
                )
        elif checkpoint_saved and save_mode == "archive":
            retention = prune_archive_checkpoints(
                archive_checkpoint_dir,
                keep_last_n=archive_keep_last_n,
            )
            if retention["removed_stale_roots"] or retention["removed_old_archive"]:
                print(
                    "[train] archive retention pruned "
                    f"stale_roots={len(retention['removed_stale_roots'])} "
                    f"old_archive={len(retention['removed_old_archive'])}",
                    flush=True,
                )

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
            if rollout_id >= args.num_critic_only_steps and not args.critic_train_only:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
            ray.get(critic_train_handle)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        _free_rollout_data_ref(rollout_data_ref)
        rollout_data_ref = None

        save_plan = resolve_save_plan(
            rollout_id=rollout_id,
            num_rollout=args.num_rollout,
            archive_interval=args.save_interval,
            full_interval=full_checkpoint_interval,
            final_save_mode=final_save_mode,
        )
        if save_plan is not None:
            save(rollout_id, save_plan=save_plan)
            if save_plan.is_final and save_plan.mode == "full":
                break

        offload_train(rollout_id)
        if args.offload_rollout:
            ray.get(rollout_manager.onload_weights.remote())
        if not args.critic_train_only:
            actor_model.update_weights()
        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)

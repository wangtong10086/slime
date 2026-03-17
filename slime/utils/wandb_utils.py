import logging
import os
import threading
from copy import deepcopy

import wandb

logger = logging.getLogger(__name__)
_FINISH_LOCK = threading.Lock()
_FINISH_CALLED = False
_SERVICE_PATCH_LOCK = threading.Lock()
_SERVICE_PATCH_INSTALLED = False


def _is_benign_wandb_teardown_error(exc: BaseException) -> bool:
    if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
        return True
    return isinstance(exc, RuntimeError) and "Event loop is closed" in str(exc)


def ensure_wandb_service_teardown_patch_installed() -> None:
    """Patch wandb service atexit teardown to silence known benign shutdown noise."""
    global _SERVICE_PATCH_INSTALLED
    with _SERVICE_PATCH_LOCK:
        if _SERVICE_PATCH_INSTALLED:
            return

        try:
            from wandb.sdk.lib.service import service_connection
        except Exception as exc:
            logger.debug("Skipping wandb service teardown patch; import failed: %s", exc)
            _SERVICE_PATCH_INSTALLED = True
            return

        original = getattr(service_connection, "_start_and_connect_service", None)
        original_teardown = getattr(service_connection.ServiceConnection, "teardown", None)
        if original is None or getattr(original, "_slime_patched", False):
            _SERVICE_PATCH_INSTALLED = True
            return

        if original_teardown is not None and not getattr(original_teardown, "_slime_patched", False):
            def _patched_teardown(self, exit_code):
                try:
                    return original_teardown(self, exit_code)
                except Exception as exc:
                    if _is_benign_wandb_teardown_error(exc):
                        logger.debug("Ignoring benign wandb service teardown error: %s", exc)
                        return None
                    raise

            _patched_teardown._slime_patched = True  # type: ignore[attr-defined]
            service_connection.ServiceConnection.teardown = _patched_teardown

        def _patched_start_and_connect_service(asyncer, settings):
            proc = service_connection.service_process.start(settings)
            client = proc.token.connect(asyncer=asyncer)
            proc.token.save_to_env()

            hooks = service_connection.ExitHooks()
            hooks.hook()

            def teardown_atexit():
                try:
                    conn.teardown(hooks.exit_code)
                except Exception as exc:
                    if _is_benign_wandb_teardown_error(exc):
                        logger.debug("Ignoring benign wandb atexit teardown error: %s", exc)
                        return
                    raise

            conn = service_connection.ServiceConnection(
                asyncer=asyncer,
                client=client,
                proc=proc,
                cleanup=lambda: service_connection.atexit.unregister(teardown_atexit),
            )

            service_connection.atexit.register(teardown_atexit)
            return conn

        _patched_start_and_connect_service._slime_patched = True  # type: ignore[attr-defined]
        _patched_start_and_connect_service._slime_original = original  # type: ignore[attr-defined]
        service_connection._start_and_connect_service = _patched_start_and_connect_service
        _SERVICE_PATCH_INSTALLED = True


def reset_wandb_finish_guard_for_tests() -> None:
    global _FINISH_CALLED
    with _FINISH_LOCK:
        _FINISH_CALLED = False


def reset_wandb_service_patch_for_tests() -> None:
    global _SERVICE_PATCH_INSTALLED
    with _SERVICE_PATCH_LOCK:
        _SERVICE_PATCH_INSTALLED = False


def finish_wandb_once() -> bool:
    """Finish the current W&B run at most once per process.

    Returns True only when this call performed the actual finish attempt.
    """
    global _FINISH_CALLED
    with _FINISH_LOCK:
        if _FINISH_CALLED:
            return False
        if getattr(wandb, "run", None) is None:
            _FINISH_CALLED = True
            return False
        _FINISH_CALLED = True
    wandb.finish()
    return True


def _is_offline_mode(args) -> bool:
    """Detect whether W&B should run in offline mode.

    Priority order:
    1) args.wandb_mode if provided
    2) WANDB_MODE environment variable
    """
    if args.wandb_mode:
        return args.wandb_mode == "offline"
    return os.environ.get("WANDB_MODE") == "offline"


def init_wandb_primary(args):
    if not args.use_wandb:
        args.wandb_run_id = None
        return

    # Set W&B mode if specified (overrides WANDB_MODE env var)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
        if args.wandb_mode == "offline":
            logger.info("W&B offline mode enabled. Data will be saved locally.")
        elif args.wandb_mode == "disabled":
            logger.info("W&B disabled mode enabled. No data will be logged.")
        elif args.wandb_mode == "online":
            logger.info("W&B online mode enabled. Data will be uploaded to cloud.")

    offline = _is_offline_mode(args)

    # Only perform explicit login when NOT offline
    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    # Prepare wandb init parameters
    # add random 6 length string with characters
    if args.wandb_random_suffix:
        group = args.wandb_group + "_" + wandb.util.generate_id()
        run_name = f"{group}-RANK_{args.rank}"
    else:
        group = args.wandb_group
        run_name = args.wandb_group

    # Prepare wandb init parameters
    init_kwargs = {
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "group": group,
        "name": run_name,
        "config": _compute_config_for_logging(args),
    }

    # Allow external launchers to pre-assign a run id so auxiliary processes
    # can attach to the same shared W&B run.
    if args.wandb_run_id:
        init_kwargs["id"] = args.wandb_run_id
        init_kwargs["resume"] = "allow"

    # Configure settings based on offline/online mode
    if offline:
        init_kwargs["settings"] = wandb.Settings(mode="offline")
    else:
        init_kwargs["settings"] = wandb.Settings(mode="shared", x_primary=True)

    # Add custom directory if specified
    if args.wandb_dir:
        # Ensure directory exists to avoid backend crashes
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir
        logger.info(f"W&B logs will be stored in: {args.wandb_dir}")

    wandb.init(**init_kwargs)

    _init_wandb_common()

    # Set wandb_run_id in args for easy access throughout the training process
    args.wandb_run_id = wandb.run.id


def _compute_config_for_logging(args):
    output = deepcopy(args.__dict__)

    whitelist_env_vars = [
        "SLURM_JOB_ID",
        # We may insert more default values here, and may also allow users to configure a whitelist
    ]
    output["env_vars"] = {k: v for k, v in os.environ.items() if k in whitelist_env_vars}

    return output


# https://docs.wandb.ai/guides/track/log/distributed-training/#track-all-processes-to-a-single-run
def init_wandb_secondary(args, router_addr=None):
    wandb_run_id = getattr(args, "wandb_run_id", None)
    if wandb_run_id is None:
        return

    # Set W&B mode if specified (same as primary)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    offline = _is_offline_mode(args)

    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    # Configure settings based on offline/online mode
    if offline:
        settings_kwargs = dict(mode="offline")
    else:
        settings_kwargs = dict(
            mode="shared",
            x_primary=False,
            x_update_finish_state=False,
        )

    if getattr(args, "sglang_enable_metrics", False) and router_addr is not None:
        logger.info(f"Forward SGLang metrics at {router_addr} to WandB.")
        settings_kwargs |= dict(
            x_stats_open_metrics_endpoints={
                "sgl_engine": f"{router_addr}/engine_metrics",
            },
            x_stats_open_metrics_filters={
                "sgl_engine.*": {},
            },
        )

    init_kwargs = {
        "id": wandb_run_id,
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "config": args.__dict__,
        "resume": "allow",
        "reinit": True,
        "settings": wandb.Settings(**settings_kwargs),
    }

    # Add custom directory if specified
    if args.wandb_dir:
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir

    wandb.init(**init_kwargs)

    _init_wandb_common()


def _init_wandb_common():
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("rollout/step")
    wandb.define_metric("rollout/*", step_metric="rollout/step")
    wandb.define_metric("env/*", step_metric="rollout/step")
    wandb.define_metric("cache/*", step_metric="rollout/step")
    wandb.define_metric("runtime/*", step_metric="rollout/step")
    wandb.define_metric("scheduler/*", step_metric="rollout/step")
    wandb.define_metric("multi_turn/*", step_metric="rollout/step")
    wandb.define_metric("passrate/*", step_metric="rollout/step")
    wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric="eval/step")
    wandb.define_metric("eval_env/*", step_metric="eval/step")
    wandb.define_metric("eval_cache/*", step_metric="eval/step")
    wandb.define_metric("eval_runtime/*", step_metric="eval/step")
    wandb.define_metric("eval_scheduler/*", step_metric="eval/step")
    wandb.define_metric("perf/*", step_metric="rollout/step")

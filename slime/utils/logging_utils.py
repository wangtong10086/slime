import logging
import os

import wandb

from . import wandb_utils
from .tensorboard_utils import _TensorboardAdapter

_LOGGER_CONFIGURED = False

_WANDB_DROP_EXACT_KEYS = {
    "scheduler/runtime_active_job_cap",
    "scheduler/config_max_parallel_env_jobs",
    "scheduler/config_max_parallel_llm_jobs",
    "scheduler/active_jobs",
    "scheduler/queued_jobs",
    "scheduler/completed_jobs",
    "scheduler/max_parallel_env_jobs",
    "scheduler/max_parallel_llm_jobs",
    "env/jit_kernel_enabled",
    "env/kernel_fallback",
    "env/runtime_reset_count",
    "env/runtime_pool_hits",
    "env/worker_reuse_count",
    "env/worker_prepare_reset_count",
    "env/runtime_instance_reuse_count",
    "env/runtime_instance_reset_count",
    "env/format_recovery_attempts",
    "env/format_recovery_successes",
    "env/format_recovery_exhausted",
    "env/reachability_audit_count",
    "eval_scheduler/runtime_active_job_cap",
    "eval_scheduler/config_max_parallel_env_jobs",
    "eval_scheduler/config_max_parallel_llm_jobs",
    "eval_scheduler/active_jobs",
    "eval_scheduler/queued_jobs",
    "eval_scheduler/completed_jobs",
    "eval_scheduler/max_parallel_env_jobs",
    "eval_scheduler/max_parallel_llm_jobs",
    "eval_env/jit_kernel_enabled",
    "eval_env/kernel_fallback",
    "eval_env/runtime_reset_count",
    "eval_env/runtime_pool_hits",
    "eval_env/worker_reuse_count",
    "eval_env/worker_prepare_reset_count",
    "eval_env/runtime_instance_reuse_count",
    "eval_env/runtime_instance_reset_count",
    "eval_env/format_recovery_attempts",
    "eval_env/format_recovery_successes",
    "eval_env/format_recovery_exhausted",
    "eval_env/reachability_audit_count",
}

_LIVEWEB_MANAGED_PREFIXES = (
    "train/",
    "rollout/",
    "perf/",
    "env/",
    "cache/",
    "runtime/",
    "scheduler/",
    "passrate/",
    "eval/",
    "eval_env/",
    "eval_cache/",
    "eval_runtime/",
    "eval_scheduler/",
)

_LIVEWEB_TRAIN_KEEP = {
    "train/loss",
    "train/pg_loss",
    "train/entropy_loss",
    "train/pg_clipfrac",
    "train/ppo_kl",
    "train/kl_loss",
    "train/grad_norm",
    "train/raw_grad_norm",
    "train/grad_clip_ratio",
    "train/learning_rate",
    "train/internal_step",
    "train/tokens_per_s",
    "train/tflops",
    "train/step_time",
    "train/data_wait_ratio",
}

_LIVEWEB_ROLLOUT_KEEP = {
    "rollout/step",
    "rollout/rewards",
    "rollout/raw_reward",
    "rollout/advantages",
    "rollout/returns",
    "rollout/response_len/mean",
    "rollout/response_len/max",
    "rollout/truncated_ratio",
    "rollout/repetition_frac",
}

_LIVEWEB_PERF_KEEP = {
    "perf/rollout_time",
    "perf/step_time",
    "perf/wait_time_ratio",
    "perf/tokens_per_gpu_per_sec",
    "perf/effective_tokens_per_gpu_per_sec",
    "perf/actor_train_tok_per_s",
    "perf/actor_train_tflops",
}

_LIVEWEB_ENV_KEEP = {
    "env/mean_score",
    "env/mean_num_tasks",
    "env/num_tasks_unknown_rate",
    "env/num_tasks_1_rate",
    "env/num_tasks_2_rate",
    "env/num_tasks_3_rate",
    "env/num_tasks_4_rate",
    "env/success_rate",
    "env/pollution_rate",
    "env/permanent_failure_rate",
    "env/invalid_output_rate",
    "env/site_unreachable_rate",
    "env/cache_error_rate",
    "env/parse_failed_rate",
    "env/llm_error_rate",
    "env/domain_unreachable_rate",
    "env/prefetch_failure_rate",
    "env/cache_fill_failure_rate",
    "env/invalid_tool_format_rate",
    "env/format_recovery_success_rate",
    "env/format_failure_recoverable_rate",
    "env/format_failure_terminal_rate",
    "env/mean_progress_score",
    "env/near_miss_rate",
    "env/wrong_domain_loop_rate",
    "env/premature_stop_rate",
    "env/wrong_path_rate",
    "env/max_steps_reached_rate",
    "env/unsupported_stop_rate",
    "env/hallucinated_plugin_mean",
    "env/google_family_offdomain_mean",
    "env/repeated_url_mean",
    "env/same_page_loop_mean",
    "env/browser_nav_time_mean",
    "env/browser_step_time_mean",
    "env/challenge_page_rate",
    "env/learning_bucket/environment_failure_rate",
    "env/learning_bucket/format_failure_rate",
    "env/reachability_env_failure_rate",
    "env/reachability_model_hallucination_rate",
}

_LIVEWEB_CACHE_KEEP = {
    "cache/hit_rate",
}

_LIVEWEB_RUNTIME_KEEP = {
    "runtime/recoverable_failure_rate",
}

_LIVEWEB_SCHEDULER_KEEP = {
    "scheduler/active_env_jobs_mean",
    "scheduler/active_llm_jobs_mean",
    "scheduler/group_fill_rate",
    "scheduler/oversample_ratio",
    "scheduler/runtime_queued_jobs",
    "scheduler/runtime_completed_jobs",
    "scheduler/runtime_requested_tokens",
    "scheduler/runtime_retained_tokens",
    "scheduler/runtime_trimmed_tokens",
    "scheduler/runtime_requested_logit_tokens",
    "scheduler/runtime_retained_logit_tokens",
    "scheduler/runtime_trimmed_logit_tokens",
}

_LIVEWEB_EVAL_ENV_KEEP = {
    "eval_env/mean_score",
    "eval_env/mean_num_tasks",
    "eval_env/num_tasks_unknown_rate",
    "eval_env/num_tasks_1_rate",
    "eval_env/num_tasks_2_rate",
    "eval_env/num_tasks_3_rate",
    "eval_env/num_tasks_4_rate",
    "eval_env/success_rate",
    "eval_env/pollution_rate",
    "eval_env/permanent_failure_rate",
    "eval_env/invalid_output_rate",
    "eval_env/site_unreachable_rate",
    "eval_env/cache_error_rate",
    "eval_env/parse_failed_rate",
    "eval_env/llm_error_rate",
    "eval_env/domain_unreachable_rate",
    "eval_env/prefetch_failure_rate",
    "eval_env/cache_fill_failure_rate",
    "eval_env/invalid_tool_format_rate",
    "eval_env/format_recovery_success_rate",
    "eval_env/format_failure_recoverable_rate",
    "eval_env/format_failure_terminal_rate",
    "eval_env/mean_progress_score",
    "eval_env/near_miss_rate",
    "eval_env/wrong_domain_loop_rate",
    "eval_env/premature_stop_rate",
    "eval_env/wrong_path_rate",
    "eval_env/max_steps_reached_rate",
    "eval_env/unsupported_stop_rate",
    "eval_env/hallucinated_plugin_mean",
    "eval_env/google_family_offdomain_mean",
    "eval_env/repeated_url_mean",
    "eval_env/same_page_loop_mean",
    "eval_env/browser_nav_time_mean",
    "eval_env/browser_step_time_mean",
    "eval_env/challenge_page_rate",
    "eval_env/learning_bucket/environment_failure_rate",
    "eval_env/learning_bucket/format_failure_rate",
    "eval_env/reachability_env_failure_rate",
    "eval_env/reachability_model_hallucination_rate",
}

_LIVEWEB_EVAL_CACHE_KEEP = {
    "eval_cache/hit_rate",
}

_LIVEWEB_EVAL_RUNTIME_KEEP = {
    "eval_runtime/recoverable_failure_rate",
}


def _is_liveweb_run(args) -> bool:
    return getattr(args, "environment_name", None) == "liveweb" or os.getenv("SLIME_ENVIRONMENT_NAME") == "liveweb"


def _keep_liveweb_metric(key: str, *, step_key: str) -> bool:
    if key == step_key:
        return True
    if key.startswith("env/failure/") or key.startswith("eval_env/failure/"):
        return True
    if key in _LIVEWEB_TRAIN_KEEP:
        return True
    if key in _LIVEWEB_ROLLOUT_KEEP:
        return True
    if key in _LIVEWEB_PERF_KEEP:
        return True
    if key in _LIVEWEB_ENV_KEEP:
        return True
    if key in _LIVEWEB_CACHE_KEEP:
        return True
    if key in _LIVEWEB_RUNTIME_KEEP:
        return True
    if key in _LIVEWEB_SCHEDULER_KEEP:
        return True
    if key in _LIVEWEB_EVAL_ENV_KEEP:
        return True
    if key in _LIVEWEB_EVAL_CACHE_KEEP:
        return True
    if key in _LIVEWEB_EVAL_RUNTIME_KEEP:
        return True
    if key == "eval/step":
        return True
    if key.startswith("eval/") and not any(
        key.startswith(prefix) for prefix in ("eval_env/", "eval_cache/", "eval_runtime/", "eval_scheduler/")
    ):
        return key.count("/") == 1
    if not key.startswith(_LIVEWEB_MANAGED_PREFIXES):
        return True
    return False


def _filter_metrics_for_wandb(args, metrics: dict, *, step_key: str) -> dict:
    if os.getenv("SLIME_WANDB_PRUNE_METRICS", "1") != "1":
        return dict(metrics)

    filtered = {}
    for key, value in metrics.items():
        if _is_liveweb_run(args):
            if not _keep_liveweb_metric(key, step_key=step_key):
                continue
        elif key != step_key and key in _WANDB_DROP_EXACT_KEYS:
            continue
        filtered[key] = value
    return filtered


# ref: SGLang
def configure_logger(prefix: str = ""):
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    _LOGGER_CONFIGURED = True

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s{prefix}] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def init_tracking(args, primary: bool = True, **kwargs):
    if getattr(args, "use_wandb", False):
        wandb_utils.ensure_wandb_service_teardown_patch_installed()
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)


def finish_tracking(args):
    if not args.use_wandb:
        return
    try:
        wandb_utils.finish_wandb_once()
    except (BrokenPipeError, ConnectionResetError) as exc:
        logging.getLogger(__name__).debug("Ignoring benign wandb teardown error: %s", exc)
    except RuntimeError as exc:
        if "Event loop is closed" in str(exc):
            logging.getLogger(__name__).debug("Ignoring benign wandb teardown runtime error: %s", exc)
            return
        logging.getLogger(__name__).exception("Failed to finish wandb run")
    except Exception:
        logging.getLogger(__name__).exception("Failed to finish wandb run")


# TODO further refactor, e.g. put TensorBoard init to the "init" part
def log(args, metrics, step_key: str):
    if args.use_wandb:
        wandb.log(_filter_metrics_for_wandb(args, metrics, step_key=step_key))

    if args.use_tensorboard:
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        _TensorboardAdapter(args).log(data=metrics_except_step, step=metrics[step_key])

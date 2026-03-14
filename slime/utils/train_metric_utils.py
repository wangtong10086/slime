import logging
from argparse import Namespace
from collections.abc import Callable
from copy import deepcopy

from slime.utils import logging_utils
from slime.utils.metric_utils import compute_rollout_step
from slime.utils.timer import Timer

logger = logging.getLogger(__name__)


def log_perf_data_raw(
    rollout_id: int, args: Namespace, is_primary_rank: bool, compute_total_fwd_flops: Callable
) -> None:
    timer_instance = Timer()
    log_dict_raw = deepcopy(timer_instance.log_dict())
    timer_instance.reset()

    if not is_primary_rank:
        return

    log_dict = {f"perf/{key}_time": val for key, val in log_dict_raw.items()}

    if ("perf/actor_train_time" in log_dict) and (compute_total_fwd_flops is not None):
        total_fwd_flops = compute_total_fwd_flops(seq_lens=timer_instance.seq_lens)

        if "perf/log_probs_time" in log_dict:
            log_dict["perf/log_probs_tflops"] = total_fwd_flops / log_dict["perf/log_probs_time"]

        if "perf/ref_log_probs_time" in log_dict:
            log_dict["perf/ref_log_probs_tflops"] = total_fwd_flops / log_dict["perf/ref_log_probs_time"]

        if log_dict["perf/actor_train_time"] > 0:
            log_dict["perf/actor_train_tflops"] = 3 * total_fwd_flops / log_dict["perf/actor_train_time"]
            log_dict["perf/actor_train_tok_per_s"] = sum(timer_instance.seq_lens) / log_dict["perf/actor_train_time"]

    if "perf/train_wait_time" in log_dict and "perf/train_time" in log_dict:
        total_time = log_dict["perf/train_wait_time"] + log_dict["perf/train_time"]
        if total_time > 0:
            log_dict["perf/step_time"] = total_time
            log_dict["perf/wait_time_ratio"] = log_dict["perf/train_wait_time"] / total_time

    logger.info(f"perf {rollout_id}: {log_dict}")

    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    logging_utils.log(args, log_dict, step_key="rollout/step")

    if args.loss_type == "sft_loss":
        sft_log_dict = {"train/step": step}

        alias_map = {
            "perf/actor_train_tok_per_s": "train/tokens_per_s",
            "perf/actor_train_tflops": "train/tflops",
            "perf/actor_train_time": "train/compute_time",
            "perf/step_time": "train/step_time",
            "perf/wait_time_ratio": "train/data_wait_ratio",
        }
        for src_key, dst_key in alias_map.items():
            if src_key in log_dict:
                sft_log_dict[dst_key] = log_dict[src_key]

        logging_utils.log(args, sft_log_dict, step_key="train/step")

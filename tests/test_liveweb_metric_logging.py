from argparse import Namespace

import pytest

from slime.utils.metric_utils import summarize_task_count_metrics
from slime.utils.train_logging_helpers import next_train_log_step, normalize_grad_norm_for_logging


def test_summarize_task_count_metrics_normalizes_known_task_distribution():
    metrics = summarize_task_count_metrics([0, 2, 3, 0, 4, 99])

    assert metrics["mean_num_tasks"] == pytest.approx(3.0)
    assert metrics["num_tasks_unknown_rate"] == pytest.approx(3 / 6)
    assert metrics["num_tasks_1_rate"] == pytest.approx(0.0)
    assert metrics["num_tasks_2_rate"] == pytest.approx(1 / 3)
    assert metrics["num_tasks_3_rate"] == pytest.approx(1 / 3)
    assert metrics["num_tasks_4_rate"] == pytest.approx(1 / 3)
    assert (
        metrics["num_tasks_1_rate"]
        + metrics["num_tasks_2_rate"]
        + metrics["num_tasks_3_rate"]
        + metrics["num_tasks_4_rate"]
    ) == pytest.approx(1.0)


def test_next_train_log_step_is_run_local_and_zero_based():
    args = Namespace()
    assert next_train_log_step(args) == 0
    assert next_train_log_step(args) == 1
    assert next_train_log_step(args) == 2


def test_normalize_grad_norm_for_logging_uses_clip_grad():
    effective, raw = normalize_grad_norm_for_logging(65.0, 1.0)
    assert effective == pytest.approx(1.0)
    assert raw == pytest.approx(65.0)


def test_normalize_grad_norm_for_logging_without_clip_keeps_raw_value():
    effective, raw = normalize_grad_norm_for_logging(0.75, None)
    assert effective == pytest.approx(0.75)
    assert raw == pytest.approx(0.75)

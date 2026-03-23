from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import EnvironmentAdapter, JobSpec, RolloutResult


logger = logging.getLogger(__name__)


@dataclass
class RuntimeJobRequest:
    group_position: int
    sample_position: int
    job: "JobSpec"


@dataclass
class RuntimeJobOutcome:
    group_position: int
    sample_position: int
    result: "RolloutResult"


class EnvironmentRuntimeWorker:
    """
    Long-lived environment runtime worker bound to a single asyncio event loop.

    The worker owns all environment async state for one scope (e.g. train_rollout
    or eval). Callers interact with it synchronously via run_coroutine_threadsafe,
    so the training loop no longer recreates a fresh event loop around reusable
    runtime objects.
    """

    def __init__(self, adapter: "EnvironmentAdapter", args: Any, *, scope: str):
        self._adapter = adapter
        self._args = args
        self._scope = scope
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._runtime: Any = None
        self._recovery_lock: asyncio.Lock | None = None
        self._prepare_calls = 0
        self._worker_reuse_count = 0
        self._worker_prepare_reset_count = 0
        self._runtime_instance_reuse_count = 0
        self._runtime_instance_reset_count = 0
        self._jit_kernel_enabled = 1 if os.getenv("LIVEWEB_RUNTIME_JIT_KERNEL_ENABLED", "1") == "1" else 0
        self._kernel_fallback = 1 if os.getenv("LIVEWEB_RUNTIME_KERNEL_FALLBACK", "0") == "1" else 0
        self._sample_wall_timeout_count = 0
        self._group_wall_timeout_count = 0
        self._abort_failed_count = 0
        self._last_pending_samples = 0
        self._last_pending_groups = 0
        self._last_oldest_pending_age_seconds = 0.0
        self._last_active_decode_requests = 0

    @property
    def scope(self) -> str:
        return self._scope

    def start(self) -> None:
        if self._thread is not None:
            return

        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._recovery_lock = asyncio.Lock()
            self._ready.set()
            try:
                loop.run_forever()
            finally:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()

        self._thread = threading.Thread(
            target=_run,
            name=f"env-runtime-{self._scope}",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()

    def stop(self) -> None:
        if self._thread is None or self._loop is None:
            return
        try:
            self._submit(self._cleanup_async())
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5.0)
            self._thread = None
            self._loop = None
            self._runtime = None
            self._recovery_lock = None
            self._ready.clear()

    def prepare(self, *, evaluation: bool, rollout_id: int | None, reset_scope: bool) -> None:
        self.start()
        if self._prepare_calls > 0:
            self._worker_reuse_count += 1
        self._prepare_calls += 1
        self._submit(
            self._prepare_async(
                evaluation=evaluation,
                rollout_id=rollout_id,
                reset_scope=reset_scope,
            )
        )

    def run_jobs(
        self,
        requests: list[RuntimeJobRequest],
        *,
        evaluation: bool,
        max_parallel_env_jobs: int,
    ) -> list[RuntimeJobOutcome]:
        self.start()
        return self._submit(
            self._run_jobs_async(
                requests,
                evaluation=evaluation,
                max_parallel_env_jobs=max_parallel_env_jobs,
            )
        )

    def recover(self, error: BaseException | None = None) -> bool:
        self.start()
        return self._submit(self._recover_async(error=error))

    def snapshot_metrics(self) -> dict[str, int]:
        self.start()
        return self._submit(self._snapshot_metrics_async())

    async def _prepare_async(self, *, evaluation: bool, rollout_id: int | None, reset_scope: bool) -> None:
        if reset_scope:
            self._worker_prepare_reset_count += 1
            if self._runtime is not None:
                self._runtime_instance_reset_count += 1
            await self._adapter.cleanup_scope(self._scope)
            self._runtime = None
        elif self._runtime is not None:
            self._runtime_instance_reuse_count += 1
        if self._runtime is None:
            self._runtime = await self._adapter.ensure_runtime(self._args, scope=self._scope)
        await self._adapter.prepare_phase(
            self._args,
            evaluation=evaluation,
            scope=self._scope,
            reset_scope=reset_scope,
        )
        self._runtime = await self._adapter.ensure_runtime(self._args, scope=self._scope)

    async def _recover_async(self, *, error: BaseException | None = None) -> bool:
        recovered = await self._adapter.recover_runtime(self._args, scope=self._scope, error=error)
        if recovered:
            self._runtime_instance_reset_count += 1
            self._runtime = await self._adapter.ensure_runtime(self._args, scope=self._scope)
        return recovered

    async def _run_jobs_async(
        self,
        requests: list[RuntimeJobRequest],
        *,
        evaluation: bool,
        max_parallel_env_jobs: int,
    ) -> list[RuntimeJobOutcome]:
        from .base import FailureKind, RolloutResult

        if self._runtime is None:
            self._runtime = await self._adapter.ensure_runtime(self._args, scope=self._scope)
        semaphore = asyncio.Semaphore(max(1, max_parallel_env_jobs))
        sample_timeout_seconds = max(
            30.0,
            float(os.getenv("LIVEWEB_RL_SAMPLE_WALL_TIMEOUT_SECONDS", "300")),
        )
        round_timeout_seconds = max(
            sample_timeout_seconds,
            float(os.getenv("LIVEWEB_RL_ROLLOUT_ROUND_TIMEOUT_SECONDS", "900")),
        )

        def _timeout_result(
            request: RuntimeJobRequest,
            *,
            failure_reason: str,
            error_message: str,
            timeout_seconds: float,
        ) -> RuntimeJobOutcome:
            extra = {
                "failure_reason": failure_reason,
                "exception_stage": "runtime_timeout",
                "runtime_scope": self._scope,
            }
            raw_result = {
                "task_name": request.job.prompt_hint or request.job.job_id,
                "score": 0.0,
                "success": False,
                "time_taken": timeout_seconds,
                "error": error_message,
                "extra": extra,
            }
            return RuntimeJobOutcome(
                group_position=request.group_position,
                sample_position=request.sample_position,
                result=RolloutResult(
                    env_name=request.job.env_name,
                    task_name=request.job.prompt_hint or request.job.job_id,
                    reward=0.0,
                    success=False,
                    time_taken=timeout_seconds,
                    failure_kind=FailureKind.ENV_RUNTIME_FAILURE,
                    failure_stage="runtime_timeout",
                    recoverable=False,
                    runtime_scope=self._scope,
                    environment_pollution=True,
                    drop_from_training=True,
                    raw_result=raw_result,
                    error=error_message,
                ),
            )

        async def _run_one(request: RuntimeJobRequest) -> RuntimeJobOutcome:
            async with semaphore:
                try:
                    result = await asyncio.wait_for(
                        self._adapter.run_job(
                            self._args,
                            self._runtime,
                            request.job,
                            evaluation=evaluation,
                        ),
                        timeout=sample_timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    self._sample_wall_timeout_count += 1
                    logger.warning(
                        "runtime_worker/sample_timeout scope=%s group=%s sample=%s timeout=%.1fs job_id=%s",
                        self._scope,
                        request.group_position,
                        request.sample_position,
                        sample_timeout_seconds,
                        request.job.job_id,
                    )
                    return _timeout_result(
                        request,
                        failure_reason="sample_wall_timeout",
                        error_message=f"Sample wall-clock timeout after {sample_timeout_seconds:.1f}s",
                        timeout_seconds=sample_timeout_seconds,
                    )
                if result.recoverable:
                    async with self._recovery_lock:
                        recovered = await self._adapter.recover_runtime(
                            self._args,
                            scope=self._scope,
                            error=None,
                        )
                        if recovered:
                            self._runtime_instance_reset_count += 1
                            self._runtime = await self._adapter.ensure_runtime(self._args, scope=self._scope)
                            result = await self._adapter.run_job(
                                self._args,
                                self._runtime,
                                request.job,
                                evaluation=evaluation,
                            )
                return RuntimeJobOutcome(
                    group_position=request.group_position,
                    sample_position=request.sample_position,
                    result=result,
                )
        start_times: dict[asyncio.Task[RuntimeJobOutcome], tuple[RuntimeJobRequest, float]] = {}
        pending_groups: set[int] = set()
        for request in requests:
            task = asyncio.create_task(_run_one(request))
            start_times[task] = (request, time.monotonic())
            pending_groups.add(request.group_position)

        outcomes: list[RuntimeJobOutcome] = []
        round_start = time.monotonic()
        max_pending_samples = 0
        max_pending_groups = 0
        max_oldest_pending_age = 0.0
        max_active_decode_requests = 0

        while start_times:
            now = time.monotonic()
            pending_samples = len(start_times)
            pending_group_count = len({req.group_position for req, _started in start_times.values()})
            oldest_pending_age = max((now - started) for _req, started in start_times.values())
            max_pending_samples = max(max_pending_samples, pending_samples)
            max_pending_groups = max(max_pending_groups, pending_group_count)
            max_oldest_pending_age = max(max_oldest_pending_age, oldest_pending_age)
            max_active_decode_requests = max(
                max_active_decode_requests,
                min(pending_samples, max_parallel_env_jobs),
            )

            self._last_pending_samples = pending_samples
            self._last_pending_groups = pending_group_count
            self._last_oldest_pending_age_seconds = oldest_pending_age
            self._last_active_decode_requests = min(pending_samples, max_parallel_env_jobs)

            round_elapsed = now - round_start
            if round_elapsed > round_timeout_seconds:
                self._group_wall_timeout_count += pending_group_count
                logger.warning(
                    "runtime_worker/round_timeout scope=%s pending_samples=%s pending_groups=%s oldest_pending_age=%.1fs round_timeout=%.1fs",
                    self._scope,
                    pending_samples,
                    pending_group_count,
                    oldest_pending_age,
                    round_timeout_seconds,
                )
                for task, (request, started) in list(start_times.items()):
                    task.cancel()
                    self._abort_failed_count += 1
                    outcomes.append(
                        _timeout_result(
                            request,
                            failure_reason="group_wall_timeout",
                            error_message=f"Rollout round timeout after {round_timeout_seconds:.1f}s",
                            timeout_seconds=now - started,
                        )
                    )
                    del start_times[task]
                break

            done, _pending = await asyncio.wait(
                list(start_times.keys()),
                timeout=min(1.0, max(0.1, round_timeout_seconds - round_elapsed)),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                continue
            for task in done:
                request, _started = start_times.pop(task)
                try:
                    outcomes.append(task.result())
                except asyncio.CancelledError:
                    self._abort_failed_count += 1
                    outcomes.append(
                        _timeout_result(
                            request,
                            failure_reason="sample_wall_timeout",
                            error_message="Sample task cancelled while waiting for runtime completion",
                            timeout_seconds=sample_timeout_seconds,
                        )
                    )
                except Exception as exc:
                    self._abort_failed_count += 1
                    logger.warning(
                        "runtime_worker/unhandled_task_failure scope=%s group=%s sample=%s error=%r",
                        self._scope,
                        request.group_position,
                        request.sample_position,
                        exc,
                    )
                    outcomes.append(
                        _timeout_result(
                            request,
                            failure_reason="runtime_worker_failure",
                            error_message=f"Runtime worker failure: {exc}",
                            timeout_seconds=0.0,
                        )
                    )

        self._last_pending_samples = 0
        self._last_pending_groups = 0
        self._last_oldest_pending_age_seconds = max_oldest_pending_age
        self._last_active_decode_requests = max_active_decode_requests
        return list(outcomes)

    async def _snapshot_metrics_async(self) -> dict[str, int]:
        metrics = dict(self._adapter.snapshot_runtime_metrics(scope=self._scope))
        metrics.update(
            {
                "worker_reuse_count": self._worker_reuse_count,
                "worker_prepare_reset_count": self._worker_prepare_reset_count,
                "runtime_instance_reuse_count": self._runtime_instance_reuse_count,
                "runtime_instance_reset_count": self._runtime_instance_reset_count,
                "jit_kernel_enabled": self._jit_kernel_enabled,
                "kernel_fallback": self._kernel_fallback,
                "sample_wall_timeout_count": self._sample_wall_timeout_count,
                "group_wall_timeout_count": self._group_wall_timeout_count,
                "abort_failed_count": self._abort_failed_count,
                "pending_samples": self._last_pending_samples,
                "pending_groups": self._last_pending_groups,
                "oldest_pending_age_seconds": self._last_oldest_pending_age_seconds,
                "active_decode_requests": self._last_active_decode_requests,
            }
        )
        return metrics

    async def _cleanup_async(self) -> None:
        await self._adapter.cleanup_scope(self._scope)
        self._runtime = None

    def _submit(self, coro):
        if self._loop is None:
            raise RuntimeError(f"Runtime worker for scope {self._scope} is not started")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result()
        except concurrent.futures.CancelledError as exc:
            raise RuntimeError(f"Runtime worker call for scope {self._scope} was cancelled") from exc

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import EnvironmentAdapter, JobSpec, RolloutResult


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
        if self._runtime is None:
            self._runtime = await self._adapter.ensure_runtime(self._args, scope=self._scope)
        semaphore = asyncio.Semaphore(max(1, max_parallel_env_jobs))

        async def _run_one(request: RuntimeJobRequest) -> RuntimeJobOutcome:
            async with semaphore:
                result = await self._adapter.run_job(
                    self._args,
                    self._runtime,
                    request.job,
                    evaluation=evaluation,
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

        outcomes = await asyncio.gather(*[_run_one(request) for request in requests])
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

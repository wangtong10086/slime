import sys
import types
from types import SimpleNamespace


_wandb_stub = types.SimpleNamespace(
    run=None,
    finish=lambda: None,
    init=lambda **kwargs: None,
    login=lambda **kwargs: None,
    define_metric=lambda *args, **kwargs: None,
    Settings=lambda **kwargs: types.SimpleNamespace(**kwargs),
    util=types.SimpleNamespace(generate_id=lambda: "stubid"),
)
sys.modules.setdefault("wandb", _wandb_stub)

from slime.utils import logging_utils, wandb_utils


def _args(use_wandb: bool = True):
    return SimpleNamespace(use_wandb=use_wandb)


def test_finish_tracking_is_idempotent(monkeypatch):
    calls = []
    wandb_utils.reset_wandb_finish_guard_for_tests()
    monkeypatch.setattr(wandb_utils.wandb, "run", object())

    def _finish():
        calls.append("finish")

    monkeypatch.setattr(wandb_utils.wandb, "finish", _finish)

    logging_utils.finish_tracking(_args())
    logging_utils.finish_tracking(_args())

    assert calls == ["finish"]


def test_finish_tracking_swallows_broken_pipe(monkeypatch):
    wandb_utils.reset_wandb_finish_guard_for_tests()
    monkeypatch.setattr(wandb_utils.wandb, "run", object())

    def _raise():
        raise BrokenPipeError("pipe closed")

    monkeypatch.setattr(wandb_utils.wandb, "finish", _raise)
    logging_utils.finish_tracking(_args())


def test_finish_tracking_uses_process_guard(monkeypatch):
    calls = []
    wandb_utils.reset_wandb_finish_guard_for_tests()
    monkeypatch.setattr(wandb_utils.wandb, "run", object())

    def _finish():
        calls.append("finish")

    monkeypatch.setattr(wandb_utils.wandb, "finish", _finish)

    assert wandb_utils.finish_wandb_once() is True
    assert wandb_utils.finish_wandb_once() is False
    assert calls == ["finish"]


def test_service_teardown_patch_is_idempotent_and_swallows_broken_pipe(monkeypatch):
    callbacks = []

    class _FakeServiceConnection:
        def __init__(self, asyncer, client, proc, cleanup=None):
            self.cleanup = cleanup

        def teardown(self, exit_code):
            raise BrokenPipeError("closed during teardown")

    fake_atexit = types.SimpleNamespace(
        register=lambda fn: callbacks.append(fn),
        unregister=lambda fn: callbacks.remove(fn) if fn in callbacks else None,
    )
    fake_service_connection = types.SimpleNamespace(
        atexit=fake_atexit,
        service_process=types.SimpleNamespace(
            start=lambda settings: types.SimpleNamespace(
                token=types.SimpleNamespace(
                    connect=lambda asyncer=None: object(),
                    save_to_env=lambda: None,
                )
            )
        ),
        ExitHooks=lambda: types.SimpleNamespace(hook=lambda: None, exit_code=0),
        ServiceConnection=_FakeServiceConnection,
        _start_and_connect_service=lambda asyncer, settings: None,
    )
    fake_package = types.ModuleType("wandb.sdk.lib.service")
    fake_package.service_connection = fake_service_connection

    monkeypatch.setitem(sys.modules, "wandb.sdk", types.ModuleType("wandb.sdk"))
    monkeypatch.setitem(sys.modules, "wandb.sdk.lib", types.ModuleType("wandb.sdk.lib"))
    monkeypatch.setitem(sys.modules, "wandb.sdk.lib.service", fake_package)
    monkeypatch.setattr(wandb_utils, "_SERVICE_PATCH_INSTALLED", False)

    wandb_utils.ensure_wandb_service_teardown_patch_installed()
    patched = fake_service_connection._start_and_connect_service
    wandb_utils.ensure_wandb_service_teardown_patch_installed()

    assert getattr(patched, "_slime_patched", False) is True

    conn = patched(asyncer=object(), settings=object())
    assert len(callbacks) == 1
    callbacks[0]()
    assert conn.cleanup is not None

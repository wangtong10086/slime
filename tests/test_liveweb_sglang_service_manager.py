import importlib.util
from pathlib import Path


SCRIPT_PATH = Path("/home/xmyf/slime/scripts/liveweb_sglang_service_manager.py")
spec = importlib.util.spec_from_file_location("liveweb_sglang_service_manager", SCRIPT_PATH)
service_manager = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(service_manager)


def test_wait_for_server_ignores_proxy_env(monkeypatch):
    class DummyResponse:
        status_code = 200

    calls = {}

    class DummyClient:
        def __init__(self, *, trust_env):
            calls["trust_env"] = trust_env

        def get(self, url, headers, timeout):
            calls["url"] = url
            calls["headers"] = headers
            return DummyResponse()

        def close(self):
            calls["closed"] = True

    monkeypatch.setattr(service_manager.httpx, "Client", DummyClient)

    assert service_manager.wait_for_server("http://127.0.0.1:16002/v1", "local-liveweb", timeout_s=1)
    assert calls["trust_env"] is False
    assert calls["url"] == "http://127.0.0.1:16002/v1/models"
    assert calls["headers"]["Authorization"] == "Bearer local-liveweb"
    assert calls["closed"] is True


def test_ensure_port_is_clear_raises_on_busy_port(monkeypatch):
    monkeypatch.setattr(service_manager, "is_port_listening", lambda port, host="127.0.0.1", timeout_s=1.0: True)

    try:
        service_manager.ensure_port_is_clear(16000)
    except RuntimeError as exc:
        assert "Port 16000 is already accepting connections" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


from types import SimpleNamespace

from slime.utils import http_utils


def test_init_http_client_disables_env_proxy(monkeypatch):
    created = {}

    class DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            created["kwargs"] = kwargs

    monkeypatch.setattr(http_utils.httpx, "AsyncClient", DummyAsyncClient)
    monkeypatch.setattr(http_utils, "_http_client", None)
    monkeypatch.setattr(http_utils, "_distributed_post_enabled", False)

    args = SimpleNamespace(
        rollout_num_gpus=8,
        sglang_server_concurrency=16,
        rollout_num_gpus_per_engine=2,
        use_distributed_post=False,
    )

    http_utils.init_http_client(args)

    assert created["kwargs"]["trust_env"] is False


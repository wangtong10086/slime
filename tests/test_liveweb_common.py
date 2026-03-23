from argparse import Namespace


def test_discover_worker_urls_uses_control_plane_headers(monkeypatch):
    from slime.rollout.liveweb_online import common

    captured: dict[str, object] = {}

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"urls": ["http://214.2.15.1:15002"]}

    class _FakeSession:
        trust_env = True

        def get(self, url, timeout=None, headers=None):
            captured["url"] = url
            captured["timeout"] = timeout
            captured["headers"] = headers
            return _FakeResponse()

    monkeypatch.setattr(common.requests, "Session", lambda: _FakeSession())

    args = Namespace(sglang_router_ip="127.0.0.1", sglang_router_port=3000, sglang_api_key="local-liveweb")
    urls = common.discover_worker_urls(args)

    assert urls == ["http://214.2.15.1:15002"]
    assert captured["url"] == "http://127.0.0.1:3000/list_workers"
    assert captured["timeout"] == 15
    assert captured["headers"]["Authorization"] == "Bearer local-liveweb"
    assert captured["headers"]["X-API-Key"] == "local-liveweb"

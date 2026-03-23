"""Unit tests for SglangConfig multi-model parsing with update_weights."""

import tempfile

import pytest
import yaml


def _write_yaml(data: dict) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(data, f)
    f.flush()
    return f.name


class TestSglangConfigUpdateWeights:
    def test_update_weights_default_true_after_resolve(self):
        """Models without explicit update_weights should resolve to True."""
        from argparse import Namespace

        from slime.backends.sglang_utils.sglang_config import SglangConfig

        path = _write_yaml(
            {
                "sglang": [
                    {
                        "name": "actor",
                        "engine_groups": [{"worker_type": "regular", "num_gpus": 4}],
                    }
                ]
            }
        )
        config = SglangConfig.from_yaml(path)
        assert len(config.models) == 1
        assert config.models[0].update_weights is None
        config.models[0].resolve(Namespace(rollout_num_gpus_per_engine=4, hf_checkpoint=None))
        assert config.models[0].update_weights is True

    def test_update_weights_explicit_false(self):
        """Models with update_weights: false should be parsed correctly."""
        from slime.backends.sglang_utils.sglang_config import SglangConfig

        path = _write_yaml(
            {
                "sglang": [
                    {
                        "name": "actor",
                        "update_weights": True,
                        "engine_groups": [{"worker_type": "regular", "num_gpus": 4}],
                    },
                    {
                        "name": "ref",
                        "update_weights": False,
                        "model_path": "/path/to/ref",
                        "engine_groups": [{"worker_type": "regular", "num_gpus": 2}],
                    },
                ]
            }
        )
        config = SglangConfig.from_yaml(path)
        assert len(config.models) == 2
        assert config.models[0].name == "actor"
        assert config.models[0].update_weights is True
        assert config.models[1].name == "ref"
        assert config.models[1].update_weights is False
        assert config.models[1].model_path == "/path/to/ref"

    def test_multi_model_total_gpus(self):
        """total_num_gpus should sum across all models."""
        from slime.backends.sglang_utils.sglang_config import SglangConfig

        path = _write_yaml(
            {
                "sglang": [
                    {
                        "name": "actor",
                        "server_groups": [{"worker_type": "regular", "num_gpus": 8}],
                    },
                    {
                        "name": "ref",
                        "update_weights": False,
                        "server_groups": [{"worker_type": "regular", "num_gpus": 4}],
                    },
                ]
            }
        )
        config = SglangConfig.from_yaml(path)
        assert config.total_num_gpus == 12


class TestGetModelUrl:
    def test_get_model_url_basic(self):
        """get_model_url should return the correct URL for a named model."""
        from argparse import Namespace

        from slime.rollout.sglang_rollout import get_model_url

        args = Namespace(
            sglang_router_ip="10.0.0.1",
            sglang_router_port=3000,
            sglang_model_routers={
                "actor": ("10.0.0.1", 3000),
                "ref": ("10.0.0.1", 3001),
            },
        )
        assert get_model_url(args, "actor") == "http://10.0.0.1:3000/generate"
        assert get_model_url(args, "ref") == "http://10.0.0.1:3001/generate"
        assert get_model_url(args, "ref", "/v1/chat/completions") == "http://10.0.0.1:3001/v1/chat/completions"

    def test_get_model_url_fallback(self):
        """get_model_url should fall back to default router if model not found."""
        from argparse import Namespace

        from slime.rollout.sglang_rollout import get_model_url

        args = Namespace(
            sglang_router_ip="10.0.0.1",
            sglang_router_port=3000,
            sglang_model_routers={"actor": ("10.0.0.1", 3000)},
        )
        assert get_model_url(args, "unknown") == "http://10.0.0.1:3000/generate"

    def test_get_model_url_no_routers(self):
        """get_model_url should work when sglang_model_routers is not set."""
        from argparse import Namespace

        from slime.rollout.sglang_rollout import get_model_url

        args = Namespace(
            sglang_router_ip="10.0.0.1",
            sglang_router_port=3000,
        )
        assert get_model_url(args, "anything") == "http://10.0.0.1:3000/generate"


class TestWorkerControlHeaders:
    def test_build_worker_control_headers_uses_arg_api_key(self, monkeypatch):
        from argparse import Namespace

        from slime.rollout.sglang_rollout import _build_worker_control_headers

        monkeypatch.delenv("LIVEWEB_API_KEY", raising=False)
        monkeypatch.delenv("API_KEY", raising=False)
        monkeypatch.delenv("SGLANG_API_KEY", raising=False)

        args = Namespace(sglang_api_key="local-liveweb")
        headers = _build_worker_control_headers(args)
        assert headers["Content-Type"] == "application/json; charset=utf-8"
        assert headers["Authorization"] == "Bearer local-liveweb"
        assert headers["X-API-Key"] == "local-liveweb"

    def test_build_worker_control_headers_falls_back_to_env(self, monkeypatch):
        from argparse import Namespace

        from slime.rollout.sglang_rollout import _build_worker_control_headers

        monkeypatch.setenv("SGLANG_API_KEY", "from-env")
        monkeypatch.delenv("LIVEWEB_API_KEY", raising=False)
        monkeypatch.delenv("API_KEY", raising=False)

        args = Namespace()
        headers = _build_worker_control_headers(args)
        assert headers["Authorization"] == "Bearer from-env"
        assert headers["X-API-Key"] == "from-env"


class TestSglangControlPlaneAuth:
    def test_raise_for_control_plane_auth_failure_raises_on_401(self):
        import requests

        from slime.backends.sglang_utils.control_plane import raise_for_control_plane_auth_failure

        request = requests.Request("GET", "http://127.0.0.1:3000/get_server_info").prepare()
        response = requests.Response()
        response.status_code = 401
        response.request = request

        with pytest.raises(PermissionError):
            raise_for_control_plane_auth_failure(response, "get_server_info")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

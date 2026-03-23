import os
from argparse import Namespace

import requests


def resolve_control_plane_api_key(args: Namespace | None = None) -> str | None:
    if args is not None:
        value = getattr(args, "sglang_api_key", None)
        if value:
            return str(value)
    return (
        os.getenv("LIVEWEB_API_KEY")
        or os.getenv("API_KEY")
        or os.getenv("SGLANG_API_KEY")
    )


def build_control_plane_headers(args: Namespace | None = None) -> dict[str, str]:
    headers = {"Content-Type": "application/json; charset=utf-8"}
    api_key = resolve_control_plane_api_key(args)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["X-API-Key"] = api_key
    return headers


def iter_control_plane_header_variants(args: Namespace | None = None) -> list[dict[str, str]]:
    api_key = resolve_control_plane_api_key(args)
    base = {"Content-Type": "application/json; charset=utf-8"}
    if not api_key:
        return [base]

    variants = [
        {
            **base,
            "Authorization": f"Bearer {api_key}",
            "X-API-Key": api_key,
        },
        {
            **base,
            "Authorization": f"Bearer {api_key}",
        },
        {
            **base,
            "X-API-Key": api_key,
        },
        {
            **base,
            "Authorization": api_key,
        },
    ]
    deduped: list[dict[str, str]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for variant in variants:
        marker = tuple(sorted(variant.items()))
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(variant)
    return deduped


def raise_for_control_plane_auth_failure(response: requests.Response, endpoint: str) -> None:
    if response.status_code not in {401, 403}:
        return
    request = getattr(response, "request", None)
    method = getattr(request, "method", "GET")
    url = getattr(request, "url", endpoint)
    raise PermissionError(
        f"SGLang control-plane authorization failed for {endpoint} "
        f"(status={response.status_code}, method={method}, url={url})"
    )

"""Post one reconstructed hook event to the Straiker coding-agent detect endpoint.

This is the same endpoint and the same ``x-tool`` routing header the native Claude Code
hook handler uses, so a gateway-reconstructed event scores through the identical backend
pipeline as an endpoint-installed hook.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Protocol, Union

import httpx
from pydantic import ValidationError

from litellm.exceptions import Timeout
from litellm.types.proxy.guardrails.guardrail_hooks.straiker import (
    StraikerDetectResponse,
    StraikerHookEvent,
)

BLOCK_ACTION = "block"


class AsyncPoster(Protocol):
    """The guardrail's shared HTTP client, whether that is a raw httpx client or
    litellm's AsyncHTTPHandler wrapper around one."""

    async def post(
        self,
        url: str,
        *,
        content: bytes,
        headers: dict[str, str],
        timeout: float,
    ) -> httpx.Response: ...


@dataclass(frozen=True, slots=True)
class DetectFailure:
    message: str


DetectOutcome = Union[StraikerDetectResponse, DetectFailure]


def _signature_headers(api_key: str, payload: str) -> dict[str, str]:
    timestamp = str(int(time.time()))
    digest = hmac.new(api_key.encode("utf-8"), f"{timestamp}.{payload}".encode("utf-8"), hashlib.sha256).hexdigest()
    return {"X-Straiker-Webhook-Signature": digest, "X-Straiker-Webhook-Timestamp": timestamp}


def build_headers(*, api_key: str, x_tool: str, payload: str, sign: bool) -> dict[str, str]:
    """``Straiker-Debug`` is not optional here: without it the coding endpoint answers
    non-tool events with an empty body and tool events with only a permission decision,
    neither of which carries the score or the block action the gateway needs."""
    base = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "x-tool": x_tool,
        "Straiker-Debug": "TRUE",
    }
    return {**base, **_signature_headers(api_key, payload)} if sign else base


async def post_hook_event(
    *,
    client: AsyncPoster,
    url: str,
    api_key: str,
    x_tool: str,
    event: StraikerHookEvent,
    sign: bool,
    timeout: float,
) -> DetectOutcome:
    payload = event.model_dump_json(exclude_none=True)
    try:
        response = await client.post(
            url,
            content=payload.encode("utf-8"),
            headers=build_headers(api_key=api_key, x_tool=x_tool, payload=payload, sign=sign),
            timeout=timeout,
        )
    except (httpx.RequestError, Timeout) as error:
        return DetectFailure(f"{type(error).__name__}: {error}")
    if response.status_code != 200:
        return DetectFailure(f"HTTP {response.status_code}: {response.text[:200]}")
    try:
        return StraikerDetectResponse.model_validate(response.json())
    except (ValidationError, json.JSONDecodeError, ValueError) as error:
        return DetectFailure(f"invalid response schema: {error}")


def is_block(outcome: DetectOutcome) -> bool:
    return isinstance(outcome, StraikerDetectResponse) and outcome.action == BLOCK_ACTION

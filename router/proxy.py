"""The async streaming proxy.

Exposes an OpenAI-compatible surface (`/v1/completions`, `/v1/chat/completions`)
so unmodified clients (`vllm bench serve` or the replay client) target the router
exactly as a single vLLM server. SSE token streams are proxied chunk-by-chunk
with no whole-response buffering, so first-token latency through the router
tracks the backend's first token.

Per-request flow:
  1. Receive request; stamp arrival (monotonic + wall).
  2. Read body once (small) — forwarded byte-identical to preserve trace fidelity.
  3. Ask the policy for a backend; acquire an in-flight slot; stamp dispatch.
  4. Propagate the request id via X-Request-Id so it appears in backend logs.
  5. Stream/forward the response, capturing first_token_time and usage.
  6. In a finally: release the slot, stamp completion, write the JSONL record.

router_overhead_ms = time spent in router logic (parse + route + cheap response
bookkeeping), excluding time awaiting the backend. It is logged per request so it
can be verified as a non-confound.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .backends import BackendPool
from .config import RouterConfig
from .policies import RequestInfo, RoutingPolicy, build_policy
from .reqlog import RequestLogger

# Hop-by-hop / recomputed headers we must not forward upstream. We also strip
# accept-encoding so backends never gzip the SSE stream (keeps passthrough raw).
_SKIP_REQUEST_HEADERS = {
    "host",
    "content-length",
    "connection",
    "transfer-encoding",
    "accept-encoding",
}
# Stripped from the upstream response before returning to the client.
# content-type is reapplied explicitly via media_type.
_SKIP_RESPONSE_HEADERS = {
    "content-length",
    "connection",
    "transfer-encoding",
    "content-encoding",
    "content-type",
}


def _stamp() -> dict[str, float]:
    return {"monotonic": time.monotonic(), "wall": time.time()}


def _maybe_usage(chunk: bytes) -> dict | None:
    """Extract a non-null `usage` object from an SSE chunk, if present.

    vLLM only emits usage in a streamed response when the client requested it
    (stream_options.include_usage). We never inject that option — it would alter
    the fixed trace — so usage is best-effort here and may be null; the
    authoritative token counts come from the backend logs joined on request_id.
    """
    if b"usage" not in chunk:
        return None
    try:
        text = chunk.decode("utf-8", "ignore")
    except Exception:  # noqa: BLE001
        return None
    found = None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except Exception:  # noqa: BLE001
            continue
        u = obj.get("usage")
        if u:
            found = u
    return found


def _upstream_headers(request: Request, req_id: str) -> dict[str, str]:
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _SKIP_REQUEST_HEADERS}
    headers["x-request-id"] = req_id
    return headers


def _response_headers(resp: httpx.Response) -> dict[str, str]:
    return {k: v for k, v in resp.headers.items() if k.lower() not in _SKIP_RESPONSE_HEADERS}


async def _handle(app: FastAPI, request: Request, path: str):
    pool: BackendPool = app.state.pool
    policy: RoutingPolicy = app.state.policy
    logger: RequestLogger = app.state.logger
    client: httpx.AsyncClient = app.state.client

    t_arrival_mono = time.monotonic()
    t_arrival_wall = time.time()

    raw = await request.body()
    try:
        body = json.loads(raw) if raw else {}
    except Exception:  # noqa: BLE001 - non-JSON bodies still get proxied verbatim
        body = {}
    stream = bool(body.get("stream", False))

    # Honor a client-supplied id for end-to-end joins; else mint one.
    req_id = request.headers.get("x-request-id") or f"router-{uuid.uuid4().hex}"

    info = RequestInfo(req_id, path, body, t_arrival_mono)

    # ---- routing decision (counts as router overhead) ----
    backend = policy.route(info)
    backend.acquire()
    t_dispatch_mono = time.monotonic()
    pre_overhead_ms = (t_dispatch_mono - t_arrival_mono) * 1000.0

    headers = _upstream_headers(request, req_id)
    url = f"{backend.base_url}{path}"

    record: dict[str, Any] = {
        "request_id": req_id,
        "policy": policy.describe(),
        "routed_backend": backend.label,
        "gpu_id": backend.gpu_id,
        "backend_index": backend.index,
        "stream": stream,
        "arrival_time": {"monotonic": t_arrival_mono, "wall": t_arrival_wall},
        "dispatch_time": {"monotonic": t_dispatch_mono, "wall": time.time()},
        "in_flight_at_dispatch": backend.in_flight,  # includes this request
        "prompt_tokens": None,
        "output_tokens": None,
        "first_token_time": None,
        "completion_time": None,
        "status_code": None,
        "router_overhead_ms": None,
        "error": None,
    }

    if stream:
        return await _handle_stream(client, url, raw, headers, backend, logger, record, pre_overhead_ms)
    return await _handle_unary(client, url, raw, headers, backend, logger, record, pre_overhead_ms)


async def _handle_stream(client, url, raw, headers, backend, logger, record, pre_overhead_ms):
    # Open the upstream stream first so we can mirror its status/headers.
    cm = client.stream("POST", url, content=raw, headers=headers)
    try:
        resp = await cm.__aenter__()
    except Exception as e:  # noqa: BLE001
        backend.release()
        record["error"] = f"{type(e).__name__}: {e}"
        record["completion_time"] = _stamp()
        record["router_overhead_ms"] = pre_overhead_ms
        logger.log(record)
        return JSONResponse({"error": {"message": f"upstream connect failed: {e}"}}, status_code=502)

    record["status_code"] = resp.status_code
    status_code = resp.status_code
    resp_headers = _response_headers(resp)
    media_type = resp.headers.get("content-type", "text/event-stream")

    async def gen():
        first_seen = False
        usage = None
        extra_overhead = 0.0
        try:
            async for chunk in resp.aiter_raw():
                if not chunk:
                    continue
                if not first_seen:
                    first_seen = True
                    record["first_token_time"] = _stamp()
                # Cheap inline bookkeeping is router overhead; measure it.
                o0 = time.monotonic()
                u = _maybe_usage(chunk)
                if u:
                    usage = u
                extra_overhead += time.monotonic() - o0
                yield chunk
        except Exception as e:  # noqa: BLE001 - client disconnect or upstream error
            record["error"] = f"{type(e).__name__}: {e}"
        finally:
            try:
                await cm.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
            backend.release()
            record["completion_time"] = _stamp()
            if usage:
                record["prompt_tokens"] = usage.get("prompt_tokens")
                record["output_tokens"] = usage.get("completion_tokens")
            record["router_overhead_ms"] = pre_overhead_ms + extra_overhead * 1000.0
            logger.log(record)

    return StreamingResponse(gen(), status_code=status_code, headers=resp_headers, media_type=media_type)


async def _handle_unary(client, url, raw, headers, backend, logger, record, pre_overhead_ms):
    try:
        resp = await client.post(url, content=raw, headers=headers)
    except Exception as e:  # noqa: BLE001
        backend.release()
        record["error"] = f"{type(e).__name__}: {e}"
        record["completion_time"] = _stamp()
        record["router_overhead_ms"] = pre_overhead_ms
        logger.log(record)
        return JSONResponse({"error": {"message": f"upstream request failed: {e}"}}, status_code=502)

    backend.release()
    # Non-streaming: the whole response arrives at once, so first token ~= completion.
    now = _stamp()
    record["first_token_time"] = now
    record["completion_time"] = now
    record["status_code"] = resp.status_code

    o0 = time.monotonic()
    content = resp.content
    try:
        data = resp.json()
        usage = data.get("usage") or {}
        record["prompt_tokens"] = usage.get("prompt_tokens")
        record["output_tokens"] = usage.get("completion_tokens")
    except Exception:  # noqa: BLE001 - non-JSON / error body
        pass
    record["router_overhead_ms"] = pre_overhead_ms + (time.monotonic() - o0) * 1000.0
    logger.log(record)

    return Response(
        content=content,
        status_code=resp.status_code,
        headers=_response_headers(resp),
        media_type=resp.headers.get("content-type"),
    )


def create_app(config: RouterConfig) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        t = config.timeouts
        timeout = httpx.Timeout(connect=t.connect_s, read=t.read_s, write=t.write_s, pool=t.pool_s)
        # No connection cap: the router must not throttle the trace it replays.
        limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
        client = httpx.AsyncClient(timeout=timeout, limits=limits)
        pool = BackendPool(config.backends)
        if config.health_check_on_startup:
            await pool.health_check(client, t.health_check_s)
        policy = build_policy(pool, config.policy.name, config.policy.params)
        logger = RequestLogger(config.log_path)

        app.state.client = client
        app.state.pool = pool
        app.state.policy = policy
        app.state.logger = logger
        app.state.config = config
        try:
            yield
        finally:
            await client.aclose()
            logger.close()

    app = FastAPI(title="variance-router", lifespan=lifespan)

    @app.get("/health")
    async def health():
        pool: BackendPool = app.state.pool
        policy: RoutingPolicy = app.state.policy
        return {
            "status": "ok",
            "policy": policy.describe(),
            "backends": [
                {
                    "backend": b.label,
                    "gpu_id": b.gpu_id,
                    "in_flight": b.in_flight,
                    "dispatched_total": b.dispatched_total,
                }
                for b in pool
            ],
        }

    @app.get("/v1/models")
    async def models(request: Request):
        # Convenience passthrough to the first backend (some clients probe this).
        pool: BackendPool = app.state.pool
        client: httpx.AsyncClient = app.state.client
        try:
            r = await client.get(f"{pool[0].base_url}/v1/models")
            return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"error": {"message": str(e)}}, status_code=502)

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await _handle(app, request, "/v1/completions")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await _handle(app, request, "/v1/chat/completions")

    return app

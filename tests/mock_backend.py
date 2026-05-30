#!/usr/bin/env python3
"""A fake vLLM-style OpenAI backend for local validation (no GPU).

Implements /health, /v1/models, /v1/completions, /v1/chat/completions with
SSE streaming, a simulated prefill delay (affecting TTFT) and per-token decode
delay (affecting TPOT), echoes X-Request-Id, and reports usage. Enough to
validate streaming passthrough, routing, in-flight accounting, and logging
end-to-end before touching real GPUs.

  python tests/mock_backend.py --port 8001 --gpu-id 0 \
      --prefill-ms 20 --token-delay-ms 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


def make_app(prefill_ms: float, token_delay_ms: float, gpu_id: int) -> FastAPI:
    app = FastAPI()
    app.state.prefill_s = prefill_ms / 1000.0
    app.state.token_delay_s = token_delay_ms / 1000.0
    app.state.gpu_id = gpu_id

    @app.get("/health")
    async def health():
        return {"status": "ok", "gpu_id": app.state.gpu_id}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": "mock-model", "object": "model"}]}

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await _serve(app, request, chat=False)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await _serve(app, request, chat=True)

    return app


async def _serve(app: FastAPI, request: Request, chat: bool):
    body = await request.json()
    req_id = request.headers.get("x-request-id", "mock-noid")
    n_tokens = int(body.get("max_tokens") or 8)
    stream = bool(body.get("stream", False))
    prompt_tokens = _estimate_prompt_tokens(body, chat)
    hdrs = {"x-request-id": req_id}

    if not stream:
        await asyncio.sleep(app.state.prefill_s + app.state.token_delay_s * n_tokens)
        text = " ".join(f"t{i}" for i in range(n_tokens))
        payload = _final_object(req_id, text, chat, prompt_tokens, n_tokens)
        return JSONResponse(payload, headers=hdrs)

    async def gen():
        # Prefill latency before the first token (the TTFT the router must mirror).
        await asyncio.sleep(app.state.prefill_s)
        for i in range(n_tokens):
            if i > 0:
                await asyncio.sleep(app.state.token_delay_s)
            chunk = _delta_object(req_id, f"t{i} ", chat)
            yield f"data: {json.dumps(chunk)}\n\n".encode()
        # Final usage chunk (mock always emits it so tests can read token counts).
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": n_tokens,
                 "total_tokens": prompt_tokens + n_tokens}
        final = _delta_object(req_id, "", chat, finish=True, usage=usage)
        yield f"data: {json.dumps(final)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers=hdrs)


def _estimate_prompt_tokens(body: dict, chat: bool) -> int:
    if chat:
        text = " ".join(m.get("content", "") for m in body.get("messages", []) if isinstance(m, dict))
    else:
        p = body.get("prompt", "")
        text = p if isinstance(p, str) else " ".join(map(str, p))
    return max(1, len(text.split()))


def _now() -> int:
    return int(time.time())


def _delta_object(req_id, text, chat, finish=False, usage=None):
    obj = {
        "id": req_id,
        "object": "chat.completion.chunk" if chat else "text_completion",
        "created": _now(),
        "model": "mock-model",
        "usage": usage,
    }
    if chat:
        delta = {} if finish else {"content": text}
        obj["choices"] = [{"index": 0, "delta": delta, "finish_reason": "stop" if finish else None}]
    else:
        obj["choices"] = [{"index": 0, "text": text, "finish_reason": "stop" if finish else None}]
    return obj


def _final_object(req_id, text, chat, prompt_tokens, completion_tokens):
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
             "total_tokens": prompt_tokens + completion_tokens}
    obj = {"id": req_id, "created": _now(), "model": "mock-model", "usage": usage}
    if chat:
        obj["object"] = "chat.completion"
        obj["choices"] = [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]
    else:
        obj["object"] = "text_completion"
        obj["choices"] = [{"index": 0, "text": text, "finish_reason": "stop"}]
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--prefill-ms", type=float, default=20.0)
    ap.add_argument("--token-delay-ms", type=float, default=5.0)
    args = ap.parse_args()
    app = make_app(args.prefill_ms, args.token_delay_ms, args.gpu_id)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

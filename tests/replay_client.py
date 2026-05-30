#!/usr/bin/env python3
"""Minimal fixed-trace replay client + client-side metrics.

Replays a pre-generated trace file (one request per line) against the router,
honoring each request's arrival timestamp, and measures TTFT / TPOT / end-to-end
latency from the client's perspective. This is a *test/validation* tool — the
real benchmarking repo owns experiment orchestration; this exists so the router
can be exercised standalone.

Trace format (JSONL), one object per line:
    {"arrival_s": 0.00, "endpoint": "/v1/completions", "body": {...}}
`arrival_s` is seconds from the start of replay. `body` is the verbatim OpenAI
request payload (must set "stream": true to measure TTFT/TPOT).

  python tests/replay_client.py --url http://127.0.0.1:8000 --trace trace.jsonl

`--make-trace out.jsonl --n 200 --rate 50 --burst-cv 2.0` generates a simple
bursty example trace (fixed: written once, then replayed identically).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path

import httpx


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #


async def _one_request(client, base_url, item, t0, results):
    target = t0 + float(item.get("arrival_s", 0.0))
    now = time.monotonic()
    if target > now:
        await asyncio.sleep(target - now)

    endpoint = item.get("endpoint", "/v1/completions")
    body = item["body"]
    stream = bool(body.get("stream", False))
    url = base_url.rstrip("/") + endpoint

    send_mono = time.monotonic()
    first_mono = None
    n_chunks = 0
    output_tokens = None
    status = None
    try:
        if stream:
            async with client.stream("POST", url, json=body) as resp:
                status = resp.status_code
                async for chunk in resp.aiter_raw():
                    if not chunk:
                        continue
                    if first_mono is None:
                        first_mono = time.monotonic()
                    n_chunks += 1
                    u = _scan_usage(chunk)
                    if u and u.get("completion_tokens") is not None:
                        output_tokens = u["completion_tokens"]
            done_mono = time.monotonic()
        else:
            resp = await client.post(url, json=body)
            status = resp.status_code
            done_mono = first_mono = time.monotonic()
            try:
                output_tokens = (resp.json().get("usage") or {}).get("completion_tokens")
            except Exception:  # noqa: BLE001
                pass
    except Exception as e:  # noqa: BLE001
        results.append({"ok": False, "error": f"{type(e).__name__}: {e}"})
        return

    ttft = (first_mono - send_mono) if first_mono else float("nan")
    e2e = done_mono - send_mono
    # Token count for TPOT: prefer reported usage, else SSE data-event count.
    n_out = output_tokens if output_tokens is not None else max(n_chunks - 1, 0)
    tpot = (e2e - ttft) / (n_out - 1) if n_out and n_out > 1 else float("nan")
    results.append({
        "ok": status is not None and status < 400,
        "status": status,
        "ttft_ms": ttft * 1000.0,
        "e2e_ms": e2e * 1000.0,
        "tpot_ms": tpot * 1000.0,
        "output_tokens": n_out,
    })


async def replay(url: str, trace: list[dict]):
    results: list[dict] = []
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=5, read=None, write=10, pool=5), limits=limits) as client:
        t0 = time.monotonic()
        tasks = [asyncio.create_task(_one_request(client, url, item, t0, results)) for item in trace]
        await asyncio.gather(*tasks)
    return results


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def _pct(values, p):
    vals = sorted(v for v in values if v == v)  # drop NaN
    if not vals:
        return float("nan")
    k = (len(vals) - 1) * (p / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return vals[int(k)]
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def summarize(results: list[dict]) -> None:
    ok = [r for r in results if r.get("ok")]
    errs = [r for r in results if not r.get("ok")]
    print(f"\nrequests: {len(results)}  ok: {len(ok)}  errors: {len(errs)}")
    if errs[:3]:
        for e in errs[:3]:
            print(f"  error sample: {e.get('error') or e.get('status')}")
    if not ok:
        return
    for metric in ("ttft_ms", "e2e_ms", "tpot_ms"):
        vals = [r[metric] for r in ok]
        print(f"  {metric:8s}  P50={_pct(vals,50):8.2f}  P95={_pct(vals,95):8.2f}  P99={_pct(vals,99):8.2f}")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _scan_usage(chunk: bytes):
    if b"usage" not in chunk:
        return None
    found = None
    for line in chunk.decode("utf-8", "ignore").splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            u = json.loads(payload).get("usage")
        except Exception:  # noqa: BLE001
            continue
        if u:
            found = u
    return found


def load_trace(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def make_trace(path: str, n: int, rate: float, burst_cv: float, max_tokens: int, chat: bool,
               model: str = "mock-model") -> None:
    """Generate a simple bursty trace deterministically (no RNG seed reliance).

    Inter-arrival gaps alternate between short (bursty) and long (quiet) to give
    a high aggregate CV without depending on a live RNG; the resulting file is
    the fixed trace replayed identically across policies.
    """
    endpoint = "/v1/chat/completions" if chat else "/v1/completions"
    mean_gap = 1.0 / rate
    short = mean_gap / max(burst_cv, 1.0)
    long = mean_gap * max(burst_cv, 1.0)
    lines = []
    t = 0.0
    for i in range(n):
        # bursts of ~5 short gaps then one long quiet gap
        gap = long if (i % 6 == 5) else short
        t += gap
        if chat:
            body = {"model": model, "stream": True, "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": f"req {i} " + "x " * 10}]}
        else:
            body = {"model": model, "stream": True, "max_tokens": max_tokens,
                    "prompt": f"req {i} " + "x " * 10}
        lines.append(json.dumps({"arrival_s": round(t, 6), "endpoint": endpoint, "body": body}))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)  # auto-create output dir
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {n} requests to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--trace", help="Trace JSONL to replay.")
    ap.add_argument("--make-trace", help="Generate a bursty example trace to this path and exit.")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--burst-cv", type=float, default=2.0)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--model", default="mock-model",
                    help="model field written into the trace; MUST match the served vLLM model.")
    args = ap.parse_args()

    if args.make_trace:
        make_trace(args.make_trace, args.n, args.rate, args.burst_cv, args.max_tokens, args.chat, args.model)
        return

    if not args.trace:
        ap.error("provide --trace to replay or --make-trace to generate one")
    trace = load_trace(args.trace)
    results = asyncio.run(replay(args.url, trace))
    summarize(results)


if __name__ == "__main__":
    main()

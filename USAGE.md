# Usage

The design rationale lives in `README.md`. This file is the operational guide.

## Install

```bash
pip install -r requirements.txt
```

Dependency-light by design: FastAPI + uvicorn + httpx + pydantic + PyYAML. No
`vllm`, no `torch`.

## Run the router

```bash
python router.py --config config.yaml      # --host/--port override the config
```

The router health-checks every backend on startup and **fails loudly** if any is
unreachable. Point your client (`vllm bench serve` or `tests/replay_client.py`)
at `http://<router-host>:<port>` exactly as if it were a single vLLM server.

## Config

See `config.example.yaml`. One file specifies the backend list, the active
policy + params, the log path, and timeouts. To compare policies apples-to-apples,
change **only** `policy:` between runs and replay the **same fixed trace**.

## Policies

| name           | role          | key params |
|----------------|---------------|------------|
| `round_robin`  | baseline      | — |
| `jsq`          | baseline      | — |
| `burst`        | intervention  | `burst_size` (primary), `active_window_ms`, `num_active_nodes`, `within_active` |
| `regime_aware` | stretch       | `low_watermark`, `high_watermark` |

`burst`: routes `burst_size` consecutive requests to the active node, then
rotates, so non-active nodes drain to pure-decode iterations. Set
`active_window_ms` to also/instead rotate on wall time (set `burst_size: null`
for time-only). `num_active_nodes > 1` keeps a small adjacent set active at once.

Add a policy by subclassing `RoutingPolicy` in `router/policies.py` and adding it
to the `POLICIES` registry — that is the only extension point.

## Logs

One JSONL record per request at `log_path`. Fields include `request_id`
(propagated to the backend via `X-Request-Id` for end-to-end joins),
`routed_backend` + `gpu_id`, monotonic+wall `arrival_time` / `dispatch_time` /
`first_token_time` / `completion_time`, `in_flight_at_dispatch`, `prompt_tokens`
/ `output_tokens` (when the backend reports usage), the `policy` in effect,
`router_overhead_ms`, `retries` (connection retries before success — see below),
and `error` (null on success).

The router keeps its idle-connection expiry below the backend's keep-alive
timeout and retries once on a fresh connection if a dispatch fails before any
response byte (a backend closing an idle keepalive socket on reuse). Such a
retry is same-backend (never re-routed) and pre-response, so it can't
double-deliver tokens. A record with `retries > 0` and `error: null` means the
retry recovered a transient connection failure; a `502` with a populated `error`
means it failed even after the retry (e.g. a backend that is down or saturated).

**Per-node arrival CV** is computed offline by the benchmarking repo from these
records: group by `routed_backend`, take the per-node sequence of
`dispatch_time.monotonic`, and compute the CV (std/mean) of consecutive
inter-dispatch gaps. Latency percentiles (TTFT/TPOT/e2e at P50/P95/P99) likewise
come from the timestamp fields. The router stays minimal and emits raw signals.

## Local validation (no GPU)

A mock vLLM-style SSE backend and a replay client let you exercise the full path
without real backends.

```bash
# 1. one-shot end-to-end smoke test (spawns mocks + router + replay, validates log)
python tests/run_integration.py

# 2. policy unit tests
python tests/test_policies.py

# 3. manual: 2 mock backends + router + a generated bursty trace
python tests/mock_backend.py --port 8001 --gpu-id 0 &
python tests/mock_backend.py --port 8002 --gpu-id 1 &
python router.py --config config.example.yaml &   # point backends at 8001/8002
python tests/replay_client.py --make-trace /tmp/trace.jsonl --n 200 --rate 50
python tests/replay_client.py --url http://127.0.0.1:8000 --trace /tmp/trace.jsonl
```

The smoke test asserts streaming is not buffered (first-token latency is
captured, not collapsed into end-to-end), router overhead is sub-millisecond,
request IDs propagate, and the burst policy actually concentrates arrivals.

## Validation sequencing (per README)

1. **Skeleton + passthrough** — single backend, confirm router TTFT/TPOT match
   direct-to-vLLM (overhead negligible). `run_integration.py` covers the
   passthrough/overhead checks against mocks.
2. **Baselines** (`round_robin`, `jsq`) over N backends on a fixed trace; confirm
   the pipeline + log joins are clean and the two baselines are ~indistinguishable.
3. **Burst** — sweep `burst_size` / `active_window_ms`; verify induced per-node CV
   rises vs. baselines, then look for the TPOT gain and TTFT trade-off.
4. **Regime-aware** — optional second intervention.

# Variance-Shaping Router for Multi-Instance vLLM Serving

## Purpose

This repo implements a lightweight HTTP router that sits between a benchmark
client and a set of independent vLLM instances. Its job is to **control the
arrival-process variance each backend instance sees** by choosing routing
policies, so we can test the hypothesis that deliberately concentrating
arrivals into per-node bursts (separated by quiet periods) improves decode
throughput (TPOT) relative to load-spreading policies like join-shortest-queue.

This router is one of three repos in the project:

- **vLLM fork** (separate repo): patched with CUDA-event-based iteration timing.
  Provides accurate per-iteration forward-pass latency and batch-composition logs.
- **Benchmarking repo** (separate repo): orchestrates experiments, generates/replays
  traces, launches instances + this router, collects and analyzes logs.
- **This repo (router)**: a standalone async HTTP proxy. Dependency-light. Does
  **not** import vllm — it speaks only the OpenAI-compatible HTTP API to backends.

The benchmarking repo owns end-to-end experiment wiring. This README documents
the router in isolation and its contract with the other two components.

## Topology

```
trace replay client  ->  THIS ROUTER  ->  vLLM instance 0 (GPU 0, port 8001)
(or vllm bench serve)         |       ->  vLLM instance 1 (GPU 1, port 8002)
                              |       ->  ... 
                              |       ->  vLLM instance N-1 (GPU N-1, port 800N)
```

- N independent `vllm serve` instances, **one per GPU**, each its own port.
- **Do NOT use vLLM data-parallel mode** (`--data-parallel-size`). DP mode does
  its own internal request distribution, which would be a second router fighting
  this one and confounding every measurement. Each backend must be an independent
  instance whose only request source is this router.
- The router is the **single routing layer**. No other component makes routing
  decisions.

## Core design principles (read before implementing)

### 1. Two distinct variance knobs — keep them separate
- **Aggregate arrival CV**: variance of the request stream hitting the router.
  This is a property of the *trace* and is held FIXED across policy comparisons.
- **Per-node arrival CV**: variance each backend sees after routing. This is what
  the *routing policy* controls and is the independent variable of the experiment.

The router must measure and log the induced per-node CV for each policy. This is
the bridge between the single-node mechanism (already established: higher per-node
CV -> more pure-decode iterations -> lower TPOT) and the multi-node intervention.
If a "burst" policy does not actually raise per-node CV vs. the baseline, the
hypothesis is falsified and we need to see that immediately.

### 2. Identical trace through every policy (apples-to-apples)
Every policy comparison MUST replay the **same fixed trace**: identical arrival
timestamps, prompts, and max_tokens. Routing policy is the ONLY thing that varies.
The trace is pre-generated and replayed from a file — arrival variance is baked
into the trace, NOT generated live. (Live generation + seed reproducibility is
unreliable end-to-end in this system; do not depend on it.)

### 3. The policy is "burst + drain", not "send bursts"
The mechanism benefits from a node alternating between a burst (prefill-heavy)
and a quiet period during which its running set drains to cheap pure-decode
iterations. Routing must therefore *concentrate arrivals temporally per node and
coordinate quiet periods across nodes* — e.g., clump requests onto node 0 while
nodes 1..N-1 drain, then move the clump to node 1, etc. Simply emitting bursts
without coordinated per-node quiet periods will not reproduce the effect.

### 4. Honest multi-metric evaluation
This intervention is a TRADE-OFF, not free latency. Concentrating arrivals is
expected to improve TPOT but worsen TTFT and possibly tail latency and peak
utilization. The router must capture enough data to report TTFT, TPOT,
end-to-end latency, and throughput, at P50/P95/P99, so the benchmarking repo can
characterize *where the trade is favorable and where it is not*.

## Router requirements

### Interface
- Expose an OpenAI-compatible endpoint (at minimum `POST /v1/completions` and
  `POST /v1/chat/completions`) so unmodified clients (`vllm bench serve` or our
  replay client) can target the router exactly as they would a single vLLM server.
- **Stream responses passthrough**: proxy SSE token streams from backend to client
  WITHOUT buffering whole responses. Buffering destroys TTFT/TPOT measurement.
  First-token latency through the router must track the backend's first token.
- The router's own added latency must be negligible vs. inference time. Measure
  and log per-request router overhead; surface it so it can be verified as a
  non-confound.

### Backend management
- Read backend instance list (host:port, associated GPU id) from a config file.
- Track per-backend in-flight request count (requests dispatched but not yet
  completed) — needed for the JSQ baseline and for logging per-node load.
- Health-check backends on startup; fail loudly if any is unreachable.

### Pluggable routing policy interface
Implement a clean policy abstraction (`route(request, backend_states) -> backend`)
with these concrete policies, swappable via config:

1. **round_robin** — baseline. Cycle through backends per request.
2. **jsq** (join-shortest-queue) — baseline. Route to backend with fewest in-flight
   requests. (Confirm this is the policy we are claiming to beat; the production
   default may differ. Document exactly which baseline each run uses.)
3. **burst** — the intervention. Route consecutive requests to the same backend in
   clumps of configurable size, cycling the "active" backend across nodes so that
   non-active nodes drain. Parameters: `burst_size` (requests per clump) and/or
   `active_window_ms` (how long a node stays active), plus how many nodes are
   active at once.
4. **regime_aware** (stretch / second policy) — route to keep each node's in-flight
   count either low (memory-bound, cheap decode) or in the efficient compute-bound
   batch range, explicitly avoiding the expensive mixed/middle regime. Framed
   directly in terms of the measured roofline transition rather than variance per se.

### Logging (per-request, router side)
Emit one structured record (JSONL) per request with at minimum:
- `request_id` (must be traceable end-to-end: client -> router -> backend -> backend logs)
- `arrival_time` (router receipt timestamp, monotonic + wall clock)
- `routed_backend` (host:port and GPU id)
- `dispatch_time`, `first_token_time`, `completion_time`
- `prompt_tokens`, `output_tokens` (from backend response usage if available)
- `policy` name and parameters in effect
- `router_overhead_ms` (time spent inside router logic, excluding backend wait)

Request IDs must propagate to the backend so these records can be joined against
each vLLM instance's iteration/request logs (produced by the CUDA-event-patched
fork). If a request cannot be traced through the full path, the experiment cannot
attribute a TPOT measurement to a routing decision — this traceability is a hard
requirement, not a nice-to-have.

### Config
Single config file (YAML or JSON) specifying:
- backend list (host, port, gpu_id)
- active policy + policy params
- output log path
- any router-level timeouts

CLI: `python router.py --config config.yaml`

## Implementation guidance
- FastAPI + httpx (async) is a sensible stack. Async streaming proxy is the only
  fiddly part — get SSE passthrough right first.
- Keep it dependency-light. No vllm import. No torch. Just an HTTP service.
- Keep total size modest (a few hundred lines). No premature abstraction beyond
  the policy interface.

## Build / validation sequencing (do in this order)
1. **Skeleton + passthrough**: router proxies a single backend, streaming works,
   TTFT/TPOT measured at the client match direct-to-vLLM numbers (router overhead
   negligible). Validate before adding any routing.
2. **Two baselines (round_robin, jsq)** across N backends, replaying a FIXED trace.
   Confirm the full pipeline — trace replay, routing, per-request logging, and
   joining router logs against backend iteration logs — produces clean,
   reproducible measurements. Expect little/no difference between two sensible
   baselines; if you can't reproduce that, the pipeline is not trustworthy yet.
   Do NOT proceed until baselines are clean.
3. **Burst policy**: add it, sweep `burst_size` / `active_window_ms`. Verify it
   raises induced per-node CV vs. baselines (necessary condition). Then look for
   the TPOT effect and the expected TTFT trade-off.
4. **Regime-aware policy** (optional second intervention) if time permits.

## Out of scope (explicitly)
- Prefix-cache / KV-cache affinity routing. Workload is synthetic with no shared
  prefixes, so any backend is equivalent for a given request — routing is purely a
  load/variance decision. Do not add cache-aware logic; it would introduce a
  competing concern that muddies the variance story.
- Autoscaling, retries with re-routing, multi-tenant fairness. Not needed for the
  experiment.
- vLLM data-parallel mode (see Topology — actively avoided).

## Success criterion for the experiment (what this router exists to show)
A clean, reproducible, multi-metric comparison demonstrating the regime in which a
variance-shaping routing policy improves per-node TPOT relative to load-spreading
baselines — connected back, at the iteration level, to the previously measured
mechanism (more homogeneous / pure-decode iterations under higher induced per-node
CV) — and an honest characterization of the TTFT / utilization trade-off and the
load regime where the trade ceases to be favorable (e.g., near saturation, where
the effect is expected to vanish).

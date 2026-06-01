# Smoke-testing against real vLLM (single GPU, e.g. A2000)

You can't run the real multi-GPU experiment on one GPU, but you *can* validate
the two things that gate everything downstream:

- **(1) Passthrough fidelity** — the router doesn't distort TTFT/TPOT. Needs one
  real vLLM instance. This is the hard gate (README build-step 1).
- **(2) Routing mechanics** — `jsq`/`burst` distribute and concentrate as logged.
  Already demonstrated against mocks (below); optionally re-confirm against two
  co-located real instances.

> The router never imports vLLM, so a **stock** `vllm serve` is fine for smoke
> tests. The CUDA-event-patched fork is only needed later for iteration-level
> TPOT attribution, not for these checks.

## 0. Prereqs

```bash
pip install -r requirements.txt        # router deps (vllm installed separately)
```

Pick a small model that fits your GPU (scale up if you have headroom). The model
id is passed explicitly to the Python tools via `--model` — **no env var is read
by the code**; it just has to match whatever `GET /v1/models` reports. The
commands below use `Qwen/Qwen3.5-9B` as the example id.

## 1. Passthrough fidelity (the important one)

**Terminal 1 — one vLLM instance:**
```bash
vllm serve Qwen/Qwen3.5-9B --port 8001 \
  --gpu-memory-utilization 0.85 --max-model-len 2048
```

**Terminal 2 — router fronting *only* that instance** (`smoke_single.yaml`):
```yaml
backends:
  - {host: 127.0.0.1, port: 8001, gpu_id: 0}
policy: {name: round_robin, params: {}}   # N=1, policy is irrelevant here
log_path: logs/router_log.jsonl
host: 127.0.0.1
port: 8000
```
```bash
python router.py --config smoke_single.yaml
```

**Terminal 3 — A/B the same trace, direct vs through-router:**
```bash
python tests/fidelity_check.py \
  --direct http://127.0.0.1:8001 \
  --router http://127.0.0.1:8000 \
  --model Qwen/Qwen3.5-9B \
  --make-trace --n 60 --rate 20 --max-tokens 64 \
  --router-log logs/router_log.jsonl
# add --chat to exercise /v1/chat/completions
```

`--model` MUST match the served model id (the default `mock-model` only works
against the mock backend, and real vLLM will 404 on a mismatch). `logs/` is
created automatically — you don't need to `mkdir` it.

**Pass looks like:** TPOT delta ≈ 0 (router never touches per-token cadence),
TTFT delta a few ms at most, `router_overhead_ms` sub-millisecond, all
`first_token_time` captured, no errors. If TTFT through the router jumps toward
e2e, streaming is being buffered — stop and investigate before any policy runs.

## 2. Routing mechanics

Already shown against mocks — `jsq` vs `burst` on one fixed trace:

```bash
python tests/run_integration.py          # spawns mocks+router, asserts the wiring
# or, to see the bridge metric (induced per-node CV) on any run's log:
python tests/inspect_log.py logs/router_log.jsonl
```

`inspect_log.py` reports the **aggregate arrival CV** (held fixed across policies)
and the **per-node CV** (the independent variable). On identical traces, `burst`
should roughly double per-node CV vs `jsq` and produce same-backend runs of
`burst_size`. If it doesn't, the intervention isn't actually shaping variance.

### Optional: two co-located real instances (mechanics only — NOT a measurement)

Sharing one GPU confounds throughput, so this only confirms the OpenAI wire
format + routing across *real* backends, not any TPOT effect:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3.5-9B --port 8001 \
  --gpu-memory-utilization 0.45 --max-model-len 1024 &
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3.5-9B --port 8002 \
  --gpu-memory-utilization 0.45 --max-model-len 1024 &
```
Point the router at both backends with `policy: {name: burst, params: {burst_size: 8}}`,
replay a trace, then `python tests/inspect_log.py logs/router_log.jsonl`.
If you OOM on a 4 GB A2000, drop `--max-model-len`, lower `--gpu-memory-utilization`,
or use a smaller model (e.g. `facebook/opt-125m`, completions-only) — or just
skip this and rely on step 1 + the mock mechanics test, which together already
cover the real wire format and the routing logic.

## Then: the real experiment

Move to N GPUs (one independent `vllm serve` per GPU, **no** `--data-parallel-size`),
keep the fixed trace, and run the README sequence: baselines (`round_robin`,
`jsq`) clean first, then sweep `burst` (`burst_size` / `active_window_ms`) and
confirm per-node CV rises before chasing the TPOT/TTFT trade-off.
```

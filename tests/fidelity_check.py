#!/usr/bin/env python3
"""Passthrough fidelity smoke test: direct-to-vLLM vs through-the-router.

This is README build-step 1 ("skeleton + passthrough"): prove that putting the
router in the path does NOT distort the measurement. It replays the SAME fixed
trace twice — once straight at a vLLM instance, once at the router (configured
with that single instance as its only backend) — and diffs TTFT / TPOT / e2e.

Because TPOT (per-token cadence) is set by the backend's decode loop, the router
must leave it essentially unchanged; TTFT may rise only by the router's
(sub-millisecond) hop overhead. If TTFT/TPOT diverge materially, streaming is
being buffered or the router is on the critical path — stop and fix before any
policy work.

Usage (point --direct at vLLM, --router at the router fronting that same vLLM):

    # terminal 1: vllm serve ... --port 8001
    # terminal 2: router config has ONE backend -> 127.0.0.1:8001 ; router on :8000
    python tests/fidelity_check.py \
        --direct http://127.0.0.1:8001 \
        --router http://127.0.0.1:8000 \
        --make-trace --n 60 --rate 20 --max-tokens 64 \
        --router-log logs/router_log.jsonl

Pass/fail is heuristic (GPU timing is noisy); the printed table is the real
artifact. Add --chat to exercise /v1/chat/completions instead of /v1/completions.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from replay_client import _pct, load_trace, make_trace, replay  # noqa: E402

METRICS = ("ttft_ms", "tpot_ms", "e2e_ms")


async def _warmup(url: str, trace: list[dict], k: int = 10) -> None:
    """Fire a batch of zero-delay requests so vLLM's first-call cost (CUDA-graph
    capture, torch.compile/autotuning, allocator + clock spin-up) is paid before
    the measured run. Cold-start otherwise lands entirely on whichever path runs
    first and masquerades as a router effect."""
    if k <= 0:
        return
    warm = []
    # Cycle the trace if it's shorter than k so warmup count is honored.
    for i in range(k):
        w = dict(trace[i % len(trace)])
        w["arrival_s"] = 0.0
        warm.append(w)
    await replay(url, warm)


def _ok_count(results: list[dict]) -> int:
    return sum(1 for r in results if r.get("ok"))


def _series(runs: list[dict], m: str, p: float) -> float:
    return _pct([x[m] for x in runs if x.get("ok")], p)


def _compare(d1: list[dict], r: list[dict], d2: list[dict]) -> bool:
    """A/B/A comparison: direct#1, router, direct#2.

    The two direct runs bracket run-to-run drift (warmup residue, clock/thermal
    movement) — the noise floor. The router hop can only ADD latency, so the
    verdict checks that the router doesn't exceed the SLOWER direct run by more
    than that noise floor (plus a small budget). Router faster than direct just
    means residual drift, not a real speedup.
    """
    sets = (("direct#1", d1), ("router", r), ("direct#2", d2))
    print()
    for name, v in sets:
        print(f"  {name:9s} ok={sum(1 for x in v if x.get('ok'))}/{len(v)}")
    if any(not any(x.get("ok") for x in v) for _, v in sets):
        print("  !! a run had no successful requests; cannot compare")
        return False

    print(f"\n{'metric':9s} {'pctl':5s} {'direct#1':>10s} {'router':>10s} {'direct#2':>10s} {'drift':>9s}")
    print("-" * 60)
    for m in METRICS:
        for p in (50, 95, 99):
            a, b, c = _series(d1, m, p), _series(r, m, p), _series(d2, m, p)
            print(f"{m:9s} P{p:<4d} {a:10.2f} {b:10.2f} {c:10.2f} {c - a:9.2f}")
        print("-" * 60)
    print("  drift = direct#2 - direct#1 (run-to-run noise floor: warmup/clocks/thermals).")
    print("  router faster than direct => residual drift, not a real speedup.\n")

    ok = True
    for m, label, floor in (("ttft_ms", "TTFT", 8.0), ("tpot_ms", "TPOT", 2.0)):
        a, b, c = _series(d1, m, 50), _series(r, m, 50), _series(d2, m, 50)
        fast, slow = min(a, c), max(a, c)
        budget = max(floor, 0.15 * fast)
        excess = b - slow  # how much SLOWER the router is than the slower direct run
        if excess > budget:
            ok = False
            print(f"  [FAIL] router {label} P50 {b:.2f}ms over direct band "
                  f"[{fast:.2f},{slow:.2f}] by +{excess:.2f}ms > budget {budget:.2f}ms")
        else:
            where = "below" if b < fast else "inside"
            print(f"  [PASS] router {label} P50 {b:.2f}ms {where} direct band "
                  f"[{fast:.2f},{slow:.2f}] (drift {slow - fast:.2f}ms) — no router cost detectable")
    return ok


def _check_router_log(path: str, expected: int) -> bool:
    import json
    try:
        recs = [json.loads(l) for l in open(path) if l.strip()]
    except FileNotFoundError:
        print(f"\n  !! router log {path} not found (skipping overhead check)")
        return True
    # Only the most recent `expected` records belong to this run.
    recs = recs[-expected:]
    ohs = [r["router_overhead_ms"] for r in recs if r.get("router_overhead_ms") is not None]
    ttft_ok = sum(1 for r in recs if r.get("first_token_time"))
    errs = [r for r in recs if r.get("error")]
    max_oh = max(ohs) if ohs else float("nan")
    print(f"\nrouter log ({len(recs)} recent records)")
    print(f"  router_overhead_ms: P50={_pct(ohs,50):.3f}  P99={_pct(ohs,99):.3f}  max={max_oh:.3f}")
    print(f"  first_token_time captured: {ttft_ok}/{len(recs)}   errors: {len(errs)}")
    ok = max_oh < 10.0 and ttft_ok == len(recs) and not errs
    print(f"  [{'PASS' if ok else 'FAIL'}] router-side: overhead<10ms, all TTFT captured, no errors")
    return ok


async def _amain(args) -> int:
    if args.make_trace:
        path = args.trace or os.path.join(tempfile.gettempdir(), "fidelity_trace.jsonl")
        make_trace(path, args.n, args.rate, args.burst_cv, args.max_tokens, args.chat, args.model)
        args.trace = path
    if not args.trace:
        print("provide --trace or --make-trace", file=sys.stderr)
        return 2

    trace = load_trace(args.trace)
    print(f"replaying {len(trace)} requests; warming BOTH paths (warmup={args.warmup})...")
    await _warmup(args.direct, trace, args.warmup)
    await _warmup(args.router, trace, args.warmup)

    # A/B/A so run-to-run drift is observable and the router is judged against it.
    print(f"D1: direct -> {args.direct}")
    d1 = await replay(args.direct, trace)
    print(f"R : router -> {args.router}")
    router = await replay(args.router, trace)
    print(f"D2: direct -> {args.direct}  (drift control)")
    d2 = await replay(args.direct, trace)

    ok = _compare(d1, router, d2)
    if args.router_log:
        ok = _check_router_log(args.router_log, len(trace)) and ok

    print(f"\n{'FIDELITY OK' if ok else 'FIDELITY CHECK FAILED — investigate before policy runs'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--direct", required=True, help="Base URL of the vLLM instance.")
    ap.add_argument("--router", required=True, help="Base URL of the router fronting that instance.")
    ap.add_argument("--trace", help="Trace JSONL to replay (same file used for both sides).")
    ap.add_argument("--router-log", help="Router JSONL log to check overhead / TTFT capture.")
    ap.add_argument("--make-trace", action="store_true", help="Generate a bursty trace first.")
    ap.add_argument("--warmup", type=int, default=10,
                    help="Warmup requests fired at BOTH paths before measuring (absorbs cold-start).")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--rate", type=float, default=20.0)
    ap.add_argument("--burst-cv", type=float, default=2.0)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--model", default="mock-model",
                    help="model field in generated requests; MUST match the served vLLM model id.")
    return asyncio.run(_amain(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())

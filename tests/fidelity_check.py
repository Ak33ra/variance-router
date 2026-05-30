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


async def _warmup(url: str, trace: list[dict], k: int = 5) -> None:
    """Fire a few zero-delay requests so vLLM's first-call compile/alloc cost
    doesn't land on the measured run."""
    warm = []
    for item in trace[:k]:
        w = dict(item)
        w["arrival_s"] = 0.0
        warm.append(w)
    if warm:
        await replay(url, warm)


def _ok_count(results: list[dict]) -> int:
    return sum(1 for r in results if r.get("ok"))


def _print_table(direct: list[dict], router: list[dict]) -> bool:
    d_ok = [r for r in direct if r.get("ok")]
    r_ok = [r for r in router if r.get("ok")]
    print(f"\nsuccessful requests   direct={len(d_ok)}/{len(direct)}   router={len(r_ok)}/{len(router)}")
    if not d_ok or not r_ok:
        print("  !! one side had no successful requests; cannot compare")
        return False

    print(f"\n{'metric':9s} {'pctl':5s} {'direct':>10s} {'router':>10s} {'delta':>10s} {'delta%':>8s}")
    print("-" * 56)
    verdict_ok = True
    for m in METRICS:
        d_vals = [r[m] for r in d_ok]
        r_vals = [r[m] for r in r_ok]
        for p in (50, 95, 99):
            d = _pct(d_vals, p)
            r = _pct(r_vals, p)
            delta = r - d
            pct = (delta / d * 100.0) if d and d == d and d != 0 else float("nan")
            print(f"{m:9s} P{p:<4d} {d:10.2f} {r:10.2f} {delta:10.2f} {pct:7.1f}%")
        print("-" * 56)

    # Heuristic gates on P50 (median is the stable signal under GPU noise).
    d_ttft = _pct([r["ttft_ms"] for r in d_ok], 50)
    r_ttft = _pct([r["ttft_ms"] for r in r_ok], 50)
    d_tpot = _pct([r["tpot_ms"] for r in d_ok], 50)
    r_tpot = _pct([r["tpot_ms"] for r in r_ok], 50)

    ttft_budget = max(8.0, 0.15 * d_ttft)
    if (r_ttft - d_ttft) > ttft_budget:
        verdict_ok = False
        print(f"  [FAIL] router TTFT P50 +{r_ttft - d_ttft:.2f}ms over direct "
              f"(budget {ttft_budget:.2f}ms) — router adding latency / buffering?")
    else:
        print(f"  [PASS] router TTFT P50 within budget (+{r_ttft - d_ttft:.2f}ms <= {ttft_budget:.2f}ms)")

    tpot_budget = max(2.0, 0.15 * d_tpot)
    if abs(r_tpot - d_tpot) > tpot_budget:
        verdict_ok = False
        print(f"  [FAIL] router TPOT P50 differs by {r_tpot - d_tpot:+.2f}ms "
              f"(budget ±{tpot_budget:.2f}ms) — per-token cadence should be untouched")
    else:
        print(f"  [PASS] router TPOT P50 within ±{tpot_budget:.2f}ms ({r_tpot - d_tpot:+.2f}ms)")

    return verdict_ok


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
        make_trace(path, args.n, args.rate, args.burst_cv, args.max_tokens, args.chat)
        args.trace = path
    if not args.trace:
        print("provide --trace or --make-trace", file=sys.stderr)
        return 2

    trace = load_trace(args.trace)
    print(f"replaying {len(trace)} requests; warming up...")
    await _warmup(args.direct, trace)

    print(f"A: direct  -> {args.direct}")
    direct = await replay(args.direct, trace)
    print(f"B: router  -> {args.router}")
    router = await replay(args.router, trace)

    ok = _print_table(direct, router)
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
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--rate", type=float, default=20.0)
    ap.add_argument("--burst-cv", type=float, default=2.0)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--chat", action="store_true")
    return asyncio.run(_amain(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())

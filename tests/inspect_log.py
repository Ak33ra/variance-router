#!/usr/bin/env python3
"""Offline analyzer for a router JSONL log.

The router deliberately emits only raw per-request records; this is the small
reference reducer that turns them into the signals the experiment cares about,
so you can eyeball a run without the full benchmarking repo:

  - per-backend dispatch counts + longest same-backend run (burst concentration),
  - induced PER-NODE arrival CV: CV of inter-dispatch gaps at each backend
    (the independent variable — burst should raise this vs. jsq/round_robin),
  - aggregate arrival CV at the router (inter-arrival gaps of the whole stream,
    held FIXED across policies — a sanity check that the trace was identical),
  - client-side-ish latency proxies from the timestamps (TTFT/e2e), and router
    overhead.

  python tests/inspect_log.py logs/router_log.jsonl

CV here is std/mean of the gap sequence (population std). Compare per-node CV
across policy runs on the SAME trace: if burst doesn't raise it, the hypothesis
is falsified — see README core principle #1.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict


def _cv(gaps: list[float]) -> float:
    if len(gaps) < 2:
        return float("nan")
    mean = sum(gaps) / len(gaps)
    if mean == 0:
        return float("nan")
    var = sum((g - mean) ** 2 for g in gaps) / len(gaps)
    return math.sqrt(var) / mean


def _gaps(times: list[float]) -> list[float]:
    t = sorted(times)
    return [b - a for a, b in zip(t, t[1:])]


def _pct(values, p):
    vals = sorted(v for v in values if v == v)
    if not vals:
        return float("nan")
    k = (len(vals) - 1) * (p / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    return vals[int(k)] if lo == hi else vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def _max_run(seq: list) -> int:
    best = cur = 0
    prev = object()
    for x in seq:
        cur = cur + 1 if x == prev else 1
        best = max(best, cur)
        prev = x
    return best


def main(path: str) -> int:
    recs = [json.loads(l) for l in open(path) if l.strip()]
    if not recs:
        print("no records")
        return 1

    policy = recs[-1].get("policy")
    print(f"records: {len(recs)}   policy: {json.dumps(policy)}")

    # Aggregate arrival CV at the router (should be ~constant across policies).
    arrivals = [r["arrival_time"]["monotonic"] for r in recs if r.get("arrival_time")]
    agg_cv = _cv(_gaps(arrivals))
    print(f"aggregate arrival CV (router ingress, held FIXED): {agg_cv:.3f}")

    # Per-backend.
    by_backend: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_backend[r["routed_backend"]].append(r)

    print(f"\n{'backend':22s} {'gpu':>3s} {'count':>6s} {'share':>6s} {'per-node CV':>12s}")
    print("-" * 56)
    for be in sorted(by_backend):
        rs = by_backend[be]
        disp = [r["dispatch_time"]["monotonic"] for r in rs if r.get("dispatch_time")]
        cv = _cv(_gaps(disp))
        gpu = rs[0].get("gpu_id")
        share = len(rs) / len(recs) * 100.0
        print(f"{be:22s} {str(gpu):>3s} {len(rs):6d} {share:5.1f}% {cv:12.3f}")

    seq = [r["routed_backend"] for r in recs]
    print(f"\nlongest same-backend run (burst concentration): {_max_run(seq)}")

    # Latency proxies from timestamps (wall clock, ms).
    ttft, e2e = [], []
    for r in recs:
        d = r.get("dispatch_time", {}).get("monotonic")
        f = (r.get("first_token_time") or {}).get("monotonic")
        c = (r.get("completion_time") or {}).get("monotonic")
        if d is not None and f is not None:
            ttft.append((f - d) * 1000.0)
        if d is not None and c is not None:
            e2e.append((c - d) * 1000.0)
    oh = [r["router_overhead_ms"] for r in recs if r.get("router_overhead_ms") is not None]
    errs = sum(1 for r in recs if r.get("error"))

    print("\nlatency proxies (dispatch->first_token / ->completion), ms")
    for name, vals in (("ttft", ttft), ("e2e", e2e)):
        if vals:
            print(f"  {name:5s} P50={_pct(vals,50):8.2f}  P95={_pct(vals,95):8.2f}  P99={_pct(vals,99):8.2f}")
    if oh:
        print(f"  router_overhead_ms  P50={_pct(oh,50):.3f}  P99={_pct(oh,99):.3f}  max={max(oh):.3f}")
    if errs:
        print(f"  errors: {errs}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python tests/inspect_log.py <router_log.jsonl>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))

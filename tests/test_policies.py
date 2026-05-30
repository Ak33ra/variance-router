#!/usr/bin/env python3
"""Pure unit tests for routing policies (no HTTP, no GPU).

Runnable with pytest *or* directly:  python tests/test_policies.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router.backends import BackendPool  # noqa: E402
from router.config import BackendConfig  # noqa: E402
from router.policies import RequestInfo, build_policy  # noqa: E402


def _pool(n=4):
    return BackendPool([BackendConfig(host="127.0.0.1", port=8001 + i, gpu_id=i) for i in range(n)])


def _req(t=0.0):
    return RequestInfo("rid", "/v1/completions", {}, t)


def test_round_robin_cycles():
    p = build_policy(_pool(3), "round_robin", {})
    seq = [p.route(_req()).index for _ in range(7)]
    assert seq == [0, 1, 2, 0, 1, 2, 0], seq


def test_jsq_picks_min_in_flight():
    pool = _pool(3)
    p = build_policy(pool, "jsq", {})
    pool[0].in_flight = 5
    pool[1].in_flight = 2
    pool[2].in_flight = 3
    assert p.route(_req()).index == 1
    # tie breaks on lowest index
    pool[1].in_flight = 3
    assert p.route(_req()).index == 1  # idx1 vs idx2 both 3 -> lower index


def test_burst_clumps_then_rotates():
    pool = _pool(3)
    p = build_policy(pool, "burst", {"burst_size": 3})
    seq = [p.route(_req()).index for _ in range(9)]
    assert seq == [0, 0, 0, 1, 1, 1, 2, 2, 2], seq


def test_burst_concentrates_load_vs_round_robin():
    # The whole point: bursting must concentrate consecutive arrivals per node.
    pool = _pool(4)
    burst = build_policy(pool, "burst", {"burst_size": 8})
    targets = [burst.route(_req()).index for _ in range(8)]
    assert len(set(targets)) == 1, "a full burst must land on a single node"


def test_burst_time_window_rotation():
    pool = _pool(2)
    p = build_policy(pool, "burst", {"burst_size": None, "active_window_ms": 100})
    # t=0..0.05s stay on node 0; cross 0.1s -> rotate to node 1
    assert p.route(_req(0.00)).index == 0
    assert p.route(_req(0.05)).index == 0
    assert p.route(_req(0.15)).index == 1
    assert p.route(_req(0.18)).index == 1


def test_burst_multi_active():
    pool = _pool(4)
    p = build_policy(pool, "burst", {"burst_size": 4, "num_active_nodes": 2})
    seq = [p.route(_req()).index for _ in range(8)]
    # 2 active nodes, round-robin within, 4 reqs then rotate by 2.
    assert seq == [0, 1, 0, 1, 2, 3, 2, 3], seq


def test_regime_aware_fills_band_then_drains():
    pool = _pool(3)
    p = build_policy(pool, "regime_aware", {"low_watermark": 2, "high_watermark": 5})
    # all empty -> promote a draining node (fullest below low; all 0 -> index 0)
    assert p.route(_req()).index == 0
    pool[0].in_flight = 3  # now in band [2,5)
    assert p.route(_req()).index == 0  # keep filling the in-band node
    pool[0].in_flight = 5  # saturated -> start filling another draining node
    assert p.route(_req()).index in (1, 2)


def test_unknown_policy_raises():
    try:
        build_policy(_pool(), "nope", {})
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown policy")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)

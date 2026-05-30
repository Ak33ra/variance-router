"""Pluggable routing policies.

A policy maps an incoming request (plus the live backend states) to exactly one
backend. The contract is ``route(request) -> BackendState`` with the live
``BackendPool`` injected at construction, so a policy can read every backend's
current in-flight count.

Routing decisions run synchronously inside the asyncio event loop and must not
block (no awaits, no I/O). This guarantees each ``route`` call is atomic with
respect to in-flight accounting and keeps router overhead negligible.

Adding a policy: subclass ``RoutingPolicy``, set a ``name``, implement
``route``, and register it in ``POLICIES`` at the bottom.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .backends import BackendPool, BackendState


class RequestInfo:
    """A lightweight, policy-facing view of an incoming request.

    Deliberately decoupled from the HTTP layer so policies are trivially
    unit-testable. ``body`` is the parsed JSON payload (or ``{}``); policies may
    inspect it (e.g. ``max_tokens``) but the workload is synthetic and prefix-free,
    so no policy here needs request content to decide.
    """

    __slots__ = ("request_id", "path", "body", "arrival_monotonic")

    def __init__(self, request_id: str, path: str, body: dict, arrival_monotonic: float) -> None:
        self.request_id = request_id
        self.path = path
        self.body = body
        self.arrival_monotonic = arrival_monotonic


class RoutingPolicy(ABC):
    name: str = "base"

    def __init__(self, pool: BackendPool, params: dict[str, Any]) -> None:
        self.pool = pool
        self.params = dict(params)

    @abstractmethod
    def route(self, request: RequestInfo) -> BackendState:
        """Return the backend this request should be dispatched to."""

    def describe(self) -> dict[str, Any]:
        """The policy identity logged with every request (name + effective params)."""
        return {"name": self.name, "params": self.effective_params()}

    def effective_params(self) -> dict[str, Any]:
        """Override to report the concrete params actually in effect."""
        return dict(self.params)


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #


class RoundRobinPolicy(RoutingPolicy):
    """Baseline: cycle through backends one request at a time."""

    name = "round_robin"

    def __init__(self, pool: BackendPool, params: dict[str, Any]) -> None:
        super().__init__(pool, params)
        self._next = 0

    def route(self, request: RequestInfo) -> BackendState:
        b = self.pool[self._next % len(self.pool)]
        self._next += 1
        return b


class JSQPolicy(RoutingPolicy):
    """Baseline: join-shortest-queue — route to the fewest in-flight requests.

    Ties break on backend index for determinism. This is the canonical
    load-spreading policy the burst intervention aims to beat; the README notes
    the production default may differ, so record which baseline each run uses
    (the policy name + params are logged per request).
    """

    name = "jsq"

    def route(self, request: RequestInfo) -> BackendState:
        return min(self.pool, key=lambda b: (b.in_flight, b.index))


# --------------------------------------------------------------------------- #
# Interventions
# --------------------------------------------------------------------------- #


class BurstPolicy(RoutingPolicy):
    """Concentrate arrivals onto an 'active' node (or small active set), then
    rotate, so non-active nodes drain to cheap pure-decode iterations.

    Params:
      burst_size (int, default 16):     requests sent to the active set before
                                        rotating. The primary knob.
      active_window_ms (float, opt):    max wall time a node stays active before
                                        rotating. Evaluated at request arrival
                                        (no background timer). Set alongside
                                        burst_size to rotate on whichever fires
                                        first; set burst_size: null to rotate on
                                        time only.
      num_active_nodes (int, default 1): how many adjacent nodes are active at
                                        once. The clump rotates by this many.
      within_active ("round_robin"|"jsq", default "round_robin"):
                                        how to pick among the active set when
                                        num_active_nodes > 1.

    Note: rotation is request-driven. If no requests arrive, the active node
    does not change — which is exactly the quiet/drain period we want.
    """

    name = "burst"

    def __init__(self, pool: BackendPool, params: dict[str, Any]) -> None:
        super().__init__(pool, params)
        bs = params.get("burst_size", 16)
        self.burst_size = None if bs is None else int(bs)
        aw = params.get("active_window_ms", None)
        self.active_window_ms = None if aw is None else float(aw)
        if self.burst_size is None and self.active_window_ms is None:
            raise ValueError("burst policy needs at least one of burst_size or active_window_ms")
        self.num_active = max(1, int(params.get("num_active_nodes", 1)))
        self.within_active = str(params.get("within_active", "round_robin"))
        if self.within_active not in ("round_robin", "jsq"):
            raise ValueError("within_active must be 'round_robin' or 'jsq'")

        self._active_start = 0
        self._count = 0  # requests sent in the current active window
        # Seeded lazily from the first request's arrival time so the window
        # clock matches the timestamps route() actually receives.
        self._window_start: float | None = None
        self._rr = 0  # round-robin cursor within the active set

    def effective_params(self) -> dict[str, Any]:
        return {
            "burst_size": self.burst_size,
            "active_window_ms": self.active_window_ms,
            "num_active_nodes": self.num_active,
            "within_active": self.within_active,
        }

    def _should_rotate(self, now: float) -> bool:
        by_count = self.burst_size is not None and self._count >= self.burst_size
        by_time = (
            self.active_window_ms is not None
            and self._window_start is not None
            and (now - self._window_start) * 1000.0 >= self.active_window_ms
        )
        return by_count or by_time

    def route(self, request: RequestInfo) -> BackendState:
        n = len(self.pool)
        now = request.arrival_monotonic
        if self._window_start is None:
            self._window_start = now
        if self._count > 0 and self._should_rotate(now):
            self._active_start = (self._active_start + self.num_active) % n
            self._count = 0
            self._window_start = now
            self._rr = 0

        k = min(self.num_active, n)
        active = [self.pool[(self._active_start + i) % n] for i in range(k)]
        if k == 1:
            chosen = active[0]
        elif self.within_active == "jsq":
            chosen = min(active, key=lambda b: (b.in_flight, b.index))
        else:
            chosen = active[self._rr % k]
            self._rr += 1

        self._count += 1
        return chosen


class RegimeAwarePolicy(RoutingPolicy):
    """Stretch / second intervention: route to keep each node either draining
    (in-flight below ``low_watermark`` -> cheap memory-bound decode) or filled
    into the efficient compute-bound band ``[low_watermark, high_watermark)``,
    while avoiding leaving nodes stuck in the expensive transition above the band.

    Heuristic (framed on the measured roofline transition, not variance per se):
      1. If any node is already in the efficient band, keep filling the fullest
         such node (concentrate to sustain a compute-bound batch).
      2. Else promote the fullest draining node (in-flight < low) into the band.
      3. Else (all nodes saturated at/above high) fall back to least-loaded.

    Params: low_watermark (int, default 4), high_watermark (int, default 32).
    These are batch-size watermarks to be calibrated to the measured roofline.
    """

    name = "regime_aware"

    def __init__(self, pool: BackendPool, params: dict[str, Any]) -> None:
        super().__init__(pool, params)
        self.low = int(params.get("low_watermark", 4))
        self.high = int(params.get("high_watermark", 32))
        if not (0 <= self.low < self.high):
            raise ValueError("regime_aware requires 0 <= low_watermark < high_watermark")

    def effective_params(self) -> dict[str, Any]:
        return {"low_watermark": self.low, "high_watermark": self.high}

    def route(self, request: RequestInfo) -> BackendState:
        in_band = [b for b in self.pool if self.low <= b.in_flight < self.high]
        if in_band:
            # Concentrate onto the fullest in-band node; tie-break low index.
            return max(in_band, key=lambda b: (b.in_flight, -b.index))
        below = [b for b in self.pool if b.in_flight < self.low]
        if below:
            return max(below, key=lambda b: (b.in_flight, -b.index))
        return min(self.pool, key=lambda b: (b.in_flight, b.index))


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

POLICIES: dict[str, type[RoutingPolicy]] = {
    cls.name: cls
    for cls in (RoundRobinPolicy, JSQPolicy, BurstPolicy, RegimeAwarePolicy)
}


def build_policy(pool: BackendPool, name: str, params: dict[str, Any]) -> RoutingPolicy:
    if name not in POLICIES:
        raise ValueError(f"Unknown policy {name!r}. Available: {sorted(POLICIES)}")
    return POLICIES[name](pool, params)

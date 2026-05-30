"""Backend instances and the pool that tracks their live load.

The pool owns per-backend in-flight counts (requests dispatched but not yet
completed). In-flight is the signal JSQ and the regime-aware policy route on,
and it is logged per request so per-node load is reconstructable offline.

All mutation of in-flight happens from inside the asyncio event loop (the
request handler), so plain integer ops are atomic — no locks needed.
"""

from __future__ import annotations

import httpx

from .config import BackendConfig


class BackendState:
    """Mutable runtime state for one backend."""

    def __init__(self, config: BackendConfig, index: int) -> None:
        self.config = config
        self.index = index
        self.in_flight = 0
        self.dispatched_total = 0

    @property
    def base_url(self) -> str:
        return self.config.base_url

    @property
    def label(self) -> str:
        return self.config.label

    @property
    def gpu_id(self) -> int | None:
        return self.config.gpu_id

    def acquire(self) -> None:
        """Mark a request dispatched to this backend."""
        self.in_flight += 1
        self.dispatched_total += 1

    def release(self) -> None:
        """Mark a dispatched request completed (or failed)."""
        self.in_flight -= 1

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<Backend {self.label} gpu={self.gpu_id} in_flight={self.in_flight}>"


class BackendPool:
    """Ordered collection of backends."""

    def __init__(self, configs: list[BackendConfig]) -> None:
        if not configs:
            raise ValueError("Backend list is empty; at least one backend is required.")
        self.backends = [BackendState(c, i) for i, c in enumerate(configs)]

    def __iter__(self):
        return iter(self.backends)

    def __len__(self) -> int:
        return len(self.backends)

    def __getitem__(self, i: int) -> BackendState:
        return self.backends[i]

    async def health_check(self, client: httpx.AsyncClient, timeout: float) -> None:
        """Probe every backend; raise loudly if any is unreachable.

        Tries ``/health`` (vLLM's liveness endpoint) and falls back to
        ``/v1/models`` so the router also works against servers that only
        expose the OpenAI surface.
        """
        unreachable: list[tuple[str, str]] = []
        for b in self.backends:
            err = await _probe(client, b.base_url, timeout)
            if err is not None:
                unreachable.append((b.label, err))
        if unreachable:
            detail = "; ".join(f"{label} ({err})" for label, err in unreachable)
            raise RuntimeError(f"Backend health check failed for: {detail}")


async def _probe(client: httpx.AsyncClient, base_url: str, timeout: float) -> str | None:
    """Return None if reachable, else a short error string."""
    last_err = "unknown error"
    for path in ("/health", "/v1/models"):
        try:
            r = await client.get(f"{base_url}{path}", timeout=timeout)
            if r.status_code < 400:
                return None
            last_err = f"{path} -> HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001 - report any connection error verbatim
            last_err = f"{path} -> {type(e).__name__}: {e}"
    return last_err

"""Per-request JSONL logger.

One structured record per request, appended as a single line. Records carry the
raw signals (routed_backend, monotonic dispatch_time, in-flight, TTFT/TPOT-able
timestamps) the benchmarking repo joins against backend iteration logs and from
which it computes induced per-node CV and latency percentiles offline.

Writes go through a lock and the stream is line-buffered so each record is
durable and intact even under abnormal shutdown.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class RequestLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.parent and not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        # buffering=1 -> line-buffered text mode (flush on every newline).
        self._fh = self.path.open("a", buffering=1, encoding="utf-8")
        self._lock = threading.Lock()

    def log(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, separators=(",", ":"))
        with self._lock:
            self._fh.write(line + "\n")

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.flush()
                self._fh.close()

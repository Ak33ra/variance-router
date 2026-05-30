#!/usr/bin/env python3
"""End-to-end smoke test (no GPU): mock backends + router + replay client.

Launches 2 mock backends and the router (as subprocesses), generates a small
bursty trace, replays it, then validates:
  - all requests succeed,
  - the router JSONL has one record per request,
  - every record is end-to-end traceable (request_id propagated) and has a
    first_token_time (streaming TTFT was captured, i.e. not buffered),
  - router overhead is small,
  - requests were spread across both backends.

  python tests/run_integration.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def _wait_http(url: str, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=1) as r:
                if r.status < 400:
                    return True
        except Exception:  # noqa: BLE001
            time.sleep(0.2)
    return False


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="vrouter-it-"))
    log_path = tmp / "router_log.jsonl"
    trace_path = tmp / "trace.jsonl"
    config_path = tmp / "config.yaml"
    procs: list[subprocess.Popen] = []
    env = dict(os.environ, PYTHONPATH=str(ROOT))

    try:
        # 1. mock backends
        for port, gpu in ((8101, 0), (8102, 1)):
            procs.append(subprocess.Popen(
                [PY, str(ROOT / "tests" / "mock_backend.py"), "--port", str(port),
                 "--gpu-id", str(gpu), "--prefill-ms", "15", "--token-delay-ms", "3"],
                env=env))
        for port in (8101, 8102):
            assert _wait_http(f"http://127.0.0.1:{port}/health"), f"mock {port} did not start"

        # 2. config + router (burst policy exercises concentration + rotation)
        config_path.write_text(
            "backends:\n"
            "  - {host: 127.0.0.1, port: 8101, gpu_id: 0}\n"
            "  - {host: 127.0.0.1, port: 8102, gpu_id: 1}\n"
            "policy: {name: burst, params: {burst_size: 5}}\n"
            f"log_path: {log_path}\n"
            "host: 127.0.0.1\n"
            "port: 8100\n"
        )
        procs.append(subprocess.Popen(
            [PY, str(ROOT / "router.py"), "--config", str(config_path), "--log-level", "warning"],
            env=env))
        assert _wait_http("http://127.0.0.1:8100/health"), "router did not start"

        # 3. trace + replay
        subprocess.check_call(
            [PY, str(ROOT / "tests" / "replay_client.py"), "--make-trace", str(trace_path),
             "--n", "40", "--rate", "80", "--max-tokens", "12"], env=env)
        subprocess.check_call(
            [PY, str(ROOT / "tests" / "replay_client.py"), "--url", "http://127.0.0.1:8100",
             "--trace", str(trace_path)], env=env)

        # 4. validate the router log
        time.sleep(0.5)
        records = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
        ok = _validate(records, expected=40)
        return 0 if ok else 1
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except Exception:  # noqa: BLE001
                p.kill()


def _validate(records: list[dict], expected: int) -> bool:
    checks: list[tuple[str, bool, str]] = []

    checks.append(("record count == requests", len(records) == expected,
                   f"got {len(records)} expected {expected}"))

    errs = [r for r in records if r.get("error")]
    checks.append(("no errored requests", not errs, f"{len(errs)} errors: {errs[:2]}"))

    traceable = all(r.get("request_id") and r["request_id"].startswith("router-") for r in records)
    checks.append(("request_ids present & propagated", traceable, "missing/empty request_id"))

    have_ttft = sum(1 for r in records if r.get("first_token_time"))
    checks.append(("first_token_time captured (streaming, not buffered)",
                   have_ttft == len(records), f"{have_ttft}/{len(records)} had TTFT"))

    overheads = [r["router_overhead_ms"] for r in records if r.get("router_overhead_ms") is not None]
    max_oh = max(overheads) if overheads else float("inf")
    checks.append(("router overhead < 50ms", max_oh < 50.0, f"max overhead {max_oh:.2f}ms"))

    backends = {r["routed_backend"] for r in records}
    checks.append(("both backends used", len(backends) >= 2, f"backends seen: {backends}"))

    # burst_size=5 means runs of consecutive same-backend dispatches.
    max_run = _max_run([r["routed_backend"] for r in records])
    checks.append(("burst concentration observed (run >= 3)", max_run >= 3,
                   f"longest same-backend run = {max_run}"))

    tokens = [r.get("output_tokens") for r in records]
    checks.append(("output_tokens recorded from usage", all(t for t in tokens),
                   f"some missing: {[t for t in tokens if not t][:3]}"))

    print()
    all_ok = True
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + ("" if passed else f"  ({detail})"))
        all_ok = all_ok and passed
    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    return all_ok


def _max_run(seq: list) -> int:
    best = cur = 0
    prev = object()
    for x in seq:
        cur = cur + 1 if x == prev else 1
        best = max(best, cur)
        prev = x
    return best


if __name__ == "__main__":
    sys.exit(main())

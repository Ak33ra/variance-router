#!/usr/bin/env python3
"""CLI entrypoint:  python router.py --config config.yaml

--host/--port override the config for ad-hoc runs.
"""

from __future__ import annotations

import argparse

import uvicorn

from router.config import load_config
from router.proxy import create_app


def main() -> None:
    ap = argparse.ArgumentParser(description="Variance-shaping router for multi-instance vLLM.")
    ap.add_argument("--config", required=True, help="Path to YAML/JSON config file.")
    ap.add_argument("--host", default=None, help="Override bind host.")
    ap.add_argument("--port", type=int, default=None, help="Override bind port.")
    ap.add_argument("--log-level", default="info", help="uvicorn log level.")
    args = ap.parse_args()

    config = load_config(args.config)
    host = args.host or config.host
    port = args.port or config.port

    app = create_app(config)
    # Single worker / single event loop: routing state and in-flight counts must
    # live in one process for correctness. Do not scale with --workers.
    uvicorn.run(app, host=host, port=port, log_level=args.log_level)


if __name__ == "__main__":
    main()

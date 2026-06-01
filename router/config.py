"""Typed configuration loading.

A single YAML (or JSON) file describes the backend list, the active routing
policy and its parameters, the output log path, and router-level timeouts.
Everything the router needs to run comes from here so that an experiment is
fully reproducible from one file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class BackendConfig(BaseModel):
    """One independent vLLM instance (one per GPU)."""

    host: str
    port: int
    gpu_id: int | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def label(self) -> str:
        return f"{self.host}:{self.port}"


class PolicyConfig(BaseModel):
    """Active policy plus its free-form parameters.

    Parameters are intentionally an open dict: each policy validates and
    documents the keys it understands (see router/policies.py). This keeps the
    config schema stable as new policies are added.
    """

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class TimeoutConfig(BaseModel):
    """httpx timeouts. ``read_s = null`` means no read timeout, which is the
    right default for long token generations (we must not abort a slow decode)."""

    connect_s: float = 5.0
    read_s: float | None = None
    write_s: float = 10.0
    pool_s: float = 5.0
    health_check_s: float = 5.0


class RouterConfig(BaseModel):
    backends: list[BackendConfig]
    policy: PolicyConfig
    log_path: str = "router_log.jsonl"
    host: str = "127.0.0.1"  # loopback: router is co-located with backends + client
    port: int = 8000
    timeouts: TimeoutConfig = Field(default_factory=TimeoutConfig)
    health_check_on_startup: bool = True


def load_config(path: str | Path) -> RouterConfig:
    """Load and validate a router config from a YAML or JSON file."""
    p = Path(path)
    text = p.read_text()
    if p.suffix == ".json":
        data = json.loads(text)
    else:
        # yaml.safe_load parses JSON too, so this also covers unknown suffixes.
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config {p} did not parse to a mapping (got {type(data).__name__}).")
    return RouterConfig.model_validate(data)

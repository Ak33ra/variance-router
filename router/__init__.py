"""Variance-shaping HTTP router for multi-instance vLLM serving.

A dependency-light async proxy that sits in front of N independent vLLM
instances and controls the per-node arrival-process variance via swappable
routing policies. See README.md for the design rationale.
"""

__all__ = ["config", "backends", "policies", "reqlog", "proxy"]

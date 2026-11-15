"""Pilot 2-day deterministic FRP operator (experimental, isolated).

Reads the production DuckDB read-only; writes only to a dedicated
output directory. Nothing here touches production analytics.
"""

METHOD_VERSION = "det-frp-oop-v1"

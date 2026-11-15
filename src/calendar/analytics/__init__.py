"""Deterministic analytical engine (Phases 2A–2E, 4B–4D).

Pure Python + DuckDB over Phase 1 normalized Parquet. No harmonization of
meaning, no invented data: every output carries versions, params, and the
execution id that produced it.
"""

from . import (anomalies, baselines, clustering, confidence, discovery,  # noqa: F401
               index, matching, metrics, runs)

"""FireAgent Calendar package.

Phase 1: data infrastructure & acquisition (FIRMS NRT + archive staging,
local-first storage, normalization foundations).

Borrowed patterns (not code) from the existing tracks:
- ingestor.py chunking/dedup semantics for the FIRMS Area API
- email_alerts.py transition-guard + fail-open ideas (Phase 5, not here)
- sar_summary.py deterministic-checks-first discipline (Phase 5, not here)

Nothing in this package imports FireAgent SAR science modules.
"""

__version__ = "0.1.0-phase1"

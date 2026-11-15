"""Acquisition layer: FIRMS NRT/SP area API + open per-country archives.

All network I/O goes through :mod:`transport` (bounded retries, timeouts,
key redaction). Raw bytes are always preserved; parsing happens later.
"""

from . import archive, nrt, sensors, transport  # noqa: F401

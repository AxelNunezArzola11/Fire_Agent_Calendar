"""NRT + Standard (SP) FIRMS Area API client.

Semantics inherited (verified, not assumed) from ``ingestor.py``:
- at most 5 days per request; longer ranges use dated ``…/{days}/{YYYY-MM-DD}``
- the date parameter is a START date: ``…/5/2026-08-01`` -> ``[08-01 … 08-05]``
- raw bytes are preserved per chunk; parsing/dedup happen downstream.

No continuous polling here (Phase 1 activates no scheduler).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import date, timedelta

from . import sensors, transport
from .. import settings

logger = logging.getLogger(__name__)

VALID_SOURCES = sensors.NRT_SOURCES + sensors.SP_SOURCES
MAX_DAYS = 5


@dataclass
class AreaRequest:
    source: str  # e.g. 'VIIRS_SNPP_NRT' | 'MODIS_SP'
    bbox: str  # 'W,S,E,N'
    days: int  # 1..5 for a single request
    start_date: date | None = None  # None -> trailing window ending today


@dataclass
class ChunkResult:
    request: AreaRequest
    content: bytes
    sha256: str
    status: str  # 'ok' | 'empty' (valid CSV, zero rows) — HTTP errors raise
    attempts: int
    elapsed_s: float
    fetched_at: str  # ISO-8601 UTC


def _validate_bbox(bbox: str) -> tuple[float, float, float, float]:
    try:
        w, s, e, n = (float(v) for v in bbox.split(","))
    except ValueError:
        raise ValueError(f"bbox must be 'W,S,E,N', got {bbox!r}") from None
    if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
        raise ValueError(f"bbox out of range or inverted: {bbox!r}")
    return w, s, e, n


def build_area_url(base_url: str, key: str, req: AreaRequest) -> str:
    """Build the Area API URL. Raises ValueError/AuthError-config on bad input."""
    if req.source not in VALID_SOURCES:
        raise ValueError(f"unknown source {req.source!r}; expected one of {VALID_SOURCES}")
    if not 1 <= req.days <= MAX_DAYS:
        raise ValueError(f"days must be 1..{MAX_DAYS}, got {req.days}")
    if not key:
        raise transport.AuthError("FIRMS_MAP_KEY is not configured")
    _validate_bbox(req.bbox)
    url = f"{base_url}/{key}/{req.source}/{req.bbox}/{req.days}"
    if req.start_date is not None:
        url += f"/{req.start_date.strftime('%Y-%m-%d')}"
    return url


def request_fingerprint(req: AreaRequest) -> str:
    """Stable idempotency key for identical logical requests."""
    start = req.start_date.isoformat() if req.start_date else "trailing"
    canonical = f"{req.source}|{req.bbox}|{req.days}|{start}"
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]


def plan_chunks(source: str, bbox: str, days: int, end: date) -> list[AreaRequest]:
    """Split a trailing *days*-long range ending *end* into ≤5-day dated chunks.

    Newest-first, non-overlapping (same verified layout as ingestor.py):
    chunk_start = end - (fetched + window - 1).
    """
    if days < 1:
        raise ValueError(f"days must be >= 1, got {days}")
    chunks: list[AreaRequest] = []
    fetched = 0
    while fetched < days:
        window = min(days - fetched, MAX_DAYS)
        start = end - timedelta(days=fetched + window - 1)
        chunks.append(AreaRequest(source, bbox, window, start))
        fetched += window
    return chunks


def fetch_window(req: AreaRequest, *, key: str | None = None) -> ChunkResult:
    """Fetch one ≤5-day window; raw bytes preserved with SHA-256."""
    from datetime import datetime, timezone

    map_key = key if key is not None else settings.FIRMS_MAP_KEY
    url = build_area_url(settings.FIRMS_AREA_BASE_URL, map_key, req)
    res = transport.fetch_bytes(
        url, timeout_s=settings.ACQUIRE_TIMEOUT_S,
        max_retries=settings.ACQUIRE_MAX_RETRIES,
        backoff_s=settings.ACQUIRE_BACKOFF_S,
        secrets=(map_key,), label=f"nrt:{req.source}",
    )
    digest = hashlib.sha256(res.content).hexdigest()
    lines = res.content.decode("utf-8", errors="replace").strip().splitlines()
    status = "empty" if len(lines) <= 1 else "ok"
    return ChunkResult(req, res.content, digest, status, res.attempts, res.elapsed_s,
                       datetime.now(timezone.utc).isoformat())


def fetch_range(source: str, bbox: str, days: int, end: date,
                *, key: str | None = None) -> list[ChunkResult]:
    """Fetch a trailing range as dated chunks (each independently hashed)."""
    return [fetch_window(req, key=key) for req in plan_chunks(source, bbox, days, end)]

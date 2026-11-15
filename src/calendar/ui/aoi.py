"""Saved Areas of Interest (Phase 3D): local JSON persistence, no accounts.

Each AOI is a named WGS84 bounding box (custom) or a reference to a
versioned preset region. Polygon AOIs are out of MVP scope (documented in
the 3D report). All geometry is validated loudly; nothing is silently fixed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..processing import regions


class AoIError(ValueError):
    """Raised for invalid AOI definitions or unreadable AOI storage."""


NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_UA = "FireAgentCalendar/1.0 (research demo)"
MAX_RADIUS_KM = 2000.0
MIN_RADIUS_KM = 0.5


def validate_point(latitude, longitude) -> tuple[float, float]:
    """Validated (lat, lon) in WGS84 ranges, else AoIError."""
    try:
        lat, lon = float(latitude), float(longitude)
    except (TypeError, ValueError):
        raise AoIError(f"coordinates must be numbers; got "
                       f"{latitude!r}/{longitude!r}")
    if not -90.0 <= lat <= 90.0:
        raise AoIError(f"latitude {lat} outside [-90, 90]")
    if not -180.0 <= lon <= 180.0:
        raise AoIError(f"longitude {lon} outside [-180, 180]")
    return lat, lon


def radius_bbox(latitude: float, longitude: float,
                radius_km: float) -> tuple[float, float, float, float]:
    """(W, S, E, N) bounding box around a point (equirectangular math).

    Radius clamped to [0.5, 2000] km; output clamped to valid ranges
    (no antimeridian wrap: boxes crossing ±180 stop at the edge and
    the caller is told via the returned bounds).
    """
    lat, lon = validate_point(latitude, longitude)
    try:
        radius = float(radius_km)
    except (TypeError, ValueError):
        raise AoIError(f"radius must be a number; got {radius_km!r}")
    if not MIN_RADIUS_KM <= radius <= MAX_RADIUS_KM:
        raise AoIError(f"radius {radius} km outside "
                       f"[{MIN_RADIUS_KM}, {MAX_RADIUS_KM}]")
    import math

    lat_deg = radius / 111.32
    lon_deg = radius / (111.32 * max(0.2, math.cos(math.radians(lat))))
    return validate_bbox((max(-180.0, lon - lon_deg),
                          max(-90.0, lat - lat_deg),
                          min(180.0, lon + lon_deg),
                          min(90.0, lat + lat_deg)))


def geocode_nominatim(place: str, *, timeout_s: float = 15.0,
                      transport=None) -> dict:
    """One Nominatim lookup (raises AoIError when unusable).

    Returns {latitude, longitude, label}. `transport` injects
    requests.get for tests. Single query per call (usage-policy
    friendly); failures report unavailability, never raise raw.
    """
    import requests

    query = (place or "").strip()
    if not query:
        raise AoIError("place name is empty")
    getter = transport or requests.get
    try:
        response = getter(
            NOMINATIM_URL,
            params={"q": query, "format": "jsonv2", "limit": 1},
            headers={"User-Agent": NOMINATIM_UA,
                     "Accept": "application/json"},
            timeout=timeout_s)
        response.raise_for_status()
        hits = response.json()
    except Exception as exc:  # noqa: BLE001 - reported as unavailable
        raise AoIError(f"geocoding unavailable: {type(exc).__name__}")
    if not hits:
        raise AoIError(f"no geocoding result for {query!r}")
    try:
        lat, lon = validate_point(hits[0]["lat"], hits[0]["lon"])
    except AoIError:
        raise AoIError(f"geocoder returned out-of-range coordinates "
                       f"for {query!r}")
    return {"latitude": lat, "longitude": lon,
            "label": str(hits[0].get("display_name", query))[:160]}


def validate_bbox(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Validate (W, S, E, N) in EPSG:4326; returns floats or raises AoIError."""
    try:
        w, s, e, n = (float(v) for v in bbox)
    except (TypeError, ValueError):
        raise AoIError(f"bbox must be four numbers (W, S, E, N); got {bbox!r}")
    for name, value, lo, hi in (("W", w, -180, 180), ("E", e, -180, 180),
                                ("S", s, -90, 90), ("N", n, -90, 90)):
        if not lo <= value <= hi:
            raise AoIError(f"bbox {name}={value} outside [{lo}, {hi}]")
    if w >= e:
        raise AoIError(f"bbox W ({w}) must be < E ({e})")
    if s >= n:
        raise AoIError(f"bbox S ({s}) must be < N ({n})")
    return (w, s, e, n)


def load_aois(path: str | Path) -> dict[str, dict]:
    """Saved AOIs by name ({} when the file does not exist yet)."""
    target = Path(path)
    if not target.is_file():
        return {}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AoIError(f"AOI file {target} is not valid JSON: {exc}")
    if not isinstance(data, dict):
        raise AoIError(f"AOI file {target} must hold an object keyed by name")
    return data


def save_aoi(path: str | Path, name: str,
             bbox: tuple[float, float, float, float]) -> dict[str, dict]:
    """Add a named custom bbox (duplicate names rejected; returns all AOIs)."""
    clean = str(name).strip()
    if not clean:
        raise AoIError("AOI name must not be empty")
    w, s, e, n = validate_bbox(bbox)
    target = Path(path)
    aois = load_aois(target)
    if clean in aois:
        raise AoIError(f"AOI {clean!r} already exists (delete it first)")
    aois[clean] = {"bbox": [w, s, e, n], "crs": "EPSG:4326",
                   "created_utc": datetime.now(timezone.utc).isoformat()}
    target.write_text(json.dumps(aois, indent=2), encoding="utf-8")
    return aois


def delete_aoi(path: str | Path, name: str) -> dict[str, dict]:
    """Remove a saved AOI (unknown names raise; returns remaining AOIs)."""
    target = Path(path)
    aois = load_aois(target)
    if name not in aois:
        raise AoIError(f"unknown AOI {name!r}")
    del aois[name]
    target.write_text(json.dumps(aois, indent=2), encoding="utf-8")
    return aois


def resolve_bbox(*, mode: str, region_id: str | None = None,
                 custom: tuple[float, float, float, float] | None = None,
                 viewport: tuple[float, float, float, float] | None = None,
                 saved: dict | None = None) -> tuple[float, float, float, float] | None:
    """Resolve the active AOI mode to a query bbox (None = full pilot)."""
    if mode == "full":
        return None
    if mode == "region":
        if region_id not in regions.REGIONS:
            raise AoIError(f"unknown region {region_id!r}")
        return regions.REGIONS[region_id].bbox
    if mode == "custom":
        if custom is None:
            raise AoIError("custom AOI selected but no bbox provided")
        return validate_bbox(custom)
    if mode == "viewport":
        if viewport is None:
            raise AoIError("viewport AOI selected but no map bounds captured")
        return validate_bbox(viewport)
    if mode == "saved":
        if not saved or "bbox" not in saved:
            raise AoIError("saved AOI selected but none is loaded")
        return validate_bbox(tuple(saved["bbox"]))
    raise AoIError(f"unknown AOI mode {mode!r}")

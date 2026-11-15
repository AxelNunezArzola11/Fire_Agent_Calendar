"""FIRMS-to-granule spatiotemporal matching (granule-level only).

Matches observations to candidate granules by acquisition time and
granule footprint. Granule-level candidacy is NOT pixel identity:
exact pixel correspondence needs pixel geolocation arrays, which
are handled (when readable) by downstream pixel work, never here.
Every match carries distance, temporal offset, and a confidence
label that says what was actually compared.
"""

from __future__ import annotations

import math


def _haversine_km(lat1: float, lon1: float, lat2: float,
                  lon2: float) -> float:
    radius = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    step_lat = p2 - p1
    step_lon = math.radians(lon2 - lon1)
    root = math.sin(step_lat / 2.0) ** 2 + math.cos(p1) * math.cos(p2) \
        * math.sin(step_lon / 2.0) ** 2
    return 2.0 * radius * math.asin(math.sqrt(root))


def match_observation(observation: dict, granules: list[dict], *,
                      max_minutes: float = 15.0,
                      max_km: float = 25.0) -> dict:
    """Best granule candidate for one observation (or explicit miss).

    observation: {latitude, longitude, acquisition_utc (ISO)}.
    granules: [{granule_ur, time_start, time_end, bbox [w,s,e,n]}].
    Returns {status, granule_ur, distance_km, offset_minutes,
    confidence, reason}; status is matched|ambiguous|unmatched.
    """
    from datetime import datetime

    def _parse(value):
        try:
            text = str(value or "").replace("Z", "+00:00")
            return datetime.fromisoformat(text)
        except ValueError:
            return None

    obs_time = _parse(observation.get("acquisition_utc"))
    if obs_time is None:
        return {"status": "unmatched", "granule_ur": None,
                "distance_km": None, "offset_minutes": None,
                "confidence": "none",
                "reason": "observation lacks a parseable timestamp"}
    try:
        lat = float(observation["latitude"])
        lon = float(observation["longitude"])
    except (KeyError, TypeError, ValueError):
        return {"status": "unmatched", "granule_ur": None,
                "distance_km": None, "offset_minutes": None,
                "confidence": "none",
                "reason": "observation lacks numeric coordinates"}
    scored = []
    for granule in granules:
        start = _parse(granule.get("time_start"))
        end = _parse(granule.get("time_end"))
        bbox = granule.get("bbox") or []
        if start is None or end is None or len(bbox) != 4:
            continue
        w, s, e, n = (float(v) for v in bbox)
        inside = w <= lon <= e and s <= lat <= n
        center_lat, center_lon = (s + n) / 2.0, (w + e) / 2.0
        distance = 0.0 if inside else _haversine_km(
            lat, lon, center_lat, center_lon)
        mid = start + (end - start) / 2
        offset = abs((obs_time - mid).total_seconds()) / 60.0
        if distance <= max_km and offset <= max_minutes:
            scored.append((offset + distance / 100.0, granule["granule_ur"],
                           distance, offset))
    if not scored:
        return {"status": "unmatched", "granule_ur": None,
                "distance_km": None, "offset_minutes": None,
                "confidence": "none",
                "reason": "no granule within tolerance"}
    scored.sort()
    if len(scored) > 1 and abs(scored[0][0] - scored[1][0]) < 1e-9:
        return {"status": "ambiguous",
                "granule_ur": scored[0][1],
                "distance_km": round(scored[0][2], 3),
                "offset_minutes": round(scored[0][3], 2),
                "confidence": "granule-level candidate (tie)",
                "reason": "two granules score identically"}
    _, best, distance, offset = scored[0]
    return {"status": "matched", "granule_ur": best,
            "distance_km": round(distance, 3),
            "offset_minutes": round(offset, 2),
            "confidence": "granule-level candidate (not pixel identity)",
            "reason": "closest granule footprint + time window"}

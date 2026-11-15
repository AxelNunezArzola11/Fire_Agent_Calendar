"""NASA CMR discovery: collections + granule inventory (no auth).

Pure query builders plus thin HTTP fetchers with injectable
transports (tests stub them; production uses requests). CMR metadata
drives the storage budget before any byte is downloaded.
"""

from __future__ import annotations

CMR_ROOT = "https://cmr.earthdata.nasa.gov/search"


def collection_query(short_name: str, version: str) -> dict:
    """CMR collection-search params (concept-id resolution)."""
    return {"short_name": short_name, "version": version}


def granule_query(concept_id: str, *, bbox, date_start: str,
                  date_end: str, page_size: int = 50,
                  page_num: int = 1) -> dict:
    """CMR granule-search params (bbox + closed date range)."""
    w, s, e, n = (float(v) for v in bbox)
    if not (w < e and s < n):
        raise ValueError(f"invalid bbox {bbox!r}")
    return {"collection_concept_id": concept_id,
            "bounding_box": f"{w},{s},{e},{n}",
            "temporal": f"{date_start}T00:00:00Z,{date_end}T23:59:59Z",
            "page_size": int(page_size), "page_num": int(page_num),
            "sort_key[]": "start_date"}


def _fetch_json(url: str, params: dict, *, timeout_s: float,
                transport) -> dict:
    import requests

    getter = transport or requests.get
    response = getter(url, params=params,
                      headers={"Accept": "application/json"},
                      timeout=timeout_s)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("CMR response is not a JSON object")
    return data


def resolve_collection(short_name: str, version: str, *,
                       timeout_s: float = 30.0,
                       transport=None) -> dict:
    """Collection concept id + title (raises _CMRError on failure)."""
    from . import products as products_mod

    products_mod.get(short_name)  # validates against the catalog
    try:
        data = _fetch_json(f"{CMR_ROOT}/collections.json",
                           collection_query(short_name, version),
                           timeout_s=timeout_s, transport=transport)
        entries = (data.get("feed") or {}).get("entry") or []
        if not entries:
            raise _CMRError(f"no CMR collection for "
                            f"{short_name} v{version}")
        first = entries[0]
        return {"concept_id": first["id"], "title": first.get("title"),
                "short_name": short_name, "version": version}
    except _CMRError:
        raise
    except Exception as exc:  # noqa: BLE001 - converted to reason
        raise _CMRError(f"collection lookup failed: "
                        f"{type(exc).__name__}: {exc}")


def _granule_size_mb(entry: dict) -> float | None:
    try:
        return float(entry.get("granule_size"))
    except (TypeError, ValueError):
        return None


def _entry_bbox(entry: dict):
    """Bounding [w,s,e,n] from boxes or polygon rings (None if absent).

    CMR polygons list lat/lon pairs; boxes list w s e n quads.
    """
    for quad in entry.get("boxes") or []:
        try:
            w, s, e, n = (float(v) for v in str(quad).split())
            return [w, s, e, n]
        except (ValueError, TypeError):
            continue
    rings = entry.get("polygons") or []
    coords = []
    for ring in rings:
        ring = ring[0] if isinstance(ring, list) and ring else ring
        try:
            nums = [float(v) for v in str(ring).split()]
        except (ValueError, TypeError):
            continue
        lats = nums[0::2]
        lons = nums[1::2]
        if lats and lons:
            coords.append((min(lons), min(lats), max(lons),
                           max(lats)))
    if not coords:
        return None
    return [min(c[0] for c in coords), min(c[1] for c in coords),
            max(c[2] for c in coords), max(c[3] for c in coords)]


def search_granules(concept_id: str, *, bbox, date_start: str,
                    date_end: str, page_size: int = 50, page_num: int = 1,
                    timeout_s: float = 30.0,
                    transport=None) -> dict:
    """One granule inventory page (metadata only, no downloads).

    Returns {granules, hits}. Each granule: {granule_ur,
    producer_granule_id, time_start/end, size_mb (or None), data_urls,
    checksum (value/type or None)}.
    """
    try:
        data = _fetch_json(
            f"{CMR_ROOT}/granules.json",
            granule_query(concept_id, bbox=bbox, date_start=date_start,
                          date_end=date_end, page_size=page_size,
                          page_num=page_num),
            timeout_s=timeout_s, transport=transport)
        feed = data.get("feed") or {}
        granules = []
        for entry in feed.get("entry") or []:
            links = [link.get("href") for link in
                     (entry.get("links") or [])
                     if isinstance(link, dict)
                     and str(link.get("rel", "")).endswith("data#")
                     and str(link.get("href", "")).startswith("http")]
            granules.append({
                "granule_ur": entry.get("title"),
                "producer_granule_id": entry.get(
                    "producer_granule_id"),
                "time_start": entry.get("time_start"),
                "time_end": entry.get("time_end"),
                "bbox": _entry_bbox(entry),
                "size_mb": _granule_size_mb(entry),
                "data_urls": links,
                "checksum": {"value": (entry.get("checksum") or {})
                             .get("value"),
                             "type": (entry.get("checksum") or {})
                             .get("algorithm")},
            })
        return {"granules": granules,
                "hits": int((feed.get("hits") or len(granules)))}
    except _CMRError:
        raise
    except Exception as exc:  # noqa: BLE001 - converted to reason
        raise _CMRError(f"granule search failed: "
                        f"{type(exc).__name__}: {exc}")


class _CMRError(Exception):
    """CMR lookup failure (message carries no secrets)."""

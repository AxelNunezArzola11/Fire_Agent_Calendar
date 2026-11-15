"""Sensor / product registry and multi-year NASA FIRMS availability model.

Five physical sensor identities, never merged or fabricated across years:
- MODIS Terra (`Terra`): operational active-fire record from 2000-11-01
- MODIS Aqua (`Aqua`): operational active-fire record from 2002-07-04
- VIIRS Suomi-NPP (`N` / `NPP`): operational 375m record from 2012-01-20
- VIIRS NOAA-20 (`NOAA-20` / `N20`): operational 375m record from 2018-04-01
- VIIRS NOAA-21 (`NOAA-21` / `N21`): operational 375m record from 2023-03-01
  (SP archive from 2023-03-01; NRT Area API feed from 2024-01-17)

Exact upstream date spans can be refreshed at runtime via the NASA FIRMS
Data Availability API (``/api/data_availability/csv/<KEY>/all``).
"""

from __future__ import annotations

import csv
import io
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone


@dataclass(frozen=True)
class Sensor:
    """One physical sensor identity."""

    id: str  # 'terra' | 'aqua' | 'snpp' | 'noaa20' | 'noaa21'
    satellite: str  # as FIRMS writes it in the CSV ('Terra', 'Aqua', 'NPP', 'NOAA-20', ...)
    instrument: str  # 'MODIS' | 'VIIRS'
    nrt_source: str  # area-API NRT source (MODIS_NRT covers Terra+Aqua)
    sp_source: str | None  # area-API Standard source; None if not published
    first_year: int  # first full year of Standard archive availability


TERRA = Sensor("terra", "Terra", "MODIS", "MODIS_NRT", "MODIS_SP", 2000)
AQUA = Sensor("aqua", "Aqua", "MODIS", "MODIS_NRT", "MODIS_SP", 2002)
SNPP = Sensor("snpp", "NPP", "VIIRS", "VIIRS_SNPP_NRT", "VIIRS_SNPP_SP", 2012)
NOAA20 = Sensor("noaa20", "NOAA-20", "VIIRS", "VIIRS_NOAA20_NRT", "VIIRS_NOAA20_SP", 2018)
NOAA21 = Sensor("noaa21", "NOAA-21", "VIIRS", "VIIRS_NOAA21_NRT", "VIIRS_NOAA21_SP", 2023)

SENSORS: dict[str, Sensor] = {s.id: s for s in (TERRA, AQUA, SNPP, NOAA20, NOAA21)}

NRT_SOURCES = ("MODIS_NRT", "VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT")
SP_SOURCES = ("MODIS_SP", "VIIRS_SNPP_SP", "VIIRS_NOAA20_SP", "VIIRS_NOAA21_SP")


@dataclass(frozen=True)
class ArchiveProduct:
    """A product published as open per-country yearly CSVs."""

    slug: str
    first_year: int


ARCHIVE_MODIS = ArchiveProduct("modis", 2000)
ARCHIVE_VIIRS_SNPP = ArchiveProduct("viirs-snpp", 2012)
ARCHIVE_PRODUCTS: dict[str, ArchiveProduct] = {
    p.slug: p for p in (ARCHIVE_MODIS, ARCHIVE_VIIRS_SNPP)
}

ARCHIVE_BASE = "https://firms.modaps.eosdis.nasa.gov/data/country"


def archive_filename(product_slug: str, year: int, country: str) -> str:
    """``viirs-snpp_2023_United_States.csv`` (country as FIRMS writes it)."""
    return f"{product_slug}_{year}_{country}.csv"


def archive_url(product_slug: str, year: int, country: str) -> str:
    """Build the open-archive URL. Raises KeyError/ValueError on bad input."""
    product = ARCHIVE_PRODUCTS[product_slug]  # KeyError if unknown slug
    if year < product.first_year:
        raise ValueError(f"{product_slug} archive starts at {product.first_year}, got {year}")
    if not country or "/" in country or "\\" in country:
        raise ValueError(f"invalid country slug: {country!r}")
    return f"{ARCHIVE_BASE}/{product_slug}/{year}/{archive_filename(product_slug, year, country)}"


# --- Multi-year product / satellite availability catalog ------------------

ALL_PRODUCTS: tuple[str, ...] = ("modis", "viirs-snpp", "viirs-noaa20", "viirs-noaa21")
ALL_SATELLITES: tuple[str, ...] = ("Terra", "Aqua", "N", "NOAA-20", "NOAA-21")

SATELLITE_LABELS: dict[str, str] = {
    "Terra": "MODIS Terra",
    "Aqua": "MODIS Aqua",
    "N": "VIIRS Suomi-NPP",
    "NPP": "VIIRS Suomi-NPP",
    "NOAA-20": "VIIRS NOAA-20",
    "N20": "VIIRS NOAA-20",
    "J1": "VIIRS NOAA-20",
    "NOAA-21": "VIIRS NOAA-21",
    "N21": "VIIRS NOAA-21",
    "J2": "VIIRS NOAA-21",
}

SATELLITE_ALIASES: dict[str, tuple[str, ...]] = {
    "Terra": ("Terra",),
    "Aqua": ("Aqua",),
    "N": ("N", "NPP"),
    "NOAA-20": ("NOAA-20", "N20", "J1"),
    "NOAA-21": ("NOAA-21", "N21", "J2"),
}

SATELLITE_PRODUCT: dict[str, str] = {
    "Terra": "modis",
    "Aqua": "modis",
    "N": "viirs-snpp",
    "NOAA-20": "viirs-noaa20",
    "NOAA-21": "viirs-noaa21",
}

PRODUCT_SATELLITES: dict[str, tuple[str, ...]] = {
    "modis": ("Terra", "Aqua"),
    "viirs-snpp": ("N",),
    "viirs-noaa20": ("NOAA-20",),
    "viirs-noaa21": ("NOAA-21",),
}

# Verified against live FIRMS /api/data_availability/csv/<KEY>/all (2026-11-15).
PRODUCT_CATALOG: dict[str, dict] = {
    "modis": {
        "label": "MODIS C6.1 (Terra + Aqua)",
        "satellites": ("Terra", "Aqua"),
        "sp_source": "MODIS_SP",
        "nrt_source": "MODIS_NRT",
        "min_date": "2000-11-01",
        "nrt_min_date": None,  # rolling NRT window (~last 75 days)
    },
    "viirs-snpp": {
        "label": "VIIRS 375m Suomi-NPP",
        "satellites": ("N",),
        "sp_source": "VIIRS_SNPP_SP",
        "nrt_source": "VIIRS_SNPP_NRT",
        "min_date": "2012-01-20",
        "nrt_min_date": None,
    },
    "viirs-noaa20": {
        "label": "VIIRS 375m NOAA-20",
        "satellites": ("NOAA-20",),
        "sp_source": "VIIRS_NOAA20_SP",
        "nrt_source": "VIIRS_NOAA20_NRT",
        "min_date": "2018-04-01",
        "nrt_min_date": None,
    },
    "viirs-noaa21": {
        "label": "VIIRS 375m NOAA-21",
        "satellites": ("NOAA-21",),
        "sp_source": "VIIRS_NOAA21_SP",
        "nrt_source": "VIIRS_NOAA21_NRT",
        "min_date": "2023-03-01",
        "nrt_min_date": "2024-01-17",
    },
}

SATELLITE_CATALOG: dict[str, dict] = {
    "Terra": {"label": "MODIS Terra", "product": "modis", "min_date": "2000-11-01"},
    "Aqua": {"label": "MODIS Aqua", "product": "modis", "min_date": "2002-07-04"},
    "N": {"label": "VIIRS Suomi-NPP", "product": "viirs-snpp", "min_date": "2012-01-20"},
    "NOAA-20": {"label": "VIIRS NOAA-20", "product": "viirs-noaa20", "min_date": "2018-04-01"},
    "NOAA-21": {"label": "VIIRS NOAA-21", "product": "viirs-noaa21", "min_date": "2023-03-01"},
}

_AVAIL_CACHE: dict[str, tuple[float, dict[str, dict[str, str]]]] = {}
_AVAIL_TTL_S = 3600.0


def canonical_satellite(raw: str, product: str = "") -> str:
    """Normalize raw FIRMS satellite identifiers into canonical UI labels."""
    val = str(raw or "").strip()
    upper = val.upper()
    if upper in ("TERRA", "T"):
        return "Terra"
    if upper in ("AQUA", "A"):
        return "Aqua"
    if upper in ("N", "NPP", "SNPP", "SUOMI-NPP"):
        return "N"
    if upper in ("NOAA-20", "NOAA20", "N20", "J1", "JPSS-1") or (
        upper == "1" and product == "viirs-noaa20"
    ):
        return "NOAA-20"
    if upper in ("NOAA-21", "NOAA21", "N21", "J2", "JPSS-2") or (
        upper == "2" and product == "viirs-noaa21"
    ):
        return "NOAA-21"
    if product == "viirs-noaa20" and not val:
        return "NOAA-20"
    if product == "viirs-noaa21" and not val:
        return "NOAA-21"
    return val


def expand_satellite_aliases(satellites: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Expand canonical satellite names to all FIRMS CSV aliases for DB queries."""
    seen: list[str] = []
    for sat in satellites:
        for alias in SATELLITE_ALIASES.get(sat, (sat,)):
            if alias not in seen:
                seen.append(alias)
    return tuple(seen)


def probe_firms_availability(
    key: str | None = None,
    *,
    timeout_s: float = 8.0,
    ttl_s: float = _AVAIL_TTL_S,
) -> dict[str, dict[str, str]]:
    """Query NASA FIRMS ``data_availability`` endpoint with process-local caching.

    Returns ``{data_id: {"min_date": "YYYY-MM-DD", "max_date": "YYYY-MM-DD"}}`` or
    ``{}`` if no key is configured or the endpoint is unreachable (never raises).
    """
    from .. import settings

    api_key = (key if key is not None else settings.get_setting("FIRMS_MAP_KEY", "")).strip()
    if not api_key:
        return {}
    now = time.monotonic()
    cached = _AVAIL_CACHE.get(api_key)
    if cached and (now - cached[0]) < ttl_s:
        return dict(cached[1])
    try:
        import requests

        resp = requests.get(
            f"https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/{api_key}/all",
            timeout=timeout_s,
        )
        if resp.status_code != 200 or not resp.text.strip():
            return {}
        parsed: dict[str, dict[str, str]] = {}
        reader = csv.DictReader(io.StringIO(resp.text.strip()))
        for row in reader:
            did = (row.get("data_id") or "").strip()
            lo = (row.get("min_date") or "").strip()[:10]
            hi = (row.get("max_date") or "").strip()[:10]
            if did and lo and hi:
                parsed[did] = {"min_date": lo, "max_date": hi}
        if parsed:
            _AVAIL_CACHE[api_key] = (now, parsed)
        return dict(parsed)
    except Exception:  # noqa: BLE001 - callers treat missing live table as fallback
        return {}


def _overlaps(a_start: str, a_end: str, b_start: str, b_end: str) -> bool:
    return a_start <= b_end and a_end >= b_start


def select_firms_sources_for_range(
    date_start: str,
    date_end: str,
    *,
    products: tuple[str, ...] = ALL_PRODUCTS,
    satellites: tuple[str, ...] | None = None,
    availability: dict[str, dict[str, str]] | None = None,
    today_utc: date | None = None,
) -> tuple[tuple[str, ...], list[str]]:
    """Select NASA FIRMS SP and/or NRT sources matching requested dates and sensors.

    Returns ``(sources, limitation_notes)``. Handles multi-year historical ranges
    (back to 2000-11-01 for MODIS), recent rolling NRT ranges, boundary overlaps
    between SP and NRT, and pre-launch dates without fabricating coverage.
    """
    today = today_utc or datetime.now(timezone.utc).date()
    today_s = today.isoformat()
    start_d = date.fromisoformat(date_start)
    end_d = date.fromisoformat(date_end)
    start_s, end_s = start_d.isoformat(), end_d.isoformat()

    active_products: list[str] = list(products) if products is not None else list(ALL_PRODUCTS)
    if satellites is not None:
        sat_prods = {
            SATELLITE_PRODUCT[s]
            for s in satellites
            if s in SATELLITE_PRODUCT
        }
        if products is None:
            active_products = [p for p in ALL_PRODUCTS if p in sat_prods]
        else:
            active_products = [p for p in active_products if p in sat_prods]

    notes: list[str] = []
    if end_d > today:
        notes.append(
            f"Requested end date {end_s} is in the future relative to UTC today ({today_s}); "
            "no future satellite observations exist."
        )

    # Check per-satellite launch boundaries when explicit satellites were requested
    sat_check_list = (
        tuple(satellites)
        if satellites is not None
        else tuple(s for p in active_products for s in PRODUCT_SATELLITES.get(p, ()))
    )
    for sat in sat_check_list:
        smeta = SATELLITE_CATALOG.get(sat)
        if not smeta:
            continue
        if end_s < smeta["min_date"]:
            notes.append(
                f"{smeta['label']} has no data for {start_s}..{end_s} "
                f"(operational archive starts {smeta['min_date']})."
            )
        elif start_s < smeta["min_date"] <= end_s:
            notes.append(
                f"{smeta['label']} coverage starts on {smeta['min_date']}; "
                f"dates before {smeta['min_date']} in this range have no observations from {smeta['label']}."
            )

    sources: list[str] = []
    # Conservative offline boundary when live availability table is absent (~95d NRT lookback)
    nrt_fallback_start = (today - timedelta(days=95)).isoformat()

    for prod in active_products:
        meta = PRODUCT_CATALOG.get(prod)
        if not meta:
            continue
        sp_src = meta["sp_source"]
        nrt_src = meta["nrt_source"]
        min_d = meta["min_date"]

        if availability:
            sp_span = availability.get(sp_src)
            nrt_span = availability.get(nrt_src)
            sp_overlaps = bool(
                sp_span and _overlaps(start_s, end_s, sp_span["min_date"], sp_span["max_date"])
            )
            sp_covers_all = bool(
                sp_span and sp_span["min_date"] <= start_s and end_s <= sp_span["max_date"]
            )
            nrt_overlaps = bool(
                nrt_span and _overlaps(start_s, end_s, nrt_span["min_date"], nrt_span["max_date"])
            )
            if sp_overlaps:
                sources.append(sp_src)
            if nrt_overlaps and not sp_covers_all:
                sources.append(nrt_src)
            if (
                not sp_overlaps
                and not nrt_overlaps
                and end_s >= min_d
            ):
                sp_hi = sp_span["max_date"] if sp_span else "unavailable"
                nrt_lo = nrt_span["min_date"] if nrt_span else "unavailable"
                notes.append(
                    f"NASA FIRMS reports no live SP ({sp_span['min_date'] if sp_span else '?'}..{sp_hi}) "
                    f"or NRT ({nrt_lo}..{nrt_span['max_date'] if nrt_span else '?'}) span covering {start_s}..{end_s} "
                    f"for {meta['label']}."
                )
        else:
            if end_s < min_d:
                continue
            sp_cutoff = (today - timedelta(days=60)).isoformat()
            if start_s <= sp_cutoff:
                sources.append(sp_src)
            if end_s >= nrt_fallback_start:
                if prod == "viirs-noaa21" and end_s < "2024-01-17":
                    pass
                else:
                    sources.append(nrt_src)

    # Deduplicate preserving order
    unique_sources = tuple(dict.fromkeys(sources))
    return unique_sources, list(dict.fromkeys(notes))


def evaluate_availability(
    date_start: str,
    date_end: str,
    products: tuple[str, ...],
    satellites: tuple[str, ...],
    *,
    local_spans: dict[str, dict[str, str]] | None = None,
    firms_key_configured: bool = False,
    region_label: str = "",
    live_dataset=None,
    availability: dict[str, dict[str, str]] | None = None,
    today_utc: date | None = None,
) -> dict:
    """Summarize exact data availability across local archives and NASA FIRMS."""
    from ..ui import query

    local_spans = local_spans or {}
    if local_spans:
        local_min = min(v["min"] for v in local_spans.values())
        local_max = max(v["max"] for v in local_spans.values())
    else:
        local_min, local_max = "unavailable", "unavailable"

    cov_start = query.coverage_for(date_start, local_spans)["status"] if local_spans else "unavailable"
    cov_end = query.coverage_for(date_end, local_spans)["status"] if local_spans else "unavailable"
    inside_local_dates = (cov_start == "complete" or cov_end == "complete") and (
        local_min != "unavailable" and _overlaps(date_start, date_end, local_min, local_max)
    )
    noaa_selected = [s for s in satellites if s in ("NOAA-20", "NOAA-21")] or [
        p for p in products if p in ("viirs-noaa20", "viirs-noaa21")
    ]
    outside_usa_region = bool(
        region_label
        and any(k in region_label.lower() for k in ("brazil", "spain", "australia", "portugal", "chile"))
    )

    firms_sources, sensor_notes = select_firms_sources_for_range(
        date_start,
        date_end,
        products=tuple(products),
        satellites=tuple(satellites),
        availability=availability,
        today_utc=today_utc,
    )

    live_matches = bool(
        live_dataset is not None
        and getattr(live_dataset, "date_start", None) == date_start
        and getattr(live_dataset, "date_end", None) == date_end
    )

    limitations: list[str] = []
    if local_min != "unavailable":
        limitations.append(
            f"Pre-indexed local DuckDB/Phase 2-5 archive covers USA MODIS C6.1 (Terra/Aqua) and "
            f"VIIRS Suomi-NPP across {local_min} to {local_max}."
        )
    else:
        limitations.append(
            "No local pre-indexed historical archive is attached in this deployment."
        )

    if firms_key_configured:
        limitations.append(
            "Multi-year & global retrieval via NASA FIRMS Area API is enabled (`FIRMS_MAP_KEY` configured): "
            "MODIS (2000-11-01+), VIIRS Suomi-NPP (2012-01-20+), VIIRS NOAA-20 (2018-04-01+), "
            "and VIIRS NOAA-21 (2023-03-01+ SP / 2024-01-17+ NRT)."
        )
    else:
        limitations.append(
            "NASA FIRMS live/multi-year retrieval is unavailable until `FIRMS_MAP_KEY` is configured "
            "in `.env` or Streamlit Secrets (supported upstream spans: MODIS 2000-11-01+, VIIRS S-NPP "
            "2012-01-20+, VIIRS NOAA-20 2018-04-01+, VIIRS NOAA-21 2023-03-01+)."
        )

    for note in sensor_notes:
        limitations.append(note)

    limitations.append(
        "Sofiev & Kouznetsov (2026) Observation Operator parameters cover MODIS and VIIRS Suomi-NPP; "
        "NOAA-20 and NOAA-21 detections are preserved in map, time-series, and export outputs without "
        "fabricated calibration coefficients."
    )

    if live_matches:
        summary = (
            f"Session FIRMS dataset active ({live_dataset.n_detections:,} detections, "
            f"{', '.join(live_dataset.sources)})."
        )
    elif sensor_notes and not firms_sources:
        summary = sensor_notes[0]
    elif inside_local_dates and not noaa_selected and not outside_usa_region:
        summary = (
            f"Local USA archive covers {date_start}..{date_end} for MODIS/S-NPP; "
            + (
                f"FIRMS API ready for `{', '.join(firms_sources)}`."
                if firms_key_configured
                else "configure `FIRMS_MAP_KEY` to add NOAA-20/21 or other years/regions."
            )
        )
    elif firms_key_configured and firms_sources:
        reasons = []
        if not inside_local_dates:
            reasons.append("date range outside local archive")
        if noaa_selected:
            reasons.append("NOAA-20/21 not in local DuckDB")
        if outside_usa_region:
            reasons.append(f"{region_label} outside local USA archive")
        why = ", ".join(reasons) or "live source selected"
        summary = (
            f"NASA FIRMS ready (`{', '.join(firms_sources)}`) for {date_start}..{date_end} ({why})."
        )
    elif not firms_key_configured:
        summary = (
            f"Dates {date_start}..{date_end} / selection require NASA FIRMS (`FIRMS_MAP_KEY` missing)."
        )
    else:
        summary = f"No NASA FIRMS sources overlap {date_start}..{date_end} for the chosen sensors."

    return {
        "date_start": date_start,
        "date_end": date_end,
        "local_span": (local_min, local_max),
        "inside_local_archive": inside_local_dates and not noaa_selected and not outside_usa_region,
        "live_loaded": live_matches,
        "firms_key_configured": firms_key_configured,
        "firms_sources": firms_sources,
        "sensor_notes": sensor_notes,
        "summary": summary,
        "limitations": limitations,
    }

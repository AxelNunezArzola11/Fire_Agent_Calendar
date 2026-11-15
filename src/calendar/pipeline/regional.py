"""Region-independent NASA FIRMS acquisition, normalization, and regional analytics.

Connects arbitrary geographic areas (preset countries/regions, custom bounding
boxes, or latitude/longitude + radius) to both:
1. Live NASA FIRMS Area API acquisition + canonical normalization when
   ``FIRMS_MAP_KEY`` is configured.
2. Deterministic Phase 2D/4B/4C-compatible time-series aggregation, trailing
   baselines, and statistical anomalies over either local DuckDB rows or newly
   retrieved regional FIRMS observations.

Scientific integrity guarantees:
- Never reuses California/USA observations for another geographic region.
- Never fabricates detections, FRP values, or baselines when observations or
  credentials are unavailable.
- Preserves provenance (``data_origin``, ``exec_id``, raw CSV SHA-256 hashes,
  source names, bounding box coordinates, and temporal coverage).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from .. import settings
from ..acquisition import nrt, sensors
from ..analytics import anomalies, baselines
from ..experimental.frp_operator_pilot import (
    operator as oop,
    results as oor,
    validation as oop_validation,
)
from ..processing import normalize
from ..ui import filters

MAX_LIVE_FIRMS_DAYS = 366


def max_live_firms_days() -> int:
    """Configured max day span per live FIRMS request (default 366)."""
    raw = settings.get_setting("CAL_FIRMS_MAX_DAYS", str(MAX_LIVE_FIRMS_DAYS))
    try:
        val = int(raw)
        return max(1, min(val, 366))
    except (TypeError, ValueError):
        return MAX_LIVE_FIRMS_DAYS


SOURCE_METADATA: dict[str, tuple[str, str]] = {
    "MODIS_NRT": ("modis", "terra_aqua"),
    "MODIS_SP": ("modis", "terra_aqua"),
    "VIIRS_SNPP_NRT": ("viirs-snpp", "snpp"),
    "VIIRS_SNPP_SP": ("viirs-snpp", "snpp"),
    "VIIRS_NOAA20_NRT": ("viirs-noaa20", "noaa20"),
    "VIIRS_NOAA20_SP": ("viirs-noaa20", "noaa20"),
    "VIIRS_NOAA21_NRT": ("viirs-noaa21", "noaa21"),
    "VIIRS_NOAA21_SP": ("viirs-noaa21", "noaa21"),
}

NRT_WINDOW_DAYS = 60


@dataclass
class RegionalDataset:
    """Normalized regional dataset ready for map, analytics, and pipeline tabs."""

    exec_id: str
    region_id: str
    region_label: str
    bbox: tuple[float, float, float, float]
    date_start: str
    date_end: str
    data_origin: str  # 'firms_live_api' | 'regional_duckdb_archive' | 'local_archive' | 'unavailable'
    sources: tuple[str, ...]
    retrieved_utc: str
    df: pd.DataFrame
    chunks: list[dict] = field(default_factory=list)
    rejected_count: int = 0
    errors: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    archive_id: str = ""
    archive_status: str = ""
    center_lat: float | None = None
    center_lon: float | None = None
    radius_km: float | None = None
    aoi_source: str = "regional"

    @property
    def n_detections(self) -> int:
        return int(len(self.df))

    def summary(self) -> dict[str, Any]:
        return {
            "exec_id": self.exec_id,
            "archive_id": self.archive_id,
            "archive_status": self.archive_status,
            "region_id": self.region_id,
            "region_label": self.region_label,
            "aoi_source": self.aoi_source,
            "bbox": list(self.bbox),
            "center_lat": self.center_lat,
            "center_lon": self.center_lon,
            "radius_km": self.radius_km,
            "date_start": self.date_start,
            "date_end": self.date_end,
            "data_origin": self.data_origin,
            "sources": list(self.sources),
            "n_detections": self.n_detections,
            "rejected_count": self.rejected_count,
            "retrieved_utc": self.retrieved_utc,
            "errors": list(self.errors),
            "limitations": list(self.limitations),
        }


def select_firms_sources_with_notes(
    date_start: str,
    date_end: str,
    *,
    products: tuple[str, ...] = sensors.ALL_PRODUCTS,
    satellites: tuple[str, ...] | None = None,
    include_noaa20: bool = False,
    include_noaa21: bool = False,
    availability: dict[str, dict[str, str]] | None = None,
    today_utc: date | None = None,
) -> tuple[tuple[str, ...], list[str]]:
    """Choose SP and/or NRT NASA FIRMS Area API sources + availability notes."""
    chosen_products = list(products) if products is not None else list(sensors.ALL_PRODUCTS)
    if include_noaa20 and "viirs-noaa20" not in chosen_products:
        chosen_products.append("viirs-noaa20")
    if include_noaa21 and "viirs-noaa21" not in chosen_products:
        chosen_products.append("viirs-noaa21")
    return sensors.select_firms_sources_for_range(
        date_start,
        date_end,
        products=tuple(chosen_products),
        satellites=satellites,
        availability=availability,
        today_utc=today_utc,
    )


def select_firms_sources(
    date_start: str,
    date_end: str,
    *,
    products: tuple[str, ...] = sensors.ALL_PRODUCTS,
    satellites: tuple[str, ...] | None = None,
    include_noaa20: bool = False,
    include_noaa21: bool = False,
    availability: dict[str, dict[str, str]] | None = None,
    today_utc: date | None = None,
) -> tuple[str, ...]:
    """Choose NRT and/or Standard (SP) Area API sources based on dates and sensors."""
    sources, _notes = select_firms_sources_with_notes(
        date_start,
        date_end,
        products=products,
        satellites=satellites,
        include_noaa20=include_noaa20,
        include_noaa21=include_noaa21,
        availability=availability,
        today_utc=today_utc,
    )
    return sources


def _execution_hash(
    bbox: tuple[float, float, float, float],
    date_start: str,
    date_end: str,
    sources: tuple[str, ...],
    digests: list[str],
) -> str:
    h = hashlib.sha256()
    h.update(
        json.dumps(
            {
                "bbox": [round(float(x), 5) for x in bbox],
                "date_start": date_start,
                "date_end": date_end,
                "sources": list(sources),
                "digests": sorted(digests),
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    return h.hexdigest()[:12]


def fetch_firms_regional(
    *,
    bbox: tuple[float, float, float, float],
    date_start: str,
    date_end: str,
    region_id: str = "custom",
    region_label: str = "Custom region",
    sources: tuple[str, ...] | None = None,
    products: tuple[str, ...] = sensors.ALL_PRODUCTS,
    satellites: tuple[str, ...] | None = None,
    availability: dict[str, dict[str, str]] | None = None,
    key: str | None = None,
    fetcher=None,
) -> RegionalDataset:
    """Retrieve and normalize NASA FIRMS observations for any global bbox + dates.

    Never fabricates observations when ``FIRMS_MAP_KEY`` is unset or the API
    returns zero rows or fails. Transport faults per source are recorded cleanly
    in ``errors``.
    """
    start_s = filters.coerce_day(date_start)
    end_s = filters.coerce_day(date_end)
    if start_s > end_s:
        raise ValueError("Start date must not be later than End date.")
    start_d = date.fromisoformat(start_s)
    end_d = date.fromisoformat(end_s)
    span_days = (end_d - start_d).days + 1
    max_days = max_live_firms_days()
    if span_days > max_days:
        raise ValueError(
            f"Live NASA FIRMS Area API retrieval is capped at {max_days} days "
            f"per request (requested {span_days} days) to respect API limits."
        )

    w, s, e, n = (float(v) for v in bbox)
    bbox_str = f"{w:.5f},{s:.5f},{e:.5f},{n:.5f}"
    map_key = key if key is not None else settings.get_setting("FIRMS_MAP_KEY", "")
    avail_map = availability
    if avail_map is None and sources is None and fetcher is None and map_key:
        avail_map = sensors.probe_firms_availability(map_key)
    if sources is not None:
        chosen_sources, source_notes = tuple(sources), []
    else:
        chosen_sources, source_notes = select_firms_sources_with_notes(
            start_s,
            end_s,
            products=products,
            satellites=satellites,
            availability=avail_map,
        )
    retrieved = datetime.now(timezone.utc).isoformat()
    exec_tag = _execution_hash((w, s, e, n), start_s, end_s, chosen_sources, [])

    if not map_key:
        return RegionalDataset(
            exec_id=exec_tag,
            region_id=region_id,
            region_label=region_label,
            bbox=(w, s, e, n),
            date_start=start_s,
            date_end=end_s,
            data_origin="firms_live_api",
            sources=chosen_sources,
            retrieved_utc=retrieved,
            df=pd.DataFrame(columns=list(normalize.CANONICAL)),
            errors=[
                "FIRMS_MAP_KEY is not configured; live NASA FIRMS retrieval "
                "across multi-year and global datasets is unavailable in this deployment."
            ],
            limitations=[
                "No live NASA FIRMS request was executed without FIRMS_MAP_KEY.",
                *source_notes,
            ],
        )

    if not chosen_sources:
        return RegionalDataset(
            exec_id=exec_tag,
            region_id=region_id,
            region_label=region_label,
            bbox=(w, s, e, n),
            date_start=start_s,
            date_end=end_s,
            data_origin="firms_live_api",
            sources=(),
            retrieved_utc=retrieved,
            df=pd.DataFrame(columns=list(normalize.CANONICAL)),
            errors=[],
            limitations=[
                "No NASA FIRMS sources cover the selected date range and satellites.",
                *source_notes,
            ],
        )

    range_fn = fetcher or nrt.fetch_range
    frames: list[pd.DataFrame] = []
    chunk_logs: list[dict] = []
    digests: list[str] = []
    rejected_total = 0
    errors: list[str] = []

    ordered_sources = sorted(
        chosen_sources,
        key=lambda src_name: (0 if src_name.endswith("_SP") else 1, chosen_sources.index(src_name)),
    )
    for src in ordered_sources:
        product, sensor = SOURCE_METADATA.get(src, ("unknown", "unknown"))
        try:
            chunks = range_fn(src, bbox_str, span_days, end_d, key=map_key)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{src}: {exc}")
            continue
        for chunk in chunks:
            digests.append(chunk.sha256)
            chunk_logs.append(
                {
                    "source": src,
                    "start_date": chunk.request.start_date.isoformat()
                    if chunk.request.start_date
                    else None,
                    "days": chunk.request.days,
                    "status": chunk.status,
                    "sha256": chunk.sha256,
                    "elapsed_s": chunk.elapsed_s,
                }
            )
            if chunk.status == "empty" or not chunk.content.strip():
                continue
            try:
                raw = pd.read_csv(io.BytesIO(chunk.content), dtype=str, keep_default_na=True)
                if raw.empty:
                    continue
                batch = normalize.normalize_frame(
                    raw,
                    product=product,
                    sensor=sensor,
                    source_file=f"firms:{src}:{bbox_str}:{start_s}_{end_s}",
                    source_sha256=chunk.sha256,
                    processing_version=f"{settings.PROCESSING_VERSION}_live",
                )
                rejected_total += len(batch.rejected)
                if not batch.accepted.empty:
                    acc = batch.accepted.copy()
                    acc["satellite"] = acc["satellite"].map(
                        lambda s_val, _prod=product: sensors.canonical_satellite(s_val, product=_prod)
                    )
                    acc["firms_source"] = src
                    frames.append(acc)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{src} parse/normalize: {exc}")

    if frames:
        merged = pd.concat(frames, ignore_index=True)
        day_col = merged["acquisition_timestamp_utc"].astype(str).str.slice(0, 10)
        mask = (
            (day_col >= start_s)
            & (day_col <= end_s)
            & (merged["longitude"] >= w)
            & (merged["longitude"] <= e)
            & (merged["latitude"] >= s)
            & (merged["latitude"] <= n)
        )
        if products is not None:
            mask &= merged["product"].isin(products)
        if satellites is not None:
            allowed_sats = {sensors.canonical_satellite(s_val) for s_val in satellites} | set(satellites)
            mask &= merged["satellite"].isin(allowed_sats)
        merged = (
            merged.loc[mask]
            .drop_duplicates(subset=["detection_id"])
            .drop_duplicates(
                subset=["product", "satellite", "acquisition_timestamp_utc", "latitude", "longitude"],
                keep="first",
            )
            .sort_values(["frp", "detection_id"], ascending=[False, True], na_position="last")
            .reset_index(drop=True)
        )
    else:
        merged = pd.DataFrame(columns=[*normalize.CANONICAL, "firms_source"])

    exec_id = _execution_hash((w, s, e, n), start_s, end_s, chosen_sources, digests)
    return RegionalDataset(
        exec_id=exec_id,
        region_id=region_id,
        region_label=region_label,
        bbox=(w, s, e, n),
        date_start=start_s,
        date_end=end_s,
        data_origin="firms_live_api",
        sources=chosen_sources,
        retrieved_utc=retrieved,
        df=merged,
        chunks=chunk_logs,
        rejected_count=rejected_total,
        errors=errors,
        limitations=[
            "Newly retrieved NASA FIRMS detections preserve source SP/NRT provenance and "
            "remain distinct from the pre-indexed local archive.",
            "MODIS and VIIRS detections and FRP values remain sensor-separated.",
            *source_notes,
        ],
    )


def _period_labels_for(day_series: pd.Series, grain: str) -> pd.Series:
    parsed = pd.to_datetime(day_series, utc=True, format="mixed", errors="coerce")
    if grain == "daily":
        return parsed.dt.strftime("%Y-%m-%d")
    if grain == "weekly":
        iso = parsed.dt.isocalendar()
        return (
            iso["year"].astype("Int64").astype(str)
            + "-W"
            + iso["week"].astype("Int64").astype(str).str.zfill(2)
        )
    if grain == "monthly":
        return parsed.dt.strftime("%Y-%m")
    raise ValueError(f"unsupported regional grain {grain!r}")


def _all_periods_between(date_start: str, date_end: str, grain: str) -> list[str]:
    start = date.fromisoformat(date_start)
    end = date.fromisoformat(date_end)
    if start > end:
        return []
    periods: list[str] = []
    seen: set[str] = set()
    cur = start
    while cur <= end:
        if grain == "daily":
            label = cur.isoformat()
        elif grain == "weekly":
            label = filters.iso_week_label(cur)
        elif grain == "monthly":
            label = f"{cur.year:04d}-{cur.month:02d}"
        else:
            raise ValueError(f"unsupported regional grain {grain!r}")
        if label not in seen:
            seen.add(label)
            periods.append(label)
        cur += timedelta(days=1)
    return periods


def build_regional_timeseries(
    df: pd.DataFrame,
    *,
    metric: str,
    scope: str,
    grain: str,
    date_start: str,
    date_end: str,
    region_label: str,
    exec_id: str = "regional",
) -> list[dict]:
    """Compute Phase 4B/4C-compatible baseline + anomaly series on any regional DataFrame.

    Returns records compatible with ``analytics_view.series_for`` / chart rendering.
    Never fabricates observations or baselines when history < MEDIAN_MIN_N (3).
    """
    if grain not in ("daily", "weekly", "monthly"):
        return []
    periods = _all_periods_between(date_start, date_end, grain)
    if not periods:
        return []

    work = df.copy()
    if not work.empty:
        work["_period"] = _period_labels_for(work["acquisition_timestamp_utc"], grain)
        work = work.dropna(subset=["_period"])
        if scope in ("modis", "viirs-snpp", "viirs-noaa20", "viirs-noaa21"):
            work = work[work["product"] == scope]
        elif scope in ("Terra", "Aqua", "N", "NOAA-20", "NOAA-21"):
            aliases = set(sensors.SATELLITE_ALIASES.get(scope, (scope,)))
            work = work[work["satellite"].astype(str).isin(aliases)]
        elif metric == "frp" and scope == "raw":
            pass
    else:
        work["_period"] = pd.Series(dtype="object")

    grouped_vals: dict[str, float] = {}
    if not work.empty:
        if metric == "frp":
            frp_num = pd.to_numeric(work["frp"], errors="coerce")
            valid_frp = work.assign(_frp=frp_num).dropna(subset=["_frp"])
            for p, part in valid_frp.groupby("_period", sort=True):
                grouped_vals[str(p)] = round(float(part["_frp"].sum()), 4)
        else:
            for p, part in work.groupby("_period", sort=True):
                grouped_vals[str(p)] = int(len(part))

    entries = [
        {"period": p, "slot": baselines.slot_index(grain, p), "value": grouped_vals.get(p, 0)}
        for p in periods
    ]
    cfg = baselines.BaselineConfig(
        method="sensor_aware",
        grain=grain,
        strategy="trailing",
        geo_scope=region_label,
        version="regional_cfg_v1",
    )
    records: list[dict] = []
    eff_metric = "frp" if metric == "frp" else "detection"
    for pos in range(len(entries)):
        base_rec = baselines.evaluate_period(
            entries,
            pos,
            cfg,
            metric=eff_metric,
            scope=scope,
            exec_id=exec_id,
        )
        anom_rec = anomalies.detect_from_baseline(base_rec, exec_id=exec_id)
        anom_rec["geo_scope"] = region_label
        anom_rec["algo_version"] = "regional_anomalies_v1"
        records.append(anom_rec)
    return records


def run_operator_on_frame(df: pd.DataFrame) -> dict:
    """Apply existing Phase 6 Observation Operator Eq.5/6/7 over regional detections."""
    records = []
    for row in df.to_dict("records"):
        verdict = oop_validation.validate_row(row)
        if not verdict["eligible"]:
            reasons = list(verdict["reasons"])
            prod = row.get("product")
            if prod in ("viirs-noaa20", "viirs-noaa21"):
                reasons = [
                    f"no published Observation Operator coefficients for {prod} "
                    "(Sofiev & Kouznetsov 2026 covers MODIS & VIIRS S-NPP)"
                    if r.startswith("unknown product")
                    else r
                    for r in reasons
                ]
            records.append(
                oor.result_record(row, None, None, reasons=tuple(reasons))
            )
            continue
        calc = oop.expected_frp(row["frp"], verdict["area"], row["product"], row["daynight"])
        regime = oop.detection_regime(
            row["frp"], verdict["area"], row["product"], row["daynight"]
        )
        records.append(
            oor.result_record(row, calc, regime, area_km2=verdict["area"])
        )
    agg = oor.aggregate_record(records)
    return {"records": records, "aggregate": agg}


# ---------------------------------------------------------------------------
# Independent Regional DuckDB Historical Archives
# ---------------------------------------------------------------------------

_REGIONAL_DETECTION_COLS: tuple[str, ...] = (
    "archive_id",
    "detection_id",
    "sensor",
    "satellite",
    "instrument",
    "product",
    "product_version",
    "acquisition_timestamp_utc",
    "original_acquisition_timestamp",
    "acq_date",
    "latitude",
    "longitude",
    "confidence",
    "frp",
    "daynight",
    "scan",
    "track",
    "firms_source",
    "source_file",
    "source_sha256",
    "data_origin",
    "exec_id",
    "ingestion_timestamp_utc",
    "processing_version",
)


def compute_archive_id(
    bbox: tuple[float, float, float, float],
    *,
    region_id: str | None = None,
    point_radius: tuple[float, float, float] | None = None,
) -> str:
    """Deterministic archive identifier isolated from the Stage 4C USA baseline."""
    rid = (region_id or "").strip().lower()
    if rid and rid not in ("custom", "full", "session", "bbox", "point_radius", "map_geometry"):
        slug = re.sub(r"[^a-z0-9_-]+", "_", rid).strip("_")
        if slug:
            return f"reg_{slug}"
    if point_radius is not None and len(point_radius) == 3:
        lat, lon, r_km = (float(point_radius[0]), float(point_radius[1]), float(point_radius[2]))
        return f"pt_{lat:.2f}_{lon:.2f}_{r_km:.1f}km"
    w, s, e, n = (round(float(v), 4) for v in bbox)
    digest = hashlib.sha256(f"{w:.4f},{s:.4f},{e:.4f},{n:.4f}".encode("utf-8")).hexdigest()[:10]
    return f"bbox_{digest}"


def _ensure_writable_parent(path: Path) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        probe = path.parent / f".write_probe_{os.getpid()}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def resolve_regional_db_path(db_path: str | Path | None = None) -> Path:
    """Resolve the independent regional DuckDB file path (never calendar.duckdb)."""
    if db_path is not None and str(db_path).strip():
        candidate = Path(str(db_path).strip())
    else:
        raw = settings.get_setting(
            "CAL_REGIONAL_DUCKDB_PATH", "regional_archives/regional.duckdb"
        )
        candidate = Path(raw or "regional_archives/regional.duckdb")
        if not candidate.is_absolute():
            candidate = settings.REPO_ROOT / candidate
    baseline_name = Path(
        settings.get_setting("CAL_DUCKDB_PATH", "calendar.duckdb") or "calendar.duckdb"
    ).name
    if candidate.name == baseline_name or candidate.name == "calendar.duckdb":
        raise ValueError(
            "Refusing to use the pinned Stage 4C USA baseline calendar.duckdb "
            "for independent regional archives."
        )
    if _ensure_writable_parent(candidate):
        return candidate
    fallback = Path(tempfile.gettempdir()) / "fireagent_regional_archives" / "regional.duckdb"
    fallback.parent.mkdir(parents=True, exist_ok=True)
    return fallback


def init_regional_db(db_path: str | Path | None = None) -> Path:
    """Create the regional DuckDB schema if missing and return its resolved path."""
    path = resolve_regional_db_path(db_path)
    con = duckdb.connect(str(path))
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS regional_archives (
                archive_id VARCHAR PRIMARY KEY,
                region_id VARCHAR,
                region_label VARCHAR,
                aoi_source VARCHAR,
                crs VARCHAR,
                lon_min DOUBLE,
                lat_min DOUBLE,
                lon_max DOUBLE,
                lat_max DOUBLE,
                center_lat DOUBLE,
                center_lon DOUBLE,
                radius_km DOUBLE,
                created_utc VARCHAR,
                updated_utc VARCHAR
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS regional_acquisitions (
                exec_id VARCHAR PRIMARY KEY,
                archive_id VARCHAR,
                date_start VARCHAR,
                date_end VARCHAR,
                products VARCHAR,
                satellites VARCHAR,
                firms_sources VARCHAR,
                n_detections INTEGER,
                rejected_count INTEGER,
                data_origin VARCHAR,
                status VARCHAR,
                retrieved_utc VARCHAR,
                errors VARCHAR,
                limitations VARCHAR,
                chunks VARCHAR
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS regional_detections (
                archive_id VARCHAR,
                detection_id VARCHAR,
                sensor VARCHAR,
                satellite VARCHAR,
                instrument VARCHAR,
                product VARCHAR,
                product_version VARCHAR,
                acquisition_timestamp_utc VARCHAR,
                original_acquisition_timestamp VARCHAR,
                acq_date VARCHAR,
                latitude DOUBLE,
                longitude DOUBLE,
                confidence VARCHAR,
                frp DOUBLE,
                daynight VARCHAR,
                scan DOUBLE,
                track DOUBLE,
                firms_source VARCHAR,
                source_file VARCHAR,
                source_sha256 VARCHAR,
                data_origin VARCHAR,
                exec_id VARCHAR,
                ingestion_timestamp_utc VARCHAR,
                processing_version VARCHAR
            )
            """
        )
    finally:
        con.close()
    return path


def _clean_str_col(series: pd.Series) -> list[str | None]:
    out: list[str | None] = []
    for val in series:
        if val is None or (isinstance(val, float) and pd.isna(val)) or pd.isna(val):
            out.append(None)
        else:
            out.append(str(val))
    return out


def _prepare_detection_frame(
    ds: RegionalDataset,
    *,
    archive_id: str,
) -> pd.DataFrame:
    df = ds.df.copy()
    if df.empty:
        return pd.DataFrame(columns=list(_REGIONAL_DETECTION_COLS))

    work = pd.DataFrame(index=df.index)
    work["archive_id"] = archive_id
    for col in (
        "detection_id",
        "sensor",
        "satellite",
        "instrument",
        "product",
        "product_version",
        "acquisition_timestamp_utc",
        "original_acquisition_timestamp",
        "confidence",
        "daynight",
        "firms_source",
        "source_file",
        "source_sha256",
        "ingestion_timestamp_utc",
        "processing_version",
    ):
        if col in df.columns:
            work[col] = _clean_str_col(df[col])
        else:
            work[col] = None

    ts_series = pd.Series(work["acquisition_timestamp_utc"], index=work.index).fillna("")
    work["acq_date"] = [t[:10] if len(t) >= 10 else None for t in ts_series]
    work["data_origin"] = ds.data_origin or "firms_live_api"
    work["exec_id"] = ds.exec_id

    for num_col in ("latitude", "longitude", "frp", "scan", "track"):
        if num_col in df.columns:
            work[num_col] = pd.to_numeric(df[num_col], errors="coerce").astype("float64")
        else:
            work[num_col] = pd.Series(float("nan"), index=work.index, dtype="float64")

    return work.loc[:, list(_REGIONAL_DETECTION_COLS)]


def _products_from_sources(sources: tuple[str, ...]) -> list[str]:
    prods: list[str] = []
    for src in sources:
        prod = SOURCE_METADATA.get(src, ("unknown", "unknown"))[0]
        if prod != "unknown" and prod not in prods:
            prods.append(prod)
    return prods


def _acq_status(ds: RegionalDataset) -> str:
    if ds.errors and ds.n_detections == 0:
        return "failed"
    if ds.errors:
        return "partial"
    if ds.n_detections == 0:
        return "empty"
    return "complete"


def store_regional_dataset(
    ds: RegionalDataset,
    *,
    db_path: str | Path | None = None,
    aoi_source: str | None = None,
    center_lat: float | None = None,
    center_lon: float | None = None,
    radius_km: float | None = None,
    requested_products: tuple[str, ...] | None = None,
    requested_satellites: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Persist regional AOI metadata, acquisition provenance, and deduplicated detections."""
    c_lat = center_lat if center_lat is not None else ds.center_lat
    c_lon = center_lon if center_lon is not None else ds.center_lon
    r_km = radius_km if radius_km is not None else ds.radius_km
    point_radius = (
        (float(c_lat), float(c_lon), float(r_km))
        if c_lat is not None and c_lon is not None and r_km is not None
        else None
    )
    archive_id = ds.archive_id or compute_archive_id(
        ds.bbox, region_id=ds.region_id, point_radius=point_radius
    )
    ds.archive_id = archive_id
    ds.center_lat = float(c_lat) if c_lat is not None else None
    ds.center_lon = float(c_lon) if c_lon is not None else None
    ds.radius_km = float(r_km) if r_km is not None else None
    eff_aoi_source = aoi_source or ds.aoi_source or "regional"
    ds.aoi_source = eff_aoi_source

    path = init_regional_db(db_path)
    now_utc = ds.retrieved_utc or datetime.now(timezone.utc).isoformat()
    w, s, e, n = (float(v) for v in ds.bbox)
    status = _acq_status(ds)
    stored_products = (
        list(requested_products)
        if requested_products is not None
        else _products_from_sources(ds.sources)
    )
    stored_sats = list(requested_satellites) if requested_satellites is not None else None

    con = duckdb.connect(str(path))
    inserted = 0
    skipped = 0
    try:
        existing_meta = con.execute(
            "SELECT created_utc FROM regional_archives WHERE archive_id = ?",
            [archive_id],
        ).fetchone()
        created_utc = existing_meta[0] if existing_meta else now_utc
        con.execute("DELETE FROM regional_archives WHERE archive_id = ?", [archive_id])
        con.execute(
            """
            INSERT INTO regional_archives (
                archive_id, region_id, region_label, aoi_source, crs,
                lon_min, lat_min, lon_max, lat_max,
                center_lat, center_lon, radius_km, created_utc, updated_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                archive_id,
                ds.region_id,
                ds.region_label,
                eff_aoi_source,
                "EPSG:4326",
                w,
                s,
                e,
                n,
                ds.center_lat,
                ds.center_lon,
                ds.radius_km,
                created_utc,
                now_utc,
            ],
        )

        con.execute("DELETE FROM regional_acquisitions WHERE exec_id = ?", [ds.exec_id])
        con.execute(
            """
            INSERT INTO regional_acquisitions (
                exec_id, archive_id, date_start, date_end, products, satellites,
                firms_sources, n_detections, rejected_count, data_origin, status,
                retrieved_utc, errors, limitations, chunks
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ds.exec_id,
                archive_id,
                ds.date_start,
                ds.date_end,
                json.dumps(stored_products),
                json.dumps(stored_sats) if stored_sats is not None else None,
                json.dumps(list(ds.sources)),
                int(ds.n_detections),
                int(ds.rejected_count),
                ds.data_origin,
                status,
                now_utc,
                json.dumps(list(ds.errors)),
                json.dumps(list(ds.limitations)),
                json.dumps(list(ds.chunks)),
            ],
        )

        prep = _prepare_detection_frame(ds, archive_id=archive_id)
        if not prep.empty:
            existing_rows = con.execute(
                """
                SELECT detection_id, product, satellite, acquisition_timestamp_utc,
                       ROUND(latitude, 5), ROUND(longitude, 5)
                FROM regional_detections
                WHERE archive_id = ?
                """,
                [archive_id],
            ).fetchall()
            seen_ids = {r[0] for r in existing_rows if r[0] is not None}
            seen_sigs = {(r[1], r[2], r[3], r[4], r[5]) for r in existing_rows}

            keep_mask: list[bool] = []
            for row in prep.itertuples(index=False):
                det_id = row.detection_id
                lat_r = round(float(row.latitude), 5) if pd.notna(row.latitude) else None
                lon_r = round(float(row.longitude), 5) if pd.notna(row.longitude) else None
                sig = (row.product, row.satellite, row.acquisition_timestamp_utc, lat_r, lon_r)
                if (det_id and det_id in seen_ids) or sig in seen_sigs:
                    keep_mask.append(False)
                    skipped += 1
                else:
                    if det_id:
                        seen_ids.add(det_id)
                    seen_sigs.add(sig)
                    keep_mask.append(True)

            to_insert = prep.loc[keep_mask].reset_index(drop=True)
            if not to_insert.empty:
                con.register("_reg_det_stage", to_insert)
                cols_sql = ", ".join(_REGIONAL_DETECTION_COLS)
                con.execute(
                    f"INSERT INTO regional_detections ({cols_sql}) "
                    f"SELECT {cols_sql} FROM _reg_det_stage"
                )
                con.unregister("_reg_det_stage")
                inserted = int(len(to_insert))
    finally:
        con.close()

    return {
        "archive_id": archive_id,
        "exec_id": ds.exec_id,
        "db_path": str(path),
        "status": status,
        "inserted": inserted,
        "skipped_duplicates": skipped,
    }


def query_regional_duckdb_detections(
    *,
    archive_id: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
    products: tuple[str, ...] | None = None,
    satellites: tuple[str, ...] | None = None,
    db_path: str | Path | None = None,
) -> pd.DataFrame:
    """Query normalized detections from the independent regional DuckDB archive."""
    empty_cols = [*normalize.CANONICAL, "firms_source", "data_origin", "exec_id", "archive_id", "acq_date"]
    try:
        path = resolve_regional_db_path(db_path)
    except Exception:
        return pd.DataFrame(columns=empty_cols)
    if not path.exists():
        return pd.DataFrame(columns=empty_cols)

    con = duckdb.connect(str(path), read_only=True)
    try:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        if "regional_detections" not in tables:
            return pd.DataFrame(columns=empty_cols)

        clauses: list[str] = ["1=1"]
        params: list[Any] = []
        if archive_id:
            clauses.append("archive_id = ?")
            params.append(archive_id)
        if bbox is not None:
            w, s, e, n = (float(v) for v in bbox)
            clauses.append("longitude BETWEEN ? AND ? AND latitude BETWEEN ? AND ?")
            params.extend([w, e, s, n])
        if date_start:
            clauses.append("acq_date >= ?")
            params.append(filters.coerce_day(date_start))
        if date_end:
            clauses.append("acq_date <= ?")
            params.append(filters.coerce_day(date_end))
        if products:
            ph = ", ".join(["?"] * len(products))
            clauses.append(f"product IN ({ph})")
            params.extend(list(products))
        if satellites:
            expanded: set[str] = set()
            for sat in satellites:
                expanded.add(sat)
                expanded.update(sensors.SATELLITE_ALIASES.get(sat, (sat,)))
                expanded.add(sensors.canonical_satellite(sat))
            sorted_sats = sorted(expanded)
            ph = ", ".join(["?"] * len(sorted_sats))
            clauses.append(f"satellite IN ({ph})")
            params.extend(sorted_sats)

        sql = (
            "SELECT * FROM regional_detections WHERE "
            + " AND ".join(clauses)
            + " ORDER BY frp DESC NULLS LAST, detection_id ASC"
        )
        df = con.execute(sql, params).df()
    finally:
        con.close()

    if df.empty:
        return pd.DataFrame(columns=empty_cols)
    return df


def _check_archive_coverage(
    con: duckdb.DuckDBPyConnection,
    archive_id: str,
    date_start: str,
    date_end: str,
    *,
    products: tuple[str, ...] | None = None,
    satellites: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    start_d = date.fromisoformat(date_start)
    end_d = date.fromisoformat(date_end)
    all_days: list[str] = []
    cur = start_d
    while cur <= end_d:
        all_days.append(cur.isoformat())
        cur += timedelta(days=1)

    req_prods = set(products) if products else set()
    req_sats = {sensors.canonical_satellite(s) for s in satellites} if satellites else set()

    acqs = con.execute(
        """
        SELECT exec_id, date_start, date_end, products, satellites, firms_sources,
               n_detections, data_origin, status, retrieved_utc, errors, limitations
        FROM regional_acquisitions
        WHERE archive_id = ? AND status IN ('complete', 'empty')
        ORDER BY retrieved_utc DESC
        """,
        [archive_id],
    ).fetchall()

    covered_days: set[str] = set()
    matched_exec_ids: list[str] = []
    matched_sources: list[str] = []
    latest_retrieved = ""

    for row in acqs:
        (
            ex_id,
            acq_s,
            acq_e,
            prods_raw,
            sats_raw,
            srcs_raw,
            _n_det,
            _origin,
            _status,
            ret_utc,
            _errs,
            _lims,
        ) = row
        if not acq_s or not acq_e:
            continue
        acq_prods = set(json.loads(prods_raw)) if prods_raw else set()
        acq_sats = (
            {sensors.canonical_satellite(s) for s in json.loads(sats_raw)}
            if sats_raw
            else set()
        )
        # Verify sensors requested were part of this prior acquisition
        if req_prods and acq_prods and not req_prods.issubset(acq_prods):
            continue
        if req_sats and acq_sats and not req_sats.issubset(acq_sats):
            continue
        ov_start = max(date_start, str(acq_s))
        ov_end = min(date_end, str(acq_e))
        if ov_start <= ov_end:
            matched_exec_ids.append(str(ex_id))
            if not latest_retrieved:
                latest_retrieved = str(ret_utc or "")
            if srcs_raw:
                for src in json.loads(srcs_raw):
                    if src not in matched_sources:
                        matched_sources.append(src)
            cur_d = date.fromisoformat(ov_start)
            end_ov = date.fromisoformat(ov_end)
            while cur_d <= end_ov:
                covered_days.add(cur_d.isoformat())
                cur_d += timedelta(days=1)

    missing_days = [d for d in all_days if d not in covered_days]
    return {
        "covered": len(missing_days) == 0 and len(all_days) > 0,
        "covered_days": len(covered_days),
        "total_days": len(all_days),
        "missing_days": missing_days,
        "matched_exec_ids": matched_exec_ids,
        "matched_sources": tuple(matched_sources),
        "retrieved_utc": latest_retrieved,
    }


def load_regional_dataset(
    archive_id: str,
    *,
    bbox: tuple[float, float, float, float],
    date_start: str,
    date_end: str,
    products: tuple[str, ...] | None = None,
    satellites: tuple[str, ...] | None = None,
    region_id: str = "custom",
    region_label: str = "Custom region",
    center_lat: float | None = None,
    center_lon: float | None = None,
    radius_km: float | None = None,
    aoi_source: str = "regional",
    db_path: str | Path | None = None,
) -> tuple[RegionalDataset | None, dict[str, Any]]:
    """Load cached observations and provenance for ``archive_id`` if window is covered."""
    start_s = filters.coerce_day(date_start)
    end_s = filters.coerce_day(date_end)
    try:
        path = resolve_regional_db_path(db_path)
    except Exception as exc:
        return None, {"covered": False, "reason": str(exc), "missing_days": []}
    if not path.exists():
        return None, {"covered": False, "reason": "regional DuckDB archive does not exist yet", "missing_days": []}

    con = duckdb.connect(str(path), read_only=True)
    try:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        if "regional_acquisitions" not in tables or "regional_detections" not in tables:
            return None, {"covered": False, "reason": "regional tables not initialized", "missing_days": []}
        cov = _check_archive_coverage(
            con,
            archive_id,
            start_s,
            end_s,
            products=products,
            satellites=satellites,
        )
    finally:
        con.close()

    cov["db_path"] = str(path)
    if not cov["covered"]:
        return None, cov

    df = query_regional_duckdb_detections(
        archive_id=archive_id,
        bbox=bbox,
        date_start=start_s,
        date_end=end_s,
        products=products,
        satellites=satellites,
        db_path=path,
    )
    exec_id = cov["matched_exec_ids"][0] if cov["matched_exec_ids"] else _execution_hash(
        bbox, start_s, end_s, cov["matched_sources"], []
    )
    ds = RegionalDataset(
        exec_id=exec_id,
        region_id=region_id,
        region_label=region_label,
        bbox=tuple(float(v) for v in bbox),
        date_start=start_s,
        date_end=end_s,
        data_origin="regional_duckdb_archive",
        sources=cov["matched_sources"],
        retrieved_utc=cov["retrieved_utc"] or datetime.now(timezone.utc).isoformat(),
        df=df,
        archive_id=archive_id,
        archive_status="loaded_from_duckdb" if not df.empty else "loaded_empty_from_duckdb",
        center_lat=center_lat,
        center_lon=center_lon,
        radius_km=radius_km,
        aoi_source=aoi_source,
        limitations=[
            f"Loaded from independent regional DuckDB archive ({archive_id}) without modifying Stage 4C USA baseline.",
            "MODIS and VIIRS detections and FRP values remain sensor-separated.",
        ],
    )
    return ds, cov


def _date_windows(date_start: str, date_end: str, max_days: int) -> list[tuple[str, str]]:
    start_d = date.fromisoformat(date_start)
    end_d = date.fromisoformat(date_end)
    out: list[tuple[str, str]] = []
    cur = start_d
    while cur <= end_d:
        win_end = min(end_d, cur + timedelta(days=max_days - 1))
        out.append((cur.isoformat(), win_end.isoformat()))
        cur = win_end + timedelta(days=1)
    return out


def ensure_regional_archive_data(
    *,
    bbox: tuple[float, float, float, float],
    date_start: str,
    date_end: str,
    region_id: str = "custom",
    region_label: str = "Custom region",
    products: tuple[str, ...] = sensors.ALL_PRODUCTS,
    satellites: tuple[str, ...] | None = None,
    center_lat: float | None = None,
    center_lon: float | None = None,
    radius_km: float | None = None,
    aoi_source: str = "regional",
    sources: tuple[str, ...] | None = None,
    availability: dict[str, dict[str, str]] | None = None,
    key: str | None = None,
    fetcher=None,
    force_refresh: bool = False,
    db_path: str | Path | None = None,
) -> RegionalDataset:
    """Resolve regional observations from DuckDB or auto-populate from NASA FIRMS.

    1. Resolves deterministic ``archive_id`` isolated from ``calendar.duckdb``.
    2. Returns cached regional DuckDB observations when the window/sensors are covered.
    3. Otherwise automatically retrieves available observations from NASA FIRMS
       (splitting multi-year windows if needed), normalizes and deduplicates them,
       persists them in ``regional.duckdb``, and returns the dataset.
    4. Never falls back to USA observations when FIRMS credentials are unavailable.
    """
    start_s = filters.coerce_day(date_start)
    end_s = filters.coerce_day(date_end)
    point_radius = (
        (float(center_lat), float(center_lon), float(radius_km))
        if center_lat is not None and center_lon is not None and radius_km is not None
        else None
    )
    archive_id = compute_archive_id(
        bbox, region_id=region_id, point_radius=point_radius
    )

    if not force_refresh:
        cached, _cov = load_regional_dataset(
            archive_id,
            bbox=bbox,
            date_start=start_s,
            date_end=end_s,
            products=products,
            satellites=satellites,
            region_id=region_id,
            region_label=region_label,
            center_lat=center_lat,
            center_lon=center_lon,
            radius_km=radius_km,
            aoi_source=aoi_source,
            db_path=db_path,
        )
        if cached is not None:
            return cached

    map_key = key if key is not None else settings.get_setting("FIRMS_MAP_KEY", "")
    if not map_key and fetcher is None:
        # Check if partial historical detections already exist in the archive
        partial_df = query_regional_duckdb_detections(
            archive_id=archive_id,
            bbox=bbox,
            date_start=start_s,
            date_end=end_s,
            products=products,
            satellites=satellites,
            db_path=db_path,
        )
        if not partial_df.empty:
            return RegionalDataset(
                exec_id=_execution_hash(bbox, start_s, end_s, (), []),
                region_id=region_id,
                region_label=region_label,
                bbox=tuple(float(v) for v in bbox),
                date_start=start_s,
                date_end=end_s,
                data_origin="regional_duckdb_archive",
                sources=(),
                retrieved_utc=datetime.now(timezone.utc).isoformat(),
                df=partial_df,
                archive_id=archive_id,
                archive_status="partial_archive_no_api_key",
                center_lat=center_lat,
                center_lon=center_lon,
                radius_km=radius_km,
                aoi_source=aoi_source,
                limitations=[
                    "Partial regional DuckDB observations returned; FIRMS_MAP_KEY is missing so missing dates could not be retrieved."
                ],
            )
        return RegionalDataset(
            exec_id=_execution_hash(bbox, start_s, end_s, (), []),
            region_id=region_id,
            region_label=region_label,
            bbox=tuple(float(v) for v in bbox),
            date_start=start_s,
            date_end=end_s,
            data_origin="unavailable",
            sources=(),
            retrieved_utc=datetime.now(timezone.utc).isoformat(),
            df=pd.DataFrame(columns=[*normalize.CANONICAL, "firms_source"]),
            archive_id=archive_id,
            archive_status="unavailable_no_api_key",
            center_lat=center_lat,
            center_lon=center_lon,
            radius_km=radius_km,
            aoi_source=aoi_source,
            errors=[
                "Regional DuckDB archive has no recorded observations for this window and "
                "FIRMS_MAP_KEY is not configured for automatic NASA FIRMS retrieval."
            ],
            limitations=[
                "External regions are isolated from the Stage 4C USA baseline and never backfilled from USA data."
            ],
        )

    eff_key = map_key or "TEST-KEY"
    max_days = max_live_firms_days()
    windows = _date_windows(start_s, end_s, max_days)
    frames: list[pd.DataFrame] = []
    all_sources: list[str] = []
    all_chunks: list[dict] = []
    all_errors: list[str] = []
    all_limitations: list[str] = []
    rejected_total = 0
    exec_ids: list[str] = []

    for w_start, w_end in windows:
        sub = fetch_firms_regional(
            bbox=bbox,
            date_start=w_start,
            date_end=w_end,
            region_id=region_id,
            region_label=region_label,
            sources=sources,
            products=products,
            satellites=satellites,
            availability=availability,
            key=eff_key,
            fetcher=fetcher,
        )
        sub.archive_id = archive_id
        sub.center_lat = center_lat
        sub.center_lon = center_lon
        sub.radius_km = radius_km
        sub.aoi_source = aoi_source
        store_regional_dataset(
            sub,
            db_path=db_path,
            aoi_source=aoi_source,
            center_lat=center_lat,
            center_lon=center_lon,
            radius_km=radius_km,
            requested_products=products,
            requested_satellites=satellites,
        )
        exec_ids.append(sub.exec_id)
        if not sub.df.empty:
            frames.append(sub.df)
        for s_name in sub.sources:
            if s_name not in all_sources:
                all_sources.append(s_name)
        all_chunks.extend(sub.chunks)
        all_errors.extend(sub.errors)
        for lim in sub.limitations:
            if lim not in all_limitations:
                all_limitations.append(lim)
        rejected_total += sub.rejected_count

    merged = (
        query_regional_duckdb_detections(
            archive_id=archive_id,
            bbox=bbox,
            date_start=start_s,
            date_end=end_s,
            products=products,
            satellites=satellites,
            db_path=db_path,
        )
        if not frames
        else pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["detection_id"])
        .sort_values(["frp", "detection_id"], ascending=[False, True], na_position="last")
        .reset_index(drop=True)
    )
    if merged.empty and frames:
        merged = pd.DataFrame(columns=[*normalize.CANONICAL, "firms_source"])

    if all_errors and merged.empty:
        arch_status = "failed_firms_retrieval"
    elif all_errors:
        arch_status = "populated_with_warnings"
    elif merged.empty:
        arch_status = "populated_empty_window"
    else:
        arch_status = "populated_from_firms"

    return RegionalDataset(
        exec_id=exec_ids[-1] if exec_ids else _execution_hash(bbox, start_s, end_s, tuple(all_sources), []),
        region_id=region_id,
        region_label=region_label,
        bbox=tuple(float(v) for v in bbox),
        date_start=start_s,
        date_end=end_s,
        data_origin="firms_live_api",
        sources=tuple(all_sources),
        retrieved_utc=datetime.now(timezone.utc).isoformat(),
        df=merged,
        chunks=all_chunks,
        rejected_count=rejected_total,
        errors=all_errors,
        limitations=all_limitations,
        archive_id=archive_id,
        archive_status=arch_status,
        center_lat=center_lat,
        center_lon=center_lon,
        radius_km=radius_km,
        aoi_source=aoi_source,
    )


def list_regional_archives(db_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Return metadata and row counts for all stored regional DuckDB archives."""
    try:
        path = resolve_regional_db_path(db_path)
    except Exception:
        return []
    if not path.exists():
        return []
    con = duckdb.connect(str(path), read_only=True)
    try:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        if "regional_archives" not in tables:
            return []
        rows = con.execute(
            """
            SELECT a.archive_id, a.region_id, a.region_label, a.aoi_source,
                   a.lon_min, a.lat_min, a.lon_max, a.lat_max,
                   a.center_lat, a.center_lon, a.radius_km,
                   COUNT(d.detection_id) AS n_detections,
                   MIN(d.acq_date) AS min_date,
                   MAX(d.acq_date) AS max_date,
                   a.updated_utc
            FROM regional_archives a
            LEFT JOIN regional_detections d ON a.archive_id = d.archive_id
            GROUP BY a.archive_id, a.region_id, a.region_label, a.aoi_source,
                     a.lon_min, a.lat_min, a.lon_max, a.lat_max,
                     a.center_lat, a.center_lon, a.radius_km, a.updated_utc
            ORDER BY a.updated_utc DESC
            """
        ).fetchall()
    finally:
        con.close()

    return [
        {
            "archive_id": r[0],
            "region_id": r[1],
            "region_label": r[2],
            "aoi_source": r[3],
            "bbox": [r[4], r[5], r[6], r[7]],
            "center_lat": r[8],
            "center_lon": r[9],
            "radius_km": r[10],
            "n_detections": int(r[11] or 0),
            "min_date": r[12],
            "max_date": r[13],
            "updated_utc": r[14],
        }
        for r in rows
    ]


def summarize_regional_distributions(df: pd.DataFrame) -> dict[str, Any]:
    """Compute sensor-separated satellite, product, day/night, and FRP distributions."""
    if df is None or df.empty:
        return {
            "by_satellite": pd.DataFrame(columns=["satellite", "product", "detections", "frp_sum", "frp_mean", "frp_median", "frp_max"]),
            "by_product": pd.DataFrame(columns=["product", "detections", "frp_sum", "frp_mean", "frp_median", "frp_max"]),
            "by_daynight": pd.DataFrame(columns=["daynight", "detections", "frp_sum"]),
            "frp_bins": pd.DataFrame(columns=["frp_bin", "modis", "viirs"]),
        }

    work = df.copy()
    work["frp_num"] = pd.to_numeric(work["frp"], errors="coerce")
    work["satellite"] = work.get("satellite", pd.Series("unknown", index=work.index)).fillna("unknown").astype(str)
    work["product"] = work.get("product", pd.Series("unknown", index=work.index)).fillna("unknown").astype(str)
    work["daynight"] = work.get("daynight", pd.Series("U", index=work.index)).fillna("U").astype(str)

    sat_rows = []
    for (sat, prod), part in work.groupby(["satellite", "product"], sort=True):
        valid_frp = part["frp_num"].dropna()
        sat_rows.append({
            "satellite": sat,
            "product": prod,
            "detections": int(len(part)),
            "frp_sum": round(float(valid_frp.sum()), 2) if not valid_frp.empty else 0.0,
            "frp_mean": round(float(valid_frp.mean()), 2) if not valid_frp.empty else 0.0,
            "frp_median": round(float(valid_frp.median()), 2) if not valid_frp.empty else 0.0,
            "frp_max": round(float(valid_frp.max()), 2) if not valid_frp.empty else 0.0,
        })
    by_sat = pd.DataFrame(sat_rows).sort_values("detections", ascending=False).reset_index(drop=True)

    prod_rows = []
    for prod, part in work.groupby("product", sort=True):
        valid_frp = part["frp_num"].dropna()
        prod_rows.append({
            "product": prod,
            "detections": int(len(part)),
            "frp_sum": round(float(valid_frp.sum()), 2) if not valid_frp.empty else 0.0,
            "frp_mean": round(float(valid_frp.mean()), 2) if not valid_frp.empty else 0.0,
            "frp_median": round(float(valid_frp.median()), 2) if not valid_frp.empty else 0.0,
            "frp_max": round(float(valid_frp.max()), 2) if not valid_frp.empty else 0.0,
        })
    by_prod = pd.DataFrame(prod_rows).sort_values("detections", ascending=False).reset_index(drop=True)

    dn_rows = []
    for dn, part in work.groupby("daynight", sort=True):
        valid_frp = part["frp_num"].dropna()
        dn_rows.append({
            "daynight": dn,
            "detections": int(len(part)),
            "frp_sum": round(float(valid_frp.sum()), 2) if not valid_frp.empty else 0.0,
        })
    by_dn = pd.DataFrame(dn_rows).sort_values("detections", ascending=False).reset_index(drop=True)

    bin_edges = [0.0, 5.0, 20.0, 50.0, 100.0, 500.0, float("inf")]
    bin_labels = ["0-5 MW", "5-20 MW", "20-50 MW", "50-100 MW", "100-500 MW", "500+ MW"]
    valid_work = work.dropna(subset=["frp_num"]).copy()
    valid_work = valid_work[valid_work["frp_num"] >= 0.0]
    if valid_work.empty:
        frp_bins = pd.DataFrame({"frp_bin": bin_labels, "modis": [0] * len(bin_labels), "viirs": [0] * len(bin_labels)})
    else:
        valid_work["frp_bin"] = pd.cut(
            valid_work["frp_num"],
            bins=bin_edges,
            labels=bin_labels,
            include_lowest=True,
            right=False,
        )
        valid_work["sensor_family"] = valid_work["product"].map(
            lambda p: "modis" if str(p).startswith("modis") else "viirs"
        )
        piv = (
            valid_work.groupby(["frp_bin", "sensor_family"], observed=False)
            .size()
            .unstack(fill_value=0)
            .reindex(bin_labels, fill_value=0)
        )
        frp_bins = pd.DataFrame({
            "frp_bin": bin_labels,
            "modis": [int(piv["modis"].get(b, 0)) if "modis" in piv.columns else 0 for b in bin_labels],
            "viirs": [int(piv["viirs"].get(b, 0)) if "viirs" in piv.columns else 0 for b in bin_labels],
        })

    return {
        "by_satellite": by_sat,
        "by_product": by_prod,
        "by_daynight": by_dn,
        "frp_bins": frp_bins,
    }

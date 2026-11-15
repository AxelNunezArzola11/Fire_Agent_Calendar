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
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from .. import settings
from ..acquisition import nrt
from ..analytics import anomalies, baselines
from ..experimental.frp_operator_pilot import (
    operator as oop,
    results as oor,
    validation as oop_validation,
)
from ..processing import normalize
from ..ui import filters

MAX_LIVE_FIRMS_DAYS = 15

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
    data_origin: str  # 'firms_live_api' | 'local_archive'
    sources: tuple[str, ...]
    retrieved_utc: str
    df: pd.DataFrame
    chunks: list[dict] = field(default_factory=list)
    rejected_count: int = 0
    errors: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    @property
    def n_detections(self) -> int:
        return int(len(self.df))


def select_firms_sources(
    date_start: str,
    date_end: str,
    *,
    products: tuple[str, ...] = ("modis", "viirs-snpp"),
    include_noaa20: bool = False,
    today_utc: date | None = None,
) -> tuple[str, ...]:
    """Choose NRT vs Standard (SP) Area API sources based on date recency."""
    today = today_utc or datetime.now(timezone.utc).date()
    end_d = date.fromisoformat(date_end)
    _ = date.fromisoformat(date_start)
    use_nrt = (today - end_d).days <= NRT_WINDOW_DAYS
    suffix = "NRT" if use_nrt else "SP"
    out: list[str] = []
    if not products or "modis" in products:
        out.append(f"MODIS_{suffix}")
    if not products or "viirs-snpp" in products:
        out.append(f"VIIRS_SNPP_{suffix}")
    if include_noaa20 or "viirs-noaa20" in products:
        out.append(f"VIIRS_NOAA20_{suffix}")
    return tuple(out) or (f"MODIS_{suffix}", f"VIIRS_SNPP_{suffix}")


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
    products: tuple[str, ...] = ("modis", "viirs-snpp"),
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
    if span_days > MAX_LIVE_FIRMS_DAYS:
        raise ValueError(
            f"Live NASA FIRMS Area API retrieval is capped at {MAX_LIVE_FIRMS_DAYS} days "
            f"per request (requested {span_days} days) to respect API limits."
        )

    w, s, e, n = (float(v) for v in bbox)
    bbox_str = f"{w:.5f},{s:.5f},{e:.5f},{n:.5f}"
    chosen_sources = sources or select_firms_sources(start_s, end_s, products=products)
    map_key = key if key is not None else settings.get_setting("FIRMS_MAP_KEY", "")
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
                "FIRMS_MAP_KEY is not configured; live global NASA FIRMS retrieval "
                "is unavailable in this deployment."
            ],
            limitations=[
                "No live NASA FIRMS request was executed without FIRMS_MAP_KEY.",
            ],
        )

    range_fn = fetcher or nrt.fetch_range
    frames: list[pd.DataFrame] = []
    chunk_logs: list[dict] = []
    digests: list[str] = []
    rejected_total = 0
    errors: list[str] = []

    for src in chosen_sources:
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
                    frames.append(batch.accepted)
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
        merged = (
            merged.loc[mask]
            .drop_duplicates(subset=["detection_id"])
            .sort_values(["frp", "detection_id"], ascending=[False, True], na_position="last")
            .reset_index(drop=True)
        )
    else:
        merged = pd.DataFrame(columns=list(normalize.CANONICAL))

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
            "Newly retrieved NASA FIRMS detections are session-scoped and distinct "
            "from the pinned 2023 USA pilot archive.",
            "MODIS and VIIRS detections and FRP values remain sensor-separated.",
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
        elif scope in ("Terra", "Aqua", "N"):
            work = work[work["satellite"] == scope]
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
            records.append(
                oor.result_record(row, None, None, reasons=tuple(verdict["reasons"]))
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

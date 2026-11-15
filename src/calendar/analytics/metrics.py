"""Phase 2D: three independent fire-activity metrics (deterministic).

Metric 1 — Detection Activity: satellite observation counts (raw and
  harmonized views; harmonized counts each match group once + singles).
Metric 2 — Hotspot Cluster Activity: cluster counts + composition (clusters
  are observation groups, NEVER labeled as individual wildfires).
Metric 3 — FRP: radiative-power sums with explicit (unit, method, sensor,
  window, scope, n) provenance; per-sensor by default; combined only with
  a recorded sensor-composition warning. Missing FRP is excluded + counted,
  never zero-filled. Repeated-observation sums are NOT total emitted energy.

Every result carries full provenance. Grains: daily/weekly/monthly/
quarterly/yearly via UTC calendar truncation (ISO weeks).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)

ALGO_VERSION = "metrics_v1"

GRAINS = ("daily", "weekly", "monthly", "quarterly", "yearly")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _period(series: pd.Series, grain: str) -> pd.Series:
    stamps = pd.to_datetime(series, utc=True)
    if grain == "daily":
        return stamps.dt.strftime("%Y-%m-%d")
    if grain == "weekly":
        iso = stamps.dt.isocalendar()
        return iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    if grain == "monthly":
        return stamps.dt.strftime("%Y-%m")
    if grain == "quarterly":
        return stamps.dt.year.astype(str) + "-Q" + stamps.dt.quarter.astype(str)
    if grain == "yearly":
        return stamps.dt.strftime("%Y")
    raise ValueError(f"unknown grain {grain!r}; expected one of {GRAINS}")


def _base_provenance(metric: str, definition: str, inputs: dict, scope: dict,
                     method: str, exec_id: str) -> dict:
    return {"metric": metric, "definition": definition, "inputs": inputs,
            "scope": scope, "method": method, "algo_version": ALGO_VERSION,
            "exec_id": exec_id, "computed_at": _utcnow()}


def detection_activity(detections_df: pd.DataFrame, members_df: pd.DataFrame,
                       *, grain: str, scope: dict, inputs: dict,
                       exec_id: str) -> dict:
    """Metric 1: raw + harmonized observation counts.

    Raw rows are per (period, product) and reconcile across grains.
    Harmonized rows are per period: matched groups count once each
    (attributed to the earliest member's period, so grains reconcile),
    ambiguous members count individually (unresolved correspondence),
    singles count once. Per-product harmonized figures are compositional
    (a shared group appears under both products) and labeled non-additive.
    """
    det = detections_df[["detection_id", "acquisition_timestamp_utc", "satellite",
                         "instrument", "product"]].copy()
    det["period"] = _period(det["acquisition_timestamp_utc"], grain)
    mem = members_df.set_index("detection_id")
    det["match_class"] = det["detection_id"].map(mem["classification"])
    det["match_id"] = det["detection_id"].map(mem["match_id"])
    # Group attribution period = earliest member's period (reconciling).
    matched_only = det[det["match_class"] == "matched"]
    group_period = matched_only.groupby("match_id")["period"].min().to_dict() \
        if len(matched_only) else {}
    rows, harmonized_rows = [], []
    for period, grp in det.groupby("period", sort=True):
        prow = {"period": period, "raw_observations": len(grp),
                "by_product": grp["product"].value_counts().to_dict(),
                "by_satellite": grp["satellite"].value_counts().to_dict(),
                "by_match_class": grp["match_class"].value_counts().to_dict()}
        rows.append(prow)
        attributed = sum(1 for _, p in group_period.items() if p == period)
        own = grp
        amb = int((own["match_class"] == "ambiguous").sum())
        singles = int(own["match_class"].isin(["MODIS-only", "VIIRS-only"]).sum())
        in_groups = own[own["match_class"] == "matched"]
        harmonized_rows.append({
            "period": period,
            "harmonized_observations": int(attributed + amb + singles),
            "matched_groups_attributed": int(attributed),
            "ambiguous_members": amb,
            "singletons": singles,
            "composition": {
                "matched_modis_members": int((in_groups["product"] == "modis").sum()),
                "matched_viirs_members": int((in_groups["product"] == "viirs-snpp").sum()),
                "ambiguous_modis": int(((own["match_class"] == "ambiguous")
                                        & (own["product"] == "modis")).sum()),
                "ambiguous_viirs": int(((own["match_class"] == "ambiguous")
                                        & (own["product"] == "viirs-snpp")).sum()),
            },
        })
    return {**_base_provenance(
        "detection_activity",
        "Raw satellite detection observations plus harmonized observations "
        "(matched groups collapsed once each by earliest-member period; "
        "ambiguous members counted individually; singles once). Observations, "
        "not fires; legitimate repeat observations are never deduplicated.",
        inputs, {**scope, "grain": grain},
        "counts + earliest-member group attribution", exec_id),
        "rows": rows, "harmonized_rows": harmonized_rows}


def cluster_activity(clusters_df: pd.DataFrame, *, grain: str, scope: dict,
                     inputs: dict, exec_id: str, only_cluster_ids=None) -> dict:
    """Metric 2: cluster counts + composition by first-observation period."""
    clusters = clusters_df.copy()
    clusters["period"] = _period(clusters["first_observation_utc"], grain)
    if only_cluster_ids is not None:
        clusters = clusters[clusters["cluster_id"].isin(set(only_cluster_ids))]
    rows = []
    for period, grp in clusters.groupby("period", sort=True):
        sizes = grp["n_detections"]
        rows.append({
            "period": period, "clusters": len(grp),
            "detections_in_clusters": int(sizes.sum()),
            "size_min": int(sizes.min()), "size_max": int(sizes.max()),
            "size_mean": round(float(sizes.mean()), 2),
            "singletons": int((sizes == 1).sum()),
            "multi_detection": int((sizes > 1).sum()),
            "modis_detections": int(grp["n_modis"].sum()),
            "viirs_detections": int(grp["n_viirs"].sum()),
            "span_h_max": round(float(grp["observation_span_h"].max()), 2),
        })
    return {**_base_provenance(
        "cluster_activity",
        "Spatial-temporal hotspot-cluster counts (observation groups, never "
        "individual wildfires), attributed to the first-observation period.",
        inputs, {**scope, "grain": grain}, "cluster-count", exec_id), "rows": rows}


def frp_summary(detections_df: pd.DataFrame, *, grain: str, scope: dict,
                inputs: dict, exec_id: str, combined: bool = False) -> dict:
    """Metric 3: FRP sums per sensor (+optional combined with composition)."""
    det = detections_df[["acquisition_timestamp_utc", "product", "satellite",
                         "frp"]].copy()
    det["period"] = _period(det["acquisition_timestamp_utc"], grain)
    det["frp"] = pd.to_numeric(det["frp"], errors="coerce")
    rows = []
    for (period, product), grp in det.groupby(["period", "product"], sort=True):
        ok = grp["frp"].dropna()
        rows.append({
            "period": period, "product": product, "unit": "MW",
            "method": "sum-of-reported-FRP (repeated observations included; "
                      "NOT total emitted energy)",
            "frp_sum": round(float(ok.sum()), 3),
            "frp_mean": round(float(ok.mean()), 3) if len(ok) else None,
            "frp_max": round(float(ok.max()), 3) if len(ok) else None,
            "n_contributing": int(len(ok)),
            "n_missing_frp": int(len(grp) - len(ok)),
            "by_satellite": {sat: round(float(g["frp"].dropna().sum()), 3)
                             for sat, g in grp.groupby("satellite", sort=True)},
        })
    result = {**_base_provenance(
        "frp_summary",
        "Reported Fire Radiative Power aggregated per sensor. Cross-sensor "
        "FRP is not directly comparable (pixel size / sensitivity differ).",
        inputs, {**scope, "grain": grain}, "per-sensor-sum", exec_id), "rows": rows}
    if combined:
        combo = det.groupby("period", sort=True)
        result["combined"] = {
            "warning": ("combined MODIS+VIIRS FRP mixes incomparable pixel "
                        "responses; use per-sensor rows for analysis"),
            "rows": [{
                "period": period,
                "frp_sum": round(float(g["frp"].dropna().sum()), 3),
                "n_contributing": int(g["frp"].notna().sum()),
                "sensor_composition": g["product"].value_counts().to_dict(),
            } for period, g in combo],
        }
    return result

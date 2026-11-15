"""Phase 2C: deterministic spatiotemporal hotspot clustering.

A cluster is a group of related satellite detections — NOT a wildfire, NOT
a fire perimeter, NOT proof of continuous burning. Durations are labeled
observation spans; representative locations are centroids, never ignitions.

Links (union-find, deterministic):
  (a) matched-group co-membership (Phase 2B matched groups only —
      ambiguous groups deliberately create NO links);
  (b) same-sensor pairs within spatial tolerance AND temporal gap
      (single-linkage chaining for multi-day continuity).
Cross-sensor pairs link ONLY via (a): matching verdicts are final here.

Every input detection lands in exactly one cluster (singletons included).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .matching import haversine_km

logger = logging.getLogger(__name__)

ALGO_VERSION = "cluster_v1"


@dataclass(frozen=True)
class ClusterParams:
    """Versioned clustering parameters (cluster_params_v1).

    Justification: 2.0 km ≈ 2× MODIS nadir footprint — adjacent MODIS
    pixels (and their VIIRS oversampling) link, while distinct neighboring
    fires ~5 km apart stay separate. 24 h gap chains consecutive daily
    overpasses (fires observed day after day) without bridging weekly gaps.
    Both are explicit, versioned, and replaceable — not universal truths.
    """

    version: str = "cluster_params_v1"
    spatial_tol_km: float = 2.0
    temporal_gap_h: float = 24.0
    geocell_deg: float = 0.1


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    keep = df[["detection_id", "latitude", "longitude", "acquisition_timestamp_utc",
               "satellite", "instrument", "product", "frp"]].copy()
    if keep["detection_id"].duplicated().any():
        raise ValueError("duplicate detection_id in clustering input")
    stamps = pd.to_datetime(keep["acquisition_timestamp_utc"], utc=True, errors="coerce")
    if stamps.isna().any():
        raise ValueError("unparseable timestamps in clustering input")
    # Resolution-agnostic epoch seconds (see matching.py note re pandas 3).
    keep["epoch_s"] = ((stamps - pd.Timestamp("1970-01-01", tz="UTC"))
                       // pd.Timedelta("1s")).astype("int64")
    keep["day_ord"] = stamps.dt.date.map(lambda d: d.toordinal())
    keep["cell_lat"] = (keep["latitude"].to_numpy() / 0.1).astype(np.int32)
    keep["cell_lon"] = (keep["longitude"].to_numpy() / 0.1).astype(np.int32)
    return keep.sort_values("detection_id").reset_index(drop=True)


def _same_sensor_links(frame: pd.DataFrame, params: ClusterParams,
                       stats: dict) -> list[tuple[int, int]]:
    """Positional index pairs (i<j) linked by same-sensor proximity."""
    lat = frame["latitude"].to_numpy(dtype=np.float64)
    lon = frame["longitude"].to_numpy(dtype=np.float64)
    epoch = frame["epoch_s"].to_numpy()
    day = frame["day_ord"].to_numpy()
    clat = frame["cell_lat"].to_numpy()
    clon = frame["cell_lon"].to_numpy()
    inst = frame["instrument"].to_numpy()
    cells: dict[tuple[int, int, int], list[int]] = {}
    for pos in range(len(frame)):
        cells.setdefault((int(day[pos]), int(clat[pos]), int(clon[pos])), []).append(pos)
    cell_index = {k: np.fromiter(v, dtype=np.int32) for k, v in cells.items()}
    links: list[tuple[int, int]] = []
    gap_s = params.temporal_gap_h * 3600
    for pos in range(len(frame)):
        parts = []
        for delta in (-1, 0, 1):
            for dlat in (-1, 0, 1):
                for dlon in (-1, 0, 1):
                    hit = cell_index.get((int(day[pos]) + delta,
                                          int(clat[pos]) + dlat,
                                          int(clon[pos]) + dlon))
                    if hit is not None:
                        parts.append(hit)
        if not parts:
            continue
        cand = np.unique(np.concatenate(parts))
        cand = cand[cand > pos]  # upper triangle only (symmetric links)
        if not len(cand):
            continue
        same = inst[cand] == inst[pos]
        dt = np.abs(epoch[cand] - int(epoch[pos]))
        pre = same & (dt <= gap_s)
        if not pre.any():
            continue
        cand = cand[pre]
        dist = haversine_km(np.full(len(cand), lat[pos]), np.full(len(cand), lon[pos]),
                            lat[cand], lon[cand])
        stats["distance_computations"] += len(cand)
        for other in cand[dist <= params.spatial_tol_km].tolist():
            links.append((pos, int(other)))
    stats["same_sensor_links"] = len(links)
    return links


def cluster_frames(detections_df: pd.DataFrame, match_groups_df: pd.DataFrame,
                   params: ClusterParams | None = None,
                   exec_id: str = "") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run clustering. Returns (clusters, membership, stats)."""
    params = params or ClusterParams()
    stats: dict = {"detections_in": len(detections_df), "distance_computations": 0,
                   "same_sensor_links": 0, "match_group_links": 0}
    frame = _prepare(detections_df)
    ids = frame["detection_id"].tolist()
    pos_of = {did: i for i, did in enumerate(ids)}
    parent = list(range(len(frame)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # (a) matched-group co-membership (ambiguous groups: no links).
    if len(match_groups_df) and "classification" in match_groups_df.columns:
        matched = match_groups_df[match_groups_df["classification"] == "matched"]
    else:
        matched = match_groups_df.iloc[0:0]  # empty: no co-membership links
    for row in matched.itertuples():
        members = json.loads(row.modis_ids) + json.loads(row.viirs_ids)
        known = [pos_of[d] for d in members if d in pos_of]
        for first, other in zip([known[0]] * max(0, len(known) - 1), known[1:]):
            union(first, other)
            stats["match_group_links"] += 1

    # (b) same-sensor spatiotemporal links.
    for i, j in _same_sensor_links(frame, params, stats):
        union(i, j)

    components: dict[int, list[int]] = {}
    for pos in range(len(frame)):
        components.setdefault(find(pos), []).append(pos)
    ordered = sorted(components.values(), key=lambda c: frame.iloc[min(c)]["detection_id"])

    match_of = {}
    for row in matched.itertuples():
        for did in json.loads(row.modis_ids) + json.loads(row.viirs_ids):
            match_of[did] = row.match_id

    clusters, membership = [], []
    for num, comp in enumerate(ordered, 1):
        cid = f"cl{num:07d}"
        sub = frame.iloc[sorted(comp)]
        member_ids = sub["detection_id"].tolist()
        frp = pd.to_numeric(sub["frp"], errors="coerce")
        ok_frp = frp.dropna()
        by_sensor = {}
        for inst, grp in sub.groupby("instrument", sort=True):
            gfrp = pd.to_numeric(grp["frp"], errors="coerce").dropna()
            by_sensor[inst] = {"n": int(len(grp)),
                               "null_frp": int(len(grp) - len(gfrp)),
                               "frp_sum": round(float(gfrp.sum()), 3) if len(gfrp) else 0.0,
                               "frp_max": round(float(gfrp.max()), 3) if len(gfrp) else None}
        stamps = pd.to_datetime(sub["acquisition_timestamp_utc"], utc=True)
        span_h = (stamps.max() - stamps.min()).total_seconds() / 3600.0 if len(sub) > 1 else 0.0
        distinct_dates = stamps.dt.date.nunique()
        span_days = max(1, int((stamps.max() - stamps.min()).total_seconds() // 86400) + 1)
        clusters.append({
            "cluster_id": cid,
            "detection_ids": json.dumps(member_ids),
            "n_detections": len(sub),
            "n_modis": int((sub["instrument"] == "MODIS").sum()),
            "n_viirs": int((sub["instrument"] == "VIIRS").sum()),
            "satellites": json.dumps(sorted(sub["satellite"].unique().tolist())),
            "first_observation_utc": stamps.min().isoformat(),
            "last_observation_utc": stamps.max().isoformat(),
            "observation_span_h": round(span_h, 3),
            "lon_min": round(float(sub["longitude"].min()), 6),
            "lon_max": round(float(sub["longitude"].max()), 6),
            "lat_min": round(float(sub["latitude"].min()), 6),
            "lat_max": round(float(sub["latitude"].max()), 6),
            "centroid_lon": round(float(sub["longitude"].mean()), 6),
            "centroid_lat": round(float(sub["latitude"].mean()), 6),
            "frp_sum": round(float(ok_frp.sum()), 3),
            "frp_mean": round(float(ok_frp.mean()), 3) if len(ok_frp) else None,
            "frp_max": round(float(ok_frp.max()), 3) if len(ok_frp) else None,
            "frp_n": int(len(ok_frp)),
            "frp_null": int(len(sub) - len(ok_frp)),
            "frp_by_sensor": json.dumps(by_sensor),
            "match_groups": json.dumps(sorted({match_of[d] for d in member_ids
                                               if d in match_of})),
            "completeness_dates": distinct_dates,
            "completeness_span_days": span_days,
            "completeness_frac": round(distinct_dates / span_days, 4),
            "algo_version": ALGO_VERSION, "params_version": params.version,
            "exec_id": exec_id,
        })
        for did in member_ids:
            membership.append({"detection_id": did, "cluster_id": cid,
                               "algo_version": ALGO_VERSION,
                               "params_version": params.version, "exec_id": exec_id})

    clusters_df = pd.DataFrame(clusters)
    membership_df = pd.DataFrame(membership)
    stats["clusters"] = len(clusters_df)
    stats["singletons"] = int((clusters_df["n_detections"] == 1).sum()) if len(clusters_df) else 0
    assert len(membership_df) == len(frame)
    assert membership_df["detection_id"].nunique() == len(frame)
    logger.info("clustered %d detections -> %d clusters (%d singletons)",
                len(frame), stats["clusters"], stats["singletons"])
    return clusters_df, membership_df, stats

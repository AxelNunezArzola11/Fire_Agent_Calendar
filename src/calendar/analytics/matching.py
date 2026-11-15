"""Phase 2B: deterministic adaptive MODIS–VIIRS spatiotemporal matching.

Cross-sensor only (same-sensor grouping is clustering, Phase 2C).
No universal fixed radius: tolerances derive from per-instrument footprints
(MODIS ~1 km, VIIRS I-band ~375 m nadir) with versioned factors.

Pipeline (all passes deterministic: sorted ids, sorted pairs, no randomness):
  1. candidate generation (day ±1 shards × 0.1° geocells × ±24 h, vectorized)
  2. mutual-best pairs -> matched (size-2 groups)
  3. uncontested single bests attach to coherent groups (1-many / many-1):
     leftover attaches to group G iff its best partner is in G, the best is
     uncontested (2nd-best outside the compete margin, or single candidate),
     and it has core edges to ALL opposite-sensor members of G (no chaining)
  4. leftovers in multi-candidate contention -> ambiguous groups
     (connected components over core edges; no forced pairing)
  5. mutual expanded-best pairs -> weak ambiguous groups (size 2, never
     chained); other expanded-only leftovers -> singleton ambiguous groups
  6. rest -> MODIS-only / VIIRS-only singletons

Every input detection lands in exactly one group-or-singleton (invariant).
Pairs live in compact numpy arrays (full-year scale); groups/members are
small Python structures.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ALGO_VERSION = "matcher_v1"

EARTH_KM = 6371.0


@dataclass(frozen=True)
class MatchParams:
    """Versioned matching parameters (match_params_v1).

    Justification: MODIS thermal pixels are ~1 km at nadir (bow-tie growth
    off-nadir); VIIRS I-band is 375 m. Core tolerance = mean footprint × 2
    (≈1.38 km M–V) admits same-overpass co-located pixels plus modest
    geolocation/footprint slop without merging neighboring fires. Time
    tolerance 12 h spans same-day polar overpass separations (Terra ~10:30,
    Aqua/S-NPP ~13:30 local); longer links belong to cluster continuity
    (Phase 2C), not observation correspondence. Expanded window (×4 space,
    24 h) marks weak evidence, never matches.
    """

    version: str = "match_params_v1"
    footprint_km: dict = field(default_factory=lambda: {"MODIS": 1.0, "VIIRS": 0.375})
    core_factor: float = 2.0
    expanded_factor: float = 4.0
    time_tol_hours: float = 12.0
    expanded_time_hours: float = 24.0
    compete_margin: float = 0.15  # 2nd-best within 15% of best -> competing
    geocell_deg: float = 0.1  # blocking cells (~11 km), 3×3 neighborhoods
    high_evidence_score: float = 0.5


def haversine_km(lat1: np.ndarray, lon1: np.ndarray,
                 lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorized great-circle distance in km."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlambda = np.radians(lon2) - np.radians(lon1)
    hav = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * EARTH_KM * np.arcsin(np.sqrt(hav))


def core_tol_km(params: MatchParams, inst_a: str = "MODIS", inst_b: str = "VIIRS") -> float:
    return ((params.footprint_km[inst_a] + params.footprint_km[inst_b]) / 2.0
            * params.core_factor)


def _near_nadir(scan: float, track: float, footprint: float) -> bool | None:
    if scan != scan or track != track:  # NaN -> geometry unavailable
        return None
    return bool(scan <= 2.0 * footprint and track <= 2.0 * footprint)


def _prepare(df: pd.DataFrame, instrument: str) -> pd.DataFrame:
    keep = df[["detection_id", "latitude", "longitude", "acquisition_timestamp_utc",
               "satellite", "instrument", "daynight", "scan", "track"]].copy()
    if keep["detection_id"].duplicated().any():
        dupes = sorted(keep.loc[keep["detection_id"].duplicated(), "detection_id"].unique())
        raise ValueError(f"duplicate detection_id in matcher input: {dupes[:5]}")
    stamps = pd.to_datetime(keep["acquisition_timestamp_utc"], utc=True, errors="coerce")
    if stamps.isna().any():
        bad = sorted(keep.loc[stamps.isna(), "detection_id"].tolist())
        raise ValueError(f"unparseable timestamps for: {bad[:5]}")
    # Resolution-agnostic epoch seconds (pandas 3 defaults to microsecond
    # resolution, so astype(int64) // 1e9 is WRONG by 1000x — see 2C report).
    keep["epoch_s"] = ((stamps - pd.Timestamp("1970-01-01", tz="UTC"))
                       // pd.Timedelta("1s")).astype("int64")
    keep["day_ord"] = stamps.dt.date.map(lambda d: d.toordinal())
    keep["cell_lat"] = (keep["latitude"].to_numpy() / 0.1).astype(np.int32)
    keep["cell_lon"] = (keep["longitude"].to_numpy() / 0.1).astype(np.int32)
    keep["instrument"] = instrument
    return keep.sort_values("detection_id").reset_index(drop=True)


_PAIR_DTYPE = np.dtype([("mpos", np.int32), ("vpos", np.int32), ("dist", np.float32),
                        ("dt", np.float32), ("score", np.float32), ("core", np.bool_)])


def _build_pairs(modis: pd.DataFrame, viirs: pd.DataFrame, params: MatchParams,
                 stats: dict) -> np.ndarray:
    """All candidate pairs as a structured array sorted by (mpos, score, vpos)."""
    tol = core_tol_km(params)
    exp_tol = tol * params.expanded_factor / params.core_factor
    t_exp = params.expanded_time_hours
    t_core = params.time_tol_hours

    v_epoch = viirs["epoch_s"].to_numpy()
    v_lat = viirs["latitude"].to_numpy(dtype=np.float64)
    v_lon = viirs["longitude"].to_numpy(dtype=np.float64)
    v_day = viirs["day_ord"].to_numpy()
    v_clat = viirs["cell_lat"].to_numpy()
    v_clon = viirs["cell_lon"].to_numpy()
    cells: dict[tuple[int, int, int], list[int]] = {}
    for vpos in range(len(viirs)):
        cells.setdefault((int(v_day[vpos]), int(v_clat[vpos]), int(v_clon[vpos])),
                         []).append(vpos)
    cell_index = {k: np.fromiter(v, dtype=np.int32) for k, v in cells.items()}

    m_epoch = modis["epoch_s"].to_numpy()
    m_lat = modis["latitude"].to_numpy(dtype=np.float64)
    m_lon = modis["longitude"].to_numpy(dtype=np.float64)
    m_day = modis["day_ord"].to_numpy()
    m_clat = modis["cell_lat"].to_numpy()
    m_clon = modis["cell_lon"].to_numpy()

    chunks: list[np.ndarray] = []
    npairs = 0
    for mpos in range(len(modis)):
        parts = []
        for delta in (-1, 0, 1):
            for dlat in (-1, 0, 1):
                for dlon in (-1, 0, 1):
                    hit = cell_index.get((int(m_day[mpos]) + delta,
                                          int(m_clat[mpos]) + dlat,
                                          int(m_clon[mpos]) + dlon))
                    if hit is not None:
                        parts.append(hit)
        if not parts:
            continue
        idx = np.unique(np.concatenate(parts))
        dt = np.abs(v_epoch[idx] - int(m_epoch[mpos])) / 3600.0
        keep = dt <= t_exp
        if not keep.any():
            continue
        idx = idx[keep]
        dt = dt[keep]
        dist = haversine_km(np.full(idx.shape, m_lat[mpos]),
                            np.full(idx.shape, m_lon[mpos]),
                            v_lat[idx], v_lon[idx]).astype(np.float32)
        in_window = dist <= exp_tol
        if not in_window.any():
            continue
        idx = idx[in_window]
        dt = dt[in_window].astype(np.float32)
        dist = dist[in_window]
        core = (dist <= tol) & (dt <= t_core)
        score = np.maximum(dist / tol, dt / t_core).astype(np.float32)
        block = np.empty(len(idx), dtype=_PAIR_DTYPE)
        block["mpos"] = np.int32(mpos)
        block["vpos"] = idx.astype(np.int32)
        block["dist"] = dist
        block["dt"] = dt
        block["score"] = score
        block["core"] = core
        # Distances beyond the expanded window never enter candidacy.
        chunks.append(block)
        npairs += len(block)
    stats["candidate_pairs"] = int(npairs)
    stats["distance_computations"] = int(npairs)
    if not chunks:
        return np.empty(0, dtype=_PAIR_DTYPE)
    pairs = np.concatenate(chunks)
    order = np.lexsort((pairs["vpos"], pairs["score"], pairs["mpos"]))
    return pairs[order]


class _PairIndex:
    """Sorted pair access per detection without materializing big dicts."""

    def __init__(self, pairs: np.ndarray):
        self.pairs = pairs
        self._m_bounds = self._bounds(pairs["mpos"]) if len(pairs) else {}
        v_order = np.lexsort((pairs["mpos"], pairs["score"], pairs["vpos"])) \
            if len(pairs) else np.empty(0, dtype=np.int64)
        self._v_order = v_order
        self._v_bounds = self._bounds(pairs["vpos"][v_order]) if len(pairs) else {}

    @staticmethod
    def _bounds(keys: np.ndarray) -> dict[int, tuple[int, int]]:
        bounds: dict[int, tuple[int, int]] = {}
        if not len(keys):
            return bounds
        edges = np.nonzero(np.diff(keys.astype(np.int64)))[0] + 1
        starts = np.concatenate([[0], edges])
        ends = np.concatenate([edges, [len(keys)]])
        for start, end in zip(starts.tolist(), ends.tolist()):
            bounds[int(keys[start])] = (start, end)
        return bounds

    def for_modis(self, mpos: int) -> np.ndarray:
        span = self._m_bounds.get(mpos)
        return self.pairs[span[0]:span[1]] if span else self.pairs[:0]

    def for_viirs(self, vpos: int) -> np.ndarray:
        span = self._v_bounds.get(vpos)
        return self.pairs[self._v_order[span[0]:span[1]]] if span else self.pairs[:0]


def _evidence(pair_score: float, mrow: pd.Series, vrow: pd.Series,
              params: MatchParams) -> str:
    m_nn = _near_nadir(mrow["scan"], mrow["track"], params.footprint_km["MODIS"])
    v_nn = _near_nadir(vrow["scan"], vrow["track"], params.footprint_km["VIIRS"])
    if (pair_score <= params.high_evidence_score and m_nn and v_nn
            and mrow["daynight"] == vrow["daynight"]):
        return "high"
    return "medium"


def match_frames(modis_df: pd.DataFrame, viirs_df: pd.DataFrame,
                 params: MatchParams | None = None,
                 exec_id: str = "") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the full matcher. Returns (groups, members, stats)."""
    params = params or MatchParams()
    stats: dict = {"candidate_pairs": 0, "distance_computations": 0,
                   "modis_in": len(modis_df), "viirs_in": len(viirs_df)}
    overlap = set(modis_df["detection_id"]) & set(viirs_df["detection_id"])
    if overlap:
        raise ValueError(f"detection_id present in both inputs: {sorted(overlap)[:5]}")
    modis = _prepare(modis_df, "MODIS")
    viirs = _prepare(viirs_df, "VIIRS")
    m_ids = modis["detection_id"].to_numpy()
    v_ids = viirs["detection_id"].to_numpy()
    m_pos = {did: i for i, did in enumerate(m_ids.tolist())}
    v_pos = {did: i for i, did in enumerate(v_ids.tolist())}
    pairs = _build_pairs(modis, viirs, params, stats)
    index = _PairIndex(pairs)

    def core_of(rows: np.ndarray) -> np.ndarray:
        return rows[rows["core"]]

    def best_of(rows: np.ndarray):
        core = core_of(rows)
        return core[0] if len(core) else None

    matched_group: dict[str, str] = {}
    groups: dict[str, dict] = {}
    gid = 0

    def _pair_records(members_m: list[int], members_v: list[int]) -> list[dict]:
        vset = set(members_v)
        out = []
        for mpos in members_m:
            for row in core_of(index.for_modis(mpos)):
                if int(row["vpos"]) in vset:
                    out.append({"modis": m_ids[int(row["mpos"])],
                                "viirs": v_ids[int(row["vpos"])],
                                "dist_km": round(float(row["dist"]), 4),
                                "dt_h": round(float(row["dt"]), 3),
                                "score": round(float(row["score"]), 4)})
        out.sort(key=lambda p: (p["score"], p["modis"], p["viirs"]))
        return out

    def new_group(members_m: list[str], members_v: list[str], classification: str,
                  evidence: str, notes: list[str],
                  extra_pairs: list[dict] | None = None,
                  candidates: list[dict] | None = None) -> str:
        nonlocal gid
        gid += 1
        match_id = f"mg{gid:07d}"
        mpos = [m_pos[d] for d in members_m]
        vpos = [v_pos[d] for d in members_v]
        all_pairs = _pair_records(mpos, vpos) + (extra_pairs or [])
        groups[match_id] = {
            "match_id": match_id, "classification": classification,
            "modis_ids": json.dumps(sorted(members_m)),
            "viirs_ids": json.dumps(sorted(members_v)),
            "pairs": json.dumps(all_pairs),
            "candidates": json.dumps(candidates or []),
            "evidence": evidence, "notes": json.dumps(notes),
            "algo_version": ALGO_VERSION, "params_version": params.version,
            "exec_id": exec_id,
        }
        for did in list(members_m) + list(members_v):
            matched_group[did] = match_id
        return match_id

    def top_candidates(did: str, limit: int = 10) -> list[dict]:
        """Best-first candidate evidence (bounded; full counts in members)."""
        is_modis = did in m_pos
        rows = index.for_modis(m_pos[did]) if is_modis else index.for_viirs(v_pos[did])
        out = []
        for row in rows[:limit]:
            partner = (v_ids[int(row["vpos"])] if is_modis else m_ids[int(row["mpos"])])
            out.append({"partner": partner, "dist_km": round(float(row["dist"]), 4),
                        "dt_h": round(float(row["dt"]), 3),
                        "score": round(float(row["score"]), 4),
                        "core": bool(row["core"])})
        return out

    # Pass 1: mutual-best pairs.
    for mid in sorted(m_pos):
        if mid in matched_group:
            continue
        top = best_of(index.for_modis(m_pos[mid]))
        if top is None:
            continue
        vid = v_ids[int(top["vpos"])]
        if vid in matched_group:
            continue
        vtop = best_of(index.for_viirs(v_pos[vid]))
        if vtop is not None and m_ids[int(vtop["mpos"])] == mid:
            ev = _evidence(float(top["score"]), modis.iloc[m_pos[mid]],
                           viirs.iloc[v_pos[vid]], params)
            new_group([mid], [vid], "matched", ev, ["mutual-best"])

    # Pass 2: uncontested bests attach to coherent groups (1-many / many-1).
    def uncontested(rows: np.ndarray) -> bool:
        core = core_of(rows)
        if len(core) <= 1:
            return True
        return bool(core[1]["score"] > core[0]["score"] * (1.0 + params.compete_margin))

    def group_coherent(did: str, is_modis: bool, match_id: str) -> bool:
        opposite = (json.loads(groups[match_id]["viirs_ids"]) if is_modis
                    else json.loads(groups[match_id]["modis_ids"]))
        mine = {r["partner"] for r in
                [{"partner": (v_ids[int(x["vpos"])] if is_modis else m_ids[int(x["mpos"])])}
                 for x in core_of(index.for_modis(m_pos[did]) if is_modis
                                  else index.for_viirs(v_pos[did]))]}
        return set(opposite) <= mine

    for did in sorted(set(m_pos) | set(v_pos)):
        if did in matched_group:
            continue
        is_modis = did in m_pos
        rows = index.for_modis(m_pos[did]) if is_modis else index.for_viirs(v_pos[did])
        top = best_of(rows)
        if top is None or not uncontested(rows):
            continue
        partner = (v_ids[int(top["vpos"])] if is_modis else m_ids[int(top["mpos"])])
        if partner not in matched_group:
            continue
        target = matched_group[partner]
        if groups[target]["classification"] != "matched":
            continue
        if not group_coherent(did, is_modis, target):
            continue
        g = groups[target]
        m_ids_list = json.loads(g["modis_ids"])
        v_ids_list = json.loads(g["viirs_ids"])
        (m_ids_list if is_modis else v_ids_list).append(did)
        for old in list(m_ids_list) + list(v_ids_list):
            matched_group.pop(old, None)
        del groups[target]
        # Fresh id (never reused: the counter only moves forward, so ids stay
        # unique; attach order is sorted, so ids stay deterministic).
        new_group(sorted(m_ids_list), sorted(v_ids_list), "matched", g["evidence"],
                  json.loads(g["notes"]) + ["secondary-attach"])

    # Pass 3: ambiguous groups = connected components over core edges.
    leftover = sorted(d for d in (set(m_pos) | set(v_pos)) if d not in matched_group)
    parent: dict[str, str] = {d: d for d in leftover}

    def find(d: str) -> str:
        while parent[d] != d:
            parent[d] = parent[parent[d]]
            d = parent[d]
        return d

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    def partners(did: str) -> list[str]:
        is_modis = did in m_pos
        rows = index.for_modis(m_pos[did]) if is_modis else index.for_viirs(v_pos[did])
        return [(v_ids[int(x["vpos"])] if is_modis else m_ids[int(x["mpos"])])
                for x in core_of(rows)]

    for did in leftover:
        for partner in partners(did):
            if partner not in matched_group:
                union(did, partner)
    components: dict[str, list[str]] = {}
    for did in leftover:
        components.setdefault(find(did), []).append(did)
    for root in sorted(components, key=lambda r: sorted(components[r])[0]):
        members_sorted = sorted(components[root])
        if not any(partners(d) for d in members_sorted):
            continue
        if len(members_sorted) == 1:
            did = members_sorted[0]
            cands = partners(did)
            if all(p in matched_group for p in cands):
                note = f"partners-matched-elsewhere:{len(cands)}"
            else:
                note = f"contested-partner:{cands[0]}"
            new_group([did] if did in m_pos else [], [] if did in m_pos else [did],
                      "ambiguous", "low", [note], candidates=top_candidates(did))
        else:
            mods = sorted(d for d in members_sorted if d in m_pos)
            virs = sorted(d for d in members_sorted if d in v_pos)
            new_group(mods, virs, "ambiguous", "low",
                      [f"members:{len(members_sorted)}", "multi-candidate-component"])

    # Pass 4: mutual expanded-best pairs -> weak groups; rest -> singletons.
    def expanded_best(did: str):
        is_modis = did in m_pos
        rows = index.for_modis(m_pos[did]) if is_modis else index.for_viirs(v_pos[did])
        rest = rows[~rows["core"]]
        if not len(rest):
            return None
        top = rest[0]
        partner = (v_ids[int(top["vpos"])] if is_modis else m_ids[int(top["mpos"])])
        return {"partner": partner, "dist_km": round(float(top["dist"]), 4),
                "dt_h": round(float(top["dt"]), 3), "score": round(float(top["score"]), 4)}

    for did in leftover:
        if did in matched_group:
            continue
        top = expanded_best(did)
        if top is None:
            continue
        partner = top["partner"]
        ptop = expanded_best(partner)
        if (partner not in matched_group and ptop is not None
                and ptop["partner"] == did):
            pair = sorted([did, partner])
            mods = [d for d in pair if d in m_pos]
            virs = [d for d in pair if d in v_pos]
            weak = {"modis": mods[0], "viirs": virs[0], "dist_km": top["dist_km"],
                    "dt_h": top["dt_h"], "score": top["score"], "window": "expanded"}
            new_group(mods, virs, "ambiguous", "low",
                      ["mutual-expanded-best-weak"], extra_pairs=[weak])
        else:
            n_exp = int((~index.for_modis(m_pos[did])["core"]).sum()) \
                if did in m_pos else int((~index.for_viirs(v_pos[did])["core"]).sum())
            new_group([did] if did in m_pos else [], [] if did in m_pos else [did],
                      "ambiguous", "low", [f"expanded-window-only:{n_exp}"])

    members = []
    for did in sorted(set(m_pos) | set(v_pos)):
        is_modis = did in m_pos
        if did in matched_group:
            classification = groups[matched_group[did]]["classification"]
        else:
            classification = "MODIS-only" if is_modis else "VIIRS-only"
        rows = index.for_modis(m_pos[did]) if is_modis else index.for_viirs(v_pos[did])
        top = best_of(rows)
        members.append({
            "detection_id": did,
            "product": "modis" if is_modis else "viirs-snpp",
            "match_id": matched_group.get(did),
            "classification": classification,
            "best_score": round(float(top["score"]), 4) if top is not None else None,
            "n_core_candidates": int(core_of(rows).shape[0]),
            "n_expanded_candidates": int(rows[~rows["core"]].shape[0]),
            "algo_version": ALGO_VERSION, "params_version": params.version,
            "exec_id": exec_id,
        })

    groups_df = pd.DataFrame(sorted(groups.values(), key=lambda g: g["match_id"]))
    members_df = pd.DataFrame(members)
    stats["groups_matched"] = int((groups_df["classification"] == "matched").sum()) \
        if len(groups_df) else 0
    stats["groups_ambiguous"] = int((groups_df["classification"] == "ambiguous").sum()) \
        if len(groups_df) else 0
    stats["singletons"] = int(members_df["classification"].isin(
        ["MODIS-only", "VIIRS-only"]).sum())
    assert len(members_df) == len(modis_df) + len(viirs_df)
    assert members_df["detection_id"].nunique() == len(members_df)
    logger.info("matched %d+%d -> %d matched groups, %d ambiguous, %d singletons "
                "(%d pairs)", len(modis_df), len(viirs_df), stats["groups_matched"],
                stats["groups_ambiguous"], stats["singletons"], stats["candidate_pairs"])
    return groups_df, members_df, stats

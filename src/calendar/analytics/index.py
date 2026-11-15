"""Phase 2E: Dual Fire Activity Index (transparent views, no opaque score).

Sensor-specific view and harmonized view are built from the same verified
frames; every figure reconciles to underlying observations (see
:func:`reconcile`). A composite numerical score is NOT computed: the
proposed formula is documented below, but weights are unset and
:func:`composite_index` refuses to run without explicit, recorded weights.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

ALGO_VERSION = "dual_index_v1"

# Proposed (NOT approved, NOT applied) composite formula, recorded so a
# future calibration decision is explicit and reviewable:
#   index = w_det * (det / det_baseline)
#         + w_clu * (clu / clu_baseline)
#         + w_frp * (frp / frp_baseline)
# with w_det + w_clu + w_frp == 1. No defaults exist on purpose.
PROPOSED_COMPOSITE_FORMULA = (
    "index = w_det*(det/det_baseline) + w_clu*(clu/clu_baseline) + "
    "w_frp*(frp/frp_baseline); weights unset (see composite_index)"
)


class UncalibratedComposite(RuntimeError):
    """Raised when a composite is requested without explicit weights."""


def sensor_view(members_df: pd.DataFrame, detections_df: pd.DataFrame) -> dict:
    """Per-product breakdown: raw, matched/ambiguous/singleton, FRP."""
    det = detections_df.set_index("detection_id")
    view: dict[str, dict] = {}
    for product, grp in members_df.groupby("product", sort=True):
        frp = pd.to_numeric(
            grp["detection_id"].map(det["frp"]), errors="coerce")
        classes = grp["classification"].value_counts().to_dict()
        view[product] = {
            "raw_observations": int(len(grp)),
            "matched_members": int(classes.get("matched", 0)),
            "ambiguous_members": int(classes.get("ambiguous", 0)),
            "singletons": int(classes.get("MODIS-only", 0) + classes.get("VIIRS-only", 0)),
            "frp_sum": round(float(frp.sum()), 3),
            "frp_n": int(frp.notna().sum()),
            "frp_missing": int(frp.isna().sum()),
        }
    return view


def harmonized_view(members_df: pd.DataFrame, match_groups_df: pd.DataFrame,
                    clusters_df: pd.DataFrame) -> dict:
    """Harmonized activity: groups once, ambiguous individually, singles once."""
    matched_groups = match_groups_df[match_groups_df["classification"] == "matched"]
    amb = int((members_df["classification"] == "ambiguous").sum())
    singles = int(members_df["classification"].isin(["MODIS-only", "VIIRS-only"]).sum())
    sizes = clusters_df["n_detections"] if len(clusters_df) else pd.Series([0])
    return {
        "matched_groups": int(len(matched_groups)),
        "ambiguous_members": amb,
        "singletons": singles,
        "harmonized_observations": int(len(matched_groups) + amb + singles),
        "matching_composition": members_df["classification"].value_counts().to_dict(),
        "clusters": int(len(clusters_df)),
        "cluster_size_min": int(sizes.min()),
        "cluster_size_max": int(sizes.max()),
        "cluster_size_mean": round(float(sizes.mean()), 2),
    }


def reconcile(*, n_input: int, members_df: pd.DataFrame,
              match_groups_df: pd.DataFrame, clusters_df: pd.DataFrame,
              membership_df: pd.DataFrame) -> dict:
    """Verify every conservation law. Returns {passed, checks[]}."""
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    check("members-cover-inputs",
          len(members_df) == n_input and members_df["detection_id"].nunique() == n_input,
          f"members={len(members_df)} inputs={n_input}")
    matched = members_df[members_df["classification"] == "matched"]
    ngroups = len(match_groups_df[match_groups_df["classification"] == "matched"])
    check("matched-members-reference-groups",
          set(matched["match_id"].unique()) <= set(match_groups_df["match_id"].unique())
          and matched["match_id"].isna().sum() == 0,
          f"matched={len(matched)} groups={ngroups}")
    check("membership-covers-inputs-once",
          len(membership_df) == n_input
          and membership_df["detection_id"].nunique() == n_input,
          f"membership={len(membership_df)}")
    check("membership-links-clusters",
          set(membership_df["cluster_id"].unique()) <= set(clusters_df["cluster_id"].unique()),
          f"clusters={len(clusters_df)}")
    check("cluster-sizes-sum-to-inputs",
          int(clusters_df["n_detections"].sum()) == n_input,
          f"sum={int(clusters_df['n_detections'].sum())}")
    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def composite_index(components: dict, baselines: dict,
                    weights: dict | None = None) -> dict:
    """Proposed composite score. REFUSES without explicit weights.

    *components*: {"det": x, "clu": y, "frp": z}; *baselines*: same keys > 0;
    *weights*: {"w_det","w_clu","w_frp"} summing to 1. The weights used are
    echoed in the output — silent defaults are impossible by construction.
    """
    if weights is None:
        raise UncalibratedComposite(
            "no approved weights exist; pass explicit weights "
            f"({PROPOSED_COMPOSITE_FORMULA})")
    if set(weights) != {"w_det", "w_clu", "w_frp"}:
        raise UncalibratedComposite(f"weights must be exactly w_det/w_clu/w_frp: {weights}")
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise UncalibratedComposite(f"weights must sum to 1: {weights}")
    if any(baselines.get(k, 0) <= 0 for k in ("det", "clu", "frp")):
        raise UncalibratedComposite(f"baselines must be positive: {baselines}")
    score = (weights["w_det"] * components["det"] / baselines["det"]
             + weights["w_clu"] * components["clu"] / baselines["clu"]
             + weights["w_frp"] * components["frp"] / baselines["frp"])
    return {"index": round(float(score), 6), "weights": dict(weights),
            "components": dict(components), "baselines": dict(baselines),
            "formula": PROPOSED_COMPOSITE_FORMULA}

"""Phase 4D: anomaly confidence components + deterministic explanations.

Confidence here = quality and statistical support of the calculation. It is
NOT a wildfire probability, danger level, or incident likelihood. No
composite score exists: component indicators are exposed separately (a
proposed composite methodology is documented for human review, and the
scaffold refuses to compute without explicit recorded weights).
"""

from __future__ import annotations

import logging

from .baselines import MEDIAN_MIN_N, ZSCORE_MIN_N

logger = logging.getLogger(__name__)

ALGO_VERSION = "confidence_v1"

DISCLAIMER = ("Anomaly confidence describes statistical support, not the "
              "probability of a wildfire, fire danger, or a real incident.")

METRIC_CAVEATS = {
    "detection": "Repeat observations of the same fire are included; counts are observations, not fires.",
    "cluster": "Clusters are observation groups (single-linkage chaining possible), not individual wildfires.",
    "frp": "Sums of reported per-observation FRP; repeated observations included, not total emitted energy.",
}

# Proposed (NOT approved, NOT applied) composite methodology, recorded for
# human review. Sub-scores would each be normalized to [0,1]:
#   n_score = min(1, reference_n / 30)            (history depth)
#   coverage_score = reference_coverage or 0      (trailing window fill)
#   variance_score = 1 if z valid else 0          (variability usable)
#   composite = w_n*n_score + w_cov*coverage_score + w_var*variance_score
# with w_n + w_cov + w_var == 1. No default weights exist on purpose.
PROPOSED_COMPOSITE_FORMULA = (
    "composite = w_n*min(1,n/30) + w_cov*coverage + w_var*(z valid); "
    "weights unset (see composite_confidence)"
)


class UncalibratedConfidence(RuntimeError):
    """Raised when a composite is requested without explicit weights."""


def components(record: dict) -> dict:
    """Transparent confidence components for one anomaly result."""
    n = record.get("reference_n") or 0
    z_status = record.get("z_status")
    eligibility = record.get("eligibility", "insufficient-history")
    if z_status == "valid":
        variance_state = "stable"
    elif eligibility == "unstable":
        variance_state = "unstable"
    else:
        variance_state = "not-enough-data"
    scope = record.get("scope", "")
    if scope in ("modis", "viirs-snpp", "Terra", "Aqua", "N", "viirs"):
        sensor_note = "single-sensor scope; no cross-sensor borrowing"
    elif scope in ("raw", "count"):
        sensor_note = "all-sensor scope (pooled observations)"
    elif scope == "harmonized":
        sensor_note = "harmonized scope per documented Phase 2E definition"
    else:
        sensor_note = "scope note unavailable"
    return {
        "reference_n": n,
        "median_gate": "pass" if n >= MEDIAN_MIN_N else "fail",
        "z_gate": "pass" if n >= ZSCORE_MIN_N else "fail",
        "reference_coverage": record.get("reference_coverage"),
        "variance_state": variance_state,
        "baseline_status": eligibility,
        "sensor_note": sensor_note,
        "metric_caveat": METRIC_CAVEATS.get(record.get("metric", ""),
                                            "metric caveat unavailable"),
        "history_depth": "single annual cycle (2023 pilot); no multi-year context",
        "disclaimer": DISCLAIMER,
        "algo_version": ALGO_VERSION,
    }


def composite_confidence(comp: dict, weights: dict | None = None) -> dict:
    """Proposed composite score. REFUSES without explicit weights.

    *comp*: components() output; *weights*: {"w_n","w_cov","w_var"} summing
    to 1. The weights used are echoed — silent defaults are impossible.
    """
    if weights is None:
        raise UncalibratedConfidence(
            "no approved weights exist; pass explicit weights "
            f"({PROPOSED_COMPOSITE_FORMULA})")
    if set(weights) != {"w_n", "w_cov", "w_var"}:
        raise UncalibratedConfidence(f"weights must be exactly w_n/w_cov/w_var: {weights}")
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise UncalibratedConfidence(f"weights must sum to 1: {weights}")
    n_score = min(1.0, (comp.get("reference_n") or 0) / 30.0)
    coverage_score = comp.get("reference_coverage") or 0.0
    variance_score = 1.0 if comp.get("variance_state") == "stable" else 0.0
    score = (weights["w_n"] * n_score + weights["w_cov"] * coverage_score
             + weights["w_var"] * variance_score)
    return {"composite": round(float(score), 6), "weights": dict(weights),
            "subscores": {"n_score": round(n_score, 4),
                          "coverage_score": round(float(coverage_score), 4),
                          "variance_score": variance_score},
            "formula": PROPOSED_COMPOSITE_FORMULA}


def _fmt(value, suffix: str = "") -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:,.2f}{suffix}"
    return f"{value}{suffix}"


def explain(record: dict) -> str:
    """Deterministic English explanation (templates only, no LLM)."""
    comp = components(record)
    lines = [
        f"Metric: {record.get('metric')} ({record.get('scope')}).",
        f"Evaluated period: {record.get('eval_period')}.",
        f"Baseline: {record.get('baseline_method')} "
        f"({record.get('reference_strategy')}, reference "
        f"{record.get('reference_start')} to {record.get('reference_end')}, "
        f"n={record.get('reference_n')}).",
        f"Observed {_fmt(record.get('observed_value'))} vs baseline "
        f"{_fmt(record.get('baseline_value'))}.",
    ]
    if record.get("pct_status") == "valid":
        lines.append(f"Percentage deviation: {_fmt(record.get('pct_deviation'), '%')} (valid).")
    else:
        lines.append(f"Percentage deviation: unavailable ({record.get('pct_status')}).")
    if record.get("z_status") == "valid":
        lines.append(f"Z-score: {_fmt(record.get('z_score'))} (valid).")
    else:
        lines.append(f"Z-score: unavailable ({record.get('z_status')}).")
    lines.append(f"Classification: {record.get('classification')}.")
    if record.get("eligibility") in ("provisional", "unstable"):
        lines.append(f"Provisional result ({record.get('eligibility')}).")
    lines.append(f"Support: {comp['reference_n']} historical observations; "
                 f"variance {comp['variance_state']}.")
    for limit in record.get("limitations", []) or []:
        lines.append(f"Limitation: {limit}.")
    lines.append(f"Caveat: {comp['metric_caveat']}")
    lines.append(DISCLAIMER)
    return "\n".join(lines)

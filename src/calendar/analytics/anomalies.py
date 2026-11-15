"""Phase 4C: deterministic statistical anomaly detection.

An anomaly = statistically unusual observation activity vs an eligible
historical reference. Never a confirmed wildfire, severity level, or
prediction. Metrics stay independent (no composite scores, no weights);
every result records its full statistical provenance.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

ALGO_VERSION = "anomalies_v1"
CLASSIFY_VERSION = "classify_v1"

# Classification thresholds (classify_v1, documented): 2-sigma is the
# conventional rarity gate; 25% is a practical magnitude floor so a
# statistically rare but tiny wiggle is not called notable. Both must
# hold (when both statistics are valid).
Z_NOTABLE = 2.0
PCT_NOTABLE = 25.0


def pct_deviation(observed: float | None,
                  baseline: float | None) -> tuple[float | None, str]:
    """Percentage deviation (or None + explicit status, never invented)."""
    if observed is None or baseline is None:
        return None, "missing-data"
    if baseline == 0:
        return None, "undefined-zero-baseline"
    return round((observed - baseline) / baseline * 100.0, 4), "valid"


def zscore(observed: float | None, mean: float | None, stdev: float | None,
           z_valid: bool) -> tuple[float | None, str]:
    """Z-score (or None + explicit status; never a silent zero)."""
    if observed is None or mean is None or stdev is None:
        return None, "missing-data"
    if not z_valid:
        return None, "unavailable"
    if stdev <= 0:
        return None, "unstable-variance"  # defense in depth (gated upstream)
    return round((observed - mean) / stdev, 4), "valid"


def classify(*, pct: float | None, pct_status: str, z: float | None,
             z_status: str, eligibility: str, observed: float | None,
             baseline: float | None) -> tuple[str, list[str]]:
    """Transparent statistical classification (neutral language only)."""
    reasons: list[str] = []
    if observed is None:
        return "missing-data", ["no observation for the evaluated period"]
    if eligibility == "insufficient-history":
        return "insufficient-evidence", ["baseline has too few references"]
    if eligibility == "unstable":
        return "unstable", ["reference variance is zero or near-zero"]
    if eligibility not in ("eligible", "provisional"):
        return "indeterminate", [f"unexpected eligibility {eligibility!r}"]
    direction = ("high" if (observed - (baseline or 0.0)) >= 0 else "low")
    if pct_status != "valid" and z_status != "valid":
        return "indeterminate", [f"pct {pct_status}", f"z {z_status}"]
    if z_status == "valid" and pct_status == "valid":
        assert pct is not None and z is not None
        if abs(z) >= Z_NOTABLE and abs(pct) >= PCT_NOTABLE:
            return f"notable-{direction}", [
                f"|z|={abs(z):.2f} >= {Z_NOTABLE}",
                f"|pct|={abs(pct):.1f}% >= {PCT_NOTABLE}%"]
        reasons.append("within typical bounds (|z|<2 or |pct|<25%)")
        return "typical", reasons
    if pct_status == "valid":
        assert pct is not None
        tag = "provisional-notable" if abs(pct) >= PCT_NOTABLE else "provisional-typical"
        reasons.append("median-only evidence (z unavailable)")
        reasons.append(f"|pct|={abs(pct):.1f}% vs {PCT_NOTABLE}%")
        return f"{tag}-{direction}" if tag == "provisional-notable" else tag, reasons
    # z-only (zero baseline): rare but well-defined.
    assert z is not None
    reasons.append("z-only evidence (pct undefined on zero baseline)")
    if abs(z) >= Z_NOTABLE:
        return f"notable-{direction}", reasons + [f"|z|={abs(z):.2f} >= {Z_NOTABLE}"]
    return "typical", reasons + ["|z| < 2"]


def anomaly_id(metric: str, scope: str, method: str, grain: str,
               period: str) -> str:
    """Deterministic readable anomaly identifier."""
    safe_scope = scope.replace("/", "-")
    return f"anom-{metric}-{safe_scope}-{method}-{grain}-{period}"


def detect_from_baseline(record: dict, *, exec_id: str = "") -> dict:
    """One anomaly result from one baseline record (4C.7 fields)."""
    observed, baseline = record.get("observed_value"), record.get("baseline_value")
    pct, pct_status = pct_deviation(observed, baseline)
    z, z_status = zscore(observed, record.get("reference_mean"),
                         record.get("reference_stdev"),
                         bool(record.get("z_valid")))
    classification, reasons = classify(
        pct=pct, pct_status=pct_status, z=z, z_status=z_status,
        eligibility=record.get("eligibility", "insufficient-history"),
        observed=observed, baseline=baseline)
    limitations = list(reasons)
    if record.get("eligibility_reasons"):
        limitations.extend(record["eligibility_reasons"])
    if record.get("reference_strategy") == "trailing":
        limitations.append("single-cycle trailing reference (no multi-year history)")
    return {
        "anomaly_id": anomaly_id(record["metric"], record["scope"],
                                 record["baseline_method"],
                                 record["config"]["grain"],
                                 record["observed_period"]),
        "exec_id": exec_id or record.get("exec_id", ""),
        "metric": record["metric"], "scope": record["scope"],
        "geo_scope": record.get("geo_scope", "USA-pilot"),
        "grain": record.get("config", {}).get("grain"),
        "eval_period": record["observed_period"], "observed_value": observed,
        "baseline_value": baseline,
        "pct_deviation": pct, "pct_status": pct_status,
        "z_score": z, "z_status": z_status,
        "baseline_method": record["baseline_method"],
        "reference_strategy": record.get("reference_strategy"),
        "reference_start": record.get("reference_start"),
        "reference_end": record.get("reference_end"),
        "reference_n": record.get("reference_n"),
        "reference_coverage": record.get("reference_coverage"),
        "classification": classification,
        "eligibility": record.get("eligibility"),
        "limitations": limitations,
        "algo_version": ALGO_VERSION,
        "classify_version": CLASSIFY_VERSION,
        "config_version": record.get("config_version"),
    }


def run_all_anomalies(baseline_records: list[dict],
                      exec_id: str = "") -> list[dict]:
    """Anomaly results for every baseline record (deterministic)."""
    out = [detect_from_baseline(record, exec_id=exec_id)
           for record in baseline_records]
    logger.info("anomalies: %d results", len(out))
    return out

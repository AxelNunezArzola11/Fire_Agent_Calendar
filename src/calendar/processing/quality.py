"""Deterministic data-quality summaries (counts only, no anomaly logic)."""

from __future__ import annotations

import pandas as pd


def quality_report(*, source_rows: int, accepted: pd.DataFrame,
                   rejected: list[dict]) -> dict:
    """Build the quality dict. Pure function; empty-safe (zeros, never crash)."""
    reasons: dict[str, int] = {}
    for item in rejected:
        reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
    report = {
        "source_rows": source_rows,
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "rejected_reasons": reasons,
        "duplicate_detection_ids": (
            int(accepted["detection_id"].duplicated().sum()) if len(accepted) else 0),
        "null_frp": int(accepted["frp"].isna().sum()) if len(accepted) else 0,
        "null_confidence": (
            int((accepted["confidence"] == "").sum()) if len(accepted) else 0),
        "sensor_distribution": (
            accepted["satellite"].value_counts().to_dict() if len(accepted) else {}),
    }
    if len(accepted):
        report["temporal_coverage"] = {
            "min": str(accepted["acquisition_timestamp_utc"].min()),
            "max": str(accepted["acquisition_timestamp_utc"].max()),
        }
        report["geographic_coverage"] = {
            "lon_min": float(accepted["longitude"].min()),
            "lon_max": float(accepted["longitude"].max()),
            "lat_min": float(accepted["latitude"].min()),
            "lat_max": float(accepted["latitude"].max()),
        }
    else:  # missing/empty input is reported as such — never as zeros-activity
        report["temporal_coverage"] = None
        report["geographic_coverage"] = None
        report["note"] = "no accepted rows: empty or fully rejected input"
    return report

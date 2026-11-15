"""Structured Observation Operator output contracts (result records).

Versioned per-observation and aggregate records with explicit units,
per-quantity status vocabulary (observed/derived/modeled/
experimental/unavailable), quality flags, and input completeness.
Unavailable science stays null, never zero. No equations here: this
module labels values computed by operator.py.
"""

from __future__ import annotations

RESULT_VERSION = "det-frp-result-v1"
FRP_UNITS = "MW"
AREA_UNITS = "km2"
RATE_UNITS = "per-MW"

STATUS_OBSERVED = "observed"
STATUS_DERIVED = "derived"
STATUS_MODELED = "modeled"
STATUS_EXPERIMENTAL = "experimental"
STATUS_UNAVAILABLE = "unavailable"


def result_record(row: dict, calc: dict | None, regime: dict | None, *,
                  area_km2=None, reasons: tuple = (),
                  l2: dict | None = None, granule_ur: str | None = None,
                  match_status: str = "unmatched") -> dict:
    """One observation-level operator record (stable key order).

    row: detection fields (detection_id, acquisition_timestamp_utc,
    sensor, satellite, product, latitude, longitude, frp, scan,
    track, daynight). calc: operator.expected_frp output or None.
    regime: operator.detection_regime output or None. area_km2: the
    pixel area used. reasons: exclusion reasons when ineligible. l2:
    matched Level-2 extract or None (all L2-backed fields stay null
    without it). granule_ur/match_status: colocation outcome.
    """
    l2 = l2 or {}
    eligible = calc is not None and regime is not None
    aod = l2.get("aod_550")
    cloud = l2.get("cloud_flag")
    return {
        "observation_id": row.get("detection_id"),
        "acquisition_utc": row.get("acquisition_timestamp_utc"),
        "sensor": row.get("sensor"),
        "satellite": row.get("satellite"),
        "product": row.get("product"),
        "latitude": row.get("latitude"),
        "longitude": row.get("longitude"),
        "observed_frp": row.get("frp"),
        "observed_frp_units": FRP_UNITS,
        "observed_frp_status": STATUS_OBSERVED,
        "detection_probability": (calc or {}).get("p"),
        "detection_probability_status": STATUS_EXPERIMENTAL
        if eligible else STATUS_UNAVAILABLE,
        "detection_limit_MW": (calc or {}).get("Dp"),
        "transition_rate_per_MW": (calc or {}).get("r"),
        "expected_retrieved_frp_MW": (calc or {}).get("E_ret"),
        "expected_frp_status": STATUS_EXPERIMENTAL
        if eligible else STATUS_UNAVAILABLE,
        "pixel_area_km2": area_km2,
        "area_units": AREA_UNITS,
        "area_method": "scan-x-track proxy (exploratory)",
        "area_status": STATUS_EXPERIMENTAL
        if eligible else STATUS_UNAVAILABLE,
        "scan_km": row.get("scan"),
        "track_km": row.get("track"),
        "daynight_flag": row.get("daynight"),
        "regime": (regime or {}).get("regime"),
        "regime_margin_MW": (regime or {}).get("margin_MW"),
        "aod_550": aod,
        "aod_source": l2.get("aod_source"),
        "aod_status": STATUS_UNAVAILABLE if aod is None
        else STATUS_OBSERVED,
        "cloud_flag": cloud,
        "cloud_source": l2.get("cloud_source"),
        "cloud_status": STATUS_UNAVAILABLE if cloud is None
        else STATUS_OBSERVED,
        "l2_granule_id": granule_ur,
        "l2_match_status": match_status,
        "equation_version": "det-frp-oop-v1 (Eq.5/6/7)",
        "result_version": RESULT_VERSION,
        "calculation_status": "ok" if eligible
        else f"excluded: {'; '.join(reasons) or 'unknown'}",
        "quality_flags": list(reasons),
        "inputs_present": sorted(
            k for k, v in (("frp", row.get("frp")),
                           ("scan", row.get("scan")),
                           ("track", row.get("track")),
                           ("daynight", row.get("daynight")),
                           ("aod_550", aod),
                           ("cloud_flag", cloud)) if v is not None),
        "limitations": "E_ret is a detection-discounted diagnostic "
        "conditional on detection; clear-sky assumed; FIRMS-input-as-P "
        "is exploratory; no universal sensor multiplier applied.",
    }


def aggregate_record(records: list[dict]) -> dict:
    """Aggregate operator statistics over result records."""
    eligible = [r for r in records
                if r.get("calculation_status") == "ok"]
    by_sensor: dict[str, dict] = {}
    regimes: dict[str, int] = {}
    for record in eligible:
        sensor = str(record.get("sensor") or "unknown")
        slot = by_sensor.setdefault(
            sensor, {"n": 0, "observed_sum_MW": 0.0,
                     "expected_sum_MW": 0.0})
        slot["n"] += 1
        slot["observed_sum_MW"] += record.get("observed_frp") or 0.0
        slot["expected_sum_MW"] += record.get(
            "expected_retrieved_frp_MW") or 0.0
        label = str(record.get("regime") or "unknown")
        regimes[label] = regimes.get(label, 0) + 1
    return {
        "result_version": RESULT_VERSION,
        "n_observations": len(records),
        "n_eligible": len(eligible),
        "n_excluded": len(records) - len(eligible),
        "by_sensor": by_sensor,
        "regime_histogram": regimes,
        "l2_coverage": {
            "with_aod": sum(1 for r in eligible
                            if r.get("aod_550") is not None),
            "with_cloud": sum(1 for r in eligible
                              if r.get("cloud_flag") is not None),
        },
        "status": STATUS_EXPERIMENTAL,
    }

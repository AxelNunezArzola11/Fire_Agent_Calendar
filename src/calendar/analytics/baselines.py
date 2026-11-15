"""Phase 4B: deterministic historical baseline engine.

Three versioned methods over pinned Phase 2D metric series. Every baseline
is leakage-free by construction (references strictly precede the evaluated
period; the eval period can never enter its own reference set) and carries
explicit eligibility: insufficient history is a first-class honest outcome,
never a silent relaxation or a fabricated value.
"""

from __future__ import annotations

import logging
import statistics as _statistics
from dataclasses import dataclass
from datetime import date

logger = logging.getLogger(__name__)

ALGO_VERSION = "baselines_v1"
CONFIG_VERSION = "baseline_cfg_v1"

METHODS = ("monthly_median", "seasonal", "sensor_aware")
STRATEGIES = ("same_month", "same_week", "trailing")
GRAINS = ("daily", "weekly", "monthly")

# Default reference strategy per method (overridable only via explicit
# config, always recorded in the output — never silently substituted).
DEFAULT_STRATEGY = {"monthly_median": "same_month",
                    "seasonal": "same_week",
                    "sensor_aware": "trailing"}

# Grains served per method under default strategies (group strategies need
# a well-defined calendar group at the grain; explicit trailing configs
# serve every grain).
METHOD_GRAINS = {"monthly_median": ("daily", "monthly"),
                 "seasonal": ("daily", "weekly"),
                 "sensor_aware": ("daily", "weekly", "monthly")}

# Trailing window sizes: a month / a quarter / half a year of recent
# history at each grain (recency-vs-representativeness tradeoff, versioned).
TRAILING_K = {"daily": 30, "weekly": 12, "monthly": 6}

# Eligibility floors (documented, not tuned): median of 3 is the smallest
# median that resists one outlier year; z needs std with >= 6 dof.
MEDIAN_MIN_N = 3
ZSCORE_MIN_N = 7
# Numerical-stability floor for variance (relative): below this, z-scores
# amplify noise, so z is marked unstable rather than reported.
VARIANCE_RTOL = 1e-9


@dataclass(frozen=True)
class BaselineConfig:
    """Versioned baseline configuration (baseline_cfg_v1)."""

    method: str
    grain: str
    strategy: str | None = None  # None -> method default (recorded anyway)
    trailing_k: int | None = None  # None -> grain default (recorded anyway)
    median_min_n: int = MEDIAN_MIN_N
    zscore_min_n: int = ZSCORE_MIN_N
    geo_scope: str = "USA-pilot"
    version: str = CONFIG_VERSION

    def resolved_strategy(self) -> str:
        if self.method not in METHODS:
            raise ValueError(f"unknown method {self.method!r}; want {METHODS}")
        strategy = self.strategy or DEFAULT_STRATEGY[self.method]
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}; want {STRATEGIES}")
        return strategy

    def resolved_k(self) -> int:
        if self.grain not in GRAINS:
            raise ValueError(f"unknown grain {self.grain!r}; want {GRAINS}")
        return self.trailing_k if self.trailing_k is not None else TRAILING_K[self.grain]


def _det_value(row: dict, scope: str):
    if scope == "raw":
        return row["raw_observations"]
    if scope in ("modis", "viirs-snpp"):
        return row["by_product"].get(scope, 0)
    if scope in ("Terra", "Aqua", "N"):
        return row["by_satellite"].get(scope, 0)
    raise ValueError(f"unknown detection scope {scope!r}")


def _clu_value(row: dict, scope: str):
    if scope == "count":
        return row["clusters"]
    if scope == "modis":
        return row["modis_detections"]
    if scope == "viirs":
        return row["viirs_detections"]
    raise ValueError(f"unknown cluster scope {scope!r}")


# (metric, scope) series supported. Harmonized detection comes from the
# harmonized rows; FRP scopes are per-sensor only (the warned combined
# total is never a statistical input).
SERIES_SPECS = (
    ("detection", "raw"), ("detection", "modis"), ("detection", "viirs-snpp"),
    ("detection", "Terra"), ("detection", "Aqua"), ("detection", "N"),
    ("detection", "harmonized"),
    ("cluster", "count"), ("cluster", "modis"), ("cluster", "viirs"),
    ("frp", "modis"), ("frp", "viirs-snpp"),
)


def slot_index(grain: str, period: str) -> int:
    """Continuous calendar slot number (orders periods across year ends)."""
    if grain == "daily":
        return date.fromisoformat(period).toordinal()
    if grain == "weekly":
        year_s, week_s = period.split("-W")
        monday = date.fromisocalendar(int(year_s), int(week_s), 1)
        return monday.toordinal() // 7
    if grain == "monthly":
        year_s, month_s = period.split("-")
        return int(year_s) * 12 + int(month_s)
    raise ValueError(f"unknown grain {grain!r}; want {GRAINS}")


def extract_series(metrics: dict, grain: str, metric: str,
                   scope: str) -> list[dict]:
    """Ordered [{period, slot, value}] for one series (gaps skipped, kept)."""
    if (metric, scope) not in SERIES_SPECS:
        raise ValueError(f"unsupported series {(metric, scope)}")
    if grain not in GRAINS:
        raise ValueError(f"unknown grain {grain!r}; want {GRAINS}")
    pack = metrics[grain]
    entries = []
    if metric == "detection" and scope == "harmonized":
        rows = pack["detection_activity"]["harmonized_rows"]
        for row in rows:
            entries.append({"period": row["period"], "value": row["harmonized_observations"]})
    elif metric == "detection":
        for row in pack["detection_activity"]["rows"]:
            entries.append({"period": row["period"],
                            "value": _det_value(row, scope)})
    elif metric == "cluster":
        for row in pack["cluster_activity"]["rows"]:
            entries.append({"period": row["period"],
                            "value": _clu_value(row, scope)})
    else:  # frp per-sensor rows
        product = "modis" if scope == "modis" else "viirs-snpp"
        for row in pack["frp_summary"]["rows"]:
            if row["product"] == product:
                entries.append({"period": row["period"], "value": row["frp_sum"]})
    for entry in entries:
        entry["slot"] = slot_index(grain, entry["period"])
    entries.sort(key=lambda e: e["slot"])
    return entries


def group_key(grain: str, period: str, strategy: str) -> tuple[int, str]:
    """(year, group) key for cross-cycle matching (loud on invalid combos).

    same_month is defined for daily/monthly grains (calendar month);
    same_week for weekly/daily grains (ISO week). Weekly periods have no
    clean month attribution and monthly periods have no week structure, so
    those combinations are rejected rather than approximated.
    """
    if strategy == "same_month":
        if grain in ("daily", "monthly"):
            return (int(period[:4]), period[5:7])
        raise ValueError(f"same_month is undefined for {grain!r} grain")
    if strategy == "same_week":
        if grain == "weekly":
            year_s, week_s = period.split("-W")
            return (int(year_s), week_s)
        if grain == "daily":
            iso_year, iso_week, _ = date.fromisoformat(period).isocalendar()
            return (iso_year, f"{iso_week:02d}")
        raise ValueError(f"same_week is undefined for {grain!r} grain")
    raise ValueError(f"group keys apply to same_month/same_week, not {strategy!r}")


def reference_set(entries: list[dict], eval_pos: int,
                  cfg: BaselineConfig) -> list[dict]:
    """Eligible references: strictly before eval_pos (leakage impossible).

    same_month / same_week match the calendar group in PRIOR cycles only
    (same-year earlier days are current-cycle recency, not history — they
    belong to the trailing strategy); trailing takes available entries
    from the K preceding calendar slots (missing slots reduce coverage,
    never zero-fill).
    """
    strategy = cfg.resolved_strategy()
    target = entries[eval_pos]
    refs = []
    if strategy == "trailing":
        lo = target["slot"] - cfg.resolved_k()
        for cand in entries[:eval_pos]:
            if lo <= cand["slot"] < target["slot"]:
                refs.append(cand)
        return refs
    want_year, want_group = group_key(cfg.grain, target["period"], strategy)
    for cand in entries[:eval_pos]:
        year, group = group_key(cfg.grain, cand["period"], strategy)
        if group == want_group and year < want_year:
            refs.append(cand)
    return refs


def summarize(values: list) -> dict:
    """Reference distribution summary (exact, deterministic)."""
    n = len(values)
    if n == 0:
        return {"n": 0, "median": None, "mean": None, "stdev": None}
    mean = _statistics.mean(values)
    return {"n": n,
            "median": float(_statistics.median(values)),
            "mean": float(mean),
            "stdev": float(_statistics.stdev(values)) if n >= 2 else None}


def assess_eligibility(summary: dict, cfg: BaselineConfig) -> dict:
    """Eligibility verdict + reasons (median gate, z gate, variance floor)."""
    n, stdev, mean, median = (summary["n"], summary["stdev"],
                              summary["mean"], summary["median"])
    reasons: list[str] = []
    baseline_ok = n >= cfg.median_min_n
    if not baseline_ok:
        reasons.append(f"insufficient-history: n={n} < {cfg.median_min_n}")
    z_ok = True
    if n < cfg.zscore_min_n:
        z_ok = False
        reasons.append(f"zscore-needs-n>={cfg.zscore_min_n}: n={n}")
    elif stdev is None or stdev <= 0:
        z_ok = False
        reasons.append("unstable-variance: stdev<=0")
    else:
        scale = max(1.0, abs(mean or 0.0), abs(median or 0.0))
        if stdev < VARIANCE_RTOL * scale:
            z_ok = False
            reasons.append("unstable-variance: near-zero stdev")
    if baseline_ok and z_ok:
        status = "eligible"
    elif baseline_ok:
        status = "unstable" if any(r.startswith("unstable-variance") for r in reasons) else "provisional"
    else:
        status = "insufficient-history"
    return {"baseline_ok": baseline_ok, "z_ok": z_ok, "status": status,
            "reasons": reasons}


def evaluate_period(entries: list[dict], eval_pos: int, cfg: BaselineConfig,
                    *, metric: str, scope: str, exec_id: str = "") -> dict:
    """One baseline record for one evaluated period (4B.5 fields)."""
    strategy = cfg.resolved_strategy()
    trailing_k = cfg.resolved_k()
    target = entries[eval_pos]
    refs = reference_set(entries, eval_pos, cfg)
    summary = summarize([r["value"] for r in refs])
    verdict = assess_eligibility(summary, cfg)
    if strategy == "trailing":
        coverage: float | None = round(len(refs) / trailing_k, 4) if trailing_k else 0.0
    else:
        coverage = None  # window fill is meaningless for group strategies
    return {
        "metric": metric, "scope": scope,
        "observed_period": target["period"], "observed_value": target["value"],
        "baseline_value": summary["median"] if verdict["baseline_ok"] else None,
        "baseline_method": cfg.method, "reference_strategy": strategy,
        "reference_start": refs[0]["period"] if refs else None,
        "reference_end": refs[-1]["period"] if refs else None,
        "reference_n": summary["n"], "reference_median": summary["median"],
        "reference_mean": summary["mean"], "reference_stdev": summary["stdev"],
        "reference_coverage": coverage,
        "sensor_scope": scope, "geo_scope": cfg.geo_scope,
        "eligibility": verdict["status"], "eligibility_reasons": verdict["reasons"],
        "z_valid": verdict["z_ok"],
        "algo_version": ALGO_VERSION, "config_version": cfg.version,
        "config": {"method": cfg.method, "grain": cfg.grain,
                   "strategy": strategy, "trailing_k": trailing_k,
                   "median_min_n": cfg.median_min_n,
                   "zscore_min_n": cfg.zscore_min_n},
        "exec_id": exec_id,
    }


def run_all_baselines(metrics: dict, *, methods: tuple[str, ...] = METHODS,
                      grains: tuple[str, ...] = GRAINS,
                      exec_id: str = "") -> list[dict]:
    """Baselines for every method x grain x series x period (deterministic).

    Only method-compatible grains run (METHOD_GRAINS); anything else would
    be an undefined calendar grouping, so it is excluded, not approximated.
    """
    records = []
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; want {METHODS}")
        for grain in grains:
            if grain not in METHOD_GRAINS[method]:
                continue
            cfg = BaselineConfig(method=method, grain=grain)
            for metric, scope in SERIES_SPECS:
                entries = extract_series(metrics, grain, metric, scope)
                for pos in range(len(entries)):
                    records.append(evaluate_period(
                        entries, pos, cfg, metric=metric, scope=scope,
                        exec_id=exec_id))
    logger.info("baselines: %d records (%d methods x %d grains x %d series)",
                len(records), len(methods), len(grains), len(SERIES_SPECS))
    return records

"""Two-day deterministic FRP pilot runner (read-only inputs).

Usage:
    .venv/bin/python -m src.calendar.experimental.frp_operator_pilot.run_pilot \
        --dates 2023-08-15 2023-08-16 --outdir pilot_outputs

Reads calendar.duckdb read-only. Writes only --outdir (CSV, JSON)
and reports/FRP_2_DAY_DETERMINISTIC_PILOT_REPORT.md. Prints the
terminal report to stdout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import duckdb

from src.calendar.experimental.frp_operator_pilot import (
    coefficients, lookup, operator, validation)

METHOD_VERSION = "det-frp-oop-v1"


def _db_hash(db_path: str) -> str:
    digest = hashlib.sha256()
    with open(db_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rows(db_path: str, dates: list[str]) -> list[dict]:
    con = duckdb.connect(db_path, read_only=True)
    try:
        rows = con.execute(
            """SELECT detection_id, product, satellite, daynight, frp,
                      scan, track, latitude, longitude,
                      acquisition_timestamp_utc,
                      substr(acquisition_timestamp_utc,1,10) AS day
               FROM detections
               WHERE substr(acquisition_timestamp_utc,1,10) IN (?, ?)
               ORDER BY day, product, detection_id""",
            dates).fetchall()
        cols = [d[0] for d in con.description]
    finally:
        con.close()
    return [dict(zip(cols, r)) for r in rows]


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "sum": 0.0, "mean": None, "median": None,
                "max": None}
    return {"n": len(values), "sum": float(sum(values)),
            "mean": float(statistics.mean(values)),
            "median": float(statistics.median(values)),
            "max": float(max(values))}


def run(dates: list[str], outdir: str, db_path: str = "calendar.duckdb",
        run_tests: bool = True) -> dict:
    """Execute the pilot; returns the full results record."""
    t0 = time.monotonic()
    started = datetime.now(timezone.utc).isoformat()
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    hash_before = _db_hash(db_path)
    rows = _load_rows(db_path, dates)
    results, excluded = [], []
    for row in rows:
        verdict = validation.validate_row(row)
        if not verdict["eligible"]:
            excluded.append({"detection_id": row["detection_id"],
                             "reasons": verdict["reasons"]})
            continue
        calc = operator.expected_frp(row["frp"], verdict["area"],
                                     row["product"], row["daynight"])
        results.append({**row, "pixel_area_km2": verdict["area"], **calc})
    # Aggregates: observed (A), expected (B), sensor (C), daily (D).
    observed = [r["frp"] for r in results]
    expected = [r["E_ret"] for r in results]
    per_sensor, per_day = {}, {}
    for sensor in ("modis", "viirs-snpp"):
        sub = [r for r in results if r["product"] == sensor]
        per_sensor[sensor] = {
            "n": len(sub),
            "observed_frp": _stats([r["frp"] for r in sub]),
            "expected_frp": _stats([r["E_ret"] for r in sub])}
    for day in dates:
        sub = [r for r in results if r["day"] == day]
        per_day[day] = {
            "n": len(sub),
            "observed_frp": _stats([r["frp"] for r in sub]),
            "expected_frp": _stats([r["E_ret"] for r in sub]),
            "by_sensor": {s: {"n": sum(1 for r in sub if r["product"] == s)}
                          for s in ("modis", "viirs-snpp")}}
    obs_sum = float(sum(observed))
    exp_sum = float(sum(expected))
    table = lookup.build_lookup()
    # Persist experimental outputs (isolated directory only).
    with open(out / "pilot_rows.csv", "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["detection_id", "day", "product", "satellite",
                         "daynight", "frp observed_MW", "pixel_area_km2",
                         "Dp_MW", "r_per_MW", "p", "E_ret_MW"])
        for r in results:
            writer.writerow([r["detection_id"], r["day"], r["product"],
                             r["satellite"], r["daynight"], r["frp"],
                             round(r["pixel_area_km2"], 4),
                             round(r["Dp"], 4), round(r["r"], 4),
                             round(r["p"], 6), round(r["E_ret"], 4)])
    with open(out / "lookup_table.json", "w", encoding="utf-8") as handle:
        json.dump(table, handle, indent=1, sort_keys=True)
    # Geometry validation vs published anchors (subset-tolerant bands).
    areas = {"modis": [r["pixel_area_km2"] for r in results
                       if r["product"] == "modis"],
             "viirs-snpp": [r["pixel_area_km2"] for r in results
                            if r["product"] == "viirs-snpp"]}
    modis_frac = (sum(1 for a in areas["modis"] if 1.0 <= a <= 1.2)
                  / max(1, len(areas["modis"])))
    viirs_frac = (sum(1 for a in areas["viirs-snpp"] if a < 0.3)
                  / max(1, len(areas["viirs-snpp"])))
    geometry = {
        "modis_frac_1_12": round(modis_frac, 4),
        "viirs_frac_lt03": round(viirs_frac, 4),
        "modis_ok": 0.15 <= modis_frac <= 0.65,
        "viirs_ok": 0.50 <= viirs_frac <= 0.95,
        "bad_geometry_rows": sum(
            1 for e in excluded if any("pixel area" in reason
                                       for reason in e["reasons"])),
    }
    hash_after = _db_hash(db_path)
    test_summary = {"ran": False}
    if run_tests:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest",
             "tests/test_frp_operator_pilot.py", "-q", "-p", "no:cacheprovider"],
            cwd=str(REPO), capture_output=True, text=True, timeout=600)
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-3:]
        test_summary = {"ran": True, "returncode": proc.returncode,
                        "tail": tail}
    elapsed = round(time.monotonic() - t0, 1)
    record = {
        "method_version": METHOD_VERSION, "dates": dates,
        "started_utc": started, "elapsed_s": elapsed,
        "db_hash_before": hash_before, "db_hash_after": hash_after,
        "n_rows": len(rows), "n_eligible": len(results),
        "n_excluded": len(excluded),
        "observed_frp": _stats(observed),
        "expected_frp": _stats(expected),
        "difference_sum": exp_sum - obs_sum,
        "pct_difference": (100.0 * (exp_sum - obs_sum) / obs_sum
                           if obs_sum else None),
        "per_sensor": per_sensor, "per_day": per_day,
        "geometry": geometry, "test_summary": test_summary,
        "coefficients_source": coefficients.SOURCE,
    }
    with open(out / "pilot_summary.json", "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
    return record


def terminal_report(record: dict) -> str:
    """Render the mandatory terminal report (PASS/PARTIAL/FAIL/BLOCKED)."""
    lines: list[str] = []
    add = lines.append
    bar = "=" * 70
    add(bar)
    add("FireAgent Calendar — 2-Day Deterministic FRP Pilot (terminal report)")
    add(bar)
    add(f"1. Repo: {REPO} | DB: calendar.duckdb "
        f"(sha {record['db_hash_before'][:12]}…)")
    add(f"2. Dates: {', '.join(record['dates'])} "
        f"({record['n_rows']} rows, {record['n_eligible']} eligible, "
        f"{record['n_excluded']} excluded)")
    obs, exp = record["observed_frp"], record["expected_frp"]
    add(f"3-4. Observed FRP: n={obs['n']} sum={obs['sum']:.1f} "
        f"mean={obs['mean']:.2f} median={obs['median']:.2f} "
        f"max={obs['max']:.1f} MW")
    add(f"5. Coefficients: det-frp-oop-v1 ← Sofiev & Kouznetsov (2026) "
        f"Tables 1-2 (MODIS/VIIRS × day/night)")
    geo = record["geometry"]
    add(f"6. Geometry: MODIS 1-1.2km² frac={geo['modis_frac_1_12']} "
        f"[{'PASS' if geo['modis_ok'] else 'FAIL'}]; VIIRS <0.3km² "
        f"frac={geo['viirs_frac_lt03']} "
        f"[{'PASS' if geo['viirs_ok'] else 'FAIL'}]; bad rows: "
        f"{geo['bad_geometry_rows']}")
    add(f"7. Expected FRP: n={exp['n']} sum={exp['sum']:.1f} "
        f"mean={exp['mean']:.2f} median={exp['median']:.2f} "
        f"max={exp['max']:.1f} MW; diff={record['difference_sum']:.1f} MW "
        f"({record['pct_difference']:.1f}%)")
    for sensor, pack in record["per_sensor"].items():
        add(f"8. {sensor}: n={pack['n']} obs={pack['observed_frp']['sum']:.1f} "
            f"exp={pack['expected_frp']['sum']:.1f} MW")
    for day, pack in record["per_day"].items():
        by = pack["by_sensor"]
        add(f"9. {day}: n={pack['n']} (modis={by['modis']['n']}, "
            f"viirs={by['viirs-snpp']['n']}) obs={pack['observed_frp']['sum']:.1f} "
            f"exp={pack['expected_frp']['sum']:.1f} MW")
    tests = record["test_summary"]
    add(f"10. Tests: {'ran rc=' + str(tests.get('returncode')) if tests['ran'] else 'NOT RUN'} "
        f"{' | '.join(tests.get('tail', []))}")
    add("11. Limitations: FIRMS-input-as-P is EXPLORATORY (selection bias: "
        "undetected fires absent); clear-sky assumed (no QA/cloud terms); "
        "D/N mapped from flags (twilight unresolved); scan×track area is "
        "distributionally validated, not per-pixel proven.")
    add(f"12. Duration: {record['elapsed_s']}s; "
        f"DB unchanged: {record['db_hash_before'] == record['db_hash_after']}")
    verdict = ("PARTIAL PASS"
               if record["n_eligible"] > 0 and geo["modis_ok"] and
               geo["viirs_ok"] and tests.get("returncode") == 0
               else "FAIL")
    add(f"13. FINAL VERDICT: {verdict} (executed end-to-end; scientific "
        f"assumptions labeled exploratory, never validated)")
    add(bar)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="2-day FRP pilot")
    parser.add_argument("--dates", nargs=2, default=["2023-08-15",
                                                     "2023-08-16"])
    parser.add_argument("--outdir", default="pilot_outputs")
    parser.add_argument("--db", default="calendar.duckdb")
    parser.add_argument("--no-tests", action="store_true")
    args = parser.parse_args(argv)
    record = run(args.dates, args.outdir, db_path=args.db,
                 run_tests=not args.no_tests)
    print(terminal_report(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Phase 4 analytical runner (reproducible, staged).

Stages: 4B (baselines), 4C (anomalies + confidence + explanations).
Each stage records its execution in analytics_runs.jsonl and writes
versioned outputs under analytics_output/<exec_id>/.

Usage: .venv/bin/python scripts/cal_analytics_4.py [4b] [4c[=baselines-exec]]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.calendar import settings  # noqa: E402
from src.calendar.analytics import anomalies, baselines, runs  # noqa: E402

RUNS = str(REPO / "analytics_runs.jsonl")
OUT_ROOT = REPO / "analytics_output"


def _latest_exec(stage: str) -> str:
    records = runs.load_runs(RUNS)
    completed = {r["exec_id"] for r in records if r.get("status") == "complete"}
    opened = [r for r in records
              if r.get("stage") == stage and r.get("exec_id") in completed]
    if not opened:
        raise SystemExit(f"BLOCKED: no completed stage {stage} run found")
    return opened[-1]["exec_id"]


def stage_4b(metrics_exec: str | None = None) -> dict:
    metrics_exec = metrics_exec or settings.CAL_EXEC_2D
    metrics_path = OUT_ROOT / metrics_exec / "metrics_2d.json"
    if not metrics_path.is_file():
        raise SystemExit(f"BLOCKED: missing metrics {metrics_path}")
    exec_id = runs.new_execution(
        RUNS, stage="4B",
        inputs={"metrics_exec": metrics_exec, "catalog": "catalog.json",
                "db": "calendar.duckdb", "scope": "full-2023-USA-pilot"},
        params={"algo": baselines.ALGO_VERSION,
                "config": baselines.CONFIG_VERSION,
                "methods": list(baselines.METHODS)})
    t0 = time.monotonic()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    t_load = time.monotonic() - t0
    records = baselines.run_all_baselines(metrics, exec_id=exec_id)
    t_base = time.monotonic() - t0 - t_load
    out_dir = OUT_ROOT / exec_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baselines_4b.json").write_text(
        json.dumps(records, indent=1, sort_keys=True))
    by_status: dict[str, int] = {}
    for record in records:
        key = f"{record['baseline_method']}:{record['eligibility']}"
        by_status[key] = by_status.get(key, 0) + 1
    summary = {"records": len(records), "by_status": by_status,
               "load_s": round(t_load, 1), "baselines_s": round(t_base, 1)}
    runs.complete_execution(RUNS, exec_id, {
        "baselines": str(out_dir / "baselines_4b.json"), **summary})
    print(f"4B {exec_id[:8]} load={t_load:.0f}s baselines={t_base:.0f}s "
          f"records={len(records)}")
    return {"exec_id": exec_id, **summary}


def stage_4c(baselines_exec: str | None = None) -> dict:
    baselines_exec = baselines_exec or _latest_exec("4B")
    baselines_path = OUT_ROOT / baselines_exec / "baselines_4b.json"
    if not baselines_path.is_file():
        raise SystemExit(f"BLOCKED: missing baselines {baselines_path}")
    exec_id = runs.new_execution(
        RUNS, stage="4C",
        inputs={"baselines_exec": baselines_exec,
                "scope": "full-2023-USA-pilot"},
        params={"algo": anomalies.ALGO_VERSION,
                "classify": anomalies.CLASSIFY_VERSION,
                "z_notable": anomalies.Z_NOTABLE,
                "pct_notable": anomalies.PCT_NOTABLE})
    t0 = time.monotonic()
    records = json.loads(baselines_path.read_text(encoding="utf-8"))
    t_load = time.monotonic() - t0
    results = anomalies.run_all_anomalies(records, exec_id=exec_id)
    t_anom = time.monotonic() - t0 - t_load
    out_dir = OUT_ROOT / exec_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "anomalies_4c.json").write_text(
        json.dumps(results, indent=1, sort_keys=True))
    by_class: dict[str, int] = {}
    for result in results:
        by_class[result["classification"]] = by_class.get(result["classification"], 0) + 1
    summary = {"results": len(results), "by_classification": by_class,
               "load_s": round(t_load, 1), "anomalies_s": round(t_anom, 1)}
    runs.complete_execution(RUNS, exec_id, {
        "anomalies": str(out_dir / "anomalies_4c.json"), **summary})
    print(f"4C {exec_id[:8]} load={t_load:.0f}s anomalies={t_anom:.0f}s "
          f"results={len(results)}")
    return {"exec_id": exec_id, **summary}


STAGES = {"4b": (stage_4b, 1), "4c": (stage_4c, 1)}


def main(argv: list[str]) -> None:
    wanted = [a.lower() for a in argv[1:]] or sorted(STAGES)
    for token in wanted:
        stage, _, args = token.partition("=")
        if stage not in STAGES:
            raise SystemExit(f"unknown stage {stage!r}; have {sorted(STAGES)}")
        func, arity = STAGES[stage]
        given = args.split(",") if args else []
        if len(given) > arity:
            raise SystemExit(f"too many args for {stage}: {token!r}")
        func(*[g or None for g in given])


if __name__ == "__main__":
    main(sys.argv)

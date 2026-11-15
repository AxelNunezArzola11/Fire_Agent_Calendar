#!/usr/bin/env python3
"""Phase 2 analytical runner (reproducible, staged).

Stages: 2A (validation), 2B (matching), 2C (clustering), 2D (metrics),
2E (dual index + reconciliation). Each stage records its execution in
analytics_runs.jsonl and writes versioned outputs under
analytics_output/<exec_id>/.

Usage: .venv/bin/python scripts/cal_analytics_2.py [2a] [2b] [2c[=match]]
       [2d[=match,cluster]] [2e[=match,cluster]]
Defaults chain from the registry (latest completed run / the match run
the cluster run used); explicit exec ids pin the lineage exactly.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.calendar.analytics import (clustering, discovery, index, matching,  # noqa: E402
                                   metrics, runs)
from src.calendar.processing import duckdb_layer  # noqa: E402

RUNS = str(REPO / "analytics_runs.jsonl")
OUT_ROOT = REPO / "analytics_output"


def stage_2a() -> dict:
    exec_id = runs.new_execution(
        RUNS, stage="2A", inputs={"catalog": "catalog.json", "db": "calendar.duckdb"})
    report = discovery.validate_inputs("catalog.json", "calendar.duckdb")
    out_dir = OUT_ROOT / exec_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "validation_2a.json").write_text(json.dumps(report, indent=2))
    runs.complete_execution(RUNS, exec_id, {"validation": str(out_dir / "validation_2a.json"),
                                            "overall": report["overall"]})
    print(f"2A {exec_id[:8]} overall={report['overall']}")
    return {"exec_id": exec_id, "overall": report["overall"]}


def stage_2b() -> dict:
    import pandas as pd

    exec_id = runs.new_execution(
        RUNS, stage="2B",
        inputs={"catalog": "catalog.json", "db": "calendar.duckdb",
                "scope": "full-2023-USA-pilot"},
        params={"algo": matching.ALGO_VERSION,
                "match_params": matching.MatchParams().version})
    t0 = time.monotonic()
    modis = duckdb_layer.fetch("calendar.duckdb", product="modis", limit=10_000_000)
    viirs = duckdb_layer.fetch("calendar.duckdb", product="viirs-snpp", limit=10_000_000)
    t_load = time.monotonic() - t0
    groups, members, stats = matching.match_frames(modis, viirs, exec_id=exec_id)
    t_match = time.monotonic() - t0 - t_load
    out_dir = OUT_ROOT / exec_id
    out_dir.mkdir(parents=True, exist_ok=True)
    groups.to_parquet(out_dir / "match_groups.parquet", index=False)
    members.to_parquet(out_dir / "match_members.parquet", index=False)
    comp = members["classification"].value_counts().to_dict()
    summary = {"modis_in": len(modis), "viirs_in": len(viirs), **stats,
               "composition": comp, "load_s": round(t_load, 1),
               "match_s": round(t_match, 1)}
    (out_dir / "matching_2b_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    runs.complete_execution(RUNS, exec_id, {
        "groups": str(out_dir / "match_groups.parquet"),
        "members": str(out_dir / "match_members.parquet"),
        "summary": str(out_dir / "matching_2b_summary.json"), **summary})
    print(f"2B {exec_id[:8]} load={t_load:.0f}s match={t_match:.0f}s " + json.dumps(comp))
    return {"exec_id": exec_id, **summary}


def _latest_exec(stage: str) -> str:
    records = runs.load_runs(RUNS)
    completed = {r["exec_id"] for r in records if r.get("status") == "complete"}
    opened = [r for r in records
              if r.get("stage") == stage and r.get("exec_id") in completed]
    if not opened:
        raise SystemExit(f"BLOCKED: no completed stage {stage} run found")
    return opened[-1]["exec_id"]


def stage_2c(match_exec: str | None = None) -> dict:
    import pandas as pd

    match_exec = match_exec or _latest_exec("2B")
    match_dir = OUT_ROOT / match_exec
    exec_id = runs.new_execution(
        RUNS, stage="2C", inputs={"match_exec": match_exec},
        params={"algo": clustering.ALGO_VERSION,
                "cluster_params": clustering.ClusterParams().version})
    t0 = time.monotonic()
    det = duckdb_layer.fetch("calendar.duckdb", limit=10_000_000)
    groups = pd.read_parquet(match_dir / "match_groups.parquet")
    t_load = time.monotonic() - t0
    clusters, membership, stats = clustering.cluster_frames(
        det, groups, exec_id=exec_id)
    t_clu = time.monotonic() - t0 - t_load
    out_dir = OUT_ROOT / exec_id
    out_dir.mkdir(parents=True, exist_ok=True)
    clusters.to_parquet(out_dir / "clusters.parquet", index=False)
    membership.to_parquet(out_dir / "cluster_members.parquet", index=False)
    sizes = clusters["n_detections"]
    summary = {**stats, "load_s": round(t_load, 1), "cluster_s": round(t_clu, 1),
               "size_min": int(sizes.min()), "size_max": int(sizes.max()),
               "size_mean": round(float(sizes.mean()), 2)}
    (out_dir / "clustering_2c_summary.json").write_text(json.dumps(summary, indent=2))
    runs.complete_execution(RUNS, exec_id, {
        "clusters": str(out_dir / "clusters.parquet"),
        "membership": str(out_dir / "cluster_members.parquet"),
        "summary": str(out_dir / "clustering_2c_summary.json"), **summary})
    print(f"2C {exec_id[:8]} load={t_load:.0f}s cluster={t_clu:.0f}s "
          f"clusters={stats['clusters']} singletons={stats['singletons']}")
    return {"exec_id": exec_id, **summary}


def _open_inputs(exec_id: str) -> dict:
    for record in runs.load_runs(RUNS):
        if record.get("exec_id") == exec_id and record.get("status") == "running":
            return dict(record.get("inputs", {}))
    raise SystemExit(f"BLOCKED: no open record for exec {exec_id[:8]}")


def _resolve_chain(match_exec: str | None, cluster_exec: str | None
                   ) -> tuple[str, str]:
    cluster_exec = cluster_exec or _latest_exec("2C")
    match_exec = match_exec or _open_inputs(cluster_exec).get("match_exec") \
        or _latest_exec("2B")
    return match_exec, cluster_exec


def stage_2d(match_exec: str | None = None, cluster_exec: str | None = None) -> dict:
    import pandas as pd

    match_exec, cluster_exec = _resolve_chain(match_exec, cluster_exec)
    match_dir, cluster_dir = OUT_ROOT / match_exec, OUT_ROOT / cluster_exec
    for needed in (match_dir / "match_members.parquet",
                   cluster_dir / "clusters.parquet"):
        if not needed.is_file():
            raise SystemExit(f"BLOCKED: missing input {needed}")
    exec_id = runs.new_execution(
        RUNS, stage="2D",
        inputs={"match_exec": match_exec, "cluster_exec": cluster_exec,
                "db": "calendar.duckdb", "scope": "full-2023-USA-pilot"},
        params={"algo": metrics.ALGO_VERSION, "grains": list(metrics.GRAINS)})
    t0 = time.monotonic()
    det = duckdb_layer.fetch("calendar.duckdb", limit=10_000_000)
    members = pd.read_parquet(match_dir / "match_members.parquet")
    clusters = pd.read_parquet(cluster_dir / "clusters.parquet")
    t_load = time.monotonic() - t0
    scope = {"region": "USA-pilot", "date_range": ("2023-01-01", "2023-12-31")}
    inputs = {"match_exec": match_exec, "cluster_exec": cluster_exec,
              "n_detections": len(det)}
    by_grain = {}
    for grain in metrics.GRAINS:
        by_grain[grain] = {
            "detection_activity": metrics.detection_activity(
                det, members, grain=grain, scope=scope, inputs=inputs,
                exec_id=exec_id),
            "cluster_activity": metrics.cluster_activity(
                clusters, grain=grain, scope=scope, inputs=inputs,
                exec_id=exec_id),
            "frp_summary": metrics.frp_summary(
                det, grain=grain, scope=scope, inputs=inputs, exec_id=exec_id,
                combined=True),
        }
    t_metrics = time.monotonic() - t0 - t_load
    out_dir = OUT_ROOT / exec_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics_2d.json").write_text(json.dumps(by_grain, indent=2))
    raw_total = sum(r["raw_observations"] for r in
                    by_grain["daily"]["detection_activity"]["rows"])
    harm_total = sum(r["harmonized_observations"] for r in
                     by_grain["daily"]["detection_activity"]["harmonized_rows"])
    summary = {"grains": list(metrics.GRAINS), "load_s": round(t_load, 1),
               "metrics_s": round(t_metrics, 1), "raw_total": raw_total,
               "harmonized_total": harm_total,
               "year_clusters": by_grain["yearly"]["cluster_activity"]["rows"][0]["clusters"]
               if by_grain["yearly"]["cluster_activity"]["rows"] else 0}
    runs.complete_execution(RUNS, exec_id, {
        "metrics": str(out_dir / "metrics_2d.json"), **summary})
    print(f"2D {exec_id[:8]} load={t_load:.0f}s metrics={t_metrics:.0f}s "
          f"raw={raw_total} harm={harm_total}")
    return {"exec_id": exec_id, **summary}


def stage_2e(match_exec: str | None = None, cluster_exec: str | None = None) -> dict:
    import pandas as pd

    match_exec, cluster_exec = _resolve_chain(match_exec, cluster_exec)
    match_dir, cluster_dir = OUT_ROOT / match_exec, OUT_ROOT / cluster_exec
    exec_id = runs.new_execution(
        RUNS, stage="2E",
        inputs={"match_exec": match_exec, "cluster_exec": cluster_exec,
                "db": "calendar.duckdb", "scope": "full-2023-USA-pilot"},
        params={"algo": index.ALGO_VERSION})
    t0 = time.monotonic()
    det = duckdb_layer.fetch("calendar.duckdb", limit=10_000_000)
    members = pd.read_parquet(match_dir / "match_members.parquet")
    groups = pd.read_parquet(match_dir / "match_groups.parquet")
    clusters = pd.read_parquet(cluster_dir / "clusters.parquet")
    membership = pd.read_parquet(cluster_dir / "cluster_members.parquet")
    t_load = time.monotonic() - t0
    n_input = int(duckdb_layer.count("calendar.duckdb"))  # independent source
    report = {
        "inputs": {"match_exec": match_exec, "cluster_exec": cluster_exec,
                   "n_input_duckdb": n_input},
        "sensor_view": index.sensor_view(members, det),
        "harmonized_view": index.harmonized_view(members, groups, clusters),
        "reconciliation": index.reconcile(
            n_input=n_input, members_df=members, match_groups_df=groups,
            clusters_df=clusters, membership_df=membership),
        "composite": {"formula": index.PROPOSED_COMPOSITE_FORMULA,
                      "status": "not-computed (no approved weights)"},
        "algo_version": index.ALGO_VERSION, "exec_id": exec_id,
    }
    out_dir = OUT_ROOT / exec_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index_2e.json").write_text(json.dumps(report, indent=2))
    summary = {"load_s": round(t_load, 1), "n_input": n_input,
               "reconciled": report["reconciliation"]["passed"]}
    runs.complete_execution(RUNS, exec_id, {
        "index": str(out_dir / "index_2e.json"), **summary})
    print(f"2E {exec_id[:8]} reconciled={summary['reconciled']} n={n_input}")
    return {"exec_id": exec_id, **summary}


STAGES = {"2a": (stage_2a, 0), "2b": (stage_2b, 0), "2c": (stage_2c, 1),
          "2d": (stage_2d, 2), "2e": (stage_2e, 2)}


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

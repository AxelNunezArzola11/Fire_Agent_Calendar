"""Date-range pipeline runner (acquire -> agents -> operator -> export).

Seven stages with per-stage status, latency, and partial-result
preservation: a failed stage never reports success, and later stages
skip honestly with reasons. Deterministic given identical dataset,
range, configuration, and model responses (execution_id derives from
the data fingerprint; only processing timestamps vary per run).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

OPERATOR_ROW_CAP = 50000
TOP_CLUSTERS = 3
TOP_EVENTS = 3
ROLLUP_CAP = 200
L2_MAX_DAYS = 7
L2_BBOX = (-125.0, 24.0, -66.0, 50.0)
L2_MAX_PAGES = 5


def build_l2_matcher(inventory: list[dict]):
    """Row matcher closure over a granule inventory (pure, tested).

    Returns matcher(row) -> (granule_ur|None, match_status). Only
    VIIRS rows are matched (the inventory is VNP14IMG); anything
    else yields (None, "unmatched").
    """
    from ..l2 import match as match_mod

    def _match(row: dict):
        if str(row.get("product") or "") != "viirs-snpp":
            return None, "unmatched"
        out = match_mod.match_observation(
            {"latitude": row.get("latitude"),
             "longitude": row.get("longitude"),
             "acquisition_utc": row.get("acquisition_timestamp_utc")},
            inventory)
        if out["status"] == "matched":
            return out["granule_ur"], "matched"
        return None, out["status"]

    return _match


def fetch_l2_inventory(date_start: str, date_end: str,
                       bbox=None) -> dict:
    """Live VNP14IMG inventory for a short range (network; may raise).

    Returns {granules, hits, truncated}. Raises on CMR failure or
    ranges longer than L2_MAX_DAYS (caller converts to a reason).
    """
    from datetime import date as _date

    from ..l2 import cmr

    span = (_date.fromisoformat(date_end)
            - _date.fromisoformat(date_start)).days + 1
    if span > L2_MAX_DAYS:
        raise ValueError(f"L2 matching capped at {L2_MAX_DAYS} days "
                         f"(asked {span})")
    concept = cmr.resolve_collection("VNP14IMG", "002",
                                     timeout_s=60.0)["concept_id"]
    granules: list[dict] = []
    hits = 0
    for page in range(1, L2_MAX_PAGES + 1):
        out = cmr.search_granules(
            concept, bbox=tuple(bbox) if bbox else L2_BBOX,
            date_start=date_start, date_end=date_end, page_size=100,
            page_num=page, timeout_s=60.0)
        hits = out["hits"]
        granules.extend(out["granules"])
        if len(granules) >= hits:
            break
    return {"granules": granules, "hits": hits,
            "truncated": len(granules) < hits}


@dataclass
class PipelineStage:
    name: str
    status: str = "pending"  # pending|running|ok|partial|failed|skipped
    detail: str = ""
    latency_ms: float = 0.0


@dataclass
class PipelineResult:
    execution_id: str
    started_utc: str
    date_start: str
    date_end: str
    config: dict
    stages: list = field(default_factory=list)
    acquisition: dict = field(default_factory=dict)
    detections: dict = field(default_factory=dict)
    events: dict = field(default_factory=dict)
    operator: dict = field(default_factory=dict)
    agent_runs: list = field(default_factory=list)
    validation: dict = field(default_factory=dict)
    metrics: list = field(default_factory=list)
    csvs: dict = field(default_factory=dict)
    limitations: list = field(default_factory=list)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _execution_id(date_start: str, date_end: str, fingerprint,
                  config: dict) -> str:
    digest = hashlib.sha256()
    digest.update(f"{date_start}|{date_end}|{fingerprint}|"
                  f"{json.dumps(config, sort_keys=True, default=str)}"
                  .encode("utf-8"))
    return digest.hexdigest()[:12]


def run_pipeline(date_start: str, date_end: str, *, events_conn,
                 chain, progress=None, operator_cap: int = OPERATOR_ROW_CAP,
                 top_clusters: int = TOP_CLUSTERS,
                 top_events: int = TOP_EVENTS,
                 l2_match: bool = False,
                 bbox: tuple | None = None,
                 live_dataset=None,
                 fetch_live_firms: bool = False,
                 region_label: str = "",
                 region_id: str | None = None,
                 map_geometry=None,
                 point_radius: tuple | None = None,
                 previous_aoi=None,
                 resolved_aoi=None,
                 firms_fetcher=None,
                 firms_key: str | None = None) -> PipelineResult:
    """Execute the full pipeline (never raises on stage faults).

    Raises ValueError only for malformed input (bad dates, inverted
    range). progress(stage_name, status, detail) is called live per
    stage transition for progressive UI display.
    """
    import pandas as pd

    from .. import settings
    from ..agents import coordinator as agents_coordinator
    from ..analytics import clustering, matching
    from ..events import association, qwen, store
    from ..experimental.frp_operator_pilot import (
        operator as oop, results as oor, validation as oop_validation)
    from ..ui import aoi as aoi_mod, filters, query
    from . import exports_csv, payloads, regional

    started = _utcnow()
    t_all = time.monotonic()
    try:
        start_day = filters.coerce_day(date_start)
        end_day = filters.coerce_day(date_end)
    except ValueError as exc:
        raise ValueError(f"invalid date range: {exc}")
    if start_day > end_day:
        raise ValueError("Start Date must not be later than End Date.")

    resolved_aoi_obj = None
    aoi_error_msg = None
    try:
        if isinstance(resolved_aoi, aoi_mod.ResolvedAoI) and (
            bbox is None or tuple(float(v) for v in bbox) == resolved_aoi.bbox
        ):
            aoi_mod.validate_bbox(resolved_aoi.bbox)
            resolved_aoi_obj = resolved_aoi
        else:
            resolved_aoi_obj = aoi_mod.resolve_aoi(
                bbox=bbox,
                region_id=region_id,
                map_geometry=map_geometry,
                point_radius=point_radius,
                previous_aoi=(
                    resolved_aoi if resolved_aoi is not None
                    else (previous_aoi if previous_aoi is not None else live_dataset)
                ),
            )
    except aoi_mod.AoIError as exc:
        aoi_error_msg = str(exc)

    eff_bbox = resolved_aoi_obj.bbox if resolved_aoi_obj is not None else (
        tuple(float(v) for v in bbox) if bbox and not aoi_error_msg else None
    )
    eff_region_label = region_label or (
        resolved_aoi_obj.label if resolved_aoi_obj is not None
        else ("Custom bbox" if eff_bbox else "USA pilot")
    )
    is_live_mode = bool(live_dataset is not None or fetch_live_firms)

    cat_path = settings.resolve_repo_path(settings.CATALOG_PATH)
    catalog = json.loads(cat_path.read_text(encoding="utf-8")) if cat_path.is_file() else {}
    spans = query.catalog_spans(catalog)
    fingerprint = query.fingerprint()
    config = {"operator_cap": operator_cap, "top_clusters": top_clusters,
              "top_events": top_events, "l2_match": bool(l2_match),
              "bbox": list(eff_bbox) if eff_bbox else None,
              "region_label": eff_region_label,
              "data_origin": "firms_live_api" if is_live_mode else "local_pilot_archive",
              "exec_2b": settings.CAL_EXEC_2B,
              "exec_2c": settings.CAL_EXEC_2C,
              "exec_2d": settings.CAL_EXEC_2D}
    if resolved_aoi_obj is not None and is_live_mode:
        config["aoi_source"] = resolved_aoi_obj.source
        if resolved_aoi_obj.region_id:
            config["region_id"] = resolved_aoi_obj.region_id
    if live_dataset is not None:
        config["firms_exec_id"] = getattr(live_dataset, "exec_id", "")
    exec_id = _execution_id(start_day, end_day, fingerprint, config)
    result = PipelineResult(exec_id, started, start_day, end_day, config)
    stages: dict[str, PipelineStage] = {}

    def _stage(name: str) -> PipelineStage:
        stage = PipelineStage(name, status="running")
        stages[name] = stage
        result.stages.append(stage)
        if progress:
            progress(name, "running", "")
        return stage

    def _done(stage: PipelineStage, status: str, detail: str,
              started_at: float) -> None:
        stage.status = status
        stage.detail = detail
        stage.latency_ms = round((time.monotonic() - started_at) * 1000,
                                 1)
        if progress:
            progress(stage.name, status, detail)

    chain_names = [getattr(p, "name", "?")
                   for p in (getattr(chain, "providers", None) or [])]

    if aoi_error_msg:
        tick = time.monotonic()
        stage = _stage("acquisition")
        _done(stage, "failed", f"Invalid AOI: {aoi_error_msg}", tick)
        _skip_rest(result, stages, progress,
                   ["detection", "events", "operator", "validation", "export", "metrics"],
                   "no data: invalid AOI configuration")
        _finalize(result)
        return result

    # -- live regional FIRMS path (outside local 2023 USA archive or explicit live fetch) --
    if live_dataset is None and fetch_live_firms:
        tick = time.monotonic()
        stage = _stage("acquisition")
        if resolved_aoi_obj is None:
            _done(
                stage,
                "failed",
                "No Area of Interest (AOI) selected: choose a region, point + radius, "
                "map view, or bounding box.",
                tick,
            )
            _skip_rest(
                result, stages, progress,
                ["detection", "events", "operator", "validation", "export", "metrics"],
                "no data: waiting for valid AOI selection",
            )
            _finalize(result)
            return result
        eff_firms_key = firms_key if firms_key is not None else settings.get_setting("FIRMS_MAP_KEY", "")
        if firms_fetcher is None and not eff_firms_key:
            _done(
                stage,
                "failed",
                "Missing configuration: FIRMS_MAP_KEY is not set in environment or Streamlit Secrets.",
                tick,
            )
            _skip_rest(
                result, stages, progress,
                ["detection", "events", "operator", "validation", "export", "metrics"],
                "no data: FIRMS_MAP_KEY is not configured",
            )
            _finalize(result)
            return result
        fetch_bbox = resolved_aoi_obj.bbox
        eff_rid = resolved_aoi_obj.region_id or region_id or resolved_aoi_obj.source
        eff_rlabel = region_label or resolved_aoi_obj.label
        try:
            live_dataset = regional.fetch_firms_regional(
                bbox=fetch_bbox,
                date_start=start_day,
                date_end=end_day,
                region_id=eff_rid,
                region_label=eff_rlabel,
                key=eff_firms_key if firms_key is not None or eff_firms_key else None,
                fetcher=firms_fetcher,
            )
        except Exception as exc:  # noqa: BLE001
            _done(stage, "failed", f"live FIRMS acquisition failed: {exc}", tick)
            _skip_rest(result, stages, progress,
                       ["detection", "events", "operator", "validation", "export", "metrics"],
                       "no data: acquisition failed")
            _finalize(result)
            return result
        if live_dataset.errors and live_dataset.n_detections == 0:
            _done(stage, "failed", "; ".join(live_dataset.errors), tick)
            _skip_rest(result, stages, progress,
                       ["detection", "events", "operator", "validation", "export", "metrics"],
                       "no data: acquisition failed")
            _finalize(result)
            return result
        result.config["firms_exec_id"] = live_dataset.exec_id
        result.config["bbox"] = list(live_dataset.bbox)
        result.config["region_label"] = live_dataset.region_label
        result.acquisition = {
            "n_detections": live_dataset.n_detections,
            "n_clusters": 0,
            "data_origin": "firms_live_api",
            "sources": list(live_dataset.sources),
            "region_id": live_dataset.region_id,
            "region_label": live_dataset.region_label,
            "bbox": list(live_dataset.bbox),
            "aoi_source": resolved_aoi_obj.source,
            "firms_exec_id": live_dataset.exec_id,
        }
        _done(stage, "ok",
              f"{live_dataset.n_detections:,} live FIRMS detections for {live_dataset.region_label} "
              f"({', '.join(live_dataset.sources) or 'none'}).", tick)

    if live_dataset is not None:
        result.config["firms_exec_id"] = live_dataset.exec_id
        result.config["bbox"] = list(live_dataset.bbox)
        result.config["region_label"] = live_dataset.region_label or eff_region_label
        if not result.acquisition:
            tick = time.monotonic()
            stage = _stage("acquisition")
            result.acquisition = {
                "n_detections": live_dataset.n_detections,
                "n_clusters": 0,
                "data_origin": "firms_live_api",
                "sources": list(live_dataset.sources),
                "region_id": live_dataset.region_id,
                "region_label": live_dataset.region_label,
                "bbox": list(live_dataset.bbox),
                "aoi_source": resolved_aoi_obj.source if resolved_aoi_obj else "session",
                "firms_exec_id": live_dataset.exec_id,
            }
            _done(stage, "ok",
                  f"{live_dataset.n_detections:,} normalized FIRMS detections ({live_dataset.region_label}).",
                  tick)
        if live_dataset.n_detections == 0:
            _skip_rest(result, stages, progress,
                       ["detection", "events", "operator", "validation"],
                       f"no detections returned for {live_dataset.region_label} ({start_day} to {end_day})")
            result.events = {"rollup": [], "rollup_truncated": False}
            result.operator = {"records": [], "aggregate": oor.aggregate_record([]),
                               "truncated": False, "cap": operator_cap,
                               "l2_matched": 0, "l2_note": "no detections to match"}
            tick = time.monotonic()
            stage = _stage("export")
            provenance = f"firms-live {live_dataset.exec_id}; {live_dataset.region_label}"
            try:
                result.csvs = {
                    "observations": exports_csv.observation_csv(
                        [], provenance=provenance, processing_utc=started, execution_id=exec_id),
                    "events": exports_csv.event_csv(
                        [], provenance=provenance, processing_utc=started, execution_id=exec_id),
                    "agents": exports_csv.agent_csv(
                        result.agent_runs, processing_utc=started, execution_id=exec_id),
                }
                _done(stage, "ok", "empty regional CSVs built (0 detections)", tick)
            except Exception as exc:  # noqa: BLE001
                _done(stage, "failed", f"export failed: {exc}", tick)
            tick = time.monotonic()
            stage = _stage("metrics")
            _compute_metrics(result, stages, t_all, chain_names, fingerprint)
            result.csvs["metrics"] = exports_csv.metrics_csv(
                result.metrics, processing_utc=started, execution_id=exec_id)
            _done(stage, "ok", f"{len(result.metrics)} metrics (0 detections)", tick)
            _finalize(result)
            return result

        # Stage 2: Cross-sensor matching + spatiotemporal clustering + detection intelligence
        tick = time.monotonic()
        stage = _stage("detection")
        df_live = live_dataset.df.drop_duplicates(subset=["detection_id"]).reset_index(drop=True)
        try:
            modis_live = df_live[df_live["product"] == "modis"]
            viirs_live = df_live[df_live["product"].astype(str).str.startswith("viirs")]
            if len(modis_live) and len(viirs_live):
                match_groups_df, _, _ = matching.match_frames(
                    modis_live, viirs_live, exec_id=live_dataset.exec_id)
            else:
                match_groups_df = pd.DataFrame(
                    columns=["match_id", "classification", "modis_ids", "viirs_ids"]
                )
            clusters_df, _, _ = clustering.cluster_frames(
                df_live, match_groups_df, exec_id=live_dataset.exec_id)
            if not clusters_df.empty:
                clusters_df = clusters_df.sort_values(
                    ["n_detections", "frp_max", "cluster_id"],
                    ascending=[False, False, True],
                    na_position="last",
                ).reset_index(drop=True)
            result.acquisition["n_clusters"] = int(len(clusters_df))
            cluster_error = ""
        except Exception as exc:  # noqa: BLE001
            clusters_df = pd.DataFrame()
            cluster_error = str(exc)

        det_runs = 0
        analyzed_cids: list[str] = []
        if not clusters_df.empty:
            for clu in clusters_df.head(max(1, top_clusters)).to_dict("records"):
                cid = clu["cluster_id"]
                member_ids = set(json.loads(clu.get("detection_ids") or "[]"))
                sub_df = df_live[df_live["detection_id"].isin(member_ids)]
                det_payload = {
                    "cluster_id": cid,
                    "cluster_ids": [cid],
                    "region": live_dataset.region_label,
                    "data_origin": "firms_live_api",
                    "n_detections": int(clu.get("n_detections", len(sub_df))),
                    "n_modis": int(clu.get("n_modis", 0)),
                    "n_viirs": int(clu.get("n_viirs", 0)),
                    "first_observed_utc": clu.get("first_observation_utc"),
                    "last_observed_utc": clu.get("last_observation_utc"),
                    "observation_span_h": clu.get("observation_span_h"),
                    "satellites": json.loads(clu.get("satellites") or "[]"),
                    "timeline": _live_cluster_timeline(sub_df),
                    "limitations": list(live_dataset.limitations),
                }
                outcome = agents_coordinator.analyze(
                    "detections", cid, det_payload, chain=chain, conn=events_conn)
                _record_run(result, outcome, "detection-intelligence", cid)
                analyzed_cids.append(cid)
                det_runs += 1
        elif not cluster_error:
            cid = f"regional-{live_dataset.exec_id}"
            det_payload = {
                "cluster_id": cid,
                "cluster_ids": [cid],
                "region": live_dataset.region_label,
                "data_origin": "firms_live_api",
                "n_detections": int(len(df_live)),
                "n_modis": int((df_live["product"] == "modis").sum()),
                "n_viirs": int(df_live["product"].astype(str).str.startswith("viirs").sum()),
                "satellites": sorted({str(s) for s in df_live["satellite"].dropna().unique()}),
                "timeline": _live_cluster_timeline(df_live),
                "limitations": list(live_dataset.limitations),
            }
            outcome = agents_coordinator.analyze(
                "detections", cid, det_payload, chain=chain, conn=events_conn)
            _record_run(result, outcome, "detection-intelligence", cid)
            analyzed_cids.append(cid)
            det_runs = 1

        result.detections = {
            "clusters_analyzed": det_runs,
            "n_clusters": int(len(clusters_df)),
            "cluster_ids": analyzed_cids,
        }
        if cluster_error:
            _done(stage, "failed", f"detection clustering failed: {cluster_error}", tick)
        else:
            _done(stage, "ok",
                  f"{len(clusters_df)} cluster(s) from {len(df_live):,} detections; "
                  f"{det_runs} agent analys(es)", tick)

        # Stage 3: Deterministic event association + event intelligence agent
        tick = time.monotonic()
        stage = _stage("events")
        event_rows: list[dict] = []
        ev_runs = 0
        rollup_truncated = len(clusters_df) > ROLLUP_CAP
        try:
            if not clusters_df.empty:
                seeded = association.seed_events(
                    clusters_df.head(ROLLUP_CAP).to_dict("records"),
                    exec_id=live_dataset.exec_id,
                    coverage_status="complete",
                )
                by_eid = {ev["event_id"]: ev for ev in seeded}
                for ev in seeded:
                    event_rows.append(_event_rollup(ev, events_conn, store))
                for row in event_rows[:top_events]:
                    payload = qwen.build_event_payload(by_eid[row["event_id"]], [])
                    outcome = agents_coordinator.analyze(
                        "event", row["event_id"], payload,
                        chain=chain, conn=events_conn)
                    _record_run(result, outcome, "event-analysis", row["event_id"])
                    row["analysis_status"] = outcome.steps[0].status
                    ev_runs += 1
            result.events = {"rollup": event_rows, "rollup_truncated": rollup_truncated}
            detail = (
                f"{len(event_rows)} events seeded from {len(clusters_df)} cluster(s)"
                + (" (truncated)" if rollup_truncated else "")
                + f", {ev_runs} analyzed"
            )
            _done(stage, "ok" if event_rows else "skipped",
                  detail if event_rows else "no clusters to seed events", tick)
        except Exception as exc:  # noqa: BLE001
            result.events = {"rollup": [], "rollup_truncated": False}
            _done(stage, "failed", f"event stage failed: {exc}", tick)

        # Stage 4: Observation operator + science observer agent
        tick = time.monotonic()
        stage = _stage("operator")
        op_truncated = len(df_live) > operator_cap
        op_out = regional.run_operator_on_frame(df_live.head(operator_cap))
        records = op_out["records"]
        agg = op_out["aggregate"]
        result.operator = {
            "records": records,
            "aggregate": agg,
            "truncated": op_truncated,
            "cap": operator_cap,
            "l2_matched": 0,
            "l2_note": "L2 matching skipped in live regional session",
        }
        sci_payload = {
            "method_version": "det-frp-oop-v1",
            "region": live_dataset.region_label,
            "n": agg["n_observations"],
            "observed_sum": sum((r.get("observed_frp") or 0.0) for r in records if r["calculation_status"] == "ok"),
            "expected_sum": sum((r.get("expected_retrieved_frp_MW") or 0.0) for r in records if r["calculation_status"] == "ok"),
            "by_sensor": {
                s: {"n": v["n"], "observed": v["observed_sum_MW"], "expected": v["expected_sum_MW"]}
                for s, v in agg["by_sensor"].items()
            },
            "mapping_applied": False,
            "inputs_missing": list(payloads.SCIENCE_MISSING_INPUTS),
            "limitations": list(live_dataset.limitations),
        }
        sci_outcome = agents_coordinator.analyze(
            "science", f"regional:{live_dataset.region_label}:{start_day}:{end_day}",
            sci_payload, chain=chain, conn=events_conn)
        _record_run(result, sci_outcome, "science-observer", f"{start_day}:{end_day}")
        _done(stage, "ok" if agg["n_eligible"] else "partial",
              f"{agg['n_eligible']}/{agg['n_observations']} eligible in {live_dataset.region_label}", tick)

        # Stage 5: Scientific & provenance validation
        tick = time.monotonic()
        stage = _stage("validation")
        checks = _validate_result(result, start_day, end_day)
        result.validation = {
            "checks": checks,
            "passed": sum(1 for c in checks if c["passed"]),
            "failed": sum(1 for c in checks if not c["passed"]),
        }
        failed = [c for c in checks if not c["passed"]]
        for row in result.events.get("rollup", []):
            row["validation_status"] = "failed" if failed else "passed"
        _done(stage, "failed" if failed else "ok",
              "; ".join(c["check"] for c in failed) or f"{len(checks)} checks passed", tick)

        # Stage 6: CSV exports (observations, events, agents, metrics added in Stage 7)
        tick = time.monotonic()
        stage = _stage("export")
        provenance = f"firms-live {live_dataset.exec_id}; {live_dataset.region_label}"
        try:
            result.csvs = {
                "observations": exports_csv.observation_csv(
                    records, provenance=provenance, processing_utc=started, execution_id=exec_id),
                "events": exports_csv.event_csv(
                    result.events.get("rollup", []),
                    provenance=provenance, processing_utc=started, execution_id=exec_id),
                "agents": exports_csv.agent_csv(
                    result.agent_runs, processing_utc=started, execution_id=exec_id),
            }
            _done(stage, "ok", "3 regional CSVs built (metrics follows)", tick)
        except Exception as exc:  # noqa: BLE001
            _done(stage, "failed", f"export failed: {exc}", tick)

        # Stage 7: Metrics
        tick = time.monotonic()
        stage = _stage("metrics")
        _compute_metrics(result, stages, t_all, chain_names, fingerprint)
        result.csvs["metrics"] = exports_csv.metrics_csv(
            result.metrics, processing_utc=started, execution_id=exec_id)
        _done(stage, "ok", f"{len(result.metrics)} metrics + metrics CSV", tick)
        _finalize(result)
        return result

    # -- acquisition -------------------------------------------------
    tick = time.monotonic()
    stage = _stage("acquisition")
    cov_start = query.coverage_for(start_day, spans)["status"]
    cov_end = query.coverage_for(end_day, spans)["status"]
    if cov_start == "unavailable" and cov_end == "unavailable":
        _done(stage, "failed",
              f"Range {start_day} to {end_day} is outside verified pilot coverage "
              "(pre-indexed 2023-01-01 to 2023-12-31 archive). Enable NASA FIRMS "
              "acquisition to process multi-year dates (MODIS 2000+, VIIRS S-NPP 2012+, "
              "NOAA-20 2018+, NOAA-21 2023+).", tick)
        _skip_rest(result, stages, progress,
                   ["detection", "events", "operator", "validation",
                    "export", "metrics"],
                   "no data: acquisition failed")
        _finalize(result)
        return result
    try:
        n_det = query.count_detections(date_start=start_day,
                                       date_end=end_day, bbox=eff_bbox)
        n_clu = query.count_clusters(date_start=start_day,
                                     date_end=end_day, bbox=eff_bbox)
    except Exception as exc:  # noqa: BLE001 - stage fault, not crash
        _done(stage, "failed", f"acquisition query failed: {exc}", tick)
        _skip_rest(result, stages, progress,
                   ["detection", "events", "operator", "validation",
                    "export", "metrics"],
                   "no data: acquisition failed")
        _finalize(result)
        return result
    result.acquisition = {"n_detections": n_det, "n_clusters": n_clu,
                          "coverage_start": cov_start,
                          "coverage_end": cov_end}
    _done(stage, "ok",
          f"{n_det:,} detections, {n_clu:,} clusters "
          f"({start_day} to {end_day}).", tick)
    if n_det == 0:
        _skip_rest(result, stages, progress,
                   ["detection", "events", "operator"],
                   "zero recorded observations in range")
        _skip_rest(result, stages, progress,
                   ["validation", "export", "metrics"],
                   "nothing to validate or export")
        _finalize(result)
        return result

    # -- detection intelligence --------------------------------------
    tick = time.monotonic()
    stage = _stage("detection")
    try:
        clusters = query.query_clusters(
            date_start=start_day, date_end=end_day, bbox=eff_bbox,
            limit=top_clusters).to_dict("records")
    except Exception as exc:  # noqa: BLE001
        clusters, cluster_error = [], str(exc)
    else:
        cluster_error = ""
    det_runs = 0
    for cluster in clusters:
        cid = cluster["cluster_id"]
        payload = payloads.build_detection_payload(cid)
        if payload is None:
            continue
        outcome = agents_coordinator.analyze(
            "detections", cid, payload, chain=chain)
        _record_run(result, outcome, "detection-intelligence", cid)
        det_runs += 1
    result.detections = {"clusters_analyzed": det_runs,
                         "cluster_ids": [c["cluster_id"]
                                         for c in clusters]}
    if cluster_error:
        _done(stage, "failed", f"cluster query failed: {cluster_error}",
              tick)
    elif det_runs == 0:
        _done(stage, "failed", "no cluster payload built", tick)
    else:
        _done(stage, "ok", f"{det_runs} cluster analys(es)", tick)

    # -- event analysis ----------------------------------------------
    tick = time.monotonic()
    stage = _stage("events")
    event_rows: list[dict] = []
    ev_runs = 0
    rollup_truncated = False
    try:
        rollup = query.query_clusters(
            date_start=start_day, date_end=end_day, bbox=eff_bbox,
            limit=ROLLUP_CAP + 1).to_dict("records")
        rollup_truncated = len(rollup) > ROLLUP_CAP
        rollup = rollup[:ROLLUP_CAP]
        for cluster in rollup:
            try:
                eid = association.event_id_for(cluster["cluster_id"])
            except ValueError:
                continue
            event = store.get_event(events_conn, eid)
            if event is None:
                continue
            event_rows.append(_event_rollup(event, events_conn, store))
        for row in sorted(event_rows,
                          key=lambda r: r.get("observation_count") or 0,
                          reverse=True)[:top_events]:
            built = payloads.build_event_payload_for(
                row["event_id"], events_conn)
            if built is None:
                continue
            _, payload = built
            outcome = agents_coordinator.analyze(
                "event", row["event_id"], payload, chain=chain)
            _record_run(result, outcome, "event-analysis",
                        row["event_id"])
            row["analysis_status"] = outcome.steps[0].status
            ev_runs += 1
        detail = (f"{len(event_rows)} events in rollup"
                  + (" (truncated)" if rollup_truncated else "")
                  + f", {ev_runs} analyzed")
        _done(stage, "ok" if event_rows else "failed",
              detail if event_rows else "no events resolved", tick)
    except Exception as exc:  # noqa: BLE001
        _done(stage, "failed", f"event stage failed: {exc}", tick)
    result.events = {"rollup": event_rows,
                     "rollup_truncated": rollup_truncated}

    # -- observation operator ----------------------------------------
    tick = time.monotonic()
    stage = _stage("operator")
    try:
        rows = _fetch_operator_rows(start_day, end_day, operator_cap,
                                    eff_bbox)
        truncated = len(rows) > operator_cap
        rows = rows[:operator_cap]
        matcher = None
        l2_note = "L2 matching off"
        if l2_match:
            try:
                inventory = fetch_l2_inventory(start_day, end_day,
                                               bbox=eff_bbox)
                matcher = build_l2_matcher(inventory["granules"])
                l2_note = (f"L2 inventory {len(inventory['granules'])}"
                           f"/{inventory['hits']} granules"
                           + (" (truncated)" if inventory["truncated"]
                              else ""))
            except Exception as exc:  # noqa: BLE001 - L2 is optional
                l2_note = f"L2 inventory failed: {exc}"
        records = []
        l2_hits = 0
        for row in rows:
            granule_ur, match_status = (None, "unmatched")
            if matcher is not None:
                try:
                    granule_ur, match_status = matcher(row)
                except Exception:  # noqa: BLE001 - per-row isolation
                    granule_ur, match_status = None, "unmatched"
                l2_hits += match_status == "matched"
            verdict = oop_validation.validate_row(row)
            if not verdict["eligible"]:
                records.append(oor.result_record(
                    row, None, None,
                    reasons=tuple(verdict["reasons"]),
                    granule_ur=granule_ur, match_status=match_status))
                continue
            calc = oop.expected_frp(row["frp"], verdict["area"],
                                    row["product"], row["daynight"])
            regime = oop.detection_regime(row["frp"], verdict["area"],
                                          row["product"], row["daynight"])
            records.append(oor.result_record(
                row, calc, regime, area_km2=verdict["area"],
                granule_ur=granule_ur, match_status=match_status))
        agg = oor.aggregate_record(records)
        result.operator = {"records": records, "aggregate": agg,
                           "truncated": truncated, "cap": operator_cap,
                           "l2_matched": l2_hits, "l2_note": l2_note}
        # Science agent narrates the aggregates (numbers precomputed).
        sci_payload = {
            "method_version": "det-frp-oop-v1",
            "n": agg["n_observations"],
            "observed_sum": sum(
                (r.get("observed_frp") or 0.0) for r in records
                if r["calculation_status"] == "ok"),
            "expected_sum": sum(
                (r.get("expected_retrieved_frp_MW") or 0.0)
                for r in records
                if r["calculation_status"] == "ok"),
            "by_sensor": {
                s: {"n": v["n"], "observed": v["observed_sum_MW"],
                    "expected": v["expected_sum_MW"]}
                for s, v in agg["by_sensor"].items()},
            "mapping_applied": False,
            "inputs_missing": list(
                payloads.SCIENCE_MISSING_INPUTS),
            "limitations": [
                "Range run shares the pilot's exploratory input "
                "interpretation."],
        }
        outcome = agents_coordinator.analyze(
            "science", f"range:{start_day}:{end_day}", sci_payload,
            chain=chain)
        _record_run(result, outcome, "science-observer",
                    f"{start_day}:{end_day}")
        _done(stage, "ok" if agg["n_eligible"] else "failed",
              f"{agg['n_eligible']}/{agg['n_observations']} eligible"
              + (" (truncated)" if truncated else "")
              + f"; {l2_note}"
              + (f", {l2_hits} L2-matched" if matcher else ""), tick)
    except Exception as exc:  # noqa: BLE001
        _done(stage, "failed", f"operator stage failed: {exc}", tick)
        result.operator = {"records": [], "aggregate": {},
                           "truncated": False, "cap": operator_cap}

    # -- validation --------------------------------------------------
    tick = time.monotonic()
    stage = _stage("validation")
    checks = _validate_result(result, start_day, end_day)
    result.validation = {"checks": checks,
                         "passed": sum(1 for c in checks
                                       if c["passed"]),
                         "failed": sum(1 for c in checks
                                       if not c["passed"])}
    failed = [c for c in checks if not c["passed"]]
    for row in result.events.get("rollup", []):
        row["validation_status"] = "failed" if failed else "passed"
    _done(stage, "failed" if failed else "ok",
          "; ".join(c["check"] for c in failed)
          or f"{len(checks)} checks passed", tick)

    # -- export ------------------------------------------------------
    tick = time.monotonic()
    stage = _stage("export")
    provenance = (f"pipeline {exec_id}; 2B {settings.CAL_EXEC_2B[:8]}, "
                  f"2C {settings.CAL_EXEC_2C[:8]}, "
                  f"2D {settings.CAL_EXEC_2D[:8]}")
    try:
        result.csvs = {
            "observations": exports_csv.observation_csv(
                result.operator.get("records", []),
                provenance=provenance, processing_utc=started,
                execution_id=exec_id),
            "events": exports_csv.event_csv(
                result.events.get("rollup", []),
                provenance=provenance, processing_utc=started,
                execution_id=exec_id),
            "agents": exports_csv.agent_csv(
                result.agent_runs, processing_utc=started,
                execution_id=exec_id),
        }
        _done(stage, "ok", "3 CSVs built (metrics follows)", tick)
    except Exception as exc:  # noqa: BLE001
        _done(stage, "failed", f"export failed: {exc}", tick)

    # -- metrics -----------------------------------------------------
    tick = time.monotonic()
    stage = _stage("metrics")
    try:
        _compute_metrics(result, stages, t_all, chain_names,
                         fingerprint)
        result.csvs["metrics"] = exports_csv.metrics_csv(
            result.metrics, processing_utc=started,
            execution_id=exec_id)
        _done(stage, "ok",
              f"{len(result.metrics)} metrics + metrics CSV", tick)
    except Exception as exc:  # noqa: BLE001
        _done(stage, "failed", f"metrics failed: {exc}", tick)

    _finalize(result)
    return result


def _skip_rest(result, stages, progress, names, reason):
    for name in names:
        stage = PipelineStage(name, status="skipped", detail=reason)
        stages[name] = stage
        result.stages.append(stage)
        if progress:
            progress(name, "skipped", reason)


def _record_run(result, outcome, agent, target):
    step = outcome.steps[0]
    result.agent_runs.append({
        "agent": agent, "target_id": target, "status": step.status,
        "provider": step.provider, "model": step.model,
        "latency_ms": step.latency_ms, "retries": step.retries,
        "response_chars": step.response_chars,
        "input_tokens": step.input_tokens,
        "generated_tokens": step.generated_tokens,
        "fallback_reason": step.fallback_reason,
        "validation": step.validation,
        "error_category": "none" if step.status == "ok"
        else ("fallback" if step.status == "fallback" else "error"),
        "summary": outcome.text,
        "evidence": list(outcome.evidence_ids),
    })


def _live_cluster_timeline(sub_df) -> list[dict]:
    """Per-UTC-day observation summary supporting MODIS + all VIIRS sensors."""
    import pandas as _pd

    if sub_df.empty:
        return []
    day_series = sub_df["acquisition_timestamp_utc"].astype(str).str.slice(0, 10)
    rows = []
    for day_val, part in sub_df.assign(_d=day_series).groupby("_d", sort=True):
        frp = _pd.to_numeric(part["frp"], errors="coerce")
        is_m = part["product"].astype(str) == "modis"
        is_v = part["product"].astype(str).str.startswith("viirs")
        rows.append({
            "day": str(day_val),
            "n": int(len(part)),
            "n_modis": int(is_m.sum()),
            "n_viirs": int(is_v.sum()),
            "frp_modis": float(frp[is_m].sum()),
            "frp_viirs": float(frp[is_v].sum()),
            "frp_missing": int(frp.isna().sum()),
        })
    return rows


def _event_rollup(event: dict, conn, store) -> dict:
    alerts = store.list_alerts(conn, event_id=event["event_id"],
                               limit=500) if conn is not None else []
    frp = event.get("frp") or {}
    sevs: dict[str, int] = {}
    for alert in alerts:
        sev = str(alert.get("severity_name", "unknown"))
        sevs[sev] = sevs.get(sev, 0) + 1
    return {
        "event_id": event["event_id"],
        "cluster_ids": event.get("cluster_ids"),
        "date_start": str(event.get("first_observed_utc") or "")[:10],
        "date_end": str(event.get("last_observed_utc") or "")[:10],
        "lifecycle_state": event.get("lifecycle_state"),
        "severity": event.get("severity_name"),
        "observation_count": event.get("detection_count"),
        "n_modis": event.get("sensor_modis_n"),
        "n_viirs": event.get("sensor_viirs_n"),
        "satellites": event.get("satellites"),
        "peak_frp": frp.get("frp_max"),
        "frp_sum": frp.get("frp_sum"),
        "frp_mean": frp.get("frp_mean"),
        "frp_missing": frp.get("frp_null"),
        "operator_expected": None,
        "operator_coverage": "not matched to operator rows",
        "completeness": event.get("completeness"),
        "n_alerts": len(alerts),
        "alert_severities": sevs,
        "analysis_status": "pending",
        "validation_status": "pending",
    }


def _fetch_operator_rows(start_day: str, end_day: str, cap: int,
                       bbox=None):
    """Detection rows ordered by (day, id) for deterministic runs."""
    import duckdb

    from .. import settings

    clauses = ["substr(acquisition_timestamp_utc,1,10) >= ?",
               "substr(acquisition_timestamp_utc,1,10) <= ?"]
    params: list = [start_day, end_day]
    if bbox:
        w, s, e, n = (float(v) for v in bbox)
        clauses.append("longitude BETWEEN ? AND ? "
                       "AND latitude BETWEEN ? AND ?")
        params.extend([w, e, s, n])
    con = duckdb.connect(settings.CAL_DUCKDB_PATH, read_only=True)
    try:
        rows = con.execute(
            f"""SELECT detection_id, product, satellite, sensor, daynight,
                      frp, scan, track, latitude, longitude,
                      acquisition_timestamp_utc
               FROM detections
               WHERE {' AND '.join(clauses)}
               ORDER BY acquisition_timestamp_utc, detection_id
               LIMIT ?""",
            [*params, cap + 1]).fetchall()
        cols = [d[0] for d in con.description]
    finally:
        con.close()
    return [dict(zip(cols, r)) for r in rows]


def _validate_result(result, start_day: str, end_day: str) -> list:
    """Deterministic pipeline checks (Agent 4's evidence role)."""
    checks = []
    records = result.operator.get("records", [])

    def _check(name: str, passed: bool, detail: str = ""):
        checks.append({"check": name, "passed": bool(passed),
                       "detail": detail})

    eligible = [r for r in records
                if r.get("calculation_status") == "ok"]
    _check("row-accounting",
           len(records) == len(eligible) + sum(
               1 for r in records
               if str(r.get("calculation_status", "")).startswith(
                   "excluded")),
           f"{len(eligible)}/{len(records)} eligible")
    _check("date-boundaries", all(
        start_day <= str(r.get("acquisition_utc") or "")[:10] <= end_day
        or not r.get("acquisition_utc") for r in records))
    _check("sensor-identity", all(
        r.get("sensor") in ("snpp", "noaa20", "noaa21", "terra_aqua", None)
        and r.get("product") in ("modis", "viirs-snpp", "viirs-noaa20", "viirs-noaa21", None)
        for r in records) and all(
        r.get("sensor") in ("snpp", "terra_aqua")
        and r.get("product") in ("modis", "viirs-snpp")
        for r in eligible))
    _check("frp-units",
           all(r.get("observed_frp_units") in ("MW", None)
               for r in records))
    _check("missing-stays-missing", all(
        not (r.get("aod_550") == 0
             and r.get("aod_status") == "unavailable")
        and not (r.get("cloud_flag") == 0
                 and r.get("cloud_status") == "unavailable")
        for r in records))
    bounds = True
    for record in eligible:
        prob = record.get("detection_probability")
        exp = record.get("expected_retrieved_frp_MW")
        obs = record.get("observed_frp")
        if prob is None or not 0.0 <= prob <= 1.0:
            bounds = False
        if exp is None or obs is None or exp < 0 or exp > obs + 1e-6:
            bounds = False
    _check("numerical-bounds", bounds)
    _check("agent-no-errors", all(
        run.get("status") != "error" for run in result.agent_runs),
           f"{sum(1 for run in result.agent_runs if run.get('status') == 'error')} error(s)")
    _check("provenance-present", bool(
        result.config.get("exec_2b") or result.config.get("firms_exec_id")))
    return checks


def _compute_metrics(result, stages, t_all, chain_names, fingerprint):
    from . import exports_csv

    total_ms = round((time.monotonic() - t_all) * 1000, 1)
    runs = result.agent_runs
    agg = (result.operator.get("aggregate") or {})
    checks = (result.validation.get("checks") or [])
    result.metrics = [
        {"metric": "total_latency_ms", "value": total_ms,
         "unit": "ms"},
        *[{"metric": f"stage_{name}_latency_ms",
           "value": stage.latency_ms, "unit": "ms"}
          for name, stage in stages.items()],
        {"metric": "n_detections", "value": result.acquisition.get(
            "n_detections", 0), "unit": "rows"},
        {"metric": "n_clusters", "value": result.acquisition.get(
            "n_clusters", 0), "unit": "rows"},
        {"metric": "operator_eligible", "value": agg.get(
            "n_eligible", 0), "unit": "rows"},
        {"metric": "operator_excluded", "value": agg.get(
            "n_excluded", 0), "unit": "rows"},
        {"metric": "l2_matched",
         "value": result.operator.get("l2_matched", 0),
         "unit": "rows"},
        {"metric": "agent_runs", "value": len(runs), "unit": "runs"},
        {"metric": "agent_fallbacks",
         "value": sum(1 for r in runs if r.get("status") != "ok"),
         "unit": "runs"},
        {"metric": "agent_errors",
         "value": sum(1 for r in runs if r.get("status") == "error"),
         "unit": "runs"},
        {"metric": "tokens_in_total",
         "value": sum(r.get("input_tokens") or 0 for r in runs),
         "unit": "tokens"},
        {"metric": "tokens_out_total",
         "value": sum(r.get("generated_tokens") or 0 for r in runs),
         "unit": "tokens"},
        {"metric": "validation_passed",
         "value": sum(1 for c in checks if c["passed"]),
         "unit": "checks"},
        {"metric": "validation_failed",
         "value": sum(1 for c in checks if not c["passed"]),
         "unit": "checks"},
        {"metric": "providers", "value": ",".join(chain_names),
         "unit": "names"},
        {"metric": "data_fingerprint",
         "value": str(fingerprint)[:64], "unit": "hash"},
        {"metric": "csv_schema", "value": exports_csv.SCHEMA_VERSION,
         "unit": "version"},
    ]


def _finalize(result):
    result.limitations = [
        "Operator outputs are experimental (exploratory FIRMS-input "
        "interpretation; clear-sky assumed).",
        "LLM narratives are validated aids; deterministic fallbacks "
        "are labeled per agent run.",
        "Level-2 fields are unmatched until granules are acquired.",
    ]

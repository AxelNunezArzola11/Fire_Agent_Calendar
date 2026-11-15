"""Targeted verification for Historical Analytics fixes, Global Regions, Granite & Fallback."""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.calendar.acquisition import nrt
from src.calendar.agents import granite, providers
from src.calendar.events import qwen
from src.calendar.pipeline import regional
from src.calendar.processing import regions
from src.calendar.ui import analytics_view, aoi, map_view, query


def test_incomplete_execution_rejected_in_analytics_status():
    st = query.artifact_status(
        "2C", "clusters.parquet", exec_id="8e3f4e4d686e47999227284b078becfc"
    )
    assert st["available"] is False
    assert st["status"] == "incomplete_execution"
    assert "running" in st["reason"]


def test_historical_analytics_window_and_float64_chart_frame():
    start, end, why = analytics_view.resolve_analytics_window(
        "2023-08-01", "2023-08-01", mode="auto"
    )
    assert (start, end, why) == ("2023-01-01", "2023-12-31", "auto-expanded")
    sample = [
        {"eval_period": "2023-01-02", "observed_value": 10, "baseline_value": None, "z_score": None, "pct_deviation": None},
        {"eval_period": "2023-01-01", "observed_value": 5, "baseline_value": None, "z_score": None, "pct_deviation": None},
        {"eval_period": "2023-01-04", "observed_value": 25, "baseline_value": 10.0, "z_score": 2.4, "pct_deviation": 150.0},
    ]
    chart = analytics_view.prepare_chart_frame(sample)
    assert list(chart["period"]) == ["2023-01-01", "2023-01-02", "2023-01-04"]
    for col in ("observed", "baseline", "z_score", "pct_deviation"):
        assert str(chart[col].dtype) == "float64"
    assert pd.isna(chart["baseline"].iloc[0])


def test_global_regions_and_firms_regional_normalization():
    for rid in ("california", "brazil", "spain", "australia"):
        bbox = aoi.resolve_bbox(mode="region", region_id=rid)
        assert len(bbox) == 4 and bbox[0] < bbox[2] and bbox[1] < bbox[3]
        layer = map_view.boundary_layer(region=regions.REGIONS[rid])
        assert len(layer._children) == 1

    # Mock live FIRMS CSV response inside Brazil bbox (never confuses California)
    csv_bytes = (
        "latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_t31,frp,daynight\n"
        "-10.5,-52.0,320.1,1.0,1.0,2026-08-01,1630,Terra,MODIS,90,6.1NRT,295.0,45.2,D\n"
        "-11.0,-51.5,315.0,1.1,1.1,2026-08-02,1700,Terra,MODIS,85,6.1NRT,292.0,31.0,D\n"
    ).encode("utf-8")

    def fake_fetch(source, bbox_str, days, end_d, *, key=None):
        _ = (bbox_str, days, key)
        req = nrt.AreaRequest(source, bbox_str, 2, date(2026, 8, 1))
        return [
            nrt.ChunkResult(
                request=req,
                content=csv_bytes if "MODIS" in source else b"",
                sha256="a" * 64,
                status="ok" if "MODIS" in source else "empty",
                attempts=1,
                elapsed_s=0.05,
                fetched_at="2026-08-03T00:00:00+00:00",
            )
        ]

    ds = regional.fetch_firms_regional(
        bbox=regions.BRAZIL.bbox,
        date_start="2026-08-01",
        date_end="2026-08-04",
        region_id="brazil",
        region_label="Brazil",
        key="DUMMY-MAP-KEY",
        fetcher=fake_fetch,
    )
    assert ds.n_detections == 2
    assert ds.data_origin == "firms_live_api"
    ts = regional.build_regional_timeseries(
        ds.df,
        metric="detection",
        scope="raw",
        grain="daily",
        date_start="2026-08-01",
        date_end="2026-08-04",
        region_label="Brazil",
        exec_id=ds.exec_id,
    )
    assert len(ts) == 4
    assert [r["observed_value"] for r in ts] == [1, 1, 0, 0]
    assert ts[3]["baseline_value"] is not None  # n=3 preceding trailing days


def test_granite_openai_and_watsonx_modes_plus_deterministic_fallback():
    oa_cfg = granite.GraniteConfig(
        api_key="DUMMY",
        url="https://granite.example.invalid/v1",
        model="ibm/granite-3-8b-instruct",
    )
    assert oa_cfg.is_openai_compatible() is True
    assert oa_cfg.status()["available"] is True

    calls = []

    def fake_post(url, *, headers, json_data=None, form_data=None, timeout):
        calls.append(url)
        class _Resp:
            status_code = 200
            def json(self):
                return {
                    "choices": [{
                        "message": {
                            "content": '{"summary": "Event ev-1 observed 4 detections.", "evidence_used": ["ev-1"], "uncertainty": ["none"], "limitations": ["Satellite observations are not confirmed fires."]}'
                        }
                    }],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 10},
                }
        return _Resp()

    res_oa = qwen.explain(
        "event",
        "ev-1",
        {
            "kind": "event",
            "event_id": "ev-1",
            "lifecycle_state": "active",
            "severity": "moderate",
            "detection_count": 4,
            "limitations": ["Satellite observations are not confirmed fires."],
        },
        ai_provider="granite",
        granite_config=oa_cfg,
        granite_post=fake_post,
    )
    assert res_oa["status"] == "ok"
    assert res_oa["provider"] == "granite"
    assert calls[0].endswith("/chat/completions")

    # Granite failure triggers deterministic rules fallback with rules applied
    res_det = qwen.explain(
        "event",
        "ev-1",
        {
            "kind": "event",
            "event_id": "ev-1",
            "lifecycle_state": "active",
            "severity": "moderate",
            "detection_count": 4,
            "sensor_modis_n": 1,
            "sensor_viirs_n": 3,
            "first_observed_utc": "2023-08-01",
            "last_observed_utc": "2023-08-01",
            "frp": {"frp_max": None},
            "limitations": ["Satellite observations are not confirmed fires."],
        },
        ai_provider="deterministic",
    )
    assert res_det["status"] == "deterministic"
    assert res_det["provider"] == "deterministic"
    assert "RULE_LIFECYCLE_STATE=active" in res_det["text"]
    assert [p.name for p in providers.default_chain("granite").providers] == ["granite"]
    assert providers.default_chain("deterministic").providers == ()


def test_multiyear_firms_source_selection_and_five_satellites():
    from pathlib import Path
    from src.calendar.acquisition import sensors
    from src.calendar.ui import filters

    # 1. All 4 products & 5 satellites are supported in UI filters and queries
    assert filters.PRODUCTS == ("modis", "viirs-snpp", "viirs-noaa20", "viirs-noaa21")
    assert filters.SATELLITES == ("Terra", "Aqua", "N", "NOAA-20", "NOAA-21")
    assert query.SATELLITES_UNAVAILABLE == ()

    # 2. Alias canonicalization matches FIRMS Area API satellite codes
    assert sensors.canonical_satellite("N20", product="viirs-noaa20") == "NOAA-20"
    assert sensors.canonical_satellite("N21", product="viirs-noaa21") == "NOAA-21"
    assert sensors.canonical_satellite("NPP", product="viirs-snpp") == "N"
    assert set(sensors.expand_satellite_aliases(("NOAA-20", "NOAA-21"))) >= {
        "NOAA-20", "N20", "J1", "NOAA-21", "N21", "J2"
    }

    # 3. Multi-year source selection respects real upstream launch dates without 2023 lock-in
    s_2005 = regional.select_firms_sources(
        "2005-07-01", "2005-07-05", products=filters.PRODUCTS, today_utc=date(2026, 11, 15)
    )
    assert s_2005 == ("MODIS_SP",)

    s_2015 = regional.select_firms_sources(
        "2015-07-01", "2015-07-05", products=filters.PRODUCTS, today_utc=date(2026, 11, 15)
    )
    assert set(s_2015) == {"MODIS_SP", "VIIRS_SNPP_SP"}

    s_2020 = regional.select_firms_sources(
        "2020-07-01", "2020-07-05", products=filters.PRODUCTS, today_utc=date(2026, 11, 15)
    )
    assert set(s_2020) == {"MODIS_SP", "VIIRS_SNPP_SP", "VIIRS_NOAA20_SP"}

    s_2024 = regional.select_firms_sources(
        "2024-08-01", "2024-08-05", products=filters.PRODUCTS, today_utc=date(2026, 11, 15)
    )
    assert set(s_2024) == {"MODIS_SP", "VIIRS_SNPP_SP", "VIIRS_NOAA20_SP", "VIIRS_NOAA21_SP"}

    s_nrt = regional.select_firms_sources(
        "2026-11-10", "2026-11-15", products=filters.PRODUCTS, today_utc=date(2026, 11, 15)
    )
    assert set(s_nrt) == {"MODIS_NRT", "VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT"}

    # 3b. When live availability shows SP covers the entire range, NRT is NOT duplicated even if NRT overlaps
    overlap_avail = {
        "VIIRS_NOAA21_SP": {"min_date": "2023-03-01", "max_date": "2026-07-31"},
        "VIIRS_NOAA21_NRT": {"min_date": "2024-01-17", "max_date": "2026-11-15"},
    }
    s_no_dup = regional.select_firms_sources(
        "2024-08-01",
        "2024-08-05",
        products=("viirs-noaa21",),
        availability=overlap_avail,
        today_utc=date(2026, 11, 15),
    )
    assert s_no_dup == ("VIIRS_NOAA21_SP",)

    # 4. Single out-of-pilot date in resolve_analytics_window does not revert to 2023
    w_s, w_e, w_why = analytics_view.resolve_analytics_window(
        "2024-08-01", "2024-08-01", mode="auto"
    )
    assert (w_s, w_e, w_why) == ("2024-08-01", "2024-08-01", "sidebar")

    # 5. Dynamic availability replaces static 2023-only copy
    avail = sensors.evaluate_availability(
        "2024-08-01", "2024-08-05",
        filters.PRODUCTS, ("NOAA-20", "NOAA-21"),
        firms_key_configured=True, region_label="Spain",
    )
    assert "VIIRS_NOAA20_SP" in avail["firms_sources"]
    assert "VIIRS_NOAA21_SP" in avail["firms_sources"]
    assert all("pilot year 2023 only" not in lim.lower() for lim in avail["limitations"])

    # 6. End-to-end FIRMS normalization + regional analytics + detection map with N20/N21 in 2024
    csv_map = {
        "VIIRS_NOAA20_SP": (
            "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight\n"
            "40.40,-3.70,335.0,0.4,0.4,2024-08-01,1300,N20,VIIRS,h,2.0NRT,295.0,19.5,D\n"
        ).encode("utf-8"),
        "VIIRS_NOAA21_SP": (
            "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight\n"
            "40.41,-3.71,338.0,0.4,0.4,2024-08-02,1315,N21,VIIRS,n,2.0NRT,296.0,22.0,D\n"
        ).encode("utf-8"),
    }

    def fake_fetch(source, bbox_str, days, end_d, *, key=None):
        _ = (days, end_d, key)
        req = nrt.AreaRequest(source, bbox_str, 2, date(2024, 8, 1))
        body = csv_map.get(source, b"")
        return [
            nrt.ChunkResult(
                request=req,
                content=body,
                sha256="b" * 64,
                status="ok" if body else "empty",
                attempts=1,
                elapsed_s=0.05,
                fetched_at="2024-08-03T00:00:00+00:00",
            )
        ]

    spain_bbox = regions.REGIONS["spain"].bbox
    ds = regional.fetch_firms_regional(
        bbox=spain_bbox,
        date_start="2024-08-01",
        date_end="2024-08-02",
        region_id="spain",
        region_label="Spain",
        products=("viirs-noaa20", "viirs-noaa21"),
        satellites=("NOAA-20", "NOAA-21"),
        key="DUMMY-MAP-KEY",
        fetcher=fake_fetch,
    )
    assert ds.n_detections == 2
    assert set(ds.df["satellite"]) == {"NOAA-20", "NOAA-21"}
    ts = query.query_regional_timeseries(
        date_start="2024-08-01",
        date_end="2024-08-02",
        bbox=spain_bbox,
        metric="detection",
        scope="NOAA-20",
        grain="daily",
        satellites=("NOAA-20", "NOAA-21"),
        extra_df=ds.df,
    )
    assert len(ts) == 2
    assert ts[0]["observed_value"] == 1
    det_map_html = map_view.build_detection_map(ds.df).get_root().render()
    assert "VIIRS NOAA-20" in det_map_html and "VIIRS NOAA-21" in det_map_html

    # 7. Hardcoded 2023-only / NOAA-disabled UI copy has been removed
    root = Path(__file__).resolve().parent.parent
    for rel in ("calendar_app.py", "src/calendar/ui/event_report.py", "src/calendar/ui/analytics_view.py"):
        txt = (root / rel).read_text(encoding="utf-8").lower()
        assert "pilot year 2023 only" not in txt
        assert "noaa-20/21 have no pilot data" not in txt
        assert "noaa-20/21 have no 2023 pilot data" not in txt


def test_centralized_aoi_resolver_priority_and_validation():
    import math
    import pytest
    from src.calendar.ui import filters

    # 1. Explicit bounding box validates and resolves
    r_bbox = aoi.resolve_aoi(bbox=(-9.0, 36.0, 3.0, 43.0))
    assert r_bbox is not None
    assert r_bbox.source == "bbox"
    assert r_bbox.bbox == (-9.0, 36.0, 3.0, 43.0)

    # 2. Named region resolves (case-insensitive / label lookup)
    r_reg = aoi.resolve_aoi(region_id="Spain")
    assert r_reg is not None
    assert r_reg.source == "region"
    assert r_reg.region_id == "spain"
    assert r_reg.bbox == regions.REGIONS["spain"].bbox

    # 3. Map geometry resolves from Leaflet bounds or GeoJSON geometry
    r_map = aoi.resolve_aoi(map_geometry={
        "_southWest": {"lat": 37.0, "lng": -6.0},
        "_northEast": {"lat": 41.0, "lng": -2.0},
    })
    assert r_map is not None
    assert r_map.source == "map_geometry"
    assert r_map.bbox == (-6.0, 37.0, -2.0, 41.0)

    r_geojson = aoi.resolve_aoi(map_geometry={
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[-4.0, 40.0], [-3.0, 40.0], [-3.0, 41.0], [-4.0, 41.0], [-4.0, 40.0]]],
        },
    })
    assert r_geojson is not None and r_geojson.bbox == (-4.0, 40.0, -3.0, 41.0)

    # 4. Point + radius resolves and passes consistency validation
    r_pr = aoi.resolve_aoi(latitude=40.4168, longitude=-3.7038, radius_km=75.0)
    assert r_pr is not None
    assert r_pr.source == "point_radius"
    assert aoi.validate_point_bbox_consistency(40.4168, -3.7038, 75.0, r_pr.bbox) == r_pr.bbox
    with pytest.raises(aoi.AoIError, match="outside bounding box"):
        aoi.validate_point_bbox_consistency(40.4168, -3.7038, 75.0, (-124.0, 32.0, -114.0, 42.0))
    with pytest.raises(aoi.AoIError, match="inconsistent"):
        aoi.validate_point_bbox_consistency(40.4168, -3.7038, 75.0, (-6.0, 38.0, -1.0, 43.0))

    # 5. Session fallback & persistence via resolve_session_aoi
    state = {filters.KEY_AOI_MODE: "point_radius", filters.KEY_AOI_LAT: -10.5,
             filters.KEY_AOI_LON: -52.0, filters.KEY_AOI_RADIUS: 60.0}
    resolved_st = aoi.resolve_session_aoi(state)
    assert resolved_st is not None and resolved_st.source == "point_radius"
    assert state[filters.KEY_RESOLVED_AOI]["bbox"] == list(resolved_st.bbox)

    # Session fallback recovers previously selected AOI when mode="auto" or fallback requested
    reused = aoi.resolve_session_aoi(
        {filters.KEY_AOI_MODE: "full", filters.KEY_RESOLVED_AOI: state[filters.KEY_RESOLVED_AOI]},
        include_session_fallback=True,
    )
    assert reused is not None and reused.source == "session" and reused.bbox == resolved_st.bbox

    # Strict 5-tier priority: bbox > region > map_geometry > point_radius > previous_aoi
    assert aoi.resolve_aoi(
        bbox=(-5.0, 38.0, -1.0, 41.0),
        region_id="brazil",
        map_geometry=(-70.0, -10.0, -60.0, -5.0),
        point_radius=(40.0, -3.0, 50.0),
        previous_aoi=r_reg,
    ).source == "bbox"
    assert aoi.resolve_aoi(
        region_id="brazil",
        map_geometry=(-70.0, -10.0, -60.0, -5.0),
        point_radius=(40.0, -3.0, 50.0),
        previous_aoi=r_reg,
    ).source == "region"
    assert aoi.resolve_aoi(
        map_geometry=(-70.0, -10.0, -60.0, -5.0),
        point_radius=(40.0, -3.0, 50.0),
        previous_aoi=r_reg,
    ).bbox == (-70.0, -10.0, -60.0, -5.0)
    assert aoi.resolve_aoi(point_radius=(40.0, -3.0, 50.0), previous_aoi=r_reg).source == "point_radius"
    assert aoi.resolve_aoi(previous_aoi=r_reg).source == "session"

    # Never silently defaults to California or CONUS when missing/invalid
    assert aoi.resolve_session_aoi({filters.KEY_AOI_MODE: "full"}) is None
    with pytest.raises(aoi.AoIError, match="No Area of Interest"):
        aoi.resolve_aoi(require=True)
    for bad_bbox in (
        (10.0, 35.0, -10.0, 42.0),  # W >= E
        (-10.0, 42.0, 10.0, 35.0),  # S >= N
        (-10.0, 40.0, -10.0, 42.0),  # zero width / zero area
        (True, 35.0, 10.0, 42.0),  # boolean
        (math.nan, 35.0, 10.0, 42.0),  # NaN
    ):
        with pytest.raises(aoi.AoIError):
            aoi.resolve_aoi(bbox=bad_bbox)


def test_granite_pipeline_automatic_aoi_and_end_to_end_stages():
    import csv
    import io
    import json as _json
    from src.calendar.events import store as ev_store
    from src.calendar.pipeline import exports_csv, runner as pipe_runner

    conn = ev_store.connect(":memory:")
    ev_store.init_schema(conn)

    seen_fetches: list[tuple[str, str]] = []

    def fake_fetch(source, bbox_str, days, end_d, *, key=None):
        _ = (days, end_d, key)
        seen_fetches.append((source, bbox_str))
        # Multi-sensor detections inside Spain & around (40.40, -3.70)
        bodies = {
            "MODIS_SP": (
                "latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_t31,frp,daynight\n"
                "40.400,-3.700,330.0,1.0,1.0,2024-08-01,1130,Terra,MODIS,90,6.1NRT,295.0,42.5,D\n"
            ),
            "VIIRS_SNPP_SP": (
                "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight\n"
                "40.402,-3.702,336.0,0.4,0.4,2024-08-01,1315,N,VIIRS,h,2.0NRT,296.0,18.2,D\n"
                "40.404,-3.703,339.0,0.4,0.4,2024-08-02,1300,N,VIIRS,n,2.0NRT,297.0,21.0,D\n"
            ),
            "VIIRS_NOAA20_SP": (
                "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight\n"
                "40.405,-3.704,335.0,0.4,0.4,2024-08-01,1320,N20,VIIRS,h,2.0NRT,295.0,19.5,D\n"
            ),
        }
        body = bodies.get(source, "").encode("utf-8")
        return [
            nrt.ChunkResult(
                request=nrt.AreaRequest(source, bbox_str, 2, date(2024, 8, 1)),
                content=body,
                sha256="c" * 64,
                status="ok" if body else "empty",
                attempts=1,
                elapsed_s=0.05,
                fetched_at="2024-08-03T00:00:00+00:00",
            )
        ]

    # 1. Missing AOI fails clearly and NEVER calls FIRMS with CONUS/California
    res_missing = pipe_runner.run_pipeline(
        "2024-08-01",
        "2024-08-02",
        events_conn=conn,
        chain=providers.ProviderChain([granite.GraniteProvider()]),
        fetch_live_firms=True,
        firms_fetcher=fake_fetch,
        firms_key="DUMMY-MAP-KEY",
    )
    assert res_missing.stages[0].status == "failed"
    assert "No Area of Interest" in res_missing.stages[0].detail
    assert all(s.status == "skipped" for s in res_missing.stages[1:])
    assert seen_fetches == []

    # 2. Invalid AOI fails clearly without executing acquisition
    res_bad = pipe_runner.run_pipeline(
        "2024-08-01",
        "2024-08-02",
        events_conn=conn,
        chain=providers.ProviderChain([granite.GraniteProvider()]),
        fetch_live_firms=True,
        bbox=(10.0, 40.0, -10.0, 42.0),
        firms_fetcher=fake_fetch,
        firms_key="DUMMY-MAP-KEY",
    )
    assert res_bad.stages[0].status == "failed"
    assert "Invalid AOI" in res_bad.stages[0].detail
    assert seen_fetches == []

    # 3. Point + radius with mocked IBM Granite OpenAI-style REST endpoint -> full pipeline 'ok'
    class _FakeHttpResp:
        def __init__(self, status_code: int, body: dict):
            self.status_code = status_code
            self._body = body

        def json(self):
            return self._body

    def _mock_granite_post(url, *, headers, json_data=None, form_data=None, timeout=30.0):
        _ = (url, headers, form_data, timeout)
        from src.calendar.agents import evidence_validator

        user_msg = json_data["messages"][1]["content"]
        raw_json = (
            user_msg.split("EVIDENCE (JSON, authoritative):\n", 1)[1]
            .split("\n\nCitable evidence values", 1)[0]
        )
        ev_payload = _json.loads(raw_json)
        citable = evidence_validator.evidence_ids(ev_payload)
        ev_ref = citable[0] if citable else (
            ev_payload.get("cluster_id")
            or ev_payload.get("event_id")
            or ev_payload.get("method_version")
            or "verified"
        )
        content = _json.dumps({
            "summary": f"Verified analysis for evidence record {ev_ref}.",
            "evidence_used": [str(ev_ref)] if citable else [],
            "uncertainty": ["Satellite hotspots are observations, not confirmed incidents."],
            "limitations": list(ev_payload.get("limitations") or ["Exploratory FRP estimate."]),
        })
        return _FakeHttpResp(200, {
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 25, "completion_tokens": 18},
        })

    g_cfg = granite.GraniteConfig(
        api_key="dummy-granite-key",
        url="https://mock-granite.example/v1",
        model="ibm/granite-4-h-small",
        endpoint_style="openai",
    )
    g_chain = providers.ProviderChain([granite.GraniteProvider(config=g_cfg, post=_mock_granite_post)])

    res_pr = pipe_runner.run_pipeline(
        "2024-08-01",
        "2024-08-02",
        events_conn=conn,
        chain=g_chain,
        point_radius=(40.40, -3.70, 50.0),
        fetch_live_firms=True,
        firms_fetcher=fake_fetch,
        firms_key="DUMMY-MAP-KEY",
    )
    assert [s.status for s in res_pr.stages] == ["ok"] * 7
    assert res_pr.config["aoi_source"] == "point_radius"
    assert res_pr.acquisition["n_detections"] == 4
    assert res_pr.acquisition["n_clusters"] >= 1
    assert len(res_pr.events["rollup"]) >= 1
    assert res_pr.validation["failed"] == 0
    assert res_pr.validation["passed"] == 8
    assert len(res_pr.agent_runs) >= 3  # detection + event + science
    assert all(r["provider"] == "granite" and r["status"] == "ok" for r in res_pr.agent_runs)
    assert set(res_pr.csvs) == {"observations", "events", "agents", "metrics"}
    obs_csv_rows = list(csv.DictReader(io.StringIO(res_pr.csvs["observations"])))
    assert len(obs_csv_rows) == 4
    ev_csv_rows = list(csv.DictReader(io.StringIO(res_pr.csvs["events"])))
    assert len(ev_csv_rows) >= 1 and list(ev_csv_rows[0].keys()) == list(exports_csv.EVENT_COLUMNS)
    # Resolved point+radius bbox reached FIRMS fetcher
    expected_pr_bbox = aoi.radius_bbox(40.40, -3.70, 50.0)
    expected_bbox_str = ",".join(f"{v:.5f}" for v in expected_pr_bbox)
    assert any(b == expected_bbox_str for _, b in seen_fetches)

    # 4. Named region + unreachable Granite -> deterministic fallback executes every stage cleanly
    def _down_granite_post(*a, **kw):
        raise ConnectionError("mock Granite unreachable")

    down_chain = providers.ProviderChain([granite.GraniteProvider(config=g_cfg, post=_down_granite_post)])
    seen_fetches.clear()
    res_reg = pipe_runner.run_pipeline(
        "2024-08-01",
        "2024-08-02",
        events_conn=conn,
        chain=down_chain,
        region_id="spain",
        fetch_live_firms=True,
        firms_fetcher=fake_fetch,
        firms_key="DUMMY-MAP-KEY",
    )
    assert [s.status for s in res_reg.stages] == ["ok"] * 7
    assert res_reg.config["region_id"] == "spain"
    assert res_reg.config["region_label"] == "Spain"
    assert all(r["status"] == "fallback" for r in res_reg.agent_runs)
    assert set(res_reg.csvs) == {"observations", "events", "agents", "metrics"}

    # 5. Zero-detection response handles empty observations honestly (never marks validation as ok)
    def fake_empty_fetch(source, bbox_str, days, end_d, *, key=None):
        _ = (days, end_d, key)
        return [
            nrt.ChunkResult(
                request=nrt.AreaRequest(source, bbox_str, 2, date(2024, 8, 1)),
                content=b"",
                sha256="e" * 64,
                status="empty",
                attempts=1,
                elapsed_s=0.02,
                fetched_at="2024-08-03T00:00:00+00:00",
            )
        ]

    res_empty = pipe_runner.run_pipeline(
        "2024-08-01",
        "2024-08-02",
        events_conn=conn,
        chain=g_chain,
        region_id="portugal",
        fetch_live_firms=True,
        firms_fetcher=fake_empty_fetch,
        firms_key="DUMMY-MAP-KEY",
    )
    by_name = {s.name: s for s in res_empty.stages}
    assert by_name["acquisition"].status == "ok"
    assert res_empty.acquisition["n_detections"] == 0
    for sk in ("detection", "events", "operator", "validation"):
        assert by_name[sk].status == "skipped"
    assert by_name["export"].status == "ok"
    assert by_name["metrics"].status == "ok"
    conn.close()


def test_independent_regional_duckdb_madrid_mexico_brazil_australia(tmp_path, monkeypatch):
    import duckdb as _duckdb
    from src.calendar import settings as _settings

    reg_db = tmp_path / "regional_archives" / "regional.duckdb"
    monkeypatch.setattr(_settings, "CAL_REGIONAL_DUCKDB_PATH", str(reg_db))

    # Regions registry includes Madrid and Mexico alongside Brazil and Australia
    for rid in ("madrid", "mexico", "brazil", "australia", "california"):
        assert rid in regions.REGIONS

    # External AOI classifier keeps Mexico, Madrid, Brazil, Australia strictly separate from USA
    madrid_pr = (40.42, -3.70, 100.0)
    madrid_bbox = aoi.radius_bbox(*madrid_pr)
    assert aoi.is_external_aoi(bbox=madrid_bbox, point_radius=madrid_pr) is True
    assert aoi.is_external_aoi(bbox=regions.MEXICO.bbox, region_id="mexico") is True
    assert aoi.is_external_aoi(bbox=regions.BRAZIL.bbox, region_id="brazil") is True
    assert aoi.is_external_aoi(bbox=regions.AUSTRALIA.bbox, region_id="australia") is True
    assert aoi.is_external_aoi(bbox=regions.CALIFORNIA.bbox, region_id="california") is False

    # Archive IDs are deterministic and distinct
    mad_id = regional.compute_archive_id(madrid_bbox, point_radius=madrid_pr)
    mex_id = regional.compute_archive_id(regions.MEXICO.bbox, region_id="mexico")
    bra_id = regional.compute_archive_id(regions.BRAZIL.bbox, region_id="brazil")
    aus_id = regional.compute_archive_id(regions.AUSTRALIA.bbox, region_id="australia")
    assert mad_id == "pt_40.42_-3.70_100.0km"
    assert (mex_id, bra_id, aus_id) == ("reg_mexico", "reg_brazil", "reg_australia")

    # Mock FIRMS feed tailored to regional coordinates (Madrid, Mexico, Brazil, Australia)
    regional_points = {
        "madrid": (40.45, -3.65, "2023-01-15", "28.4", "MODIS_SP", "Terra", "MODIS", "88"),
        "mexico": (19.50, -99.10, "2023-01-16", "41.2", "VIIRS_SNPP_SP", "N", "VIIRS", "h"),
        "brazil": (-12.50, -55.20, "2023-01-17", "95.0", "MODIS_SP", "Aqua", "MODIS", "92"),
        "australia": (-33.50, 150.80, "2023-01-18", "120.5", "VIIRS_NOAA20_SP", "NOAA-20", "VIIRS", "h"),
    }

    fetch_calls: list[tuple[str, str]] = []

    def fake_fetch(source, bbox_str, days, end_d, *, key=None):
        _ = (days, end_d, key)
        fetch_calls.append((source, bbox_str))
        w, s, e, n = (float(x) for x in bbox_str.split(","))
        lines = ["latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_t31,frp,daynight"]
        for _tag, (lat, lon, dt_s, frp_s, exp_src, sat, inst, conf) in regional_points.items():
            if w <= lon <= e and s <= lat <= n and source == exp_src:
                lines.append(f"{lat},{lon},325.0,1.0,1.0,{dt_s},1330,{sat},{inst},{conf},6.1,295.0,{frp_s},D")
        content = ("\n".join(lines) + "\n").encode("utf-8") if len(lines) > 1 else b""
        req = nrt.AreaRequest(source, bbox_str, 41, date(2023, 1, 1))
        return [
            nrt.ChunkResult(
                request=req,
                content=content,
                sha256="b" * 64,
                status="ok" if content else "empty",
                attempts=1,
                elapsed_s=0.01,
                fetched_at="2023-02-11T00:00:00+00:00",
            )
        ]

    # Populate Madrid point+radius automatically on empty regional archive
    ds_mad = regional.ensure_regional_archive_data(
        bbox=madrid_bbox,
        date_start="2023-01-01",
        date_end="2023-02-10",
        region_id="point_radius",
        region_label="Point (40.42, -3.70) + 100.0 km",
        center_lat=40.42,
        center_lon=-3.70,
        radius_km=100.0,
        aoi_source="point_radius",
        products=("modis", "viirs-snpp"),
        key="DUMMY-MAP-KEY",
        fetcher=fake_fetch,
        db_path=reg_db,
    )
    assert ds_mad.data_origin == "firms_live_api"
    assert ds_mad.archive_id == mad_id
    assert ds_mad.archive_status == "populated_from_firms"
    assert ds_mad.n_detections == 1
    assert ds_mad.df.iloc[0]["firms_source"] == "MODIS_SP"

    # Second call with no API key loads directly from independent regional DuckDB archive
    fetch_calls.clear()
    ds_mad_cached = regional.ensure_regional_archive_data(
        bbox=madrid_bbox,
        date_start="2023-01-01",
        date_end="2023-02-10",
        region_id="point_radius",
        region_label="Point (40.42, -3.70) + 100.0 km",
        center_lat=40.42,
        center_lon=-3.70,
        radius_km=100.0,
        aoi_source="point_radius",
        products=("modis", "viirs-snpp"),
        key="",
        db_path=reg_db,
    )
    assert ds_mad_cached.data_origin == "regional_duckdb_archive"
    assert ds_mad_cached.archive_id == mad_id
    assert ds_mad_cached.n_detections == 1
    assert fetch_calls == []

    # Populate Mexico, Brazil, Australia without mixing observations or touching calendar.duckdb
    for rid, reg in (("mexico", regions.MEXICO), ("brazil", regions.BRAZIL), ("australia", regions.AUSTRALIA)):
        ds_r = regional.ensure_regional_archive_data(
            bbox=reg.bbox,
            date_start="2023-01-01",
            date_end="2023-02-10",
            region_id=rid,
            region_label=reg.label,
            key="DUMMY-MAP-KEY",
            fetcher=fake_fetch,
            db_path=reg_db,
        )
        assert ds_r.n_detections == 1
        assert ds_r.archive_id == f"reg_{rid}"

    archives = {a["archive_id"]: a for a in regional.list_regional_archives(reg_db)}
    assert {mad_id, mex_id, bra_id, aus_id}.issubset(archives)
    assert archives[mad_id]["center_lat"] == 40.42 and archives[mad_id]["radius_km"] == 100.0

    # Verify table schemas and non-duplication when forcing a repeat store over same window
    con = _duckdb.connect(str(reg_db), read_only=True)
    try:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        assert {"regional_archives", "regional_acquisitions", "regional_detections"}.issubset(tables)
        mad_rows = con.execute("SELECT COUNT(*) FROM regional_detections WHERE archive_id = ?", [mad_id]).fetchone()[0]
        assert mad_rows == 1
    finally:
        con.close()

    dup_info = regional.store_regional_dataset(ds_mad, db_path=reg_db)
    assert dup_info["inserted"] == 0 and dup_info["skipped_duplicates"] == 1

    # Query layer for Mexico NEVER leaks USA baseline calendar.duckdb detections
    mex_ts = query.query_regional_timeseries(
        date_start="2023-01-01",
        date_end="2023-02-10",
        bbox=regions.MEXICO.bbox,
        region_id="mexico",
        region_label="Mexico",
        archive_id="reg_mexico",
        regional_db_path=reg_db,
    )
    assert len(mex_ts) == 41
    assert sum(r["observed_value"] for r in mex_ts) == 1

    # Uncached archive without API key fails honestly (unavailable, never USA fallback)
    uncached = regional.ensure_regional_archive_data(
        bbox=(10.0, 45.0, 12.0, 47.0),
        date_start="2024-06-01",
        date_end="2024-06-05",
        region_id="custom",
        region_label="Alps Test",
        key="",
        db_path=reg_db,
    )
    assert uncached.data_origin == "unavailable"
    assert uncached.archive_status == "unavailable_no_api_key"
    assert uncached.n_detections == 0

    # Distribution summarizer separates MODIS & VIIRS and builds FRP bins
    combined_df = pd.concat([ds_mad.df, ds_mad.df.assign(
        detection_id="v1", product="viirs-snpp", sensor="snpp", satellite="N", instrument="VIIRS", frp=60.0
    )], ignore_index=True)
    dist = regional.summarize_regional_distributions(combined_df)
    assert set(dist["by_product"]["product"]) == {"modis", "viirs-snpp"}
    assert dist["frp_bins"]["modis"].sum() == 1
    assert dist["frp_bins"]["viirs"].sum() == 1
    # Refuses to point regional DuckDB to pinned Stage 4C USA baseline
    import pytest as _pytest
    with _pytest.raises(ValueError):
        regional.resolve_regional_db_path("calendar.duckdb")

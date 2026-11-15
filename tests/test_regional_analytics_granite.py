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

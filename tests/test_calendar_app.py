"""Phase 3F + 5F AppTest journeys: the real app, headless, no browser needed.

Each test drives `calendar_app.py` end to end (widgets, queries, sync).
Slow by design (first run builds the default map); keep journeys few and
themed. Browser-pixel verification is a 3F manual step, not asserted here.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "calendar_app.py")
TIMEOUT = 180


def _captions(at: AppTest) -> list[str]:
    return [str(c.value) for c in at.caption]


def test_startup_calendar_and_grain_switch():
    at = AppTest.from_file(APP)
    at.run(timeout=TIMEOUT)
    assert at.exception == []
    assert [t.label for t in at.tabs] == ["Fire Calendar", "Activity Map",
                                           "Historical Analytics",
                                           "Events & Alerts", "Data Export",
                                           "About"]
    assert any("not equivalent to confirmed individual wildfires" in str(m.value)
               for m in at.info)
    # Daily-map default: single initial day shared by range + caption.
    assert at.session_state["cal_range_start"] == "2023-08-01"
    assert at.session_state["cal_range_end"] == "2023-08-01"
    assert any("Showing acquisition date 2023-08-01 (UTC)" in str(m.value)
               for m in at.caption)
    assert any("does not currently implement autonomous multi-agent "
               "orchestration" in str(m.value) for m in at.markdown)
    at.radio(key="cal_grain").set_value("weekly").run(timeout=TIMEOUT)
    assert at.exception == []
    assert any("2023-W31" in str(m.value) for m in at.markdown)
    at.radio(key="cal_grain").set_value("monthly").run(timeout=TIMEOUT)
    assert at.exception == []
    assert any("### 2023" in str(m.value) for m in at.markdown)
    at.radio(key="cal_grain").set_value("daily").run(timeout=TIMEOUT)
    at.button(key="cal_next_day").click().run(timeout=TIMEOUT)
    assert at.exception == []
    assert at.session_state["cal_period"] == "2023-08-02"
    at.button(key="cal_prev_month").click().run(timeout=TIMEOUT)
    assert at.exception == []
    assert any("July 2023" in str(m.value) for m in at.markdown)


def test_map_filters_detail_and_unknown_id():
    at = AppTest.from_file(APP)
    at.run(timeout=TIMEOUT)
    at.text_input(key="cal_range_start").set_value("2023-08-15").run(timeout=TIMEOUT)
    at.text_input(key="cal_range_end").set_value("2023-08-15").run(timeout=TIMEOUT)
    at.multiselect(key="cal_products").set_value(["modis"]).run(timeout=TIMEOUT)
    assert at.exception == []
    assert any("362 detections" in c for c in _captions(at))  # pinned day count
    assert any("In current sensor selection" in c for c in _captions(at))
    at.radio(key="cal_aoi_mode").set_value("region").run(timeout=TIMEOUT)
    assert at.exception == []
    assert any("Area filter active (region)" in c for c in _captions(at))
    at.text_input(key="cal_id_search").set_value("cl0000161").run(timeout=TIMEOUT)
    assert at.exception == []
    assert any("cl0000161" in str(m.value) for m in at.markdown)
    assert any("Very large observation group" in str(m.value) for m in at.warning)
    at.text_input(key="cal_id_search").set_value("no-such-id").run(timeout=TIMEOUT)
    assert at.exception == []
    assert any("No detection with id" in str(m.value) for m in at.warning)
    # Detection detail by id (real id from the canonical store).
    from src.calendar.ui import query as _query

    det_id = _query.query_detections(date_start="2023-08-15",
                                     date_end="2023-08-15", limit=1).iloc[0]["detection_id"]
    at.text_input(key="cal_id_search").set_value(det_id).run(timeout=TIMEOUT)
    assert at.exception == []
    assert any(f"Detection {det_id}" in str(m.value) for m in at.markdown)
    # Cluster mode + FRP density toggle render without errors.
    at.radio(key="cal_mode").set_value("clusters").run(timeout=TIMEOUT)
    assert at.exception == []
    assert any("Cluster Mode" in str(m.value) for m in at.markdown)
    at.radio(key="cal_mode").set_value("detections").run(timeout=TIMEOUT)
    at.checkbox(key="cal_map_density").set_value(True).run(timeout=TIMEOUT)
    assert at.exception == []
    assert not at.error  # map builds cleanly (no construction failure panel)
    # Empty (out-of-coverage) and invalid ranges fail loudly.
    at.text_input(key="cal_range_start").set_value("2024-01-01").run(timeout=TIMEOUT)
    at.text_input(key="cal_range_end").set_value("2024-01-02").run(timeout=TIMEOUT)
    assert any("outside verified pilot coverage" in str(m.value).lower()
               for m in list(at.error) + list(at.info))
    at.text_input(key="cal_range_start").set_value("junk").run(timeout=TIMEOUT)
    assert any("Invalid filter" in str(m.value) for m in at.error)
    # Empty in-coverage selection states zero activity (never blank).
    at.text_input(key="cal_range_start").set_value("2023-08-15").run(timeout=TIMEOUT)
    at.text_input(key="cal_range_end").set_value("2023-08-15").run(timeout=TIMEOUT)
    at.multiselect(key="cal_products").set_value([]).run(timeout=TIMEOUT)
    assert at.exception == []
    assert any("No records match these filters" in str(m.value)
               for m in at.info)


def test_sync_both_directions():
    at = AppTest.from_file(APP)
    at.run(timeout=TIMEOUT)
    at.text_input(key="cal_range_start").set_value("2023-08-15").run(timeout=TIMEOUT)
    at.text_input(key="cal_range_end").set_value("2023-08-15").run(timeout=TIMEOUT)
    at.button(key="cal_map_to_cal").click().run(timeout=TIMEOUT)
    assert at.exception == []
    assert at.session_state["cal_period"] == "2023-08-15"
    assert at.session_state["cal_grain"] == "daily"
    at.radio(key="cal_grain").set_value("weekly").run(timeout=TIMEOUT)
    at.button(key="cal_to_map").click().run(timeout=TIMEOUT)
    assert at.exception == []
    assert at.session_state["cal_range_start"] == "2023-07-31"
    assert at.session_state["cal_range_end"] == "2023-08-06"


def test_analytics_tab_methods_and_detail():
    at = AppTest.from_file(APP)
    at.run(timeout=TIMEOUT)
    assert at.exception == []
    assert any("do not establish wildfire occurrence" in str(m.value)
               for m in at.info)
    # Historical exploration is explicit: widen the default single day.
    at.text_input(key="cal_range_start").set_value("2023-01-01").run(
        timeout=TIMEOUT)
    at.text_input(key="cal_range_end").set_value("2023-12-31").run(
        timeout=TIMEOUT)
    assert at.exception == []
    # Multi-year method: honest insufficient-history panel, no crash.
    at.selectbox(key="cal_an_method").set_value("monthly_median").run(timeout=TIMEOUT)
    assert at.exception == []
    assert any("insufficient history" in str(m.value).lower() for m in at.warning)
    # Trailing method: notables table + inspectable detail.
    at.selectbox(key="cal_an_method").set_value("sensor_aware").run(timeout=TIMEOUT)
    assert at.exception == []
    assert any("Notable periods" in str(m.value) for m in at.markdown)
    options = at.selectbox(key="cal_an_selected").options
    assert len(options) > 0
    at.selectbox(key="cal_an_selected").set_value(options[0]).run(timeout=TIMEOUT)
    assert at.exception == []
    assert any(str(m.value).startswith("### Anomaly anom-") for m in at.markdown)
    assert any("statistical support, not the probability" in str(m.value)
               for m in list(at.caption) + list(at.markdown))
    # 6C Journey 2: reference interval shown with the evaluated period.
    assert any("Reference:" in str(m.value) for m in at.markdown)


def _seed_events_db(path: str, cluster_id: str = "cl0000001") -> dict:
    """Tiny synthetic events DB for the 5F journey (isolated, disposable)."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.calendar.events import association, store

    event_id = f"ev-{cluster_id}-a1"
    conn = store.connect(path)
    store.init_schema(conn)
    high = association.build_event_record(
        {"cluster_id": cluster_id, "n_detections": 500, "n_modis": 200,
         "n_viirs": 300, "satellites": '["Aqua", "N"]',
         "first_observation_utc": "2023-08-01T06:00:00+00:00",
         "last_observation_utc": "2023-08-15T06:00:00+00:00",
         "observation_span_h": 336.0, "lon_min": -120.0, "lon_max": -119.0,
         "lat_min": 38.0, "lat_max": 39.0, "centroid_lon": -119.5,
         "centroid_lat": 38.5, "frp_sum": 5000.0, "frp_mean": 10.0,
         "frp_max": 300.0, "frp_n": 500, "frp_null": 0,
         "frp_by_sensor": '{"MODIS": {"frp_max": 100.0}, '
                          '"VIIRS": {"frp_max": 300.0}}',
         "match_groups": '["mg0000001"]', "completeness_dates": 15,
         "completeness_span_days": 15, "completeness_frac": 1.0,
         "algo_version": "cluster_v1", "params_version": "cluster_params_v1",
         "exec_id": "exec2c"},
        exec_id="exec5a")
    high["lifecycle_state"] = "Pending Closure"
    high["severity_name"] = "High"
    high["severity_level"] = 3
    high["review_required"] = True
    store.upsert_event(conn, high)
    low = association.build_event_record(
        {"cluster_id": "cl0000002", "n_detections": 1, "n_modis": 1,
         "n_viirs": 0, "satellites": '["Terra"]',
         "first_observation_utc": "2023-12-30T06:00:00+00:00",
         "last_observation_utc": "2023-12-30T06:00:00+00:00",
         "observation_span_h": 0.0, "lon_min": -110.0, "lon_max": -110.0,
         "lat_min": 40.0, "lat_max": 40.0, "centroid_lon": -110.0,
         "centroid_lat": 40.0, "frp_sum": 5.0, "frp_mean": 5.0,
         "frp_max": 5.0, "frp_n": 1, "frp_null": 0,
         "frp_by_sensor": '{"MODIS": {"frp_max": 5.0}}',
         "match_groups": '[]', "completeness_dates": 1,
         "completeness_span_days": 1, "completeness_frac": 1.0,
         "algo_version": "cluster_v1",
         "params_version": "cluster_params_v1",
         "exec_id": "exec2c"},
        exec_id="exec5a")
    store.upsert_event(conn, low)
    store.upsert_alert(conn, {
        "alert_id": f"alt-asp-{event_id}-r1",
        "alert_type": "Activity Spike",
        "severity_name": "High", "severity_level": 3,
        "event_id": event_id, "scope": "event",
        "metric": "detection", "grain": "event", "period": "2023-08-15",
        "trigger": {"n_detections": 500}, "baseline_ref": {},
        "data_quality": {}, "explanation": "seeded test alert",
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T00:00:00+00:00",
        "rule_version": "alert_rules_v1", "status": "open",
        "fingerprint": "seedfp1"})
    store.upsert_alert(conn, {
        "alert_id": "alt-han-seeded-notable-r1",
        "alert_type": "Historical Anomaly", "severity_name": "Moderate",
        "severity_level": 2, "event_id": None, "scope": "raw",
        "metric": "detection", "grain": "daily", "period": "2023-08-15",
        "trigger": {"observed_value": 500.0}, "baseline_ref": {},
        "data_quality": {}, "explanation": "seeded anomaly alert",
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T00:00:00+00:00",
        "rule_version": "alert_rules_v1", "status": "open",
        "fingerprint": "seedfp2"})
    conn.close()
    return {"event": event_id,
            "alert": f"alt-asp-{event_id}-r1",
            "anomaly_alert": "alt-han-seeded-notable-r1"}


def test_events_tab_sections_and_flows(tmp_path, monkeypatch):
    from src.calendar.ui import query as _query

    # Seed under a real cluster id so the Qwen event dropdown (built
    # from real date-range clusters) can resolve the synthetic event.
    real_cluster = _query.query_clusters(
        date_start="2023-08-01", date_end="2023-08-01",
        limit=1).iloc[0]["cluster_id"]
    db_path = str(tmp_path / "journey.sqlite")
    ids = _seed_events_db(db_path, cluster_id=real_cluster)
    # Patch the live settings attribute (the module is already imported;
    # env-only patching would not take effect in-process).
    monkeypatch.setattr("src.calendar.settings.CAL_EVENTS_DB_PATH", db_path)
    at = AppTest.from_file(APP)
    at.run(timeout=TIMEOUT)
    assert at.exception == []
    assert "Events & Alerts" in [t.label for t in at.tabs]
    assert any("not confirmed wildfires" in str(m.value) for m in at.info)
    # Events: inspect the seeded High event, record a review.
    at.text_input(key="cal_ev_selected").set_value(ids["event"]).run(
        timeout=TIMEOUT)
    assert at.exception == []
    assert any(f"### Event {ids['event']}" in str(m.value)
               for m in at.markdown)
    assert any("Human review is required" in str(m.value) for m in at.warning)
    at.text_input(key="cal_ev_reviewer").set_value("journey").run(
        timeout=TIMEOUT)
    at.button(key="cal_ev_review").click().run(timeout=TIMEOUT)
    assert at.exception == []
    assert any("Review recorded" in str(m.value) for m in at.success)
    # 6B: cluster-id cross-reference resolves to the same event detail.
    at.text_input(key="cal_ev_selected").set_value(real_cluster).run(
        timeout=TIMEOUT)
    assert at.exception == []
    assert any(f"### Event {ids['event']}" in str(m.value)
               for m in at.markdown)
    # 6C Journey 3: honest semantics on the event record.
    assert any("contained or extinguished" in str(m.value)
               for m in list(at.caption) + list(at.markdown))
    assert not any("confirmed wildfire" in str(m.value).lower()
                   for m in list(at.markdown) + list(at.caption))
    # Alerts: inspect and acknowledge the seeded alert.
    at.radio(key="cal_ev_section").set_value("Alerts").run(timeout=TIMEOUT)
    assert at.exception == []
    at.text_input(key="cal_al_selected").set_value(ids["alert"]).run(
        timeout=TIMEOUT)
    assert at.exception == []
    assert any(f"### Alert {ids['alert']}" in str(m.value)
               for m in at.markdown)
    # 6B/6C Journey 4: retrospective-only caption on Spike alerts.
    assert any("Retrospective-only classification" in str(m.value)
               for m in at.caption)
    at.button(key="cal_al_ack").click().run(timeout=TIMEOUT)
    assert at.exception == []
    assert any("Acknowledged" in str(m.value) for m in at.success)
    # 6C Journey 4: anomaly alert carries the trailing-reference caption.
    at.text_input(key="cal_al_selected").set_value(
        ids["anomaly_alert"]).run(timeout=TIMEOUT)
    assert at.exception == []
    assert any("strictly precedes the evaluation period" in str(m.value)
               for m in at.caption)
    assert any("pilot-wide" in str(m.value) for m in at.markdown)
    # Notifications: SMTP unavailable path, validation, token errors.
    # Hermetic: force the unavailable state so no real email can send
    # regardless of local .env (unavailable-path coverage by design).
    monkeypatch.setattr("src.calendar.settings.CAL_SMTP_ENABLED", "false")
    monkeypatch.setattr("src.calendar.settings.CAL_SMTP_USER", "")
    monkeypatch.setattr("src.calendar.settings.CAL_SMTP_PASSWORD", "")
    at.radio(key="cal_ev_section").set_value("Notifications").run(
        timeout=TIMEOUT)
    assert at.exception == []
    assert any("Email delivery unavailable" in str(m.value)
               for m in at.warning)
    at.text_input(key="cal_ntf_email").set_value("bad-address").run(
        timeout=TIMEOUT)
    at.button(key="cal_ntf_sub").click().run(timeout=TIMEOUT)
    assert any("invalid email" in str(m.value).lower() for m in at.error)
    at.text_input(key="cal_ntf_email").set_value("journey@example.com").run(
        timeout=TIMEOUT)
    at.button(key="cal_ntf_sub").click().run(timeout=TIMEOUT)
    assert at.exception == []
    assert any("pending" in str(m.value).lower()
               for m in list(at.info) + list(at.success))
    at.text_input(key="cal_ntf_token").set_value("bogus").run(
        timeout=TIMEOUT)
    at.button(key="cal_ntf_confirm").click().run(timeout=TIMEOUT)
    assert any("unknown or consumed" in str(m.value).lower()
               for m in at.error)
    # Qwen: unavailable path with deterministic fallback.
    at.radio(key="cal_ev_section").set_value("Qwen Intelligence").run(
        timeout=TIMEOUT)
    assert at.exception == []
    assert any("Qwen interpretation unavailable" in str(m.value)
               for m in at.warning)
    at.selectbox(key="cal_qw_target").set_value(ids["event"]).run(
        timeout=TIMEOUT)
    # Explain-gating: merely opening the section never invokes Qwen.
    import sqlite3 as _sqlite

    _probe = _sqlite.connect(db_path)
    assert _probe.execute("SELECT COUNT(*) FROM qwen_log").fetchone()[0] == 0
    _probe.close()
    at.button(key="cal_qw_go").click().run(timeout=TIMEOUT)
    assert at.exception == []
    assert any("Deterministic explanation" in str(m.value)
               for m in at.markdown)
    _probe = _sqlite.connect(db_path)
    assert _probe.execute("SELECT COUNT(*) FROM qwen_log").fetchone()[0] == 1
    _probe.close()
    # 6C Journey 6: evidence identifiers accompany the fallback.
    assert any("Evidence references:" in str(m.value) for m in at.caption)
    assert any(ids["event"] in str(m.value) for m in at.caption)
    # 6B: map cluster detail cross-references its activity event.
    at.text_input(key="cal_id_search").set_value("cl0000161").run(
        timeout=TIMEOUT)
    assert at.exception == []
    assert any("Activity event ev-cl0000161-a1" in str(m.value)
               for m in at.caption)
    # Custom alerts: create a rule, run it, see the hit (isolated DB).
    at.radio(key="cal_ev_section").set_value("Custom Alerts").run(
        timeout=TIMEOUT)
    assert at.exception == []
    assert any("no background scheduler" in str(m.value).lower()
               for m in at.caption)
    at.text_input(key="cal_cr_name").set_value("journey rule")
    at.selectbox(key="cal_cr_op").set_value("gt")
    at.text_input(key="cal_cr_threshold").set_value("0")
    at.form_submit_button[0].click().run(timeout=TIMEOUT)
    assert at.exception == []
    assert any("Rule saved" in str(m.value) for m in at.success)
    run_buttons = [b for b in at.button if b.label == "Run now"]
    assert len(run_buttons) == 1
    run_buttons[0].click().run(timeout=TIMEOUT)
    assert at.exception == []
    assert any("Triggered:" in str(m.value) for m in at.success)


def test_pipeline_section_renders_and_rejects_bad_input():
    at = AppTest.from_file(APP)
    at.run(timeout=TIMEOUT)
    assert at.exception == []
    at.radio(key="cal_ev_section").set_value("Pipeline").run(
        timeout=TIMEOUT)
    assert at.exception == []
    assert any("Start Date" in str(w.label)
               for w in at.text_input)
    assert any(b.label == "Run Analysis" for b in at.button)
    assert any("Granite" in str(m.value) for m in at.caption)
    at.text_input(key="cal_pipe_start").set_value("junk").run(
        timeout=TIMEOUT)
    at.button(key="cal_pipe_go").click().run(timeout=TIMEOUT)
    assert at.exception == []
    assert any("invalid date range" in str(m.value).lower()
               for m in at.error)


def test_pipeline_run_offline_downloads(monkeypatch):
    from src.calendar.agents import granite as granite_mod

    monkeypatch.setattr(
        granite_mod.GraniteConfig, "from_settings",
        classmethod(lambda cls: granite_mod.GraniteConfig()))
    at = AppTest.from_file(APP)
    at.run(timeout=TIMEOUT)
    at.radio(key="cal_ev_section").set_value("Pipeline").run(
        timeout=TIMEOUT)
    at.text_input(key="cal_pipe_start").set_value("2023-08-15").run(
        timeout=TIMEOUT)
    at.text_input(key="cal_pipe_end").set_value("2023-08-15").run(
        timeout=TIMEOUT)
    at.button(key="cal_pipe_go").click().run(timeout=TIMEOUT)
    assert at.exception == []
    labels = [str(b.label) for b in at.download_button]
    assert sum("Download" in label and "CSV" in label
               for label in labels) == 4
    assert any("Stages" in str(m.value) for m in at.markdown)
    assert any("Agent executions" in str(m.value)
               for m in at.markdown)


def test_pipeline_downloads_preserve_session(monkeypatch):
    from src.calendar.agents import granite as granite_mod
    from src.calendar.pipeline import runner as pipe_runner

    monkeypatch.setattr(
        granite_mod.GraniteConfig, "from_settings",
        classmethod(lambda cls: granite_mod.GraniteConfig()))
    calls = []
    real_run = pipe_runner.run_pipeline

    def _counting(*args, **kwargs):
        calls.append(1)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(pipe_runner, "run_pipeline", _counting)
    at = AppTest.from_file(APP)
    at.run(timeout=TIMEOUT)
    at.radio(key="cal_ev_section").set_value("Pipeline").run(
        timeout=TIMEOUT)
    at.text_input(key="cal_pipe_start").set_value("2023-08-15").run(
        timeout=TIMEOUT)
    at.text_input(key="cal_pipe_end").set_value("2023-08-15").run(
        timeout=TIMEOUT)
    at.button(key="cal_pipe_go").click().run(timeout=TIMEOUT)
    assert at.exception == []
    assert len(calls) == 1
    first = at.session_state["cal_pipe_result"].execution_id
    # Navigate away and back: same stored execution, no recompute.
    at.radio(key="cal_ev_section").set_value("Alerts").run(
        timeout=TIMEOUT)
    at.radio(key="cal_ev_section").set_value("Pipeline").run(
        timeout=TIMEOUT)
    assert at.exception == []
    assert at.session_state["cal_pipe_result"].execution_id == first
    assert len(calls) == 1
    labels = [str(b.label) for b in at.download_button]
    assert sum("Download" in label and "CSV" in label
               for label in labels) == 4
    assert any(first in str(m.value) for m in at.caption)


def test_export_tab_count_prepare_download():
    at = AppTest.from_file(APP)
    at.run(timeout=TIMEOUT)
    assert at.exception == []
    assert "Data Export" in [t.label for t in at.tabs]
    at.text_input(key="cal_range_start").set_value("2023-08-15").run(
        timeout=TIMEOUT)
    at.text_input(key="cal_range_end").set_value("2023-08-15").run(
        timeout=TIMEOUT)
    at.selectbox(key="cal_ex_category").set_value("modis").run(
        timeout=TIMEOUT)
    assert at.exception == []
    assert any(m.label == "Rows matching" and m.value == "362"
               for m in at.metric)
    at.button(key="cal_ex_prepare").click().run(timeout=TIMEOUT)
    assert at.exception == []
    assert any("Prepared 362 of 362 rows" in str(m.value)
               for m in at.caption)
    assert any("Download CSV" in str(b.label) for b in at.download_button)
    assert any("README" in str(b.label) for b in at.download_button)

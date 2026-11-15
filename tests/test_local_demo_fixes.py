"""Local demo fixes: map rendering, Qwen/SMTP activation, log redaction.

Isolated regression coverage for the pre-demo bug-fix task. Map tests use
small synthetic frames (builders only) plus read-only pilot queries;
Qwen/SMTP tests use monkeypatched settings and stub transports — no
external calls, no emails, no production writes.
"""

import importlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.calendar.ui import map_view


def _det_frame(rows):
    return pd.DataFrame(rows, columns=[
        "detection_id", "satellite", "instrument", "product",
        "acquisition_timestamp_utc", "latitude", "longitude",
        "confidence", "frp", "daynight"])


def _det_row(lat=38.5, lon=-119.5, did="d1"):
    return {"detection_id": did, "satellite": "Aqua", "instrument": "MODIS",
            "product": "modis",
            "acquisition_timestamp_utc": "2023-08-15T06:00:00+00:00",
            "latitude": lat, "longitude": lon, "confidence": "80",
            "frp": 10.0, "daynight": "D"}


def _clu_frame(rows):
    return pd.DataFrame(rows, columns=[
        "cluster_id", "n_detections", "n_modis", "n_viirs",
        "first_observation_utc", "last_observation_utc",
        "observation_span_h", "lon_min", "lon_max", "lat_min", "lat_max",
        "centroid_lon", "centroid_lat", "frp_sum"])


def _clu_row(clu="cl0000001", n=5, clon=-119.5, clat=38.5):
    return {"cluster_id": clu, "n_detections": n, "n_modis": 2, "n_viirs": 3,
            "first_observation_utc": "2023-08-15T06:00:00+00:00",
            "last_observation_utc": "2023-08-15T06:00:00+00:00",
            "observation_span_h": 0.0, "lon_min": clon - 0.1,
            "lon_max": clon + 0.1, "lat_min": clat - 0.1,
            "lat_max": clat + 0.1, "centroid_lon": clon,
            "centroid_lat": clat, "frp_sum": 50.0}


class TestSanitizeFrame:
    def test_clean_rows_pass(self):
        df = _det_frame([_det_row(), _det_row(lat=40.0, did="d2")])
        clean, dropped = map_view.sanitize_frame(df, ("latitude", "longitude"))
        assert dropped == 0
        assert len(clean) == 2

    def test_nan_and_none_dropped(self):
        df = _det_frame([_det_row(), _det_row(lat=float("nan"), did="bad1"),
                         _det_row(lon=None, did="bad2")])
        clean, dropped = map_view.sanitize_frame(df, ("latitude", "longitude"))
        assert dropped == 2
        assert clean["detection_id"].tolist() == ["d1"]

    def test_infinite_dropped(self):
        df = _det_frame([_det_row(lat=float("inf"), did="bad")])
        _, dropped = map_view.sanitize_frame(df, ("latitude", "longitude"))
        assert dropped == 1

    def test_string_coords_coerced(self):
        df = _det_frame([_det_row(lat="38.5", lon="-119.5")])
        clean, dropped = map_view.sanitize_frame(df, ("latitude", "longitude"))
        assert (dropped, len(clean)) == (0, 1)

    def test_unparseable_strings_dropped(self):
        df = _det_frame([_det_row(lat="north", did="bad")])
        _, dropped = map_view.sanitize_frame(df, ("latitude", "longitude"))
        assert dropped == 1

    def test_missing_columns_drop_all(self):
        df = pd.DataFrame([{"a": 1}])
        clean, dropped = map_view.sanitize_frame(df, ("latitude", "longitude"))
        assert (len(clean), dropped) == (0, 1)

    def test_empty_frame(self):
        clean, dropped = map_view.sanitize_frame(_det_frame([]),
                                                 ("latitude", "longitude"))
        assert (len(clean), dropped) == (0, 0)

    def test_multi_pair_all_must_pass(self):
        df = _clu_frame([_clu_row(), _clu_row(clu="bad", clon=float("nan"))])
        clean, dropped = map_view.sanitize_frame(
            df, ("centroid_lat", "centroid_lon"), ("lat_min", "lon_min"),
            ("lat_max", "lon_max"))
        assert dropped == 1
        assert clean["cluster_id"].tolist() == ["cl0000001"]

    def test_input_never_mutated(self):
        df = _det_frame([_det_row(lat=float("nan"), did="bad")])
        before = df.copy()
        map_view.sanitize_frame(df, ("latitude", "longitude"))
        pd.testing.assert_frame_equal(df, before)


class TestMapBuilders:
    def test_detection_markers_match_rows(self):
        df = _det_frame([_det_row(), _det_row(lat=40.0, did="d2")])
        html = map_view.build_detection_map(df).get_root().render()
        assert html.count("L.circleMarker") == 2
        # CARTO excluded by default: keyless endpoint returns API-KEY
        # watermarks (pixel-verified); OSM is the working default.
        assert "tile.openstreetmap.org" in html
        assert "basemaps.cartocdn.com" not in html

    def test_cluster_markers_and_extents(self):
        df = _clu_frame([_clu_row(), _clu_row(clu="cl0000002")])
        html = map_view.build_cluster_map(df).get_root().render()
        assert html.count("L.circleMarker") == 2
        assert html.count("L.rectangle") >= 2  # extents + boundary
        assert "Not a fire perimeter" in html

    def test_canvas_hint_present(self):
        df = _det_frame([_det_row()])
        html = map_view.build_detection_map(df).get_root().render()
        assert "preferCanvas" in html

    def test_sensor_modes_preserved(self):
        df = _det_frame([_det_row()])
        sensor = map_view.build_detection_map(df, color_mode="sensor")
        frp = map_view.build_detection_map(df, color_mode="frp")
        assert sensor.get_root().render().count("L.circleMarker") == 1
        assert frp.get_root().render().count("L.circleMarker") == 1

    def test_empty_frame_returns_base_map(self):
        param = map_view.build_detection_map(_det_frame([]))
        html = param.get_root().render()
        assert "L.circleMarker" not in html
        assert param.dropped_rows == 0
        cparam = map_view.build_cluster_map(_clu_frame([]))
        assert cparam.dropped_rows == 0

    def test_invalid_coords_dropped_not_rendered(self):
        df = _det_frame([_det_row(), _det_row(lat=float("nan"), did="bad")])
        param = map_view.build_detection_map(df)
        html = param.get_root().render()
        assert param.dropped_rows == 1
        assert html.count("L.circleMarker") == 1
        assert "[nan" not in html and "nan," not in html

    def test_columnless_frame_does_not_raise(self):
        param = map_view.build_detection_map(pd.DataFrame([{"a": 1}]))
        assert param.dropped_rows == 1
        cparam = map_view.build_cluster_map(pd.DataFrame())
        assert cparam.dropped_rows == 0

    def test_large_group_warning_preserved(self):
        df = _clu_frame([_clu_row(n=3872)])
        html = map_view.build_cluster_map(df).get_root().render()
        assert "Very large observation group" in html


class TestMapDataKey:
    def test_stable_for_identical_inputs(self):
        df = _det_frame([_det_row(), _det_row(did="d2")])
        first = map_view.map_data_key(mode="detections", df=df,
                                      id_col="detection_id",
                                      display=("sensor",))
        second = map_view.map_data_key(mode="detections", df=df,
                                       id_col="detection_id",
                                       display=("sensor",))
        assert first == second
        assert first.startswith("cal_map_")

    def test_changes_with_data_mode_or_display(self):
        df = _det_frame([_det_row(), _det_row(did="d2")])
        base = map_view.map_data_key(mode="detections", df=df,
                                     id_col="detection_id", display=())
        assert map_view.map_data_key(
            mode="detections", df=df.iloc[[0]],
            id_col="detection_id", display=()) != base
        assert map_view.map_data_key(
            mode="clusters", df=df, id_col="detection_id",
            display=()) != base
        assert map_view.map_data_key(
            mode="detections", df=df, id_col="detection_id",
            display=("frp",)) != base

    def test_empty_frame_has_stable_key(self):
        first = map_view.map_data_key(mode="detections",
                                      df=_det_frame([]),
                                      id_col="detection_id")
        second = map_view.map_data_key(mode="detections",
                                       df=_det_frame([]),
                                       id_col="detection_id")
        assert first == second


def _reload_settings():
    import src.calendar.settings as settings

    return importlib.reload(settings)


class TestQwenActivation:
    def test_api_key_falls_back_to_existing_credential(self, monkeypatch):
        import src.calendar.settings as settings

        original = settings.CAL_QWEN_API_KEY
        monkeypatch.setattr(settings, "CAL_QWEN_API_KEY", original)
        monkeypatch.delenv("CAL_QWEN_API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "opaque-test-key")
        reloaded = _reload_settings()
        assert reloaded.CAL_QWEN_API_KEY == "opaque-test-key"

    def test_explicit_key_wins_over_fallback(self, monkeypatch):
        import src.calendar.settings as settings

        original = settings.CAL_QWEN_API_KEY
        monkeypatch.setattr(settings, "CAL_QWEN_API_KEY", original)
        monkeypatch.setenv("CAL_QWEN_API_KEY", "opaque-explicit-key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "opaque-test-key")
        reloaded = _reload_settings()
        assert reloaded.CAL_QWEN_API_KEY == "opaque-explicit-key"

    def test_enabled_but_incomplete_names_missing_fields(self, monkeypatch):
        from src.calendar.events import qwen

        monkeypatch.setattr("src.calendar.settings.CAL_QWEN_ENABLED", "true")
        monkeypatch.setattr("src.calendar.settings.CAL_QWEN_BASE_URL", "")
        monkeypatch.setattr("src.calendar.settings.CAL_QWEN_API_KEY", "")
        monkeypatch.setattr("src.calendar.settings.CAL_QWEN_MODEL", "")
        status = qwen.QwenConfig.from_settings().status()
        assert status["available"] is False
        assert "CAL_QWEN_BASE_URL" in status["reason"]
        assert "CAL_QWEN_API_KEY" in status["reason"]
        assert "CAL_QWEN_MODEL" in status["reason"]

    def test_enabled_complete_reports_available(self, monkeypatch):
        from src.calendar.events import qwen

        monkeypatch.setattr("src.calendar.settings.CAL_QWEN_ENABLED", "true")
        monkeypatch.setattr("src.calendar.settings.CAL_QWEN_BASE_URL",
                            "http://localhost/v1")
        monkeypatch.setattr("src.calendar.settings.CAL_QWEN_API_KEY", "k")
        monkeypatch.setattr("src.calendar.settings.CAL_QWEN_MODEL", "m")
        status = qwen.QwenConfig.from_settings().status()
        assert status["available"] is True

    def test_no_secret_values_in_status_reason(self, monkeypatch):
        from src.calendar.events import qwen

        monkeypatch.setattr("src.calendar.settings.CAL_QWEN_ENABLED", "true")
        monkeypatch.setattr("src.calendar.settings.CAL_QWEN_BASE_URL",
                            "http://localhost/v1")
        monkeypatch.setattr("src.calendar.settings.CAL_QWEN_API_KEY",
                            "opaque-secret-value")
        monkeypatch.setattr("src.calendar.settings.CAL_QWEN_MODEL", "")
        status = qwen.QwenConfig.from_settings().status()
        assert "opaque-secret-value" not in status["reason"]
        assert "CAL_QWEN_MODEL" in status["reason"]


class TestSmtpActivation:
    def test_disabled_reports_disabled(self, monkeypatch):
        from src.calendar.events import notify

        monkeypatch.setattr("src.calendar.settings.CAL_SMTP_ENABLED", "false")
        status = notify.smtp_status()
        assert status == {"available": False,
                          "reason": "CAL_SMTP_ENABLED is not true"}

    def test_enabled_incomplete_names_gap(self, monkeypatch):
        from src.calendar.events import notify

        monkeypatch.setattr("src.calendar.settings.CAL_SMTP_ENABLED", "true")
        monkeypatch.setattr("src.calendar.settings.CAL_SMTP_USER", "")
        monkeypatch.setattr("src.calendar.settings.CAL_SMTP_PASSWORD", "")
        status = notify.smtp_status()
        assert status["available"] is False
        assert "credentials" in status["reason"].lower()

    def test_enabled_complete_reports_available(self, monkeypatch):
        from src.calendar.events import notify

        monkeypatch.setattr("src.calendar.settings.CAL_SMTP_ENABLED", "true")
        monkeypatch.setattr("src.calendar.settings.CAL_SMTP_USER",
                            "opaque-smtp-user")
        monkeypatch.setattr("src.calendar.settings.CAL_SMTP_PASSWORD",
                            "opaque-smtp-secret")
        monkeypatch.setattr("src.calendar.settings.CAL_SMTP_HOST", "h")
        monkeypatch.setattr("src.calendar.settings.CAL_SMTP_PORT", 587)
        status = notify.smtp_status()
        assert status["available"] is True
        assert "opaque-smtp-user" not in status["reason"]
        assert "opaque-smtp-secret" not in status["reason"]

    def test_send_refuses_without_config(self, monkeypatch):
        from src.calendar.events import notify

        monkeypatch.setattr("src.calendar.settings.CAL_SMTP_ENABLED", "false")
        assert notify.send_via_smtp("a@example.com", "s", "b") is False


class TestLogRedaction:
    def test_redact_masks_addresses(self):
        from src.calendar.events.notify import redact

        assert redact("recipient refused: user@example.com (550)") == \
            "recipient refused: [redacted-address] (550)"
        assert redact("no address here") == "no address here"
        assert "@" not in redact("a@b.co and x@y.io failed")

    def test_sender_exception_address_never_logged(self, memdb_protected,
                                                   caplog):
        import logging

        from src.calendar.events import alerts, notify, store

        pending = notify.subscribe(memdb_protected, "victim@example.com")
        notify.confirm(memdb_protected, pending["confirm_token"])
        alerts.persist_alerts(memdb_protected, [{
            "alert_id": "alt-redact-1", "alert_type": "Activity Spike",
            "severity_name": "High", "severity_level": 3,
            "event_id": None, "scope": "event", "metric": "detection",
            "grain": "event", "period": "2023-08-15", "trigger": {},
            "baseline_ref": {}, "data_quality": {}, "explanation": "t",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00+00:00",
            "rule_version": "alert_rules_v1", "status": "open",
            "fingerprint": "fp-r"}], exec_id="e1",
            created_at="2024-01-10T00:00:00+00:00")

        def boom(to, subject, body):
            raise ConnectionError(
                "recipient refused: victim@example.com (550)")

        with caplog.at_level(logging.ERROR, logger="src.calendar.events.notify"):
            stats = notify.process_outbox(memdb_protected, kind="immediate",
                                          sender=boom)
        assert stats["failed"] == 1
        assert "victim@example.com" not in caplog.text
        assert "[redacted-address]" in caplog.text
        rows = store.list_notifications(memdb_protected)
        assert rows[0]["status"] != "sent"
        assert rows[0]["sent_at"] is None


@pytest.fixture()
def memdb_protected():
    import sqlite3

    from src.calendar.events import store

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    store.init_schema(conn)
    yield conn
    conn.close()


def _component_pipeline(param):
    """Mirror st_folium's pre-frontend transforms (installed 0.27.4).

    Runs exactly the pure-Python steps streamlit-folium executes before
    handing the map to its Leaflet frontend: root render, map render,
    html/header/script extraction, id, and bounds. Anything that breaks
    here would break the browser map with no visible error.
    """
    from streamlit_folium import (_get_header, _get_html, _get_map_string,
                                  get_full_id)

    param.get_root().render()
    param.render()
    html = _get_html(param)
    header = _get_header(param)
    leaflet = _get_map_string(param)
    m_id = get_full_id(param)
    bounds = param.get_bounds()
    return {"html": html, "header": header, "leaflet": leaflet, "id": m_id,
            "bounds": bounds}


class TestComponentApi:
    def test_native_width_uses_stretch_not_deprecated(self):
        import inspect

        import streamlit as st

        assert "width" in inspect.signature(st.dataframe).parameters
        root = Path(__file__).resolve().parent.parent / "src" / "calendar"
        offenders = []
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in str(path):
                continue
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                stripped = line.strip()
                if "use_container_width" in stripped and not stripped.startswith("#"):
                    callers = path.read_text().splitlines()[
                        max(0, lineno - 4):lineno]
                    if not any("st_folium" in prev for prev in callers):
                        offenders.append(f"{path.name}:{lineno}")
        assert offenders == []

    def test_st_folium_call_matches_installed_signature(self):
        import inspect

        from streamlit_folium import st_folium

        params = inspect.signature(st_folium).parameters
        assert "use_container_width" in params  # custom-component API
        assert params["width"].annotation in ("int | None", int, None) or \
            "int" in str(params["width"].annotation)

    def test_detection_pipeline_real_data(self):
        from src.calendar.ui import map_view, query

        df = query.query_detections(date_start="2023-08-15",
                                    date_end="2023-08-15",
                                    products=("modis",), limit=5000)
        assert len(df) == 362
        import folium

        param = map_view.build_detection_map(df)
        # Mirror the app: legend element attached before component render.
        param.get_root().html.add_child(
            folium.Element(map_view.legend_html("detections", "sensor")))
        out = _component_pipeline(param)
        assert out["leaflet"].count("L.circleMarker") == 362
        import re as _re
        assert not _re.findall(r"[\[:,]nan[\],\s}]", out["leaflet"].lower())
        assert out["id"] and out["header"] and "Satellite detections" in out["html"]
        (s, w), (n, e) = out["bounds"]
        assert -90 <= s <= n <= 90 and -180 <= w <= e <= 180
        assert (n - s) > 0 and (e - w) > 0  # non-zero dimensions

    def test_cluster_pipeline_real_data(self):
        from src.calendar.ui import map_view, query

        df = query.query_clusters(date_start="2023-08-15",
                                  date_end="2023-08-15", limit=5000,
                                  columns=query.CLUSTER_MAP_COLUMNS)
        assert len(df) > 0
        out = _component_pipeline(map_view.build_cluster_map(df))
        assert out["leaflet"].count("L.circleMarker") == len(df)
        assert "L.rectangle" in out["leaflet"]
        import re as _re
        assert not _re.findall(r"[\[:,]nan[\],\s}]", out["leaflet"].lower())
        (s, w), (n, e) = out["bounds"]
        assert (n - s) > 0 and (e - w) > 0

    def test_empty_pipeline_has_bounds_fallback(self):
        from src.calendar.ui import map_view

        out = _component_pipeline(
            map_view.build_detection_map(_det_frame([])))
        assert out["leaflet"]  # base map script still produced
        assert out["id"]


class TestMapCopyIsolation:
    def test_copy_renders_identical_markers(self):
        from src.calendar.ui import map_view

        df = _det_frame([_det_row(), _det_row(did="d2")])
        first = _component_pipeline(map_view.copy_map(
            map_view.build_detection_map(df)))
        second = _component_pipeline(map_view.copy_map(
            map_view.build_detection_map(df)))
        assert first["leaflet"].count("L.circleMarker") == 2
        assert second["leaflet"].count("L.circleMarker") == 2

    def test_concurrent_copies_never_mutate_shared_state(self):
        import threading

        from src.calendar.ui import map_view

        df = _det_frame([_det_row(did=f"d{i}") for i in range(30)])
        pristine = map_view.build_detection_map(df)
        errors = []

        def worker():
            try:
                for _ in range(2):
                    _component_pipeline(map_view.copy_map(pristine))
            except Exception as exc:  # noqa: BLE001 - collected, asserted
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        assert errors == []
        # The pristine original still renders (never handed out, never hurt).
        assert _component_pipeline(map_view.copy_map(pristine))["leaflet"]


class TestStreamlitSkills:
    def test_official_skill_installed_at_project_scope(self):
        skill = (Path(__file__).resolve().parent.parent / ".claude" /
                 "skills" / "developing-with-streamlit" / "SKILL.md")
        assert skill.is_file()
        text = skill.read_text(encoding="utf-8")
        assert "name: developing-with-streamlit" in text
        assert "use_container_width" in text  # width guidance present

    def test_skill_references_present(self):
        refs = (Path(__file__).resolve().parent.parent / ".claude" /
                "skills" / "developing-with-streamlit" / "references")
        for name in ("testing.md", "data-display.md", "performance.md"):
            assert (refs / name).is_file(), name


class TestExportBuilders:
    def test_csv_exact_values(self):
        from src.calendar.ui import export_view

        text = export_view.rows_to_csv(
            [{"a": "x", "b": 1.5}, {"a": "y,z", "b": 2}],
            ["a", "b"])
        assert text.split("\n")[0] == "a,b"
        assert '"y,z"' in text  # quoting preserved

    def test_missing_stays_missing_never_zero(self):
        import csv as _csv
        import io as _io

        from src.calendar.ui import export_view

        text = export_view.rows_to_csv(
            [{"frp": None}, {"frp": float("nan")}, {"frp": 0.0}],
            ["frp"])
        rows = list(_csv.reader(_io.StringIO(text)))
        assert [r[0] for r in rows[1:]] == ["", "", "0.0"]

    def test_encoding_is_utf8_sig(self):
        from src.calendar.ui import export_view

        payload = export_view.encode_csv("a,b\n")
        assert payload[:3] == b"\xef\xbb\xbf"
        assert payload[3:].decode("utf-8") == "a,b\n"

    def test_filename_sanitized(self):
        from src.calendar.ui import export_view

        name = export_view.export_filename("harmonized", "2023-08-15",
                                           "2023-08-15")
        assert name == "fireagent-calendar-harmonized-2023-08-15_2023-08-15.csv"

    def test_readme_carries_provenance_and_glossary(self):
        from src.calendar.ui import export_view

        text = export_view.readme_text(
            category="modis", filters_used={"date_start": "2023-08-15"},
            total=10, exported=10, truncated=None)
        for token in ("WGS84", "UTC acquisition", "MW", "matcher_v1",
                      "cluster_v1", "detection_id", "never total emitted",
                      "TRUNCATED:", "2A", "2B", "2C"):
            if token == "TRUNCATED:":
                assert token not in text
            else:
                assert token in text, token

    def test_readme_notes_truncation(self):
        from src.calendar.ui import export_view

        text = export_view.readme_text(
            category="combined", filters_used={}, total=200000,
            exported=100000, truncated="row cap")
        assert "TRUNCATED: row cap" in text

    def test_readme_carries_citations(self):
        from src.calendar.ui import export_view

        text = export_view.readme_text(
            category="modis", filters_used={}, total=10, exported=10,
            truncated=None)
        for token in ("How to cite", "FIRMS", "Sofiev",
                      "gmd-19-10123-2026"):
            assert token in text, token

    def test_export_is_deterministic(self):
        from src.calendar.ui import export_view

        selection = {"date_start": "2023-08-15", "date_end": "2023-08-15",
                     "products": ("modis", "viirs-snpp"),
                     "satellites": ("Terra", "Aqua", "N")}
        first = export_view.build_export("modis", selection, None)
        second = export_view.build_export("modis", selection, None)
        assert first["csv_text"] == second["csv_text"]
        assert (first["total"], first["exported"], first["filename"]) == (
            second["total"], second["exported"], second["filename"])
        assert first["total"] == 362  # pinned demo-day count
        readme = [r for r in (first["readme"], second["readme"])]
        stripped = ["\n".join(line for line in text.split("\n")
                              if not line.startswith("Generated (UTC):"))
                    for text in readme]
        assert stripped[0] == stripped[1]

    def test_unknown_category_rejected(self):
        from src.calendar.ui import export_view

        with pytest.raises(ValueError, match="unknown export category"):
            export_view.build_export("bogus", {"date_start": "2023-08-15",
                                               "date_end": "2023-08-15",
                                               "products": ("modis",),
                                               "satellites": ("Terra",)},
                                     None)


class TestExportIntegration:
    def test_modis_day_export_matches_count(self):
        from src.calendar.ui import export_view

        selection = {"date_start": "2023-08-15", "date_end": "2023-08-15",
                     "products": ("modis",), "satellites": ("Terra", "Aqua")}
        payload = export_view.build_export("modis", selection, None)
        assert payload["total"] == payload["exported"] == 362
        assert payload["truncated"] is None
        lines = payload["csv_text"].strip().split("\n")
        assert len(lines) == 363  # header + rows
        assert lines[0].split(",")[0] == "detection_id"
        assert "match_classification" in lines[0]

    def test_harmonized_carries_partners(self):
        from src.calendar.ui import export_view

        selection = {"date_start": "2023-08-15", "date_end": "2023-08-15",
                     "products": ("modis", "viirs-snpp"),
                     "satellites": ("Terra", "Aqua", "N")}
        payload = export_view.build_export("harmonized", selection, None)
        assert payload["exported"] == payload["total"] > 0
        assert "partner_ids" in payload["csv_text"].split("\n")[0]

    def test_clusters_day_export(self):
        from src.calendar.ui import export_view

        selection = {"date_start": "2023-08-15", "date_end": "2023-08-15",
                     "products": ("modis", "viirs-snpp"),
                     "satellites": ("Terra", "Aqua", "N")}
        payload = export_view.build_export("clusters", selection, None)
        assert payload["exported"] == payload["total"] > 0
        assert payload["csv_text"].split("\n")[0].split(",")[0] == "cluster_id"

    def test_empty_export_message_shape(self):
        from src.calendar.ui import export_view

        selection = {"date_start": "2023-08-15", "date_end": "2023-08-15",
                     "products": (), "satellites": ("Terra",)}
        payload = export_view.build_export("modis", selection, None)
        assert payload["total"] == payload["exported"] == 0
        assert payload["csv_text"].strip().split("\n")[0].startswith(
            "detection_id")


def _rule(**over):
    base = {"name": "demo rule", "description": "", "metric": "detection",
            "scope": "raw", "grain": "daily", "period": "2023-08-15",
            "operator": "gt", "threshold": 0.0, "anomaly_class": None,
            "severity_name": "Moderate", "notify": False, "enabled": True}
    base.update(over)
    return base


@pytest.fixture()
def _docs():
    from src.calendar.ui import query

    return {"metrics": query.load_metrics(),
            "anomalies": query.load_anomalies()}


@pytest.fixture()
def crdb():
    import sqlite3

    from src.calendar.events import store

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    store.init_schema(conn)
    yield conn
    conn.close()


class TestCustomRules:
    def test_valid_rule_normalized(self):
        from src.calendar.events import custom

        rule = custom.validate_rule(_rule())
        assert rule["rule_id"].startswith("crt-")
        assert rule["geo_scope"] == "USA-pilot"
        assert rule["version"] == "custom_rules_v1"

    def test_rejects_pooling_and_inventions(self):
        from src.calendar.events import custom

        with pytest.raises(ValueError, match="unsupported series"):
            custom.validate_rule(_rule(metric="detection", scope="modis+N"))
        with pytest.raises(ValueError, match="unsupported series"):
            custom.validate_rule(_rule(metric="frp", scope="harmonized"))

    def test_rejects_bad_grain_period_operator(self):
        from src.calendar.events import custom

        with pytest.raises(ValueError, match="grain"):
            custom.validate_rule(_rule(grain="yearly"))
        with pytest.raises(ValueError, match="not a valid"):
            custom.validate_rule(_rule(period="2023-13-99"))
        with pytest.raises(ValueError, match="together"):
            custom.validate_rule(_rule(operator="gt", threshold=None))
        with pytest.raises(ValueError, match="operator"):
            custom.validate_rule(_rule(operator="~=", threshold=1.0))
        with pytest.raises(ValueError, match="anomaly class"):
            custom.validate_rule(_rule(operator=None, threshold=None,
                                        anomaly_class="extreme"))
        with pytest.raises(ValueError, match="needs a threshold"):
            custom.validate_rule(_rule(operator=None, threshold=None,
                                        anomaly_class=None))

    def test_crud_lifecycle(self, crdb):
        from src.calendar.events import custom, store

        record = custom.validate_rule(_rule())
        record.update({"created_at": "2024-01-01T00:00:00+00:00",
                       "updated_at": "2024-01-01T00:00:00+00:00"})
        store.upsert_custom_rule(crdb, record)
        assert store.get_custom_rule(crdb, record["rule_id"])["name"] == \
            "demo rule"
        assert len(store.list_custom_rules(crdb)) == 1
        store.set_custom_rule_enabled(crdb, record["rule_id"], False)
        assert store.get_custom_rule(crdb, record["rule_id"])["enabled"] == 0
        assert store.delete_custom_rule(crdb, record["rule_id"]) is True
        assert store.get_custom_rule(crdb, record["rule_id"]) is None
        assert store.delete_custom_rule(crdb, record["rule_id"]) is False


class TestCustomEval:
    def test_threshold_triggers_and_misses(self, _docs):
        from src.calendar.events import custom

        entries = [e for e in custom.baselines.extract_series(
            _docs["metrics"], "daily", "detection", "raw")
            if e["period"] == "2023-08-15"]
        value = entries[0]["value"]
        hit = custom.evaluate_rule(
            custom.validate_rule(_rule(threshold=value - 1)),
            metrics=_docs["metrics"], anomalies=_docs["anomalies"])
        assert hit["triggered"] is True and hit["value"] == value
        miss = custom.evaluate_rule(
            custom.validate_rule(_rule(threshold=value + 1)),
            metrics=_docs["metrics"], anomalies=_docs["anomalies"])
        assert miss["triggered"] is False

    def test_missing_value_never_triggers(self, _docs):
        from src.calendar.events import custom

        outcome = custom.evaluate_rule(
            custom.validate_rule(_rule(period="1999-01-01")),
            metrics=_docs["metrics"], anomalies=_docs["anomalies"])
        assert outcome["triggered"] is False
        assert "missing" in outcome["reason"]

    def test_anomaly_class_match_and_mismatch(self, _docs):
        from src.calendar.events import custom

        hit = custom.evaluate_rule(
            custom.validate_rule(_rule(period="2023-02-06", operator=None,
                                        threshold=None,
                                        anomaly_class="notable-high")),
            metrics=_docs["metrics"], anomalies=_docs["anomalies"])
        assert hit["triggered"] is True
        assert hit["anomaly_id"].endswith("2023-02-06")
        miss = custom.evaluate_rule(
            custom.validate_rule(_rule(period="2023-02-06", operator=None,
                                        threshold=None,
                                        anomaly_class="typical")),
            metrics=_docs["metrics"], anomalies=_docs["anomalies"])
        assert miss["triggered"] is False

    def test_leakage_guard_refuses(self, _docs):
        from src.calendar.events import custom

        poisoned = [dict(a) for a in _docs["anomalies"]]
        for row in poisoned:
            if row.get("eval_period") == "2023-02-06":
                row["reference_end"] = "2023-02-06"
        outcome = custom.evaluate_rule(
            custom.validate_rule(_rule(period="2023-02-06", operator=None,
                                        threshold=None,
                                        anomaly_class="notable-high")),
            metrics=_docs["metrics"], anomalies=poisoned)
        assert outcome["triggered"] is False
        assert "leakage" in outcome["reason"]

    def test_run_records_eval_and_hit_idempotent(self, crdb, _docs):
        from src.calendar.events import custom, store

        record = custom.validate_rule(_rule(threshold=0.0))
        record.update({"created_at": "t", "updated_at": "t"})
        store.upsert_custom_rule(crdb, record)
        first = custom.run_evaluation(
            crdb, record["rule_id"], metrics=_docs["metrics"],
            anomalies=_docs["anomalies"])
        second = custom.run_evaluation(
            crdb, record["rule_id"], metrics=_docs["metrics"],
            anomalies=_docs["anomalies"])
        assert first["triggered"] and second["triggered"]
        assert first["hit_id"] == second["hit_id"]
        assert len(store.list_custom_hits(crdb)) == 1  # deduplicated
        assert len(store.list_custom_evals(crdb, record["rule_id"])) == 2
        store.set_custom_hit_status(crdb, first["hit_id"], "acknowledged")
        assert store.get_custom_hit(crdb, first["hit_id"])["status"] == \
            "acknowledged"

    def test_disabled_and_unknown_rules_rejected(self, crdb, _docs):
        from src.calendar.events import custom, store

        record = custom.validate_rule(_rule())
        record.update({"created_at": "t", "updated_at": "t", "enabled": False})
        store.upsert_custom_rule(crdb, record)
        with pytest.raises(ValueError, match="disabled"):
            custom.run_evaluation(crdb, record["rule_id"],
                                  metrics=_docs["metrics"],
                                  anomalies=_docs["anomalies"])
        with pytest.raises(ValueError, match="unknown custom rule"):
            custom.run_evaluation(crdb, "crt-nope", metrics=_docs["metrics"],
                                  anomalies=_docs["anomalies"])


class TestCustomNotify:
    def _hit_rule(self, crdb):
        from src.calendar.events import custom, store

        record = custom.validate_rule(_rule(threshold=0.0, notify=True))
        record.update({"created_at": "t", "updated_at": "t"})
        store.upsert_custom_rule(crdb, record)
        return record

    def test_smtp_unavailable_keeps_queue(self, crdb, _docs, monkeypatch):
        from src.calendar.events import custom, notify, store

        monkeypatch.setattr("src.calendar.settings.CAL_SMTP_ENABLED", "false")
        record = self._hit_rule(crdb)
        outcome = custom.run_evaluation(
            crdb, record["rule_id"], metrics=_docs["metrics"],
            anomalies=_docs["anomalies"])
        assert outcome["triggered"]
        assert len(store.list_custom_notifications(crdb, status="queued")) == 1
        stats = custom.send_pending_custom(crdb, sender=notify.send_via_smtp)
        assert stats == {"sent": 0, "failed": 0, "skipped": 0,
                         "reason": "SMTP unavailable: queue retained, "
                         "nothing sent"}
        assert len(store.list_custom_notifications(crdb, status="queued")) == 1

    def test_stub_delivery_marks_sent_idempotent(
            self, crdb, _docs, monkeypatch):
        from src.calendar.events import custom, notify, store

        monkeypatch.setattr("src.calendar.settings.CAL_SMTP_ENABLED", "true")
        monkeypatch.setattr("src.calendar.settings.CAL_SMTP_USER", "u")
        monkeypatch.setattr("src.calendar.settings.CAL_SMTP_PASSWORD", "p")
        pending = notify.subscribe(crdb, "op@example.com")
        notify.confirm(crdb, pending["confirm_token"])
        record = self._hit_rule(crdb)
        custom.run_evaluation(crdb, record["rule_id"],
                              metrics=_docs["metrics"],
                              anomalies=_docs["anomalies"])
        sent = []
        stats = custom.send_pending_custom(
            crdb, sender=lambda to, s, b: sent.append((to, s, b)) or True)
        assert stats["sent"] == 1 and sent[0][0] == "op@example.com"
        assert "Custom alert" in sent[0][1]
        assert "USER-DEFINED" in sent[0][2]
        assert "not proof of a wildfire" in sent[0][2]
        again = custom.send_pending_custom(
            crdb, sender=lambda to, s, b: True)
        assert again == {"sent": 0, "failed": 0, "skipped": 0}

    def test_no_subscribers_keeps_queue(self, crdb, _docs, monkeypatch):
        from src.calendar.events import custom, store

        monkeypatch.setattr("src.calendar.settings.CAL_SMTP_ENABLED", "true")
        monkeypatch.setattr("src.calendar.settings.CAL_SMTP_USER", "u")
        monkeypatch.setattr("src.calendar.settings.CAL_SMTP_PASSWORD", "p")
        record = self._hit_rule(crdb)
        custom.run_evaluation(crdb, record["rule_id"],
                              metrics=_docs["metrics"],
                              anomalies=_docs["anomalies"])
        stats = custom.send_pending_custom(
            crdb, sender=lambda to, s, b: True)
        assert stats["sent"] == 0
        assert "no confirmed subscribers" in stats["reason"]
        assert len(store.list_custom_notifications(crdb, status="queued")) == 1

    def test_retry_budget_then_failed(self, crdb, _docs, monkeypatch):
        from src.calendar.events import custom, notify, store

        monkeypatch.setattr("src.calendar.settings.CAL_SMTP_ENABLED", "true")
        monkeypatch.setattr("src.calendar.settings.CAL_SMTP_USER", "u")
        monkeypatch.setattr("src.calendar.settings.CAL_SMTP_PASSWORD", "p")
        pending = notify.subscribe(crdb, "op@example.com")
        notify.confirm(crdb, pending["confirm_token"])
        record = self._hit_rule(crdb)
        custom.run_evaluation(crdb, record["rule_id"],
                              metrics=_docs["metrics"],
                              anomalies=_docs["anomalies"])
        for _ in range(4):
            custom.send_pending_custom(
                crdb, sender=lambda to, s, b: False, max_attempts=3)
        rows = store.list_custom_notifications(crdb)
        assert len(rows) == 1
        assert rows[0]["status"] == "failed"
        assert rows[0]["sent_at"] is None


def _catalog(*entries):
    return {"data": [
        {"id": mid, "context_length": 32000,
         "architecture": {"input_modalities": ["text"],
                           "output_modalities": ["text"]},
         "pricing": {"prompt": prompt, "completion": completion}}
        for mid, prompt, completion in entries]}


def _stub_transport(catalog, *, key_ok=True, calls=None):
    import requests

    def fake_get(url, headers=None, timeout=None):
        (calls if calls is not None else []).append(url)
        if url.endswith("/models"):
            return _StubResponse(catalog)
        if url.endswith("/auth/key"):
            if "Authorization" not in (headers or {}):
                raise requests.HTTPError("401")
            if not key_ok:
                raise requests.HTTPError("401 Unauthorized")
            return _StubResponse({"data": {"label": "stub-key"}})
        raise AssertionError(f"unexpected discovery URL {url}")

    return fake_get


class _StubResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class TestModelDiscovery:
    def test_key_sent_without_exposure(self):
        from src.calendar.events import models

        calls, seen = [], {}

        def spy(url, headers=None, timeout=None):
            seen.update(headers or {})
            calls.append(url)
            return _StubResponse({"data": []} if url.endswith("/models")
                                  else {"data": {"label": "k"}})

        out = models.discover_models("opaque-key", transport=spy)
        assert seen.get("Authorization") == "Bearer opaque-key"
        assert "opaque-key" not in json.dumps(out)
        assert all(u.endswith("/models") or u.endswith("/auth/key")
                   for u in calls)  # no completion call

    def test_free_qwen_preferred(self):
        from src.calendar.events import models

        catalog = _catalog(
            ("qwen/qwen-paid", "0.0001", "0.0002"),
            ("qwen/qwen-free", "0", "0"),
            ("other/free", "0", "0"))
        out = models.discover_models(
            "k", transport=_stub_transport(catalog))
        picked = models.select_model(out["models"])
        assert picked["model_id"] == "qwen/qwen-free"
        assert picked["price_class"] == "free"
        assert picked["needs_approval"] is False

    def test_cheapest_paid_needs_approval(self):
        from src.calendar.events import models

        catalog = _catalog(
            ("qwen/qwen-a", "0.001", "0.002"),
            ("qwen/qwen-b", "0.0001", "0.0001"))
        out = models.discover_models(
            "k", transport=_stub_transport(catalog))
        picked = models.select_model(out["models"])
        assert picked["model_id"] == "qwen/qwen-b"
        assert picked["price_class"] == "paid"
        assert picked["needs_approval"] is True
        approved = models.select_model(out["models"], allow_paid=True)
        assert approved["needs_approval"] is False

    def test_non_qwen_free_never_selected_as_qwen(self):
        from src.calendar.events import models

        catalog = _catalog(("other/free", "0", "0"),
                            ("openrouter/free", "0", "0"))
        out = models.discover_models(
            "k", transport=_stub_transport(catalog))
        picked = models.select_model(out["models"])
        assert picked["model_id"] is None
        assert "no Qwen chat model" in picked["reason"]

    def test_non_chat_models_ignored(self):
        from src.calendar.events import models

        catalog = {"data": [{
            "id": "qwen/qwen-image", "context_length": 100,
            "architecture": {"input_modalities": ["text"],
                             "output_modalities": ["image"]},
            "pricing": {"prompt": "0", "completion": "0"}}]}
        out = models.discover_models(
            "k", transport=_stub_transport(catalog))
        assert models.select_model(out["models"])["model_id"] is None

    def test_auth_failure_reported(self):
        from src.calendar.events import models

        out = models.discover_models(
            "bad", transport=_stub_transport(_catalog(), key_ok=False))
        assert out["key_authorized"] is False

    def test_timeout_and_network_failure_fall_back(self):
        from src.calendar.events import models

        def boom(url, headers=None, timeout=None):
            raise TimeoutError("timed out")

        out = models.discover_models("k", transport=boom)
        assert out["models"] == []
        assert "timed out" in out["error"]
        assert models.select_model(out["models"])["model_id"] is None

    def test_cache_ttl_and_refresh(self, crdb):
        import time

        from src.calendar.events import models, store

        catalog = _catalog(("qwen/qwen-free", "0", "0"))
        record = models.refresh_discovery(
            crdb, api_key="k",
            transport=_stub_transport(catalog))
        assert record["selection"]["model_id"] == "qwen/qwen-free"
        assert record["key_authorized"] is True
        assert models.load_discovery(crdb)["selection"]["model_id"] == \
            "qwen/qwen-free"
        stale = models.load_discovery(
            crdb, now_ts=time.time() + models.CACHE_TTL_S + 1)
        assert stale is None
        assert store.meta_get(crdb, models.META_KEY)["catalog_count"] == 1

    def test_paid_approval_flow(self, crdb):
        from src.calendar.events import models

        catalog = _catalog(("qwen/qwen-paid", "0.001", "0.002"))
        transport = _stub_transport(catalog)
        first = models.refresh_discovery(crdb, api_key="k",
                                         transport=transport)
        assert first["selection"]["needs_approval"] is True
        model_id = first["selection"]["model_id"]
        from src.calendar.events.qwen import QwenConfig

        cfg = QwenConfig(enabled=True, base_url="u", api_key="k", model="",
                         timeout_s=5.0)
        assert models.effective_model(crdb, cfg)[0] is None
        models.approve_paid_model(crdb, model_id)
        second = models.refresh_discovery(crdb, api_key="k",
                                          transport=transport)
        assert second["selection"]["needs_approval"] is False
        assert models.effective_model(crdb, cfg)[0] == model_id

    def test_effective_model_prefers_discovery_then_config(self, crdb):
        from src.calendar.events import models
        from src.calendar.events.qwen import QwenConfig

        catalog = _catalog(("qwen/qwen-free", "0", "0"))
        models.refresh_discovery(crdb, api_key="k",
                                 transport=_stub_transport(catalog))
        cfg = QwenConfig(enabled=True, base_url="u", api_key="k",
                         model="explicit-model", timeout_s=5.0)
        assert models.effective_model(crdb, cfg)[0] == "qwen/qwen-free"


class TestFreeModelPolicy:
    def _cfg(self, model):
        from src.calendar.events.qwen import QwenConfig

        return QwenConfig(enabled=True, base_url="u", api_key="k",
                          model=model, timeout_s=5.0)

    def test_explicit_free_name_never_implies_free(self):
        from src.calendar.events import models

        out = models.validate_explicit_model([], "qwen/free-sounding")
        assert out["verdict"] == "unknown"

    def test_explicit_paid_blocked_without_approval(self, crdb):
        from src.calendar.events import models

        catalog = _catalog(("qwen/qwen3.7-flash", "0.001", "0.002"))
        models.refresh_discovery(crdb, api_key="k",
                                 transport=_stub_transport(catalog))
        model, reason = models.effective_model(
            crdb, self._cfg("qwen/qwen3.7-flash"))
        assert model is None
        assert "blocked" in reason and "paid" in reason

    def test_explicit_paid_allowed_with_approval(self, crdb):
        from src.calendar.events import models

        catalog = _catalog(("qwen/qwen3.7-flash", "0.001", "0.002"))
        models.refresh_discovery(crdb, api_key="k",
                                 transport=_stub_transport(catalog))
        models.approve_paid_model(crdb, "qwen/qwen3.7-flash")
        models.refresh_discovery(crdb, api_key="k",
                                 transport=_stub_transport(catalog))
        model, _ = models.effective_model(
            crdb, self._cfg("other-model"))
        # approved paid discovery selection wins over explicit config
        assert model == "qwen/qwen3.7-flash"

    def test_explicit_missing_pricing_blocked(self, crdb):
        from src.calendar.events import models

        catalog = _catalog(("qwen/qwen-noprice", None, None))
        models.refresh_discovery(crdb, api_key="k",
                                 transport=_stub_transport(catalog))
        model, reason = models.effective_model(
            crdb, self._cfg("qwen/qwen-noprice"))
        assert model is None
        assert "not free" in reason

    def test_explicit_malformed_pricing_blocked(self, crdb):
        from src.calendar.events import models

        catalog = _catalog(("qwen/qwen-weird", "free", {"x": 1}))
        models.refresh_discovery(crdb, api_key="k",
                                 transport=_stub_transport(catalog))
        model, reason = models.effective_model(
            crdb, self._cfg("qwen/qwen-weird"))
        assert model is None
        assert "not free" in reason

    def test_explicit_without_record_blocked(self, crdb):
        from src.calendar.events import models

        model, reason = models.effective_model(
            crdb, self._cfg("qwen/whatever"))
        assert model is None
        assert "no discovery record" in reason

    def test_no_config_no_record_reports_plainly(self, crdb):
        from src.calendar.events import models

        model, reason = models.effective_model(crdb, self._cfg(""))
        assert model is None
        assert "no model discovered or configured" in reason

    def test_paid_selection_without_explicit_names_approval(self, crdb):
        from src.calendar.events import models

        catalog = _catalog(("qwen/qwen3.7-flash", "0.001", "0.002"))
        models.refresh_discovery(crdb, api_key="k",
                                 transport=_stub_transport(catalog))
        model, reason = models.effective_model(crdb, self._cfg(""))
        assert model is None
        assert "approval" in reason

    def test_discovery_failure_selects_nothing(self, crdb):
        from src.calendar.events import models

        def boom(url, headers=None, timeout=None):
            raise ConnectionError("offline")

        record = models.refresh_discovery(crdb, api_key="k",
                                          transport=boom)
        assert record["selection"]["model_id"] is None
        assert "failed" in record["error"]
        model, _ = models.effective_model(crdb, self._cfg(""))
        assert model is None

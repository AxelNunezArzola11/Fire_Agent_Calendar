"""Regression tests for Streamlit Community Cloud storage & UI safety."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import duckdb
import pytest

from src.calendar import settings
from src.calendar.events import store as events_store
from src.calendar.ui import (
    analytics_view,
    events_view,
    export_view,
    filters,
    map_view,
    query,
)


REPO_ROOT = settings.REPO_ROOT


class _FakeCol:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeStreamlit:
    """Minimal Streamlit stub capturing warnings, info, errors, and widgets."""

    def __init__(self, initial_state: dict | None = None):
        self.session_state: dict = dict(initial_state or {})
        self.warnings: list[str] = []
        self.infos: list[str] = []
        self.errors: list[str] = []
        self.captions: list[str] = []
        self.writes: list[str] = []
        self.headers: list[str] = []
        self.dataframes: list[object] = []
        self.charts: list[object] = []
        self.metrics: list[tuple[str, str]] = []
        self.download_buttons: list[dict] = []
        self.radio_overrides: dict[str, object] = {}

    def columns(self, spec, **kwargs):
        count = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
        return [_FakeCol() for _ in range(count)]

    def warning(self, msg, **kwargs):
        self.warnings.append(str(msg))

    def info(self, msg, **kwargs):
        self.infos.append(str(msg))

    def error(self, msg, **kwargs):
        self.errors.append(str(msg))

    def caption(self, msg, **kwargs):
        self.captions.append(str(msg))

    def write(self, msg, **kwargs):
        self.writes.append(str(msg))

    def header(self, msg, **kwargs):
        self.headers.append(str(msg))

    def subheader(self, msg, **kwargs):
        self.headers.append(str(msg))

    def markdown(self, msg, **kwargs):
        self.writes.append(str(msg))

    def success(self, msg, **kwargs):
        self.writes.append(str(msg))

    def divider(self):
        return None

    def metric(self, label, value, **kwargs):
        self.metrics.append((str(label), str(value)))

    def dataframe(self, data, **kwargs):
        self.dataframes.append(data)

    def plotly_chart(self, fig, **kwargs):
        self.charts.append(fig)

    def form(self, key, **kwargs):
        return _FakeCol()

    def expander(self, label, **kwargs):
        return _FakeCol()

    def form_submit_button(self, label="Submit", **kwargs):
        return False

    def radio(self, label, options, index=0, key=None, **kwargs):
        if key and key in self.radio_overrides:
            val = self.radio_overrides[key]
        elif key and key in self.session_state:
            val = self.session_state[key]
        else:
            val = options[index]
        if key:
            self.session_state[key] = val
        return val

    def checkbox(self, label, value=False, key=None, **kwargs):
        if key and key in self.session_state:
            return bool(self.session_state[key])
        if key:
            self.session_state[key] = bool(value)
        return bool(value)

    def selectbox(self, label, options, index=0, key=None, **kwargs):
        opts = list(options)
        if key and key in self.session_state and self.session_state[key] in opts:
            return self.session_state[key]
        val = opts[index]
        if key:
            self.session_state[key] = val
        return val

    def multiselect(self, label, options, default=None, key=None, **kwargs):
        val = list(default) if default is not None else []
        if key:
            val = list(self.session_state.get(key, val))
            self.session_state[key] = val
        return val

    def number_input(self, label, value=0, key=None, **kwargs):
        if key:
            val = self.session_state.get(key, value)
            self.session_state[key] = val
            return val
        return value

    def text_input(self, label, value="", key=None, **kwargs):
        if key:
            val = self.session_state.get(key, value)
            self.session_state[key] = val
            return val
        return value

    def button(self, label, key=None, **kwargs):
        return False

    def download_button(self, label, data=None, file_name=None, **kwargs):
        self.download_buttons.append({"label": label, "file_name": file_name})
        return False

    def spinner(self, text=""):
        return _FakeCol()


def _install_fake_st(monkeypatch, initial_state: dict | None = None) -> _FakeStreamlit:
    fake = _FakeStreamlit(initial_state)
    fake_mod = types.ModuleType("streamlit")
    for name in dir(fake):
        if not name.startswith("_"):
            setattr(fake_mod, name, getattr(fake, name))
    fake_mod.session_state = fake.session_state

    # map_view imports st_folium unconditionally at top of render_map_tab
    fake_folium = types.ModuleType("streamlit_folium")
    fake_folium.st_folium = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "streamlit", fake_mod)
    monkeypatch.setitem(sys.modules, "streamlit_folium", fake_folium)
    return fake


def _create_sample_duckdb(path: Path) -> None:
    conn = duckdb.connect(str(path))
    try:
        conn.execute(
            """
            CREATE TABLE detections (
                detection_id VARCHAR,
                sensor VARCHAR,
                satellite VARCHAR,
                instrument VARCHAR,
                product VARCHAR,
                acquisition_timestamp_utc VARCHAR,
                latitude DOUBLE,
                longitude DOUBLE,
                confidence VARCHAR,
                frp DOUBLE,
                daynight VARCHAR,
                acq_date_utc VARCHAR,
                year INTEGER,
                month INTEGER
            )
            """
        )
        conn.execute(
            """
            INSERT INTO detections VALUES
            ('det-1', 'MODIS', 'Terra', 'MODIS', 'modis',
             '2023-08-01T18:00:00Z', 39.5, -121.2, '85', 42.5, 'D', '2023-08-01',
             2023, 8)
            """
        )
    finally:
        conn.close()


def _create_exec_artifact(
    analytics_root: Path, exec_id: str, filename: str, payload: object
) -> Path:
    exec_dir = analytics_root / exec_id
    exec_dir.mkdir(parents=True, exist_ok=True)
    target = exec_dir / filename
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target


def test_1_duckdb_file_exists_and_opens_successfully(tmp_path, monkeypatch):
    sample_db = tmp_path / "valid_calendar.duckdb"
    _create_sample_duckdb(sample_db)
    monkeypatch.setattr(settings, "CAL_DUCKDB_PATH", str(sample_db))

    status = query.duckdb_status()
    assert status["available"] is True
    assert status["status"] == "ok"
    assert status["env_var"] == "CAL_DUCKDB_PATH"

    total = query.count_detections(date_start="2023-08-01", date_end="2023-08-01")
    assert total == 1
    df = query.query_detections(date_start="2023-08-01", date_end="2023-08-01")
    assert len(df) == 1
    assert df.iloc[0]["detection_id"] == "det-1"


def test_2_duckdb_file_does_not_exist(tmp_path, monkeypatch):
    missing_db = tmp_path / "does_not_exist.duckdb"
    monkeypatch.setattr(settings, "CAL_DUCKDB_PATH", str(missing_db))

    status = query.duckdb_status()
    assert status["available"] is False
    assert status["status"] == "missing"
    assert "CAL_DUCKDB_PATH" in status["remediation"]
    # Must never silently create an empty database file
    assert not missing_db.exists()

    with pytest.raises(query.DataUnavailableError):
        query.count_detections(date_start="2023-08-01", date_end="2023-08-01")
    with pytest.raises(query.DataUnavailableError):
        query.query_detections(date_start="2023-08-01", date_end="2023-08-01")
    with pytest.raises(query.DataUnavailableError):
        query.detection_by_id("det-1")
    assert not missing_db.exists()


def test_3_duckdb_path_invalid_or_inaccessible(tmp_path, monkeypatch):
    # Corrupt non-DuckDB file
    corrupt_db = tmp_path / "corrupt.duckdb"
    corrupt_db.write_text("not a valid duckdb binary file", encoding="utf-8")
    monkeypatch.setattr(settings, "CAL_DUCKDB_PATH", str(corrupt_db))
    corrupt_status = query.duckdb_status()
    assert corrupt_status["available"] is False
    assert corrupt_status["status"] == "invalid"
    with pytest.raises(query.DataUnavailableError):
        query.detection_by_id("det-1")

    # Directory instead of a file
    dir_status = query.duckdb_status(str(tmp_path))
    assert dir_status["available"] is False
    assert dir_status["status"] == "invalid"

    # Empty initialized DuckDB file without detections table/view
    empty_db = tmp_path / "empty.duckdb"
    duckdb.connect(str(empty_db)).close()
    empty_status = query.duckdb_status(str(empty_db))
    assert empty_status["available"] is False
    assert empty_status["status"] == "invalid"

    # Unconfigured empty setting
    monkeypatch.setattr(settings, "CAL_DUCKDB_PATH", "")
    monkeypatch.delenv("CAL_DUCKDB_PATH", raising=False)
    unconf_status = query.duckdb_status()
    assert unconf_status["available"] is False
    assert unconf_status["status"] == "unconfigured"


def test_4_analytical_execution_exists_and_artifact_available(tmp_path, monkeypatch):
    analytics_root = tmp_path / "analytics_output"
    exec_id = "461ecbc98ca14f11910e8bcda7254027"
    sample_anomalies = [
        {
            "anomaly_id": "anom-test-1",
            "period_key": "2023-08-01",
            "metric_name": "detection_count",
            "observed_value": 100.0,
            "expected_value": 20.0,
            "z_score": 4.2,
            "severity": "elevated",
            "is_anomalous": True,
            "baseline_available": True,
        }
    ]
    expected_file = _create_exec_artifact(
        analytics_root, exec_id, "anomalies_4c.json", sample_anomalies
    )
    monkeypatch.setattr(settings, "CAL_ANALYTICS_ROOT", str(analytics_root))
    monkeypatch.setattr(settings, "CAL_EXEC_4C", exec_id)

    status = query.artifact_status("4C", "anomalies_4c.json")
    assert status["available"] is True
    assert status["status"] == "ok"
    assert status["exec_id"] == exec_id
    assert query.analytics_path(exec_id, "anomalies_4c.json") == expected_file

    loaded = query.load_anomalies()
    assert len(loaded) == 1
    assert loaded[0]["anomaly_id"] == "anom-test-1"


def test_5_analytical_execution_exists_but_artifact_missing(tmp_path, monkeypatch):
    analytics_root = tmp_path / "analytics_output"
    exec_id = "abcd1234abcd1234abcd1234abcd1234"
    (analytics_root / exec_id).mkdir(parents=True)
    runs_file = tmp_path / "analytics_runs.jsonl"
    runs_file.write_text("", encoding="utf-8")

    monkeypatch.setattr(settings, "CAL_ANALYTICS_ROOT", str(analytics_root))
    monkeypatch.setattr(settings, "CAL_RUNS_PATH", str(runs_file))
    monkeypatch.setattr(settings, "CAL_EXEC_4C", exec_id)

    # Case A: execution directory exists, artifact file missing
    st_dir = query.artifact_status("4C", "anomalies_4c.json")
    assert st_dir["available"] is False
    assert st_dir["status"] == "missing_artifact"
    with pytest.raises(query.DataUnavailableError):
        query.analytics_path(exec_id, "anomalies_4c.json")
    with pytest.raises(query.DataUnavailableError):
        query.load_anomalies()

    # Case B: directory removed, but execution registered in analytics_runs.jsonl
    (analytics_root / exec_id).rmdir()
    runs_file.write_text(json.dumps({"exec_id": exec_id, "stage": "4C"}) + "\n", encoding="utf-8")
    st_run = query.artifact_status("4C", "anomalies_4c.json")
    assert st_run["available"] is False
    assert st_run["registered_in_runs"] is True
    assert st_run["status"] == "missing_artifact"

    # Case C: neither directory nor registry entry exists
    runs_file.write_text("", encoding="utf-8")
    st_none = query.artifact_status("4C", "anomalies_4c.json")
    assert st_none["available"] is False
    assert st_none["status"] == "missing_execution"


def test_6_execution_id_not_configured(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "CAL_ANALYTICS_ROOT", str(tmp_path))
    monkeypatch.setattr(settings, "CAL_EXEC_4C", "")
    monkeypatch.delenv("CAL_EXEC_4C", raising=False)

    status = query.artifact_status("4C", "anomalies_4c.json")
    assert status["available"] is False
    assert status["status"] == "unconfigured"
    assert status["env_var"] == "CAL_EXEC_4C"

    with pytest.raises(query.DataUnavailableError):
        query.analytics_path("", "anomalies_4c.json")
    with pytest.raises(query.DataUnavailableError):
        query.load_anomalies()

    # Path traversal / invalid exec_id must also be rejected cleanly
    bad_status = query.artifact_status("4C", "anomalies_4c.json", exec_id="../outside")
    assert bad_status["available"] is False
    assert bad_status["status"] == "invalid_exec_id"
    with pytest.raises(query.DataUnavailableError):
        query.analytics_path("../outside", "anomalies_4c.json")


def test_7_local_storage_works_without_cloud_config(monkeypatch):
    # Committed Stage 2D & 2E artifacts must resolve without Streamlit Cloud config
    assert query.artifact_status("2D", query.METRICS_FILENAME)["available"] is True
    assert query.artifact_status("2E", query.INDEX_FILENAME)["available"] is True
    metrics = query.load_metrics()
    index = query.load_index()
    assert "daily" in metrics and "monthly" in metrics
    assert index["exec_id"] == settings.CAL_EXEC_2E

    overview = query.storage_overview()
    assert overview["artifacts"]["2D_metrics"]["available"] is True
    assert overview["artifacts"]["2E_index"]["available"] is True

    # If local DuckDB and Parquet artifacts are present in this checkout, verify they work end-to-end
    if settings.resolve_repo_path(settings.CAL_DUCKDB_PATH).exists():
        assert overview["duckdb"]["available"] is True
        assert query.count_detections(date_start="2023-08-01", date_end="2023-08-05") >= 0
        payload = export_view.build_export(
            "modis",
            {
                "date_start": "2023-08-01",
                "date_end": "2023-08-02",
                "products": ("modis",),
                "satellites": ("Terra", "Aqua"),
                "modis_conf_min": None,
                "viirs_conf": None,
            },
            None,
        )
        assert "csv_text" in payload and "readme" in payload

    if settings.resolve_repo_path(
        f"{settings.CAL_ANALYTICS_ROOT}/{settings.CAL_EXEC_4C}/anomalies_4c.json"
    ).exists():
        assert isinstance(query.load_anomalies(), list)

    # Also verify settings.get_setting reads Streamlit secrets cleanly with env precedence
    fake_st = types.ModuleType("streamlit")
    fake_st.secrets = {
        "CAL_EXEC_4C": "secret-top-level",
        "calendar": {"CAL_EXEC_2C": "secret-nested"},
    }
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    monkeypatch.delenv("CAL_EXEC_4C", raising=False)
    monkeypatch.delenv("CAL_EXEC_2C", raising=False)
    # Precedence: monkeypatch.setattr(settings, ...) > os.environ > st.secrets > default
    assert settings.get_setting("CAL_EXEC_4C") == "secret-top-level"
    assert settings.get_setting("CAL_EXEC_2C") == "secret-nested"
    monkeypatch.setenv("CAL_EXEC_4C", "env-wins")
    assert settings.get_setting("CAL_EXEC_4C") == "env-wins"
    monkeypatch.setattr(settings, "CAL_EXEC_4C", "attr-wins")
    assert settings.get_setting("CAL_EXEC_4C") == "attr-wins"


def test_8_cloud_mode_handles_missing_storage_without_crashing(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    cloud_analytics = tmp_path / "analytics_output"
    for exec_id, fname in (
        (settings.CAL_EXEC_2D, query.METRICS_FILENAME),
        (settings.CAL_EXEC_2E, query.INDEX_FILENAME),
    ):
        src = REPO_ROOT / "analytics_output" / exec_id / fname
        dst = cloud_analytics / exec_id / fname
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())

    missing_duckdb = tmp_path / "missing_cloud.duckdb"
    missing_events = tmp_path / "missing_events.sqlite"
    monkeypatch.setattr(settings, "CAL_DUCKDB_PATH", str(missing_duckdb))
    monkeypatch.setattr(settings, "CAL_EVENTS_DB_PATH", str(missing_events))
    monkeypatch.setattr(settings, "CAL_ANALYTICS_ROOT", str(cloud_analytics))
    monkeypatch.setattr(settings, "CAL_RUNS_PATH", str(tmp_path / "no_runs.jsonl"))

    at = AppTest.from_file(str(REPO_ROOT / "calendar_app.py"), default_timeout=120)
    at.run()

    assert not at.exception
    assert not at.error
    # App renders unavailable diagnostics across Map, Analytics, Events, and Export
    warning_text = "\n".join(w.value for w in at.warning)
    assert "unavailable" in warning_text.lower()
    # Must never create empty databases on ephemeral cloud disk
    assert not missing_duckdb.exists()
    assert not missing_events.exists()
    # Must not emit fabricated export CSV download buttons
    export_labels = [b.label for b in at.download_button]
    assert "Download CSV" not in export_labels


def test_9_map_handles_unavailable_data(tmp_path, monkeypatch):
    missing_db = tmp_path / "missing_map.duckdb"
    monkeypatch.setattr(settings, "CAL_DUCKDB_PATH", str(missing_db))
    monkeypatch.setattr(settings, "CAL_ANALYTICS_ROOT", str(tmp_path / "no_analytics"))
    monkeypatch.setattr(settings, "CAL_RUNS_PATH", str(tmp_path / "no_runs.jsonl"))

    called = {"det": 0, "clu": 0, "det_map": 0, "clu_map": 0}

    def _fail_if_called(name):
        def _fn(*args, **kwargs):
            called[name] += 1
            raise AssertionError(f"Callback {name} should not run when source is unavailable")
        return _fn

    cb = map_view.MapCallbacks(
        fetch_det=_fail_if_called("det"),
        count_det=_fail_if_called("det"),
        fetch_clu=_fail_if_called("clu"),
        count_clu=_fail_if_called("clu"),
        by_det=lambda *_: None,
        member=lambda *_: None,
        group=lambda *_: None,
        by_clu=lambda *_: None,
        build_det_map=_fail_if_called("det_map"),
        build_clu_map=_fail_if_called("clu_map"),
        fetch_cluster_dets=lambda *_: None,
    )
    spans = {"date_min": "2023-08-01", "date_max": "2023-08-05"}

    # Detection Mode unavailable
    fake_det = _install_fake_st(
        monkeypatch,
        {
            filters.KEY_MODE: "detections",
            map_view.KEY_SELECTED_ID: "stale-det-id",
        },
    )
    map_view.render_map_tab(spans, cb)
    assert fake_det.errors == []
    assert fake_det.session_state[map_view.KEY_SELECTED_ID] is None
    assert any("unavailable" in w.lower() for w in fake_det.warnings)
    assert any("CAL_DUCKDB_PATH" in i for i in fake_det.infos)
    assert all(v == 0 for v in called.values())

    # Cluster Mode unavailable
    fake_clu = _install_fake_st(
        monkeypatch,
        {
            filters.KEY_MODE: "clusters",
            map_view.KEY_SELECTED_ID: "cl0000001",
        },
    )
    map_view.render_map_tab(spans, cb)
    assert fake_clu.errors == []
    assert fake_clu.session_state[map_view.KEY_SELECTED_ID] is None
    assert any("unavailable" in w.lower() for w in fake_clu.warnings)
    assert any("clusters.parquet" in i for i in fake_clu.infos)
    assert all(v == 0 for v in called.values())


def test_10_analytics_handles_unavailable_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "CAL_ANALYTICS_ROOT", str(tmp_path / "missing_analytics"))
    monkeypatch.setattr(settings, "CAL_RUNS_PATH", str(tmp_path / "no_runs.jsonl"))

    fake = _install_fake_st(monkeypatch)
    analytics_view.render_analytics_tab(None)
    assert fake.errors == []
    assert any("unavailable" in w.lower() for w in fake.warnings)
    assert any(
        "anomalies_4c.json" in i and "CAL_EXEC_4C" in i for i in fake.infos
    )
    assert fake.charts == []
    assert fake.dataframes == []

    # Unconfigured CAL_EXEC_4C
    monkeypatch.setattr(settings, "CAL_EXEC_4C", "")
    fake_unconf = _install_fake_st(monkeypatch)
    analytics_view.render_analytics_tab(None)
    assert fake_unconf.errors == []
    assert any("unconfigured" in w.lower() for w in fake_unconf.warnings)
    assert fake_unconf.charts == []


def test_11_events_handles_unavailable_artifacts_and_zero_events(tmp_path, monkeypatch):
    # Case A: Missing events SQLite database
    missing_events = tmp_path / "missing_events.sqlite"
    monkeypatch.setattr(settings, "CAL_EVENTS_DB_PATH", str(missing_events))
    monkeypatch.setattr(settings, "CAL_ANALYTICS_ROOT", str(tmp_path / "no_analytics"))
    monkeypatch.setattr(settings, "CAL_RUNS_PATH", str(tmp_path / "no_runs.jsonl"))

    assert events_view._db_ok() is False
    ev_missing = query.events_db_status()
    assert ev_missing["available"] is False
    assert ev_missing["status"] == "missing"

    fake_missing = _install_fake_st(monkeypatch)
    events_view.render_events_tab(metrics=None, anomalies=None)
    assert fake_missing.errors == []
    assert any("not zero detected events" in w.lower() for w in fake_missing.warnings)
    assert not missing_events.exists()

    # Case B: Corrupt/non-schema SQLite file must report invalid
    bad_sqlite = tmp_path / "bad_events.sqlite"
    bad_sqlite.write_text("not sqlite", encoding="utf-8")
    monkeypatch.setattr(settings, "CAL_EVENTS_DB_PATH", str(bad_sqlite))
    assert query.events_db_status()["status"] == "invalid"
    assert events_view._db_ok() is False

    # Case C: Real initialized events SQLite database containing 0 rows
    real_empty_db = tmp_path / "empty_events.sqlite"
    conn = events_store.connect(str(real_empty_db))
    events_store.init_schema(conn)
    conn.close()
    monkeypatch.setattr(settings, "CAL_EVENTS_DB_PATH", str(real_empty_db))

    assert query.events_db_status()["available"] is True
    assert events_view._db_ok() is True

    fake_empty = _install_fake_st(monkeypatch)
    fake_empty.radio_overrides[events_view.KEY_EV_SECTION] = "Events"
    events_view.render_events_tab(metrics=None, anomalies=None)
    assert fake_empty.errors == []
    assert any(
        "no events detected" in msg.lower() for msg in fake_empty.infos
    )

    # Case D: Custom Alerts handles missing 2D metrics and 4C anomalies cleanly
    fake_custom = _install_fake_st(monkeypatch)
    fake_custom.radio_overrides[events_view.KEY_EV_SECTION] = "Custom Alerts"
    events_view.render_events_tab(metrics=None, anomalies=None)
    assert fake_custom.errors == []
    assert any(
        "2d" in w.lower() and "4c" in w.lower() and "unavailable" in w.lower()
        for w in fake_custom.warnings
    )


def test_12_export_refuses_misleading_output_when_source_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "CAL_DUCKDB_PATH", str(tmp_path / "no_db.duckdb"))
    monkeypatch.setattr(settings, "CAL_ANALYTICS_ROOT", str(tmp_path / "no_analytics"))
    monkeypatch.setattr(settings, "CAL_RUNS_PATH", str(tmp_path / "no_runs.jsonl"))

    selection = {
        "date_start": "2023-08-01",
        "date_end": "2023-08-02",
        "products": ("modis", "viirs-snpp"),
        "satellites": ("Terra", "Aqua", "N"),
        "modis_conf_min": None,
        "viirs_conf": ("l", "n", "h"),
    }

    for category in export_view.CATEGORIES:
        src_st = export_view.export_source_status(category)
        assert src_st["available"] is False
        assert len(src_st["missing"]) >= 1
        with pytest.raises(query.DataUnavailableError):
            export_view.build_export(category, selection, None)

        fake = _install_fake_st(
            monkeypatch,
            {
                export_view.KEY_EX_CATEGORY: category,
                export_view.KEY_EX_PAYLOAD: {"csv_text": "stale"},
            },
        )
        export_view.render_export_tab()
        assert fake.errors == []
        assert fake.session_state[export_view.KEY_EX_PAYLOAD] is None
        assert any("export unavailable" in w.lower() for w in fake.warnings)
        assert fake.download_buttons == []
        assert fake.metrics == []

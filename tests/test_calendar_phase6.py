"""Phase 6 tests: integration labels, cross-references, reconciliation.

All checks are read-only against production data (JSON/parquet/SQLite
aggregates) or synthetic in-memory fixtures. Nothing here writes to
production stores, regenerates outputs, or touches external services.
"""

import json
import sqlite3
from pathlib import Path

import duckdb

from src.calendar.events import alerts, association

REPO = Path(__file__).resolve().parent.parent


def _event_fixture():
    return {
        "event_id": "ev-cl0000001-a1", "cluster_ids": ["cl0000001"],
        "detection_count": 500, "first_observed_utc": "2023-08-01T06:00:00+00:00",
        "last_observed_utc": "2023-08-15T06:00:00+00:00",
        "sensor_modis_n": 200, "sensor_viirs_n": 300,
    }


class TestRetrospectiveLabel:
    def test_spike_template_carries_retrospective_only(self):
        text = alerts.explain_event_alert(
            "Activity Spike", "High", _event_fixture(),
            {"n_detections": 500}, {"Moderate": 9.0, "High": 28.0,
                                    "Critical": 172.0}, 137640)
        assert "Retrospective-only classification" in text
        assert "not validated" in text and "real-time" in text

    def test_frp_template_carries_retrospective_only(self):
        text = alerts.explain_event_alert(
            "High FRP", "High", _event_fixture(), {"frp_max": 300.0},
            {"Moderate": 53.4, "High": 168.2, "Critical": 626.9}, 137640)
        assert "Retrospective-only classification" in text

    def test_anomaly_template_unchanged_scope(self):
        # 4C-derived explanations keep their trailing-reference language;
        # the retrospective sentence belongs to population bands only.
        record = {"metric": "detection", "scope": "raw",
                  "eval_period": "2023-08-15", "observed_value": 500.0,
                  "baseline_value": 100.0, "baseline_method": "sensor_aware",
                  "reference_strategy": "trailing", "reference_n": 30,
                  "classification": "notable-high", "eligibility": "eligible",
                  "z_score": 3.5, "z_status": "valid",
                  "pct_deviation": 400.0, "pct_status": "valid"}
        text = alerts.explain_anomaly_alert(record, "High")
        assert "Retrospective-only" not in text
        assert "|z| >= 2" in text


class TestClusterEventCrossReference:
    def test_map_reference_derives_event_id(self):
        assert association.event_id_for("cl0000161") == "ev-cl0000161-a1"

    def test_referenced_event_exists_in_store(self):
        conn = sqlite3.connect(f"file:{REPO / 'calendar_events.sqlite'}?mode=ro",
                               uri=True)
        try:
            row = conn.execute(
                "SELECT event_id FROM events WHERE event_id = ?",
                (association.event_id_for("cl0000161"),)).fetchone()
        finally:
            conn.close()
        assert row is not None


class TestScientificReconciliation:
    def test_raw_totals_match_catalog(self):
        catalog = json.loads((REPO / "catalog.json").read_text(
            encoding="utf-8"))
        accepted = sum(v["record_counts"]["accepted"]
                       for d in catalog["datasets"]
                       for v in d["versions"][-1:])
        conn = duckdb.connect(str(REPO / "calendar.duckdb"), read_only=True)
        try:
            live = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
            by_product = dict(conn.execute(
                "SELECT product, COUNT(*) FROM detections GROUP BY product"
            ).fetchall())
        finally:
            conn.close()
        assert live == accepted == 482672
        assert set(by_product) == {"modis", "viirs-snpp"}

    def test_cluster_totals_match_phase2(self):
        conn = duckdb.connect()
        try:
            clusters, members = conn.execute(
                "SELECT COUNT(*), SUM(n_detections) FROM read_parquet(?)",
                [str(REPO / "analytics_output/7caf584581684169820505f8b73672aa"
                     / "clusters.parquet")]).fetchone()
        finally:
            conn.close()
        assert (clusters, members) == (137640, 482672)

    def test_harmonized_never_double_counts(self):
        metrics = json.loads((REPO / "analytics_output/9741f859517c418fa15ff6b6cbe078e2"
                              / "metrics_2d.json").read_text(encoding="utf-8"))
        for grain, pack in metrics.items():
            det = pack["detection_activity"]
            raw = sum(r["raw_observations"] for r in det["rows"])
            harm = sum(r["harmonized_observations"]
                       for r in det["harmonized_rows"])
            assert harm <= raw, grain  # collapsing groups only removes dupes
            assert raw == 482672, grain  # every grain reconciles to inputs

    def test_phase4_outputs_unchanged(self):
        anomalies = json.loads((REPO / "analytics_output/461ecbc98ca14f11910e8bcda7254027"
                                / "anomalies_4c.json").read_text(
                                    encoding="utf-8"))
        assert len(anomalies) == 14700
        for row in anomalies:
            ref_end = row.get("reference_end") or ""
            assert ref_end < row["eval_period"] or ref_end == "", row["anomaly_id"]

    def test_phase5_event_counts_reconcile(self):
        conn = sqlite3.connect(f"file:{REPO / 'calendar_events.sqlite'}?mode=ro",
                               uri=True)
        try:
            n = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            total = conn.execute(
                "SELECT SUM(detection_count) FROM events").fetchone()[0]
            states = dict(conn.execute(
                "SELECT lifecycle_state, COUNT(*) FROM events GROUP BY 1"
            ).fetchall())
            alerts_n = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
            missing_frp = conn.execute(
                "SELECT COUNT(*) FROM events WHERE "
                "json_extract(frp, '$.frp_max') IS NULL").fetchone()[0]
        finally:
            conn.close()
        assert n == 137640
        assert total == 482672
        assert sum(states.values()) == 137640
        assert set(states) <= {"Active", "Monitoring", "Pending Closure",
                               "Resolved"}
        assert alerts_n == 15057
        assert missing_frp == 0  # pilot FRP complete; nulls stay null by code

    def test_sensors_distinguishable_everywhere(self):
        conn = sqlite3.connect(f"file:{REPO / 'calendar_events.sqlite'}?mode=ro",
                               uri=True)
        try:
            mixed = conn.execute(
                "SELECT COUNT(*) FROM events WHERE sensor_modis_n + sensor_viirs_n "
                "!= detection_count").fetchone()[0]
        finally:
            conn.close()
        assert mixed == 0

    def test_provenance_exec_ids_present(self):
        conn = sqlite3.connect(f"file:{REPO / 'calendar_events.sqlite'}?mode=ro",
                               uri=True)
        try:
            events_bare = conn.execute(
                "SELECT COUNT(*) FROM events WHERE "
                "json_extract(provenance, '$.created_by_exec') IS NULL OR "
                "json_extract(provenance, '$.cluster_exec') IS NULL").fetchone()[0]
            alerts_bare = conn.execute(
                "SELECT COUNT(*) FROM alerts WHERE rule_version IS NULL OR "
                "rule_version = ''").fetchone()[0]
        finally:
            conn.close()
        assert events_bare == 0
        assert alerts_bare == 0

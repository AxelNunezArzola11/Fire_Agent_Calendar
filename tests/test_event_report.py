"""Event report builder tests: pure markdown assembly (no DB, no Streamlit)."""

from src.calendar.ui import event_report


def _event():
    return {
        "event_id": "ev-cl0000001-a1",
        "lifecycle_state": "Pending Closure",
        "severity_name": "High",
        "detection_count": 500,
        "sensor_modis_n": 200,
        "sensor_viirs_n": 300,
        "first_observed_utc": "2023-08-01T06:00:00+00:00",
        "last_observed_utc": "2023-08-15T06:00:00+00:00",
        "satellites": ["Aqua", "N"],
        "cluster_ids": ["cl0000001"],
        "activity": {"n": 500},
        "frp": {"frp_sum": 5000.0, "frp_max": 300.0, "frp_n": 500,
                "frp_null": 0},
        "bbox": [-120.0, 38.0, -119.0, 39.0],
        "centroid": [-119.5, 38.5],
        "completeness": {"frac": 1.0},
        "ambiguity": {"flags": []},
        "severity_provenance": "nearest-rank bands p95/99/99.9",
        "provenance": {"assoc": "assoc_v1", "exec_5a": "270adf51"},
        "detection_resolution": {"via": "Phase 2C"},
    }


def _inputs():
    return {
        "transitions": [
            {"at_utc": "2023-09-01T00:00:00+00:00",
             "prev_state": "Active", "new_state": "Monitoring",
             "reason": "quiet 7d"},
            {"at_utc": "2023-09-20T00:00:00+00:00",
             "prev_state": "Monitoring", "new_state": "Pending Closure",
             "reason": "quiet 21d"},
        ],
        "alerts": [
            {"alert_id": "alt-asp-ev-cl0000001-a1-r1",
             "alert_type": "Activity Spike", "severity_name": "High",
             "status": "open", "period": "2023-08-15",
             "explanation": "n_detections=500 vs band High>=28"},
        ],
        "exec_chain": {"2B": "bfef579a", "2C": "7caf5845",
                       "4B": "61722b76", "4C": "461ecbc9",
                       "5A": "270adf51", "5B": "7b4bc145",
                       "5C": "e78f50c5"},
        "generated_at": "2024-01-01T00:00:00+00:00",
    }


class TestEventReport:
    def test_byte_identical_regeneration(self):
        first = event_report.event_report_markdown(_event(), **_inputs())
        second = event_report.event_report_markdown(_event(), **_inputs())
        assert first == second
        assert "2024-01-01T00:00:00+00:00" in first

    def test_sections_and_evidence(self):
        text = event_report.event_report_markdown(_event(), **_inputs())
        for token in ("ev-cl0000001-a1", "Pending Closure", "High",
                      "Observation summary", "Lifecycle history",
                      "Linked alerts", "Methods and provenance",
                      "Limitations", "How to cite",
                      "Active → Monitoring", "alt-asp-ev-cl0000001-a1-r1",
                      "bfef579a", "e78f50c5", "Sofiev", "FIRMS"):
            assert token in text, token

    def test_guardrails_present(self):
        text = event_report.event_report_markdown(_event(), **_inputs())
        for token in ("not confirmed wildfires", "not perimeters",
                      "contained or extinguished",
                      "not directly comparable"):
            assert token in text, token

    def test_empty_history_and_alerts(self):
        kw = _inputs()
        kw["transitions"] = []
        kw["alerts"] = []
        text = event_report.event_report_markdown(_event(), **kw)
        assert "No transitions yet" in text
        assert "No linked alerts" in text

    def test_missing_fields_tolerated(self):
        event = {"event_id": "ev-x", "lifecycle_state": "Active"}
        text = event_report.event_report_markdown(
            event, transitions=[], alerts=[], exec_chain={},
            generated_at="2024-01-01T00:00:00+00:00")
        assert "ev-x" in text and "unavailable" in text

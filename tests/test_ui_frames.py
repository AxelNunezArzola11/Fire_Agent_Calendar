"""UI frame schemas + Arrow compatibility (reproduces the reported bug).

The production failure: pipeline metrics mix provider names
('granite'), numbers (14.6), booleans, and nulls in one `value`
column -> pandas object dtype -> PyArrow infers double -> the
string cells raise ArrowInvalid at render time.
"""

import pandas as pd
import pyarrow as pa
import pytest

from src.calendar.ui import frames


def _metrics():
    return [
        {"metric": "total_latency_ms", "value": 15401.0, "unit": "ms"},
        {"metric": "providers", "value": "granite", "unit": "names"},
        {"metric": "ratio", "value": 14.6, "unit": "x"},
        {"metric": "missing_thing", "value": None, "unit": ""},
        {"metric": "flag_on", "value": True, "unit": ""},
        {"metric": "count", "value": 7, "unit": "rows"},
    ]


def _assert_arrow_ok(df, table):
    assert frames.check_arrow_compatible(df, table) == []
    pa.Table.from_pandas(df, preserve_index=False)  # must not raise


class TestMetricsFrame:
    def test_mixed_types_convert(self):
        df = frames.metrics_frame(_metrics())
        _assert_arrow_ok(df, "metrics")

    def test_numerics_stay_numeric(self):
        df = frames.metrics_frame(_metrics())
        assert str(df["numeric_value"].dtype) == "Float64"
        by_metric = {m: v for m, v in
                     zip(df["metric"], df["numeric_value"])}
        assert by_metric["ratio"] == 14.6
        assert by_metric["count"] == 7.0
        assert pd.isna(by_metric["providers"])
        assert pd.isna(by_metric["missing_thing"])
        assert pd.isna(by_metric["flag_on"])  # bools are not numbers

    def test_text_preserved_verbatim(self):
        df = frames.metrics_frame(_metrics())
        by_metric = {m: v for m, v in
                     zip(df["metric"], df["text_value"])}
        assert by_metric["providers"] == "granite"
        assert by_metric["ratio"] == "14.6"
        assert by_metric["missing_thing"] is pd.NA

    def test_raw_dicts_would_fail(self):
        # Proves the test reproduces the production failure mode.
        raw = pd.DataFrame([{"metric": m["metric"], "value": m["value"],
                             "unit": m["unit"]} for m in _metrics()])
        with pytest.raises(pa.ArrowInvalid):
            pa.Table.from_pandas(raw, preserve_index=False)
        diagnostics = frames.check_arrow_compatible(raw, "metrics-raw")
        assert diagnostics
        assert any("metrics-raw.value" in line for line in diagnostics)


class TestAgentRunsFrame:
    def _runs(self):
        return [{"agent": "detection-intelligence", "target_id": "c1",
                 "status": "fallback", "provider": "granite",
                 "model": "ibm/granite-4-h-small", "latency_ms": 29293.9,
                 "retries": 1, "input_tokens": 44,
                 "generated_tokens": None,
                 "fallback_reason": "empty response",
                 "validation": "no provider usable"},
                {"agent": "event-analysis", "target_id": "e1",
                 "status": "ok", "provider": "", "model": "",
                 "latency_ms": 12.0, "retries": 0, "input_tokens": None,
                 "generated_tokens": None, "fallback_reason": "",
                 "validation": "all checks passed"}]

    def test_mixed_telemetry_converts(self):
        df = frames.agent_runs_frame(self._runs())
        _assert_arrow_ok(df, "agent-runs")

    def test_telemetry_types(self):
        df = frames.agent_runs_frame(self._runs())
        assert str(df["latency_ms"].dtype) == "Float64"
        assert str(df["retries"].dtype) == "Int64"
        assert str(df["tokens_in"].dtype) == "Int64"
        assert df["tokens_out"].isna().all()  # nulls stay null
        assert df["provider"].tolist() == ["granite", "—"]


class TestStagesAndEventsFrames:
    def test_stages_convert(self):
        import types

        stages = [types.SimpleNamespace(
            name="acquisition", status="ok", latency_ms=190.1,
            detail="4,783 detections")]
        _assert_arrow_ok(frames.stages_frame(stages), "stages")

    def test_events_convert_with_nulls(self):
        events = [{"event_id": "ev-1", "lifecycle_state": "Active",
                   "severity_name": "High",
                   "first_observed_utc": "2023-08-15T06:00:00+00:00",
                   "last_observed_utc": None, "detection_count": 5,
                   "sensor_modis_n": 2, "sensor_viirs_n": None}]
        df = frames.events_frame(events)
        _assert_arrow_ok(df, "events")
        assert df["VIIRS"].isna().all()


class TestEventDetailFrames:
    def test_lifecycle_convert(self):
        history = [{"at_utc": "2023-08-15T06:00:00+00:00",
                    "prev_state": "Active", "new_state": "Pending Closure",
                    "reason": "quiet-period"},
                   {"at_utc": None, "prev_state": None,
                    "new_state": "Resolved", "reason": None}]
        df = frames.lifecycle_frame(history)
        _assert_arrow_ok(df, "lifecycle")
        assert df["at"].tolist()[0] == "2023-08-15"

    def test_reviews_convert_with_nulls(self):
        reviews = [{"at_utc": "2023-08-16T00:00:00+00:00",
                    "reviewer_label": "duty-analyst",
                    "decision": "approved", "note": None}]
        df = frames.reviews_frame(reviews)
        _assert_arrow_ok(df, "reviews")
        assert df["note"].isna().all()

    def test_alert_lists_convert(self):
        alerts = [{"alert_id": "alt-1", "alert_type": "Activity Spike",
                   "severity_name": "High", "period": "2023-08-15",
                   "event_id": "ev-1", "status": "open"},
                  {"alert_id": "alt-2", "alert_type": "High FRP",
                   "severity_name": "Moderate", "period": None,
                   "event_id": None, "status": "acknowledged"}]
        listed = frames.alerts_list_frame(alerts)
        _assert_arrow_ok(listed, "alerts-list")
        assert listed["event"].tolist()[1] == "pilot-wide"
        linked = frames.linked_alerts_frame(alerts)
        _assert_arrow_ok(linked, "alerts-linked")

    def test_alert_history_convert(self):
        history = [{"at_utc": "2023-08-15T00:00:00+00:00",
                    "action": "acknowledged"},
                   {"at_utc": None, "action": None}]
        _assert_arrow_ok(frames.alert_history_frame(history),
                         "alert-history")

    def test_notifications_convert(self):
        rows = [{"created_at": "2023-08-15T06:00:00+00:00",
                 "alert_id": "alt-1", "kind": "spike",
                 "status": "sent", "attempts": 2},
                {"created_at": None, "alert_id": None,
                 "kind": None, "status": "skipped",
                 "attempts": None}]
        df = frames.notifications_frame(rows)
        _assert_arrow_ok(df, "notifications")
        assert str(df["attempts"].dtype) == "Int64"
        assert df["attempts"].isna().tolist()[1]


class TestCustomFrames:
    def test_rules_convert_with_bools(self):
        rules = [{"rule_id": "cr-1", "name": "r1",
                  "severity_name": "High", "enabled": True,
                  "notify": False},
                 {"rule_id": "cr-2", "name": None,
                  "severity_name": "Moderate", "enabled": 1,
                  "notify": 0}]
        df = frames.custom_rules_frame(rules, ["cond-a", "cond-b"])
        _assert_arrow_ok(df, "custom-rules")
        assert str(df["enabled"].dtype) == "boolean"
        assert df["enabled"].tolist() == [True, True]
        assert df["notify"].tolist() == [False, False]

    def test_evals_convert(self):
        evals = [{"at_utc": "2023-08-15T06:00:00+00:00",
                  "triggered": True, "reason": "over threshold"},
                 {"at_utc": None, "triggered": False,
                  "reason": None}]
        df = frames.custom_evals_frame(evals)
        _assert_arrow_ok(df, "custom-evals")
        assert df["triggered"].tolist() == [True, False]

    def test_hits_convert(self):
        hits = [{"hit_id": "crh-1", "rule_name": "r1",
                 "period": "2023-08-15", "severity_name": "High",
                 "status": "open", "summary": "triggered"},
                {"hit_id": None, "rule_name": None,
                 "period": None, "severity_name": None,
                 "status": None, "summary": None}]
        _assert_arrow_ok(frames.custom_hits_frame(hits),
                         "custom-hits")


class TestAboutSourcesFrame:
    def test_convert(self):
        rows = [{"Feature": "Calendar metrics",
                 "Source": "metrics_2d.json", "Execution": "9741f859"},
                {"Feature": None, "Source": None,
                 "Execution": None}]
        df = frames.about_sources_frame(rows)
        _assert_arrow_ok(df, "about-sources")
        assert df["Feature"].tolist()[0] == "Calendar metrics"


class TestNotablesFrame:
    def test_mixed_observed_and_unavailable(self):
        def _fmt(value):
            return "unavailable" if value is None else str(value)

        rows = [{"eval_period": "2023-08-15",
                 "observed_value": 500, "baseline_value": 100.5,
                 "pct_deviation": None, "z_score": 3.2,
                 "classification": "notable-high"},
                {"eval_period": "2023-08-16",
                 "observed_value": 12.5, "baseline_value": None,
                 "pct_deviation": 400.0, "z_score": None,
                 "classification": "provisional-notable-low"}]
        df = frames.notables_frame(rows, _fmt)
        _assert_arrow_ok(df, "notables")
        assert str(df["observed"].dtype) == "Float64"
        assert df["observed"].tolist() == [500.0, 12.5]
        assert df["baseline"].tolist()[1] == "unavailable"

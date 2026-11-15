"""Phase 5 tests: events, lifecycle, alerts, notifications, Qwen, UI.

Conventions follow tests/test_calendar_phase4.py: synthetic fixtures only
(isolated unit tests may use synthetic data; production never does),
deterministic assertions, sqlite DBs in tmp_path. Real-data integration is
covered by the scripts/cal_events_5.py runs recorded in the stage reports.
"""

import json
import sqlite3

import pytest

from src.calendar.events import (alerts, association, lifecycle, model,
                                 notify, qwen, store)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _cluster(cluster_id="cl0000001", n=5, modis=2, viirs=3,
             first="2023-08-15T06:00:00+00:00",
             last="2023-08-17T06:00:00+00:00", span_h=48.0,
             match_groups='["mg0000001"]', completeness_frac=1.0,
             frp_sum=120.5, frp_mean=24.1, frp_max=60.0, frp_n=5,
             frp_null=0, satellites='["Aqua", "N"]'):
    return {
        "cluster_id": cluster_id, "n_detections": n, "n_modis": modis,
        "n_viirs": viirs, "satellites": satellites,
        "first_observation_utc": first, "last_observation_utc": last,
        "observation_span_h": span_h, "lon_min": -120.0, "lon_max": -119.0,
        "lat_min": 38.0, "lat_max": 39.0, "centroid_lon": -119.5,
        "centroid_lat": 38.5, "frp_sum": frp_sum, "frp_mean": frp_mean,
        "frp_max": frp_max, "frp_n": frp_n, "frp_null": frp_null,
        "frp_by_sensor": '{"MODIS": {"n": 2}, "VIIRS": {"n": 3}}',
        "match_groups": match_groups, "completeness_dates": 3,
        "completeness_span_days": 3, "completeness_frac": completeness_frac,
        "algo_version": "cluster_v1", "params_version": "cluster_params_v1",
        "exec_id": "exec2c",
    }


@pytest.fixture()
def memdb():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    store.init_schema(conn)
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# 5A.1 Event identity
# ---------------------------------------------------------------------------

class TestEventIdentity:
    def test_stable_deterministic_id(self):
        assert association.event_id_for("cl0000161") == "ev-cl0000161-a1"
        assert (association.event_id_for("cl0000161")
                == association.event_id_for("cl0000161"))

    def test_id_embeds_source_cluster(self):
        event_id = association.event_id_for("cl0000042")
        assert "cl0000042" in event_id  # lineage visible in the ID

    def test_invalid_cluster_id_rejected(self):
        for bad in ("", None, 123):
            with pytest.raises(ValueError):
                association.event_id_for(bad)

    def test_distinct_clusters_distinct_events(self):
        assert (association.event_id_for("cl0000001")
                != association.event_id_for("cl0000002"))

    def test_rebuild_keeps_ids(self, memdb):
        first = association.build_event_record(_cluster(), exec_id="exec-a")
        second = association.build_event_record(_cluster(), exec_id="exec-b")
        assert first["event_id"] == second["event_id"]


# ---------------------------------------------------------------------------
# 5A.2 Association methodology
# ---------------------------------------------------------------------------

class TestAssociation:
    def test_one_event_per_cluster_no_merging(self):
        events = association.seed_events(
            [_cluster("cl0000001"), _cluster("cl0000002")], exec_id="e1")
        assert [e["event_id"] for e in events] == ["ev-cl0000001-a1",
                                                  "ev-cl0000002-a1"]
        assert events[0]["cluster_ids"] == ["cl0000001"]
        assert events[1]["cluster_ids"] == ["cl0000002"]

    def test_duplicate_cluster_refused(self):
        with pytest.raises(ValueError, match="duplicate cluster_id"):
            association.seed_events([_cluster("cl0000001"),
                                     _cluster("cl0000001")], exec_id="e1")

    def test_missing_lineage_rejected(self):
        bad = _cluster()
        del bad["cluster_id"]
        with pytest.raises(ValueError, match="lineage"):
            association.build_event_record(bad, exec_id="e1")

    def test_missing_timestamps_rejected(self):
        bad = _cluster()
        bad["last_observation_utc"] = None
        with pytest.raises(ValueError, match="timestamps"):
            association.build_event_record(bad, exec_id="e1")

    def test_singleton_flagged_never_dropped(self):
        event = association.build_event_record(
            _cluster("cl0000009", n=1, modis=1, viirs=0, match_groups="[]",
                     span_h=0.0, satellites='["Terra"]'), exec_id="e1")
        assert "singleton-observation" in event["ambiguity"]["flags"]
        assert "single-sensor-only" in event["ambiguity"]["flags"]
        assert event["detection_count"] == 1

    def test_no_cross_sensor_match_flagged(self):
        event = association.build_event_record(
            _cluster(match_groups="[]"), exec_id="e1")
        assert "no-cross-sensor-match" in event["ambiguity"]["flags"]

    def test_gappy_span_flagged(self):
        event = association.build_event_record(
            _cluster(completeness_frac=0.5), exec_id="e1")
        assert "gappy-observation-span" in event["ambiguity"]["flags"]

    def test_very_large_group_reuses_map_constant(self):
        from src.calendar.ui.map_view import LARGE_CLUSTER_WARN

        big = association.build_event_record(
            _cluster(n=LARGE_CLUSTER_WARN + 1, modis=50,
                     viirs=LARGE_CLUSTER_WARN - 49), exec_id="e1")
        assert "very-large-group" in big["ambiguity"]["flags"]
        small = association.build_event_record(
            _cluster(n=LARGE_CLUSTER_WARN, modis=50,
                     viirs=LARGE_CLUSTER_WARN - 50), exec_id="e1")
        assert "very-large-group" not in small["ambiguity"]["flags"]

    def test_same_overpass_group_flagged(self):
        event = association.build_event_record(_cluster(span_h=0.0),
                                               exec_id="e1")
        assert "same-overpass-group" in event["ambiguity"]["flags"]

    def test_clean_multi_sensor_group_unflagged(self):
        event = association.build_event_record(_cluster(), exec_id="e1")
        assert event["ambiguity"]["flags"] == []

    def test_sensor_identity_preserved(self):
        event = association.build_event_record(_cluster(), exec_id="e1")
        assert event["sensor_modis_n"] == 2
        assert event["sensor_viirs_n"] == 3
        assert event["satellites"] == ["Aqua", "N"]

    def test_frp_aggregation_semantics_explicit(self):
        event = association.build_event_record(_cluster(), exec_id="e1")
        assert "NOT total emitted energy" in event["frp"]["aggregation"]
        assert event["frp"]["unit"] == "MW"
        assert event["frp"]["by_sensor"]["MODIS"]["n"] == 2

    def test_missing_frp_preserved_as_missing(self):
        event = association.build_event_record(
            _cluster(frp_mean=None, frp_max=None, frp_n=0, frp_null=5,
                     frp_sum=0.0), exec_id="e1")
        assert event["frp"]["frp_mean"] is None
        assert event["frp"]["frp_max"] is None
        assert event["frp"]["frp_n"] == 0
        assert event["frp"]["frp_null"] == 5

    def test_detection_resolution_pointer(self):
        event = association.build_event_record(_cluster(), exec_id="e1")
        assert event["detection_refs"] is None  # resolved on demand
        assert event["detection_resolution"]["cluster_id"] == "cl0000001"
        assert event["detection_resolution"]["exec_2c"] == "exec2c"

    def test_seed_state_and_severity_defaults(self):
        event = association.build_event_record(_cluster(), exec_id="e1")
        assert event["lifecycle_state"] == "Active"
        assert event["severity_name"] == "Informational"
        assert event["severity_level"] == 0
        assert event["last_evaluated_utc"] is None

    def test_provenance_versions(self):
        event = association.build_event_record(_cluster(), exec_id="exec-5a")
        prov = event["provenance"]
        assert prov["created_by_exec"] == "exec-5a"
        assert prov["assoc"] == model.ASSOC_VERSION
        assert prov["cluster_algo"] == "cluster_v1"
        assert prov["cluster_params"] == "cluster_params_v1"
        assert event["model_version"] == model.MODEL_VERSION
        assert event["assoc_version"] == model.ASSOC_VERSION


# ---------------------------------------------------------------------------
# 5A.3 Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_schema_idempotent(self, memdb):
        store.init_schema(memdb)  # second init is a no-op
        assert store.count_events(memdb) == 0

    def test_upsert_round_trip(self, memdb):
        record = association.build_event_record(_cluster(), exec_id="e1")
        store.upsert_event(memdb, record)
        back = store.get_event(memdb, record["event_id"])
        assert back["event_id"] == record["event_id"]
        assert back["cluster_ids"] == ["cl0000001"]
        assert back["frp"]["frp_max"] == 60.0
        assert back["ambiguity"]["flags"] == []

    def test_upsert_idempotent(self, memdb):
        record = association.build_event_record(_cluster(), exec_id="e1")
        store.upsert_event(memdb, record)
        store.upsert_event(memdb, record)
        assert store.count_events(memdb) == 1

    def test_batch_upsert(self, memdb):
        records = association.seed_events(
            [_cluster(f"cl000000{i}") for i in range(1, 6)], exec_id="e1")
        assert store.upsert_events(memdb, records) == 5
        assert store.count_events(memdb) == 5
        assert store.upsert_events(memdb, records) == 5  # re-run: same rows
        assert store.count_events(memdb) == 5

    def test_unknown_event_returns_none(self, memdb):
        assert store.get_event(memdb, "ev-nope-a1") is None

    def test_invalid_state_rejected(self, memdb):
        record = association.build_event_record(_cluster(), exec_id="e1")
        record["lifecycle_state"] = "Burning"
        with pytest.raises(ValueError, match="unknown lifecycle state"):
            store.upsert_event(memdb, record)

    def test_transition_requires_current_state(self, memdb):
        record = association.build_event_record(_cluster(), exec_id="e1")
        store.upsert_event(memdb, record)
        with pytest.raises(ValueError, match="stale transition"):
            store.record_transition(
                memdb, event_id=record["event_id"], prev_state="Monitoring",
                new_state="Resolved", reason="x", evidence={}, completeness={},
                algo_version="lifecycle_v1", config_version="lifecycle_cfg_v1",
                exec_id="e2")
        # No partial write: state unchanged, no audit row.
        assert store.get_event(memdb, record["event_id"])["lifecycle_state"] \
            == "Active"
        assert store.list_transitions(memdb, record["event_id"]) == []

    def test_transition_atomic_with_audit(self, memdb):
        record = association.build_event_record(_cluster(), exec_id="e1")
        store.upsert_event(memdb, record)
        trans_id = store.record_transition(
            memdb, event_id=record["event_id"], prev_state="Active",
            new_state="Monitoring", reason="quiet-7d", evidence={"quiet_d": 9},
            completeness={"coverage_status": "complete"},
            algo_version="lifecycle_v1", config_version="lifecycle_cfg_v1",
            exec_id="e2")
        assert trans_id == 1
        assert store.get_event(memdb, record["event_id"])["lifecycle_state"] \
            == "Monitoring"
        history = store.list_transitions(memdb, record["event_id"])
        assert len(history) == 1
        assert history[0]["reason"] == "quiet-7d"
        assert history[0]["evidence"] == {"quiet_d": 9}

    def test_recovery_rerun_safe(self, memdb):
        # An interrupted-then-rerun seed converges to identical state.
        half = association.seed_events([_cluster("cl0000001")], exec_id="e1")
        store.upsert_events(memdb, half)
        full = association.seed_events(
            [_cluster("cl0000001"), _cluster("cl0000002")], exec_id="e1")
        store.upsert_events(memdb, full)
        assert store.count_events(memdb) == 2
        assert store.get_event(memdb, "ev-cl0000001-a1")["cluster_ids"] == \
            ["cl0000001"]

    def test_list_events_bounded_and_ordered(self, memdb):
        records = association.seed_events(
            [_cluster(f"cl000000{i}") for i in range(1, 4)], exec_id="e1")
        store.upsert_events(memdb, records)
        store.update_event_fields(memdb, "ev-cl0000002-a1",
                                  {"severity_name": "High"})
        listed = store.list_events(memdb, limit=10)
        assert [e["event_id"] for e in listed][0] == "ev-cl0000002-a1"
        with pytest.raises(ValueError):
            store.list_events(memdb, limit=0)
        with pytest.raises(ValueError):
            store.list_events(memdb, limit=5001)

    def test_event_id_immutable(self, memdb):
        record = association.build_event_record(_cluster(), exec_id="e1")
        store.upsert_event(memdb, record)
        with pytest.raises(ValueError, match="immutable"):
            store.update_event_fields(memdb, record["event_id"],
                                      {"event_id": "ev-other-a1"})

    def test_meta_round_trip(self, memdb):
        store.meta_set(memdb, "exec_5a", "abc123")
        assert store.meta_get(memdb, "exec_5a") == "abc123"
        assert store.meta_get(memdb, "missing") is None

    def test_batch_transaction_joins(self, memdb):
        # Helpers join an ambient transaction: one commit for the group,
        # and a rollback discards the whole group (crash recovery).
        first = association.build_event_record(_cluster("cl0000001"),
                                               exec_id="e1")
        second = association.build_event_record(_cluster("cl0000002"),
                                                exec_id="e1")
        with store.batch(memdb):
            store.upsert_event(memdb, first)
            store.upsert_event(memdb, second)
        assert store.count_events(memdb) == 2
        # A failed batch rolls back as a unit; the DB keeps converging.
        with pytest.raises(RuntimeError):
            with store.batch(memdb):
                store.upsert_event(
                    memdb,
                    association.build_event_record(_cluster("cl0000003"),
                                                   exec_id="e1"))
                raise RuntimeError("simulated crash")
        assert store.count_events(memdb) == 2

    def test_list_events_id_order_stable(self, memdb):
        records = association.seed_events(
            [_cluster("cl0000003"), _cluster("cl0000001"),
             _cluster("cl0000002")], exec_id="e1")
        store.upsert_events(memdb, records)
        ordered = store.list_events(memdb, limit=10, order="id")
        assert [e["event_id"] for e in ordered] == ["ev-cl0000001-a1",
                                                   "ev-cl0000002-a1",
                                                   "ev-cl0000003-a1"]
        with pytest.raises(ValueError, match="order"):
            store.list_events(memdb, limit=10, order="bogus")

    def test_file_db_uses_wal_and_timeout(self, tmp_path):
        conn = store.connect(str(tmp_path / "wal.sqlite"))
        try:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            assert mode.lower() == "wal"
            assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 60000
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# 5A.4 Model vocabulary
# ---------------------------------------------------------------------------

class TestModel:
    def test_severity_scale_exact(self):
        assert model.SEVERITY_LEVELS == {"Informational": 0, "Low": 1,
                                         "Moderate": 2, "High": 3,
                                         "Critical": 4}
        assert "fire danger rating" in model.SEVERITY_SCALE_NOTE.lower() \
            or "Not a fire danger" in model.SEVERITY_SCALE_NOTE

    def test_lifecycle_states_exact(self):
        assert list(model.LIFECYCLE_STATES) == ["Active", "Monitoring",
                                                "Pending Closure",
                                                "Resolved"]
        assert "extinguish" in model.RESOLVED_DISCLAIMER

    def test_alert_types_exact(self):
        assert list(model.ALERT_TYPES) == ["Activity Spike",
                                           "Cluster Anomaly", "High FRP",
                                           "Historical Anomaly",
                                           "Data Quality"]

    def test_validators_reject_unknown(self):
        with pytest.raises(ValueError):
            model.check_severity("Extreme")
        with pytest.raises(ValueError):
            model.check_state("Burning")
        with pytest.raises(ValueError):
            model.check_alert_type("Evacuation")


# ---------------------------------------------------------------------------
# 5B.1 Lifecycle rules
# ---------------------------------------------------------------------------

CTX_OK = {"as_of": "2023-12-31", "coverage_max": "2023-12-31",
          "coverage_complete": True}
CTX_STALE = {"as_of": "2024-02-15", "coverage_max": "2023-12-31",
             "coverage_complete": False}


def _event(last="2023-12-30T06:00:00+00:00", state="Active",
           severity="Informational", evaluated=None, review="none",
           first="2023-12-29T06:00:00+00:00"):
    record = association.build_event_record(
        _cluster(first=first, last=last), exec_id="e1")
    record["lifecycle_state"] = state
    record["severity_name"] = severity
    record["severity_level"] = model.SEVERITY_LEVELS[severity]
    record["last_evaluated_utc"] = evaluated
    record["review_status"] = review
    return record


class TestLifecycleRules:
    def test_targets_by_quiet_days(self):
        cfg = lifecycle.LifecycleConfig()
        assert cfg.target_state(0) == "Active"
        assert cfg.target_state(6) == "Active"
        assert cfg.target_state(7) == "Monitoring"
        assert cfg.target_state(20) == "Monitoring"
        assert cfg.target_state(21) == "Pending Closure"
        assert cfg.target_state(44) == "Pending Closure"
        assert cfg.target_state(45) == "Resolved"
        assert cfg.target_state(400) == "Resolved"

    def test_active_sustained_when_recent(self):
        decision = lifecycle.evaluate(_event(), ctx=dict(CTX_OK))
        assert (decision["prev_state"], decision["new_state"]) == \
            ("Active", "Active")
        assert decision["changed"] is False
        assert decision["reason"] == "state-sustained"

    def test_closure_advances_one_step(self):
        decision = lifecycle.evaluate(
            _event(last="2023-01-01T06:00:00+00:00"), ctx=dict(CTX_OK))
        assert (decision["prev_state"], decision["new_state"]) == \
            ("Active", "Monitoring")
        assert decision["evidence"]["target_state"] == "Resolved"
        assert decision["evidence"]["quiet_days"] == 364

    def test_monitoring_to_pending_single_step(self):
        decision = lifecycle.evaluate(
            _event(last="2023-06-01T06:00:00+00:00", state="Monitoring"),
            ctx=dict(CTX_OK))
        assert decision["new_state"] == "Pending Closure"

    def test_pending_to_resolved(self):
        decision = lifecycle.evaluate(
            _event(last="2023-06-01T06:00:00+00:00",
                   state="Pending Closure"), ctx=dict(CTX_OK))
        assert decision["new_state"] == "Resolved"
        assert decision["reason"] == "quiet-period-elapsed"

    def test_activity_side_applies_immediately(self):
        decision = lifecycle.evaluate(
            _event(state="Monitoring"), ctx=dict(CTX_OK))
        assert decision["new_state"] == "Active"
        assert decision["reason"] == "evidence-supports-alert-state"

    def test_as_of_precedes_observation_holds(self):
        decision = lifecycle.evaluate(
            _event(state="Monitoring"),
            ctx={"as_of": "2023-01-01", "coverage_max": "2023-12-31",
                 "coverage_complete": True})
        assert decision["new_state"] == "Active"
        assert decision["reason"] == "as-of-precedes-observation"

    def test_decision_carries_versions(self):
        decision = lifecycle.evaluate(_event(), ctx=dict(CTX_OK))
        assert decision["algo_version"] == model.LIFECYCLE_VERSION
        assert decision["config_version"] == model.LIFECYCLE_CFG_VERSION
        assert decision["completeness"]["coverage_status"] == "complete"


# ---------------------------------------------------------------------------
# 5B.2 Missing data can never close
# ---------------------------------------------------------------------------

class TestMissingData:
    def test_stale_active_steps_to_monitoring_hold(self):
        decision = lifecycle.evaluate(_event(), ctx=dict(CTX_STALE))
        assert decision["new_state"] == "Monitoring"
        assert decision["reason"] == "data-hold-stale-coverage"
        assert decision["evidence"]["held_closure"] is True

    def test_stale_monitoring_preserved(self):
        decision = lifecycle.evaluate(_event(state="Monitoring"),
                                      ctx=dict(CTX_STALE))
        assert decision["new_state"] == "Monitoring"
        assert decision["reason"] == "insufficient-evidence-hold"

    def test_stale_pending_never_resolves(self):
        decision = lifecycle.evaluate(
            _event(last="2023-01-01T06:00:00+00:00",
                   state="Pending Closure"), ctx=dict(CTX_STALE))
        assert decision["new_state"] == "Pending Closure"
        assert decision["evidence"]["held_closure"] is True

    def test_stale_resolved_stays_resolved(self):
        decision = lifecycle.evaluate(
            _event(state="Resolved",
                   evaluated="2023-12-31T23:59:59+00:00"),
            ctx=dict(CTX_STALE))
        assert decision["new_state"] == "Resolved"

    def test_incomplete_coverage_holds_despite_quiet(self):
        ctx = {"as_of": "2023-12-31", "coverage_max": "2023-12-31",
               "coverage_complete": False}
        decision = lifecycle.evaluate(
            _event(last="2023-01-01T06:00:00+00:00"), ctx=ctx)
        assert decision["new_state"] == "Monitoring"
        assert decision["reason"] == "data-hold-stale-coverage"


# ---------------------------------------------------------------------------
# 5B.3 Review gate for High/Critical
# ---------------------------------------------------------------------------

class TestReviewGate:
    def test_high_holds_pending_without_review(self):
        decision = lifecycle.evaluate(
            _event(last="2023-01-01T06:00:00+00:00",
                   state="Pending Closure", severity="High"),
            ctx=dict(CTX_OK))
        assert decision["new_state"] == "Pending Closure"
        assert decision["reason"] == "review-required-hold"
        assert decision["review_required"] is True

    def test_critical_holds_pending_without_review(self):
        decision = lifecycle.evaluate(
            _event(last="2023-01-01T06:00:00+00:00",
                   state="Pending Closure", severity="Critical"),
            ctx=dict(CTX_OK))
        assert decision["review_required"] is True
        assert decision["new_state"] == "Pending Closure"

    def test_approved_review_allows_resolution(self):
        decision = lifecycle.evaluate(
            _event(last="2023-01-01T06:00:00+00:00",
                   state="Pending Closure", severity="High",
                   review="approved"), ctx=dict(CTX_OK))
        assert decision["new_state"] == "Resolved"
        assert decision["review_performed"] is True

    def test_rejected_review_still_holds(self):
        decision = lifecycle.evaluate(
            _event(last="2023-01-01T06:00:00+00:00",
                   state="Pending Closure", severity="High",
                   review="rejected"), ctx=dict(CTX_OK))
        assert decision["new_state"] == "Pending Closure"
        assert decision["review_required"] is True

    def test_moderate_resolves_without_review(self):
        decision = lifecycle.evaluate(
            _event(last="2023-01-01T06:00:00+00:00",
                   state="Pending Closure", severity="Moderate"),
            ctx=dict(CTX_OK))
        assert decision["new_state"] == "Resolved"

    def test_review_recorded_auditable(self, memdb):
        record = _event(severity="High")
        store.upsert_event(memdb, record)
        review_id = store.record_review(
            memdb, event_id=record["event_id"], reviewer_label="analyst-1",
            decision="approved", note="imagery checked")
        assert review_id == 1
        reviews = store.list_reviews(memdb, record["event_id"])
        assert reviews[0]["reviewer_label"] == "analyst-1"
        assert store.get_event(memdb, record["event_id"])["review_status"] \
            == "approved"
        with pytest.raises(ValueError, match="decision"):
            store.record_review(memdb, event_id=record["event_id"],
                                reviewer_label="x", decision="maybe")


# ---------------------------------------------------------------------------
# 5B.4 Reopening on new activity
# ---------------------------------------------------------------------------

class TestReopening:
    def test_resolved_reopens_on_new_activity(self):
        decision = lifecycle.evaluate(
            _event(state="Resolved",
                   evaluated="2023-06-01T23:59:59+00:00",
                   last="2023-12-30T06:00:00+00:00"), ctx=dict(CTX_OK))
        assert decision["new_state"] == "Active"
        assert decision["reason"] == "new-qualifying-activity"
        assert decision["evidence"]["reopened"] is True

    def test_pending_reactivates_on_new_activity(self):
        decision = lifecycle.evaluate(
            _event(state="Pending Closure",
                   evaluated="2023-06-01T23:59:59+00:00",
                   last="2023-12-30T06:00:00+00:00"), ctx=dict(CTX_OK))
        assert decision["new_state"] == "Active"
        assert decision["evidence"]["reopened"] is False

    def test_no_reopen_without_new_observation(self):
        decision = lifecycle.evaluate(
            _event(state="Resolved",
                   evaluated="2023-12-31T23:59:59+00:00",
                   last="2023-01-01T06:00:00+00:00",
                   first="2023-01-01T06:00:00+00:00"), ctx=dict(CTX_OK))
        assert decision["new_state"] == "Resolved"
        assert decision["reason"] == "state-sustained"

    def test_first_evaluation_is_not_new_activity(self):
        # last_evaluated None: initial assessment, never a reopen.
        decision = lifecycle.evaluate(
            _event(state="Active", evaluated=None), ctx=dict(CTX_OK))
        assert decision["reason"] == "state-sustained"


# ---------------------------------------------------------------------------
# 5B.5 Audit + batch convergence + reseed safety
# ---------------------------------------------------------------------------

class TestLifecycleAudit:
    def test_unchanged_decision_stamps_evaluation(self, memdb):
        record = _event()
        store.upsert_event(memdb, record)
        decision = lifecycle.evaluate(record, ctx=dict(CTX_OK))
        assert lifecycle.apply_decision(memdb, decision, "exec-5b") is False
        back = store.get_event(memdb, record["event_id"])
        assert back["last_evaluated_utc"] == "2023-12-31T23:59:59+00:00"
        assert store.list_transitions(memdb, record["event_id"]) == []

    def test_changed_decision_audited(self, memdb):
        record = _event(last="2023-01-01T06:00:00+00:00")
        store.upsert_event(memdb, record)
        decision = lifecycle.evaluate(record, ctx=dict(CTX_OK))
        assert lifecycle.apply_decision(memdb, decision, "exec-5b",
                                        at_utc="2023-12-31T23:59:59+00:00") \
            is True
        history = store.list_transitions(memdb, record["event_id"])
        assert len(history) == 1
        row = history[0]
        assert (row["prev_state"], row["new_state"]) == ("Active",
                                                         "Monitoring")
        assert row["reason"] == "quiet-period-elapsed"
        assert row["evidence"]["quiet_days"] == 364
        assert row["algo_version"] == "lifecycle_v1"
        assert row["config_version"] == "lifecycle_cfg_v1"
        assert row["exec_id"] == "exec-5b"

    def test_batch_converges_and_rerun_quiet(self, tmp_path):
        db = str(tmp_path / "lc.sqlite")
        conn = store.connect(db)
        store.init_schema(conn)
        old = association.build_event_record(
            _cluster("cl0000001", first="2023-01-01T06:00:00+00:00",
                     last="2023-01-02T06:00:00+00:00"), exec_id="e1")
        new = association.build_event_record(
            _cluster("cl0000002", first="2023-12-30T06:00:00+00:00",
                     last="2023-12-31T06:00:00+00:00"), exec_id="e1")
        store.upsert_events(conn, [old, new])
        conn.close()
        runs_path = str(tmp_path / "runs.jsonl")
        first = lifecycle.run_lifecycle(db, "2023-12-31", runs_path)
        assert first["states"] == {"Active": 1, "Resolved": 1}
        assert first["transitions"] == 3  # full ladder walked with audit
        conn = store.connect(db)
        ladder = [t["new_state"] for t in
                  store.list_transitions(conn, "ev-cl0000001-a1")]
        assert ladder == ["Monitoring", "Pending Closure", "Resolved"]
        conn.close()
        second = lifecycle.run_lifecycle(db, "2023-12-31", runs_path)
        assert second["transitions"] == 0  # quiescent: rerun is a no-op

    def test_reseed_preserves_lifecycle(self):
        managed = _event(state="Monitoring", severity="High",
                         evaluated="2023-12-31T23:59:59+00:00",
                         review="approved")
        managed["anomaly_refs"] = ["anom-x"]
        fresh = association.build_event_record(_cluster(), exec_id="e2")
        merged = association.merge_reseed(managed, fresh, exec_id="e2")
        for field in association.RESEED_PRESERVED:
            assert merged[field] == managed[field], field
        assert merged["provenance"]["created_by_exec"] == "e1"
        assert merged["provenance"]["updated_by_exec"] == "e2"
        with pytest.raises(ValueError, match="identical event_id"):
            association.merge_reseed(
                managed,
                association.build_event_record(_cluster("cl0000002"),
                                               exec_id="e2"), exec_id="e2")


# ---------------------------------------------------------------------------
# 5C.1 Distribution bands
# ---------------------------------------------------------------------------

class TestBands:
    def test_nearest_rank_exact(self):
        values = [float(i) for i in range(1, 101)]
        assert alerts.nearest_rank_threshold(values, 95.0) == 95.0
        assert alerts.nearest_rank_threshold(values, 99.0) == 99.0
        assert alerts.nearest_rank_threshold(values, 50.0) == 50.0

    def test_nearest_rank_small_population(self):
        assert alerts.nearest_rank_threshold([1.0, 2.0, 3.0, 4.0],
                                             95.0) == 4.0

    def test_nearest_rank_rejects_empty_and_bad_pct(self):
        with pytest.raises(ValueError, match="empty population"):
            alerts.nearest_rank_threshold([], 95.0)
        for bad in (0.0, 100.0, -5.0):
            with pytest.raises(ValueError, match="outside"):
                alerts.nearest_rank_threshold([1.0], bad)

    def test_band_boundaries(self):
        bands = {"Critical": 100.0, "High": 50.0, "Moderate": 10.0}
        assert alerts.band_for(100.0, bands) == "Critical"
        assert alerts.band_for(99.9, bands) == "High"
        assert alerts.band_for(50.0, bands) == "High"
        assert alerts.band_for(10.0, bands) == "Moderate"
        assert alerts.band_for(9.99, bands) is None

    def test_alert_id_stable_and_typed(self):
        first = alerts.alert_id_for("High FRP", "ev-cl0000001-a1")
        assert first == alerts.alert_id_for("High FRP", "ev-cl0000001-a1")
        assert first.startswith("alt-frp-")
        assert alerts.alert_id_for("Activity Spike", "x").startswith("alt-asp-")
        with pytest.raises(ValueError):
            alerts.alert_id_for("Evacuation", "x")


# ---------------------------------------------------------------------------
# 5C.2 Per-event alerts
# ---------------------------------------------------------------------------

BANDS = {"size": {"Critical": 1000.0, "High": 100.0, "Moderate": 10.0},
         "frp": {"Critical": 500.0, "High": 200.0, "Moderate": 50.0},
         "n_events": 100, "n_frp": 90}


def _big_event(cluster_id="cl0000001", n=5000, frp_max=10.0):
    return association.build_event_record(
        _cluster(cluster_id, n=n, modis=n // 2, viirs=n - n // 2,
                 frp_max=frp_max, frp_mean=frp_max / 2, frp_sum=frp_max * n,
                 frp_n=n), exec_id="e1")


class TestEventAlerts:
    def test_spike_bands(self, memdb):
        store.upsert_event(memdb, _big_event(n=5000, frp_max=10.0))
        store.upsert_event(memdb, _cluster_event("cl0000002"))
        records = alerts.generate_event_alerts(
            memdb, BANDS, exec_id="e5c", created_at="2024-01-01T00:00:00+00:00")
        by_id = {r["alert_id"]: r for r in records}
        spike = by_id["alt-asp-ev-cl0000001-a1-r1"]
        assert spike["severity_name"] == "Critical"
        assert spike["alert_type"] == "Activity Spike"
        assert spike["event_id"] == "ev-cl0000001-a1"
        assert "not a fire perimeter" in spike["explanation"].lower() \
            or "perimeter" in spike["explanation"]
        # Small event: below every band, no alert.
        assert "alt-asp-ev-cl0000002-a1-r1" not in by_id

    def test_high_frp_bands(self, memdb):
        store.upsert_event(memdb, _big_event(n=5, frp_max=300.0))
        records = alerts.generate_event_alerts(
            memdb, BANDS, exec_id="e5c", created_at="2024-01-01T00:00:00+00:00")
        by_id = {r["alert_id"]: r for r in records}
        frp = by_id["alt-frp-ev-cl0000001-a1-r1"]
        assert frp["severity_name"] == "High"
        assert frp["trigger"]["frp_max"] == 300.0
        assert "MW" in frp["explanation"]

    def test_missing_frp_never_alerts(self, memdb):
        record = association.build_event_record(
            _cluster(frp_max=None, frp_mean=None, frp_n=0, frp_null=5,
                     frp_sum=0.0), exec_id="e1")
        store.upsert_event(memdb, record)
        records = alerts.generate_event_alerts(
            memdb, BANDS, exec_id="e5c", created_at="2024-01-01T00:00:00+00:00")
        assert [r for r in records if r["alert_type"] == "High FRP"] == []

    def test_sensor_maxima_kept_separate(self, memdb):
        record = _big_event(n=5, frp_max=300.0)
        record["frp"]["by_sensor"] = {"MODIS": {"frp_max": 100.0},
                                      "VIIRS": {"frp_max": 300.0}}
        store.upsert_event(memdb, record)
        records = alerts.generate_event_alerts(
            memdb, BANDS, exec_id="e5c", created_at="2024-01-01T00:00:00+00:00")
        frp = next(r for r in records if r["alert_type"] == "High FRP")
        assert frp["trigger"]["by_sensor_max"] == {"MODIS": 100.0,
                                                  "VIIRS": 300.0}
        assert "not directly comparable" in frp["explanation"]

    def test_event_bands_derived_deterministically(self, memdb):
        for i in range(1, 21):
            store.upsert_event(
                memdb, _big_event(f"cl00000{i:02d}", n=i, frp_max=float(i)))
        bands = alerts._event_bands(memdb, alerts.AlertConfig())
        assert bands["n_events"] == 20
        assert bands["n_frp"] == 20
        assert bands["size"]["Moderate"] == 19.0  # nearest-rank p95 of 1..20
        again = alerts._event_bands(memdb, alerts.AlertConfig())
        assert again == bands


def _cluster_event(cluster_id):
    return association.build_event_record(_cluster(cluster_id), exec_id="e1")


# ---------------------------------------------------------------------------
# 5C.3 Phase 4C-derived alerts (methodology reuse, verbatim)
# ---------------------------------------------------------------------------

def _anomaly(anomaly_id="anom-detection-raw-sensor_aware-daily-2023-08-15",
             metric="detection", scope="raw", classification="notable-high",
             observed=500.0, baseline=100.0, eligibility="eligible",
             method="sensor_aware", grain="daily", period="2023-08-15"):
    return {
        "anomaly_id": anomaly_id, "metric": metric, "scope": scope,
        "eval_period": period, "observed_value": observed,
        "baseline_value": baseline, "pct_deviation": 400.0,
        "pct_status": "valid", "z_score": 3.5, "z_status": "valid",
        "baseline_method": method, "reference_strategy": "trailing",
        "reference_start": "2023-07-16", "reference_end": "2023-08-14",
        "reference_n": 30, "classification": classification,
        "eligibility": eligibility, "limitations": [], "grain": grain,
        "exec_id": "exec4c",
    }


class TestAnomalyAlerts:
    def test_mapping_to_lowest_defensible_severity(self):
        rows = [
            _anomaly("a1", classification="notable-high"),
            _anomaly("a2", classification="provisional-notable-high"),
            _anomaly("a3", classification="notable-low"),
            _anomaly("a4", classification="provisional-notable-low"),
        ]
        records = alerts.generate_anomaly_alerts(
            rows, exec_id="e5c", created_at="2024-01-01T00:00:00+00:00")
        by_id = {r["baseline_ref"]["anomaly_id"]: r["severity_name"]
                 for r in records}
        assert by_id == {"a1": "High", "a2": "Moderate", "a3": "Low",
                         "a4": "Informational"}

    def test_non_actionable_produce_nothing(self):
        rows = [_anomaly(f"n{i}", classification=c) for i, c in
                enumerate(alerts.NON_ACTIONABLE)]
        records = alerts.generate_anomaly_alerts(
            rows, exec_id="e5c", created_at="2024-01-01T00:00:00+00:00")
        assert records == []

    def test_unknown_classification_produce_nothing(self):
        rows = [_anomaly("u1", classification="mystery")]
        assert alerts.generate_anomaly_alerts(
            rows, exec_id="e5c", created_at="2024-01-01T00:00:00+00:00") == []

    def test_cluster_metric_routes_to_cluster_type(self):
        rows = [_anomaly("c1", metric="cluster", scope="count")]
        (record,) = alerts.generate_anomaly_alerts(
            rows, exec_id="e5c", created_at="2024-01-01T00:00:00+00:00")
        assert record["alert_type"] == "Cluster Anomaly"
        assert record["event_id"] is None  # global: no fabricated link
        assert "no single event is implicated" in record["explanation"]

    def test_scopes_stay_distinguishable(self):
        rows = [_anomaly("s1", scope="modis"),
                _anomaly("s2", scope="harmonized")]
        records = alerts.generate_anomaly_alerts(
            rows, exec_id="e5c", created_at="2024-01-01T00:00:00+00:00")
        assert {r["scope"] for r in records} == {"modis", "harmonized"}

    def test_explanation_carries_rule_and_disclaimer(self):
        (record,) = alerts.generate_anomaly_alerts(
            [_anomaly()], exec_id="e5c", created_at="2024-01-01T00:00:00+00:00")
        assert "|z| >= 2" in record["explanation"]
        assert "not the probability" in record["explanation"]


# ---------------------------------------------------------------------------
# 5C.4 Data-quality alerts
# ---------------------------------------------------------------------------

class TestDataQualityAlerts:
    def test_pilot_complete_coverage_yields_zero(self):
        records = alerts.generate_data_quality_alerts(
            exec_id="e5c", created_at="2024-01-01T00:00:00+00:00")
        assert records == []  # honest zero: 365/365 both sensors

    def test_partial_coverage_detected(self, tmp_path, monkeypatch):
        catalog = {"datasets": [
            {"dataset_id": "modis", "versions": [
                {"coverage": {"temporal": {"min": "2023-01-01",
                                           "max": "2023-12-31"}}}]},
            {"dataset_id": "viirs-snpp", "versions": [
                {"coverage": {"temporal": {"min": "2023-01-01",
                                           "max": "2023-06-30"}}}]},
        ]}
        path = tmp_path / "catalog.json"
        path.write_text(json.dumps(catalog), encoding="utf-8")
        monkeypatch.setattr("src.calendar.settings.CATALOG_PATH", str(path))
        records = alerts.generate_data_quality_alerts(
            exec_id="e5c", created_at="2024-01-01T00:00:00+00:00")
        assert len(records) == 184  # 2023-07-01..2023-12-31
        assert records[0]["alert_type"] == "Data Quality"
        assert records[0]["event_id"] is None
        assert "not a wildfire activity alert" in records[0]["explanation"]

    def test_explanation_never_claims_activity(self):
        text = alerts.explain_data_quality_alert(
            "Partial sensor coverage",
            {"severity": "Low", "description": "x"})
        assert "never interpreted as zero activity" in text


# ---------------------------------------------------------------------------
# 5C.5 Dedup, cooldown, ack/resolve, history
# ---------------------------------------------------------------------------

def _alert(alert_id="alt-asp-ev-cl0000001-a1-r1", severity="High",
           fingerprint="fp1", status="open"):
    return {
        "alert_id": alert_id, "alert_type": "Activity Spike",
        "severity_name": severity,
        "severity_level": model.SEVERITY_LEVELS[severity],
        "event_id": "ev-cl0000001-a1", "scope": "event",
        "metric": "detection", "grain": "event", "period": "2023-08-15",
        "trigger": {"n_detections": 500}, "baseline_ref": {},
        "data_quality": {}, "explanation": "test",
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T00:00:00+00:00",
        "rule_version": "alert_rules_v1", "status": status,
        "fingerprint": fingerprint,
    }


class TestAlertPersistence:
    def test_create_then_suppress_duplicate(self, memdb):
        first = alerts.persist_alerts(memdb, [_alert()], exec_id="e1",
                                      created_at="2024-01-10T00:00:00+00:00")
        assert first == {"created": 1, "suppressed": 0, "re_alerted": 0,
                         "reopened": 0}
        second = alerts.persist_alerts(memdb, [_alert()], exec_id="e1",
                                       created_at="2024-01-10T00:00:00+00:00")
        assert second["suppressed"] == 1
        history = store.list_alert_history(memdb, _alert()["alert_id"])
        assert [h["action"] for h in history] == ["created"]  # silent rerun

    def test_material_change_re_alerts(self, memdb):
        alerts.persist_alerts(memdb, [_alert()], exec_id="e1",
                              created_at="2024-01-10T00:00:00+00:00")
        changed = _alert(severity="Critical", fingerprint="fp2")
        stats = alerts.persist_alerts(memdb, [changed], exec_id="e2",
                                      created_at="2024-01-11T00:00:00+00:00")
        assert stats["re_alerted"] == 1
        back = store.get_alert(memdb, changed["alert_id"])
        assert back["severity_name"] == "Critical"
        assert back["status"] == "open"
        actions = [h["action"] for h in
                   store.list_alert_history(memdb, changed["alert_id"])]
        assert actions == ["created", "re-alert"]

    def test_resolved_cooldown_suppresses(self, memdb):
        alerts.persist_alerts(memdb, [_alert()], exec_id="e1",
                              created_at="2024-01-10T00:00:00+00:00")
        store.set_alert_status(memdb, _alert()["alert_id"], "resolved",
                               note="done", exec_id="e1",
                               at_utc="2024-01-10T01:00:00+00:00")
        changed = _alert(severity="Critical", fingerprint="fp2")
        stats = alerts.persist_alerts(memdb, [changed], exec_id="e2",
                                      created_at="2024-01-12T00:00:00+00:00")
        assert stats["suppressed"] == 1
        assert store.get_alert(memdb, changed["alert_id"])["status"] == \
            "resolved"

    def test_resolved_reopens_after_cooldown(self, memdb):
        alerts.persist_alerts(memdb, [_alert()], exec_id="e1",
                              created_at="2024-01-01T00:00:00+00:00")
        store.set_alert_status(memdb, _alert()["alert_id"], "resolved",
                               exec_id="e1",
                               at_utc="2024-01-01T01:00:00+00:00")
        changed = _alert(severity="Critical", fingerprint="fp2")
        stats = alerts.persist_alerts(memdb, [changed], exec_id="e2",
                                      created_at="2024-01-20T00:00:00+00:00")
        assert stats["reopened"] == 1
        assert store.get_alert(memdb, changed["alert_id"])["status"] == "open"

    def test_ack_and_resolve_flow(self, memdb):
        alerts.persist_alerts(memdb, [_alert()], exec_id="e1",
                              created_at="2024-01-10T00:00:00+00:00")
        alert_id = _alert()["alert_id"]
        store.set_alert_status(memdb, alert_id, "acknowledged", note="seen",
                               at_utc="2024-01-10T02:00:00+00:00")
        assert store.get_alert(memdb, alert_id)["status"] == "acknowledged"
        store.set_alert_status(memdb, alert_id, "resolved", note="done",
                               at_utc="2024-01-10T03:00:00+00:00")
        assert store.get_alert(memdb, alert_id)["status"] == "resolved"
        actions = [h["action"] for h in
                   store.list_alert_history(memdb, alert_id)]
        assert actions == ["created", "acknowledged", "resolved"]

    def test_history_survives_event_state_change(self, memdb):
        record = _event()
        store.upsert_event(memdb, record)
        alerts.persist_alerts(memdb, [_alert()], exec_id="e1",
                              created_at="2024-01-10T00:00:00+00:00")
        store.record_transition(
            memdb, event_id=record["event_id"], prev_state="Active",
            new_state="Monitoring", reason="x", evidence={},
            completeness={}, algo_version="lifecycle_v1",
            config_version="lifecycle_cfg_v1", exec_id="e2")
        history = store.list_alert_history(memdb, _alert()["alert_id"])
        assert [h["action"] for h in history] == ["created"]


# ---------------------------------------------------------------------------
# 5C.6 Severity sync + review-gate escalation
# ---------------------------------------------------------------------------

class TestSeveritySync:
    def test_max_open_alert_wins(self, memdb):
        store.upsert_event(memdb, _event())
        alerts.persist_alerts(
            memdb, [_alert("a1", severity="Moderate", fingerprint="f1"),
                    _alert("a2", severity="Critical", fingerprint="f2")],
            exec_id="e1", created_at="2024-01-10T00:00:00+00:00")
        # Re-point both alerts at the same event.
        memdb.execute("UPDATE alerts SET event_id = ?",
                      (_event()["event_id"],))
        memdb.commit()
        result = alerts.sync_event_severity(memdb, exec_id="e5c")
        assert result["updated"] == 1
        back = store.get_event(memdb, _event()["event_id"])
        assert back["severity_name"] == "Critical"
        assert "a2" in back["severity_provenance"]

    def test_resolved_alerts_ignored(self, memdb):
        store.upsert_event(memdb, _event())
        alerts.persist_alerts(memdb, [_alert(fingerprint="f1")], exec_id="e1",
                              created_at="2024-01-10T00:00:00+00:00")
        memdb.execute("UPDATE alerts SET event_id = ?",
                      (_event()["event_id"],))
        store.set_alert_status(memdb, _alert()["alert_id"], "resolved",
                               at_utc="2024-01-10T02:00:00+00:00")
        alerts.sync_event_severity(memdb, exec_id="e5c")
        assert store.get_event(memdb, _event()["event_id"])[
            "severity_name"] == "Informational"

    def test_unalerted_events_stay_informational(self, memdb):
        store.upsert_event(memdb, _event())
        result = alerts.sync_event_severity(memdb, exec_id="e5c")
        assert result["updated"] == 0
        assert store.get_event(memdb, _event()["event_id"])[
            "severity_name"] == "Informational"

    def test_resolved_high_escalates_for_review(self, memdb):
        record = _event(state="Resolved", severity="Informational",
                        evaluated="2023-12-31T23:59:59+00:00")
        store.upsert_event(memdb, record)
        alert = _alert(severity="High", fingerprint="f9")
        alert["event_id"] = record["event_id"]
        alerts.persist_alerts(memdb, [alert], exec_id="e1",
                              created_at="2024-01-10T00:00:00+00:00")
        result = alerts.sync_event_severity(
            memdb, exec_id="e5c", created_at="2024-01-10T00:00:00+00:00")
        assert result["escalated"] == 1
        back = store.get_event(memdb, record["event_id"])
        assert back["lifecycle_state"] == "Pending Closure"
        assert back["review_required"] == 1
        history = store.list_transitions(memdb, record["event_id"])
        assert history[-1]["reason"] == "severity-escalation-review"

    def test_resolved_high_with_approval_stays_resolved(self, memdb):
        record = _event(state="Resolved", severity="Informational",
                        evaluated="2023-12-31T23:59:59+00:00", review="approved")
        store.upsert_event(memdb, record)
        alert = _alert(severity="High", fingerprint="f9")
        alert["event_id"] = record["event_id"]
        alerts.persist_alerts(memdb, [alert], exec_id="e1",
                              created_at="2024-01-10T00:00:00+00:00")
        result = alerts.sync_event_severity(memdb, exec_id="e5c")
        assert result["escalated"] == 0
        assert store.get_event(memdb, record["event_id"])[
            "lifecycle_state"] == "Resolved"


# ---------------------------------------------------------------------------
# 5D.1 Subscription workflow
# ---------------------------------------------------------------------------

class TestSubscribe:
    def test_subscribe_pending_with_tokens(self, memdb):
        result = notify.subscribe(memdb, "User@Example.com")
        assert result["status"] == "pending"
        assert result["confirm_token"] and result["manage_token"]
        assert result["confirm_token"] != result["manage_token"]
        back = store.get_subscription(memdb, result["sub_id"])
        assert back["email"] == "user@example.com"  # normalized
        assert back["status"] == "pending"

    def test_invalid_email_rejected(self, memdb):
        for bad in ("", "no-at-sign", "a@b", "x" * 300 + "@example.com",
                    "user @example.com"):
            with pytest.raises(ValueError, match="invalid email"):
                notify.subscribe(memdb, bad)

    def test_only_hashes_stored(self, memdb):
        result = notify.subscribe(memdb, "a@example.com")
        back = store.get_subscription(memdb, result["sub_id"])
        assert back["confirm_token_hash"] != result["confirm_token"]
        assert len(back["confirm_token_hash"]) == 64
        assert result["confirm_token"] not in json.dumps(back)

    def test_tokens_unique(self, memdb):
        first = notify.subscribe(memdb, "a@example.com")
        second = notify.subscribe(memdb, "b@example.com")
        assert first["confirm_token"] != second["confirm_token"]
        assert first["manage_token"] != second["manage_token"]

    def test_resubscribe_pending_refreshes(self, memdb):
        first = notify.subscribe(memdb, "a@example.com")
        second = notify.subscribe(memdb, "a@example.com")
        assert second["sub_id"] == first["sub_id"]
        assert second["confirm_token"] != first["confirm_token"]
        old = notify.confirm(memdb, first["confirm_token"])
        assert old["status"] == "invalid"  # superseded token dies

    def test_resubscribe_confirmed_stays(self, memdb):
        pending = notify.subscribe(memdb, "a@example.com")
        notify.confirm(memdb, pending["confirm_token"])
        again = notify.subscribe(memdb, "a@example.com")
        assert again["status"] == "confirmed"
        assert again["confirm_token"] is None


class TestConfirm:
    def test_confirm_happy_path(self, memdb):
        pending = notify.subscribe(memdb, "a@example.com")
        result = notify.confirm(memdb, pending["confirm_token"])
        assert result["status"] == "confirmed"
        back = store.get_subscription(memdb, pending["sub_id"])
        assert back["status"] == "confirmed"
        assert back["confirm_token_hash"] is None  # one-time: consumed
        assert back["confirmed_at"]

    def test_confirm_one_time(self, memdb):
        pending = notify.subscribe(memdb, "a@example.com")
        notify.confirm(memdb, pending["confirm_token"])
        again = notify.confirm(memdb, pending["confirm_token"])
        assert again["status"] == "invalid"

    def test_confirm_unknown_token(self, memdb):
        assert notify.confirm(memdb, "bogus")["status"] == "invalid"
        assert notify.confirm(memdb, "")["status"] == "invalid"

    def test_confirm_expired(self, memdb):
        pending = notify.subscribe(memdb, "a@example.com")
        memdb.execute("UPDATE subscriptions SET confirm_expires_utc = ? "
                      "WHERE sub_id = ?", ("2000-01-01T00:00:00+00:00",
                                           pending["sub_id"]))
        memdb.commit()
        result = notify.confirm(memdb, pending["confirm_token"])
        assert result["status"] == "expired"
        assert store.get_subscription(memdb, pending["sub_id"])["status"] \
            == "pending"

    def test_confirm_after_unsubscribe_invalid(self, memdb):
        pending = notify.subscribe(memdb, "a@example.com")
        notify.unsubscribe(memdb, pending["manage_token"])
        result = notify.confirm(memdb, pending["confirm_token"])
        assert result["status"] == "invalid"


class TestUnsubscribe:
    def test_unsubscribe_happy_path(self, memdb):
        pending = notify.subscribe(memdb, "a@example.com")
        notify.confirm(memdb, pending["confirm_token"])
        result = notify.unsubscribe(memdb, pending["manage_token"])
        assert result["status"] == "unsubscribed"
        assert store.get_subscription(memdb, pending["sub_id"])[
            "unsubscribed_at"]

    def test_unsubscribe_idempotent(self, memdb):
        pending = notify.subscribe(memdb, "a@example.com")
        notify.unsubscribe(memdb, pending["manage_token"])
        again = notify.unsubscribe(memdb, pending["manage_token"])
        assert again["status"] == "unsubscribed"

    def test_unsubscribe_unknown_token(self, memdb):
        assert notify.unsubscribe(memdb, "bogus")["status"] == "invalid"

    def test_unsubscribed_can_resubscribe(self, memdb):
        pending = notify.subscribe(memdb, "a@example.com")
        notify.unsubscribe(memdb, pending["manage_token"])
        again = notify.subscribe(memdb, "a@example.com")
        assert again["status"] == "pending"


class TestPrefs:
    def test_set_digest_weekly(self, memdb):
        pending = notify.subscribe(memdb, "a@example.com")
        result = notify.set_prefs(memdb, pending["manage_token"],
                                  {"digest": "weekly"})
        assert result["status"] == "ok"
        assert result["prefs"]["digest"] == "weekly"

    def test_bad_digest_rejected(self, memdb):
        pending = notify.subscribe(memdb, "a@example.com")
        result = notify.set_prefs(memdb, pending["manage_token"],
                                  {"digest": "hourly"})
        assert result["status"] == "invalid"

    def test_informational_opt_in(self, memdb):
        pending = notify.subscribe(memdb, "a@example.com")
        notify.set_prefs(memdb, pending["manage_token"],
                         {"include_informational": True})
        back = store.get_subscription(memdb, pending["sub_id"])
        assert back["prefs"]["include_informational"] is True

    def test_unknown_token_rejected(self, memdb):
        assert notify.set_prefs(memdb, "bogus", {})["status"] == "invalid"


# ---------------------------------------------------------------------------
# 5D.2 Policy, delivery, retries, content
# ---------------------------------------------------------------------------

def _confirmed(memdb, email="a@example.com"):
    pending = notify.subscribe(memdb, email)
    notify.confirm(memdb, pending["confirm_token"])
    return store.get_subscription(memdb, pending["sub_id"])


class TestPolicy:
    def test_pending_and_unsubscribed_get_nothing(self, memdb):
        pending = notify.subscribe(memdb, "a@example.com")
        sub = store.get_subscription(memdb, pending["sub_id"])
        assert notify.wants_alert(sub, _alert(severity="Critical")) is False
        notify.unsubscribe(memdb, pending["manage_token"])
        sub = store.get_subscription(memdb, pending["sub_id"])
        assert notify.wants_alert(sub, _alert(severity="Critical")) is False

    def test_immediate_and_digest_matrix(self, memdb):
        sub = _confirmed(memdb)
        assert notify.wants_alert(sub, _alert(severity="Critical")) is True
        assert notify.wants_alert(sub, _alert(severity="High")) is True
        assert notify.wants_alert(sub, _alert(severity="Moderate")) is True
        assert notify.wants_alert(sub, _alert(severity="Low")) is True
        assert notify.wants_alert(sub,
                                  _alert(severity="Informational")) is False

    def test_informational_opt_in(self, memdb):
        sub = _confirmed(memdb)
        updated = dict(sub)
        updated["prefs"] = {**sub["prefs"], "include_informational": True}
        assert notify.wants_alert(updated,
                                  _alert(severity="Informational")) is True


class TestDelivery:
    def test_immediate_sends_high_and_critical(self, memdb):
        sub = _confirmed(memdb)
        alerts.persist_alerts(
            memdb, [_alert("h1", severity="High", fingerprint="fh"),
                    _alert("c1", severity="Critical", fingerprint="fc"),
                    _alert("m1", severity="Moderate", fingerprint="fm")],
            exec_id="e1", created_at="2024-01-10T00:00:00+00:00")
        sent = []
        stats = notify.process_outbox(
            memdb, kind="immediate",
            sender=lambda to, subject, body: sent.append((to, subject)) or True)
        assert stats["sent"] == 2
        assert {s for _, s in sent} != set()
        assert all(sub["email"] == to for to, _ in sent)

    def test_rerun_never_duplicates(self, memdb):
        _confirmed(memdb)
        alerts.persist_alerts(memdb, [_alert()], exec_id="e1",
                              created_at="2024-01-10T00:00:00+00:00")
        sender = lambda to, subject, body: True  # noqa: E731
        first = notify.process_outbox(memdb, kind="immediate", sender=sender)
        second = notify.process_outbox(memdb, kind="immediate", sender=sender)
        assert first["sent"] == 1
        assert second == {"sent": 0, "failed": 0, "skipped": 1}

    def test_changed_evidence_remails(self, memdb):
        _confirmed(memdb)
        alerts.persist_alerts(memdb, [_alert()], exec_id="e1",
                              created_at="2024-01-10T00:00:00+00:00")
        sender = lambda to, subject, body: True  # noqa: E731
        notify.process_outbox(memdb, kind="immediate", sender=sender)
        changed = _alert(severity="Critical", fingerprint="fp2")
        alerts.persist_alerts(memdb, [changed], exec_id="e2",
                              created_at="2024-01-11T00:00:00+00:00")
        stats = notify.process_outbox(memdb, kind="immediate", sender=sender)
        assert stats["sent"] == 1

    def test_failure_retries_then_fails(self, memdb):
        _confirmed(memdb)
        alerts.persist_alerts(memdb, [_alert()], exec_id="e1",
                              created_at="2024-01-10T00:00:00+00:00")
        sender = lambda to, subject, body: False  # noqa: E731
        for _ in range(notify.MAX_ATTEMPTS):
            notify.process_outbox(memdb, kind="immediate", sender=sender)
        rows = store.list_notifications(memdb)
        assert len(rows) == 1
        assert rows[0]["attempts"] == notify.MAX_ATTEMPTS
        assert rows[0]["status"] in ("failed", "queued")
        assert rows[0]["sent_at"] is None  # never marked delivered

    def test_sender_exception_is_failure_not_crash(self, memdb):
        _confirmed(memdb)
        alerts.persist_alerts(memdb, [_alert()], exec_id="e1",
                              created_at="2024-01-10T00:00:00+00:00")

        def boom(to, subject, body):
            raise ConnectionError("smtp down")

        stats = notify.process_outbox(memdb, kind="immediate", sender=boom)
        assert stats["failed"] == 1

    def test_digest_batches_moderate_and_low(self, memdb):
        _confirmed(memdb)
        alerts.persist_alerts(
            memdb, [_alert("m1", severity="Moderate", fingerprint="fm"),
                    _alert("l1", severity="Low", fingerprint="fl")],
            exec_id="e1", created_at="2024-01-10T00:00:00+00:00")
        sent = []
        stats = notify.process_outbox(
            memdb, kind="digest",
            sender=lambda to, subject, body: sent.append(body) or True)
        assert stats["sent"] == 1  # one email per subscriber
        assert "m1" in sent[0] and "l1" in sent[0]

    def test_digest_rerun_silent(self, memdb):
        _confirmed(memdb)
        alerts.persist_alerts(memdb, [_alert(severity="Low")], exec_id="e1",
                              created_at="2024-01-10T00:00:00+00:00")
        sender = lambda to, subject, body: True  # noqa: E731
        notify.process_outbox(memdb, kind="digest", sender=sender)
        second = notify.process_outbox(memdb, kind="digest", sender=sender)
        assert second["skipped"] == 1

    def test_unsubscribed_gets_nothing(self, memdb):
        pending = notify.subscribe(memdb, "a@example.com")
        notify.confirm(memdb, pending["confirm_token"])
        notify.unsubscribe(memdb, pending["manage_token"])
        alerts.persist_alerts(memdb, [_alert(severity="Critical")],
                              exec_id="e1",
                              created_at="2024-01-10T00:00:00+00:00")
        calls = []
        stats = notify.process_outbox(
            memdb, kind="immediate",
            sender=lambda to, s, b: calls.append(to) or True)
        assert stats == {"sent": 0, "failed": 0, "skipped": 0}
        assert calls == []

    def test_smtp_unavailable_never_marks_sent(self, memdb, monkeypatch):
        # Hermetic: pin the unavailable state regardless of local .env.
        monkeypatch.setattr("src.calendar.settings.CAL_SMTP_ENABLED", "false")
        monkeypatch.setattr("src.calendar.settings.CAL_SMTP_USER", "")
        monkeypatch.setattr("src.calendar.settings.CAL_SMTP_PASSWORD", "")
        _confirmed(memdb)
        alerts.persist_alerts(memdb, [_alert()], exec_id="e1",
                              created_at="2024-01-10T00:00:00+00:00")
        assert notify.smtp_status()["available"] is False  # no creds in test
        stats = notify.process_outbox(memdb, kind="immediate")
        assert stats["sent"] == 0
        rows = store.list_notifications(memdb)
        assert all(r["sent_at"] is None for r in rows)
        assert all(r["status"] != "sent" for r in rows)


class TestContent:
    def test_confirmation_email(self):
        subject, body = notify.confirmation_email("sub1", "tok123")
        assert "Confirm" in subject
        assert "tok123" in body
        assert "48 hours" in body
        assert "one" in body.lower() or "once" in body

    def test_immediate_content(self, memdb):
        record = _event(severity="High")
        store.upsert_event(memdb, record)
        alert = _alert(severity="High")
        alert["event_id"] = record["event_id"]
        subject, body = notify.render_immediate(alert, record, "manage1")
        assert "High" in subject
        assert record["event_id"] in body
        assert "Approx. loc." in body
        assert "not an ignition" in body
        assert "manage1" in body
        assert "confirmed ground truth" in body

    def test_immediate_without_event(self):
        subject, body = notify.render_immediate(_alert(), None)
        assert "alt-asp" in subject
        assert "Evidence:" in body

    def test_digest_content(self):
        subject, body = notify.render_digest([_alert()], "2024-01-10")
        assert "digest" in subject.lower()
        assert _alert()["alert_id"] in body


# ---------------------------------------------------------------------------
# 5E.1 Qwen configuration + payloads
# ---------------------------------------------------------------------------

class TestQwenConfig:
    def test_disabled_by_default(self):
        status = qwen.QwenConfig.from_settings().status()
        assert status["available"] is False

    def test_enabled_but_incomplete(self, monkeypatch):
        monkeypatch.setattr("src.calendar.settings.CAL_QWEN_ENABLED", "true")
        monkeypatch.setattr("src.calendar.settings.CAL_QWEN_BASE_URL", "")
        monkeypatch.setattr("src.calendar.settings.CAL_QWEN_API_KEY", "")
        monkeypatch.setattr("src.calendar.settings.CAL_QWEN_MODEL", "")
        status = qwen.QwenConfig.from_settings().status()
        assert status["available"] is False
        assert "missing configuration" in status["reason"]

    def test_fully_configured(self, monkeypatch):
        monkeypatch.setattr("src.calendar.settings.CAL_QWEN_ENABLED", "true")
        monkeypatch.setattr("src.calendar.settings.CAL_QWEN_BASE_URL",
                            "http://localhost:11434/v1")
        monkeypatch.setattr("src.calendar.settings.CAL_QWEN_API_KEY", "k")
        monkeypatch.setattr("src.calendar.settings.CAL_QWEN_MODEL", "qwen")
        status = qwen.QwenConfig.from_settings().status()
        assert status["available"] is True


def _payload():
    return qwen.build_event_payload(_event(severity="High"), [_alert()],
                                    [_anomaly()])


class TestPayloads:
    def test_event_payload_verified_fields_only(self):
        payload = _payload()
        assert payload["event_id"] == "ev-cl0000001-a1"
        assert payload["severity"] == "High"
        assert payload["frp"]["frp_max"] == 60.0
        assert payload["alerts"][0]["alert_id"] == _alert()["alert_id"]
        assert payload["anomalies"][0]["classification"] == "notable-high"
        assert any("not confirmed fires" in line
                   for line in payload["limitations"])

    def test_alert_payload_with_and_without_event(self):
        with_event = qwen.build_alert_payload(_alert(), _event())
        assert with_event["event"]["event_id"] == "ev-cl0000001-a1"
        assert "not a fire danger rating" in " ".join(
            with_event["limitations"])
        solo = qwen.build_alert_payload(_alert(), None)
        assert solo["event"] is None


# ---------------------------------------------------------------------------
# 5E.2 Response validation
# ---------------------------------------------------------------------------

def _valid_response():
    return {"summary": "Event ev-cl0000001-a1 shows 5 satellite observations "
                       "on 2023-08-15 with maximum FRP 60 MW.",
            "evidence_used": ["ev-cl0000001-a1",
                              "anom-detection-raw-sensor_aware-daily-2023-08-15"],
            "uncertainty": ["Single-day span; no trend visible."],
            "limitations": ["Satellite observations are not confirmed "
                            "fires."]}


class TestValidation:
    def test_valid_passes(self):
        verdict = qwen.validate_response(_valid_response(), _payload())
        assert verdict == {"valid": True, "reason": "all checks passed"}

    def test_non_object_rejected(self):
        assert qwen.validate_response([1, 2], _payload())["valid"] is False

    def test_missing_keys_rejected(self):
        response = _valid_response()
        del response["uncertainty"]
        verdict = qwen.validate_response(response, _payload())
        assert verdict["valid"] is False
        assert "uncertainty" in verdict["reason"]

    def test_empty_summary_rejected(self):
        response = _valid_response()
        response["summary"] = "  "
        assert qwen.validate_response(response, _payload())["valid"] is False

    def test_banned_claims_rejected(self):
        for banned in ("the fire is contained", "PERIMETER mapped",
                       "evacuation recommended", "fire danger is extreme",
                       "predicted to spread", "NASA assessment confirms",
                       "confirmed wildfire", "will spread north"):
            response = _valid_response()
            response["summary"] = f"Activity noted; {banned}."
            verdict = qwen.validate_response(response, _payload())
            assert verdict["valid"] is False, banned
            assert "banned claim" in verdict["reason"]

    def test_unsupported_evidence_rejected(self):
        response = _valid_response()
        response["evidence_used"] = ["ev-nope-a1"]
        verdict = qwen.validate_response(response, _payload())
        assert verdict["valid"] is False
        assert "unsupported evidence" in verdict["reason"]

    def test_fabricated_numbers_rejected(self):
        response = _valid_response()
        response["summary"] = ("Event shows 99999 satellite observations "
                               "with maximum FRP 187.4 MW.")
        verdict = qwen.validate_response(response, _payload())
        assert verdict["valid"] is False
        assert "untraceable numbers" in verdict["reason"]

    def test_dates_and_ids_are_citable(self):
        response = _valid_response()
        response["summary"] = ("Cluster cl0000001 observed on 2023-08-15: "
                               "5 observations.")
        response["evidence_used"] = ["ev-cl0000001-a1"]
        assert qwen.validate_response(response, _payload())["valid"] is True

    def test_contradictory_numbers_rejected(self):
        # 60 MW is the verified max; claiming 61 MW must fail.
        response = _valid_response()
        response["summary"] = "Maximum FRP 61 MW observed."
        verdict = qwen.validate_response(response, _payload())
        assert verdict["valid"] is False


# ---------------------------------------------------------------------------
# 5E.3 Explain: fallback discipline + logging
# ---------------------------------------------------------------------------

def _enabled_config():
    return qwen.QwenConfig(enabled=True, base_url="http://x/v1",
                           api_key="k", model="qwen-test", timeout_s=5.0)


class TestExplain:
    def test_disabled_falls_back(self, memdb):
        result = qwen.explain("event", "ev-cl0000001-a1", _payload(),
                              config=qwen.QwenConfig(enabled=False),
                              conn=memdb)
        assert result["status"] == "disabled"
        assert result["label"] == qwen.FALLBACK_LABEL
        assert "ev-cl0000001-a1" in result["text"]
        assert result["model"] == ""
        logs = store.list_qwen_log(memdb, target_id="ev-cl0000001-a1")
        assert logs[0]["status"] == "disabled"

    def test_timeout_falls_back(self, memdb):
        def slow(prompt, config):
            raise TimeoutError("request timed out")

        result = qwen.explain("event", "ev-cl0000001-a1", _payload(),
                              config=_enabled_config(), transport=slow,
                              conn=memdb)
        assert result["status"] == "timeout"
        assert result["label"] == qwen.FALLBACK_LABEL
        assert result["latency_ms"] is not None

    def test_transport_error_falls_back(self, memdb):
        def broken(prompt, config):
            raise ConnectionError("connection refused")

        result = qwen.explain("event", "ev-cl0000001-a1", _payload(),
                              config=_enabled_config(), transport=broken,
                              conn=memdb)
        assert result["status"] == "error"
        assert "ev-cl0000001-a1" in result["text"]  # fallback still explains

    def test_invalid_json_falls_back(self, memdb):
        result = qwen.explain(
            "event", "ev-cl0000001-a1", _payload(),
            config=_enabled_config(),
            transport=lambda prompt, config: "not json at all", conn=memdb)
        assert result["status"] == "invalid"
        assert "unparseable" in result["validation"]

    def test_banned_output_falls_back(self, memdb):
        bad = dict(_valid_response(), summary="The fire is contained.")
        result = qwen.explain(
            "event", "ev-cl0000001-a1", _payload(),
            config=_enabled_config(),
            transport=lambda prompt, config: json.dumps(bad), conn=memdb)
        assert result["status"] == "invalid"
        assert "banned claim" in result["validation"]

    def test_valid_output_labeled_ai(self, memdb):
        good = _valid_response()
        result = qwen.explain(
            "event", "ev-cl0000001-a1", _payload(),
            config=_enabled_config(),
            transport=lambda prompt, config: json.dumps(good), conn=memdb)
        assert result["status"] == "ok"
        assert result["label"] == qwen.AI_LABEL
        assert result["model"] == "qwen-test"
        assert "Evidence used:" in result["text"]
        assert "Uncertainty:" in result["text"]
        assert "Limitations:" in result["text"]
        logs = store.list_qwen_log(memdb, target_id="ev-cl0000001-a1")
        assert logs[0]["status"] == "ok"

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="kind"):
            qwen.explain("fire", "x", {}, config=_enabled_config())

    def test_qwen_never_mutates_store(self, memdb):
        before = store.count_events(memdb)
        qwen.explain("event", "ev-cl0000001-a1", _payload(),
                     config=qwen.QwenConfig(enabled=False), conn=memdb)
        assert store.count_events(memdb) == before


class TestFallback:
    def test_event_fallback_facts(self):
        text = qwen.deterministic_fallback("event", _payload())
        assert "ev-cl0000001-a1" in text
        assert "5 satellite observations" in text
        assert "60" in text and "MW" in text
        assert "Limitations:" in text

    def test_event_fallback_missing_frp(self):
        payload = _payload()
        payload["frp"] = {"frp_max": None, "frp_n": 0, "frp_null": 5}
        text = qwen.deterministic_fallback("event", payload)
        assert "missing, not zero" in text

    def test_alert_fallback_facts(self):
        payload = qwen.build_alert_payload(_alert(), None)
        text = qwen.deterministic_fallback("alert", payload)
        assert _alert()["alert_id"] in text
        assert "no single event is implicated" in text


class TestValidationReceipt:
    def test_check_catalog_complete(self):
        checks = qwen.describe_checks()
        assert len(checks) == 4
        joined = " ".join(c["check"] + " " + c["rule"] for c in checks)
        for token in ("evidence_used", "perimeters",
                      str(len(qwen.BANNED_PHRASES)), "payload"):
            assert token in joined, token

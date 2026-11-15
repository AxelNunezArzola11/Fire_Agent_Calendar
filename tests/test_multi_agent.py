"""Multi-agent tests: contracts, providers, agents, coordinator.

No network, no DB fixtures except tmp sqlite for audit rows. Provider
behavior is scripted; a scripted success is never a live test.
"""

import json

import pytest

from src.calendar.agents import (contracts, coordinator, detection_intel,
                                event_analysis, evidence_validator,
                                providers, science_observer)


class ScriptedProvider(providers.Provider):
    def __init__(self, script, name="scripted",
                 available=(True, "scripted")):
        self.name = name
        self._script = list(script)
        self._available = available
        self.calls = []

    def available(self):
        return self._available

    def complete(self, prompt, system):
        self.calls.append((prompt, system))
        if not self._script:
            raise AssertionError("over-called: script exhausted")
        outcome = self._script.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _attempt(text, provider="scripted"):
    return providers.Attempt(provider, "mock-model", True, 1.0,
                             text=text)


def _detection_payload():
    return {
        "cluster_id": "cl0000161", "cluster_ids": ["cl0000161"],
        "n_detections": 500, "n_modis": 200, "n_viirs": 300,
        "first_observed_utc": "2023-08-01T06:00:00+00:00",
        "last_observed_utc": "2023-08-15T06:00:00+00:00",
        "observation_span_h": 336.0, "satellites": ["Aqua", "N"],
        "timeline": [
            {"day": "2023-08-01", "n": 200, "n_modis": 100,
             "n_viirs": 100, "frp_modis": 1000.0, "frp_viirs": 500.0,
             "frp_missing": 0},
            {"day": "2023-08-15", "n": 300, "n_modis": 100,
             "n_viirs": 200, "frp_modis": 2000.0, "frp_viirs": 1500.0,
             "frp_missing": 2},
        ],
        "limitations": ["Observations are not confirmed fires."],
    }


def _valid_detection_json():
    return json.dumps({
        "summary": "Cluster cl0000161 shows 500 observations over 2 days.",
        "evidence_used": ["cl0000161"],
        "uncertainty": ["Single-linkage chaining may merge activity."],
        "limitations": ["Observations are not confirmed fires."]})


def _event_payload():
    from src.calendar.events import qwen as qwen_mod

    return qwen_mod.build_event_payload({
        "event_id": "ev-cl0000001-a1",
        "lifecycle_state": "Active", "severity_name": "High",
        "detection_count": 5, "sensor_modis_n": 2,
        "sensor_viirs_n": 3,
        "first_observed_utc": "2023-08-15T06:00:00+00:00",
        "last_observed_utc": "2023-08-15T06:00:00+00:00",
        "satellites": ["Aqua"], "cluster_ids": ["cl0000001"],
        "frp": {"frp_max": 60.0, "frp_n": 5, "frp_null": 0},
        "centroid": [0.0, 0.0], "ambiguity": {"flags": []},
        "completeness": {}, "anomaly_refs": [],
        "provenance": {}}, [])


def _alert_payload():
    from src.calendar.events import qwen as qwen_mod

    return qwen_mod.build_alert_payload({
        "alert_id": "alt-1", "alert_type": "Activity Spike",
        "severity_name": "High", "status": "open", "scope": "event",
        "metric": "detection", "period": "2023-08-15",
        "trigger": {"n": 500}, "baseline_ref": {}, "data_quality": {},
        "rule_version": "alert_rules_v1"}, None)


def _science_payload():
    return {
        "method_version": "det-frp-oop-v1", "n": 4783,
        "observed_sum": 108027.2, "expected_sum": 106573.4,
        "by_sensor": {
            "modis": {"n": 925, "observed": 65700.9,
                      "expected": 64842.2},
            "viirs": {"n": 3858, "observed": 42326.3,
                      "expected": 41731.2}},
        "inputs_missing": ["QA flags", "cloud fraction"],
    }


class TestContracts:
    def test_bad_kind_rejected(self):
        with pytest.raises(ValueError, match="unknown agent kind"):
            contracts.AgentRequest("bogus", "x", {})

    def test_empty_target_rejected(self):
        with pytest.raises(ValueError, match="target_id"):
            contracts.AgentRequest("event", "", {})

    def test_non_dict_payload_rejected(self):
        for bad in (None, "x", [1], 42):
            with pytest.raises(ValueError, match="payload must be"):
                contracts.AgentRequest("event", "x", bad)

    def test_missing_payload_keys_rejected(self):
        with pytest.raises(ValueError, match="required keys"):
            contracts.AgentRequest("detections", "x", {"n_detections": 1})
        with pytest.raises(ValueError, match="required keys"):
            contracts.AgentRequest("science", "x", {"observed_sum": 1.0})

    def test_ok_requires_provider(self):
        with pytest.raises(ValueError, match="must name their provider"):
            contracts.AgentResponse("a", "ok", "text", validation="v")

    def test_finding_severity_enforced(self):
        with pytest.raises(ValueError, match="bad severity"):
            contracts.Finding("r", "f", severity="bogus")
        with pytest.raises(ValueError, match="non-empty"):
            contracts.Finding(" ", "f")

    def test_response_holes_closed(self):
        good = {"agent": "a", "status": "fallback", "summary": "s",
                "fallback_reason": "r", "validation": "v"}
        with pytest.raises(ValueError, match="non-empty string"):
            contracts.AgentResponse(**{**good, "summary": "  "})
        with pytest.raises(ValueError, match="tuple of strings"):
            contracts.AgentResponse(**{**good,
                                       "evidence_used": ("a", 1)})
        with pytest.raises(ValueError, match="tuple of Finding"):
            contracts.AgentResponse(**{**good, "findings": ("x",)})
        with pytest.raises(ValueError, match="latency_ms"):
            contracts.AgentResponse(**{**good, "latency_ms": -1.0})
        with pytest.raises(ValueError, match="fallback_reason"):
            contracts.AgentResponse(**{**good, "fallback_reason": " "})


class TestRedact:
    @pytest.mark.parametrize("dirty", [
        "store api_key=SECRET123 failed",
        "token: TOKEN456 missing",
        "'access_token': 'ABC123'",
        '{"apikey": "XYZ789", "x": 1}',
        "auth Bearer TOKEN456 down",
    ])
    def test_secret_shapes_scrubbed(self, dirty):
        clean = providers.redact(dirty)
        assert "[redacted]" in clean
        for token in ("SECRET123", "TOKEN456", "ABC123", "XYZ789"):
            assert token not in clean

    def test_truncation_documented(self):
        assert len(providers.redact("x" * 600)) == 500


class TestProviderChain:
    def test_failover_identity_and_calls(self):
        first = ScriptedProvider(
            [providers.Attempt("first", "m", False, 1.0, error="boom")],
            name="first")
        second = ScriptedProvider([_attempt("hi", "second")],
                                  name="second")
        attempts = providers.ProviderChain(
            [first, second]).complete("p", "s")
        assert [a.provider for a in attempts] == ["first", "second"]
        assert [a.ok for a in attempts] == [False, True]
        assert len(first.calls) == 1 and len(second.calls) == 1

    def test_faulty_provider_isolated(self):
        bad = ScriptedProvider([RuntimeError("bad")])
        attempts = providers.ProviderChain([bad]).complete("p", "s")
        assert attempts[0].ok is False
        assert "RuntimeError" in attempts[0].error

    def test_default_chain_qwen_only_on_import_failure(
            self, monkeypatch):
        import sys

        import src.calendar.agents as agents_pkg

        # Poison both lookup paths: from-imports resolve via the parent
        # package attribute once the real module was imported earlier.
        monkeypatch.setitem(sys.modules,
                            "src.calendar.agents.granite", object())
        monkeypatch.setattr(agents_pkg, "granite", object(),
                            raising=False)
        chain = providers.default_chain()
        assert [p.name for p in chain.providers] == ["qwen"]


class TestQwenTransport:
    def _config(self, **over):
        import types

        args = {"base_url": "http://example.invalid",
                "api_key": "DUMMY", "model": "m", "timeout_s": 5.0}
        args.update(over)
        return types.SimpleNamespace(**args)

    def test_bad_timeout_rejected(self):
        with pytest.raises(ValueError, match="timeout_s"):
            providers._qwen_transport("p", "s", self._config(timeout_s=0))

    def test_malformed_schema_rejected(self, monkeypatch):
        import types

        seen = {}

        def _post(url, *, headers, json, timeout):
            seen["timeout"] = timeout
            return types.SimpleNamespace(
                status_code=200, content=b"{}", json=lambda: {},
                raise_for_status=lambda: None)

        import requests

        monkeypatch.setattr(requests, "post", _post)
        with pytest.raises(ValueError, match="malformed"):
            providers._qwen_transport("p", "s", self._config())
        assert seen["timeout"] == 5.0

    def test_oversize_rejected(self, monkeypatch):
        import types

        def _post(url, *, headers, json, timeout):
            return types.SimpleNamespace(
                status_code=200, content=b"x" * (256 * 1024 + 1),
                json=lambda: {}, raise_for_status=lambda: None)

        import requests

        monkeypatch.setattr(requests, "post", _post)
        with pytest.raises(ValueError, match="size cap"):
            providers._qwen_transport("p", "s", self._config())

    def test_success_path(self, monkeypatch):
        import types

        def _post(url, *, headers, json, timeout):
            assert headers["Authorization"].startswith("Bearer ")
            assert json["model"] == "m"
            return types.SimpleNamespace(
                status_code=200, content=b"{}",
                json=lambda: {"choices": [
                    {"message": {"content": "ok-text"}}]},
                raise_for_status=lambda: None)

        import requests

        monkeypatch.setattr(requests, "post", _post)
        text, usage = providers._qwen_transport(
            "p", "s", self._config())
        assert text == "ok-text"
        assert usage == {"input_tokens": None,
                         "generated_tokens": None}


class TestPromptGuardrails:
    def test_no_perimeter_word_rule(self):
        for system in (detection_intel.SYSTEM, event_analysis.SYSTEM,
                       science_observer.SYSTEM):
            assert "Never write any of these words" in system
            for token in ("perimeter", "containment", "contained",
                          "extinguish"):
                assert token in system
            assert "observation extent" in system

    def test_perimeter_summary_rejected_even_when_negated(self):
        # Strict policy: the ban is lexical, including disclaimers.
        body = json.dumps({
            "summary": "This is not a perimeter.",
            "evidence_used": [], "uncertainty": [], "limitations": []})
        verdict = evidence_validator.check_llm_json(body, {})
        assert verdict["valid"] is False
        assert "banned claim" in verdict["reason"]


class TestDetectionAgent:
    def test_fallback_without_providers(self):
        request = contracts.AgentRequest(
            "detections", "cl0000161", _detection_payload())
        response = detection_intel.analyze(
            request, providers.ProviderChain([]))
        assert response.status == "fallback"
        assert "cl0000161" in response.summary
        assert "500 satellite observations" in response.summary
        assert "2023-08-15" in response.summary
        assert response.evidence_used
        assert len(response.findings) == 4

    def test_ok_path_validates_and_prompts(self):
        provider = ScriptedProvider([_attempt(_valid_detection_json())])
        request = contracts.AgentRequest(
            "detections", "cl0000161", _detection_payload())
        response = detection_intel.analyze(
            request, providers.ProviderChain([provider]))
        assert response.status == "ok"
        assert response.provider == "scripted"
        assert response.validation == "all checks passed"
        assert len(provider.calls) == 1
        prompt, system = provider.calls[0]
        assert "cl0000161" in prompt and "EVIDENCE" in prompt
        assert "Citable evidence values" in prompt
        assert system == detection_intel.SYSTEM

    def test_banned_claim_falls_back_clean(self):
        bad_summary = "The fire perimeter is contained."
        bad = json.dumps({
            "summary": bad_summary, "evidence_used": ["cl0000161"],
            "uncertainty": [], "limitations": []})
        request = contracts.AgentRequest(
            "detections", "cl0000161", _detection_payload())
        chain = providers.ProviderChain([ScriptedProvider([_attempt(bad)])])
        response = detection_intel.analyze(request, chain)
        assert response.status == "fallback"
        assert "banned claim" in response.fallback_reason
        assert bad_summary not in response.summary
        assert response.evidence_used == evidence_validator.evidence_ids(
            _detection_payload())
        assert response.provider == "scripted"  # last-attempted

    def test_validity_failover_to_next_provider(self):
        bad = json.dumps({"summary": "The perimeter is contained.",
                          "evidence_used": [], "uncertainty": [],
                          "limitations": []})
        first = ScriptedProvider([_attempt(bad, "first")], name="first")
        second = ScriptedProvider([_attempt(_valid_detection_json(),
                                            "second")], name="second")
        request = contracts.AgentRequest(
            "detections", "cl0000161", _detection_payload())
        response = detection_intel.analyze(
            request, providers.ProviderChain([first, second]))
        assert response.status == "ok"
        assert response.provider == "second"
        assert len(first.calls) == 1 and len(second.calls) == 1

    def test_wrong_kind_rejected(self):
        request = contracts.AgentRequest(
            "event", "x", {"kind": "event"})
        with pytest.raises(ValueError, match="cannot handle"):
            detection_intel.analyze(request, providers.ProviderChain([]))

    def test_malformed_timeline_defensive(self):
        payload = {"cluster_id": "c", "n_detections": 3,
                   "timeline": [{"n": 1}, "junk", {"day": "2023-01-01"}],
                   "satellites": "Aqua"}
        findings = detection_intel.analyze_findings(payload)
        # count + span + frp-availability; bad rows filtered, no sensor
        # finding (string satellites are not char-joined).
        assert len(findings) == 3
        assert "A, q, u, a" not in " ".join(f.finding for f in findings)


class TestEventAgent:
    def test_fallback_reuses_qwen_template(self):
        request = contracts.AgentRequest(
            "event", "ev-cl0000001-a1", _event_payload())
        response = event_analysis.analyze(
            request, providers.ProviderChain([]))
        assert response.status == "fallback"
        assert "ev-cl0000001-a1" in response.summary
        assert "Limitations:" in response.summary
        assert len(response.findings) == 3

    def test_ok_path(self):
        body = json.dumps({
            "summary": "Event ev-cl0000001-a1 is Active with 5 obs.",
            "evidence_used": ["ev-cl0000001-a1"],
            "uncertainty": ["u"], "limitations": ["l"]})
        request = contracts.AgentRequest(
            "event", "ev-cl0000001-a1", _event_payload())
        response = event_analysis.analyze(
            request, providers.ProviderChain([ScriptedProvider(
                [_attempt(body)])]))
        assert response.status == "ok"

    def test_alert_kind_fallback(self):
        request = contracts.AgentRequest("event", "alt-1",
                                         _alert_payload())
        response = event_analysis.analyze(
            request, providers.ProviderChain([]))
        assert response.status == "fallback"
        assert "alt-1" in response.summary
        assert len(response.findings) == 1

    def test_unknown_narrative_kind_raises(self):
        request = contracts.AgentRequest(
            "event", "x", {"kind": "bogus", "event_id": "x"})
        with pytest.raises(ValueError, match="cannot narrate"):
            event_analysis.analyze(request, providers.ProviderChain([]))


class TestScienceAgent:
    def test_rules_and_limitations(self):
        request = contracts.AgentRequest(
            "science", "frp-pilot", _science_payload())
        response = science_observer.analyze(
            request, providers.ProviderChain([]))
        assert response.status == "fallback"
        assert "-1.35%" in response.summary
        assert "not comparable" in response.summary
        assert "QA flags" in response.summary
        assert "det-frp-oop-v1" in response.summary
        assert "x1.2" in response.summary
        cautions = [f for f in response.findings
                    if f.severity == "caution"]
        assert len(cautions) == 2

    def test_zero_observed_no_percentage(self):
        payload = _science_payload()
        payload["observed_sum"] = 0.0
        request = contracts.AgentRequest("science", "x", payload)
        response = science_observer.analyze(
            request, providers.ProviderChain([]))
        assert "no percentage stated" in response.summary
        assert "nan%" not in response.summary.lower()

    def test_mistyped_shapes_defensive(self):
        payload = {"observed_sum": 10.0, "expected_sum": 9.0,
                   "by_sensor": ["x"], "inputs_missing": "QA",
                   "method_version": None}
        request = contracts.AgentRequest("science", "x", payload)
        response = science_observer.analyze(
            request, providers.ProviderChain([]))
        assert response.status == "fallback"
        assert "unstated" in response.summary


class TestValidator:
    @pytest.mark.parametrize("raw", ["not json", "[]", "123", '"x"',
                                     '{"summary": 1}'])
    def test_unparseable_and_non_object(self, raw):
        verdict = evidence_validator.check_llm_json(raw, {})
        assert verdict["valid"] is False

    def test_missing_keys_and_types(self):
        assert evidence_validator.check_llm_json(
            json.dumps({"summary": "s"}), {})["valid"] is False
        assert evidence_validator.check_llm_json(json.dumps({
            "summary": "s", "evidence_used": "x", "uncertainty": [],
            "limitations": []}), {})["valid"] is False

    def test_malformed_findings_rejected(self):
        assert evidence_validator.check_findings(None)["valid"] is True
        assert evidence_validator.check_findings(
            [contracts.Finding("r" * 201, "f")])["valid"] is False
        assert evidence_validator.check_findings(
            ["not-a-finding"])["valid"] is False

        def _boom():
            raise RuntimeError("gen")
            yield

        assert evidence_validator.check_findings(_boom())["valid"] is False

    def test_evidence_ids_fail_closed(self):
        assert evidence_validator.evidence_ids(None) == ()
        assert evidence_validator.evidence_ids("x") == ()

    def test_fence_edges_only(self):
        body = json.dumps({
            "summary": "mentions ```json blocks``` inline with 5 obs.",
            "evidence_used": [], "uncertainty": [], "limitations": []})
        verdict = evidence_validator.check_llm_json(
            "```json\n" + body + "\n```",
            {"detection_count": 5})
        assert verdict["valid"] is True
        assert "```json blocks```" in verdict["parsed"]["summary"]

    def test_prose_wrapped_json_repaired(self):
        body = json.dumps({
            "summary": "Cluster shows 5 obs.", "evidence_used": [],
            "uncertainty": [], "limitations": []})
        verdict = evidence_validator.check_llm_json(
            "Here is the analysis:\n" + body + "\nHope this helps.",
            {"detection_count": 5})
        assert verdict["valid"] is True

    def test_garbage_without_json_fails(self):
        verdict = evidence_validator.check_llm_json(
            "no braces at all", {})
        assert verdict["valid"] is False


class TestCoordinator:
    def test_unknown_kind_rejected(self):
        with pytest.raises(ValueError, match="unknown agent kind"):
            coordinator.analyze("bogus", "x", {})

    def test_none_payload_rejected_at_boundary(self):
        with pytest.raises(ValueError, match="payload must be"):
            coordinator.analyze("detections", "x", None)

    def test_routing_and_trace(self):
        result = coordinator.analyze(
            "detections", "cl0000161", _detection_payload(),
            chain=providers.ProviderChain([]))
        assert result.kind == "detections"
        assert len(result.steps) == 1
        assert result.steps[0].agent == "detection-intelligence"
        assert result.steps[0].status == "fallback"
        assert result.evidence_ids

    def test_worker_fault_redacted(self, monkeypatch):
        def _boom(request, chain):
            raise RuntimeError("store api_key=SECRET123 failed")

        monkeypatch.setattr(detection_intel, "analyze_findings", _boom)
        result = coordinator.analyze(
            "detections", "cl0000161", _detection_payload(),
            chain=providers.ProviderChain([]))
        assert result.steps[0].status == "error"
        blob = result.text + result.steps[0].fallback_reason
        assert "SECRET123" not in blob
        assert "no analysis is available" in result.text

    def test_audit_rows_written(self, tmp_path):
        from src.calendar.events import store as store_mod

        db = str(tmp_path / "audit.sqlite")
        conn = store_mod.connect(db)
        store_mod.init_schema(conn)
        result = coordinator.analyze(
            "event", "ev-1",
            {"kind": "event", "event_id": "ev-1",
             "lifecycle_state": "Active"},
            chain=providers.ProviderChain([]), conn=conn)
        rows = store_mod.list_qwen_log(conn, target_type="agent:event-analysis")
        conn.close()
        assert len(rows) == 1
        assert rows[0]["status"] == "fallback"
        assert result.audit_refs == (f"qwen_log:{rows[0]['log_id']}",)

    def test_audit_failure_isolated(self, tmp_path):
        from src.calendar.events import store as store_mod

        db = str(tmp_path / "audit.sqlite")
        conn = store_mod.connect(db)
        store_mod.init_schema(conn)
        conn.close()  # writes now fail; analysis must survive
        result = coordinator.analyze(
            "event", "ev-1",
            {"kind": "event", "event_id": "ev-1"},
            chain=providers.ProviderChain([]), conn=conn)
        assert result.steps[0].status == "fallback"
        assert result.audit_refs == ()

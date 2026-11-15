"""Phase 4 tests: baselines, anomalies, confidence (4B/4C/4D units + integration).

Synthetic fixtures test mathematics only; they never enter real datasets.
Real-data tests read the pinned 2D metrics (read-only).
"""

import pytest

from src.calendar.analytics import anomalies, baselines, confidence


def _entries(periods, values):
    """Hand-built ordered entries (caller guarantees chronological order)."""
    return [{"period": p, "slot": i, "value": v}
            for i, (p, v) in enumerate(zip(periods, values))]


def _daily_entries(values, start="2023-01-01"):
    from datetime import date, timedelta

    base = date.fromisoformat(start)
    return [{"period": (base + timedelta(days=i)).isoformat(),
             "slot": (base + timedelta(days=i)).toordinal(), "value": v}
            for i, v in enumerate(values)]


class TestGroupKeys:
    def test_month_keys(self):
        assert baselines.group_key("daily", "2023-08-15", "same_month") == (2023, "08")
        assert baselines.group_key("monthly", "2022-01", "same_month") == (2022, "01")

    def test_week_keys_iso_edge(self):
        # 2023-01-01 is ISO 2022-W52: daily and weekly keys must agree.
        assert baselines.group_key("daily", "2023-01-01", "same_week") == (2022, "52")
        assert baselines.group_key("weekly", "2022-W52", "same_week") == (2022, "52")

    def test_invalid_combos_rejected(self):
        with pytest.raises(ValueError, match="undefined for 'weekly'"):
            baselines.group_key("weekly", "2023-W01", "same_month")
        with pytest.raises(ValueError, match="undefined for 'monthly'"):
            baselines.group_key("monthly", "2023-01", "same_week")


class TestReferenceSets:
    def test_same_month_prior_cycles_only(self):
        entries = _entries(
            ["2021-08", "2022-08", "2023-07", "2023-08"],
            [10, 20, 99, 999])
        cfg = baselines.BaselineConfig(method="monthly_median", grain="monthly")
        refs = baselines.reference_set(entries, 3, cfg)
        assert [r["value"] for r in refs] == [10, 20]  # eval + other months out
        refs = baselines.reference_set(entries, 2, cfg)
        assert refs == []  # no prior July anywhere

    def test_same_week_prior_cycles_only(self):
        entries = _entries(["2021-W31", "2022-W31", "2023-W30", "2023-W31"],
                           [10, 20, 99, 999])
        cfg = baselines.BaselineConfig(method="seasonal", grain="weekly")
        refs = baselines.reference_set(entries, 3, cfg)
        assert [r["value"] for r in refs] == [10, 20]

    def test_trailing_window_exact_membership(self):
        entries = _daily_entries(list(range(40)))
        cfg = baselines.BaselineConfig(method="sensor_aware", grain="daily",
                                       trailing_k=10)
        refs = baselines.reference_set(entries, 20, cfg)
        assert [r["value"] for r in refs] == list(range(10, 20))
        refs = baselines.reference_set(entries, 5, cfg)
        assert [r["value"] for r in refs] == list(range(0, 5))  # short history

    def test_trailing_skips_gaps_without_zero_fill(self):
        full = _daily_entries([10.0] * 40)
        gapped = [e for i, e in enumerate(full) if i not in (30, 31)]
        cfg = baselines.BaselineConfig(method="sensor_aware", grain="daily",
                                       trailing_k=10)
        refs = baselines.reference_set(gapped, 37, cfg)  # eval = day 39
        assert len(refs) == 8  # 2 missing slots simply absent
        assert all(r["value"] == 10.0 for r in refs)

    def test_no_future_references_possible(self):
        entries = _daily_entries(list(range(30)))
        cfg = baselines.BaselineConfig(method="sensor_aware", grain="daily")
        for pos in (0, 7, 29):
            for ref in baselines.reference_set(entries, pos, cfg):
                assert ref["slot"] < entries[pos]["slot"]


class TestEligibility:
    def test_insufficient_history(self):
        cfg = baselines.BaselineConfig(method="sensor_aware", grain="daily")
        verdict = baselines.assess_eligibility(
            baselines.summarize([1.0, 2.0]), cfg)
        assert verdict["status"] == "insufficient-history"
        assert verdict["baseline_ok"] is False and verdict["z_ok"] is False

    def test_median_ok_z_needs_seven(self):
        cfg = baselines.BaselineConfig(method="sensor_aware", grain="daily")
        verdict = baselines.assess_eligibility(
            baselines.summarize([1.0, 2.0, 3.0, 4.0]), cfg)
        assert verdict["status"] == "provisional"
        assert verdict["baseline_ok"] is True and verdict["z_ok"] is False
        verdict = baselines.assess_eligibility(
            baselines.summarize([float(v) for v in range(7)]), cfg)
        assert verdict["status"] == "eligible"

    def test_zero_and_near_zero_variance_unstable(self):
        cfg = baselines.BaselineConfig(method="sensor_aware", grain="daily")
        verdict = baselines.assess_eligibility(
            baselines.summarize([5.0] * 10), cfg)
        assert verdict["status"] == "unstable"
        assert verdict["baseline_ok"] is True and verdict["z_ok"] is False
        tiny = [100.0 + i * 1e-12 for i in range(10)]
        verdict = baselines.assess_eligibility(baselines.summarize(tiny), cfg)
        assert verdict["status"] == "unstable"


class TestEvaluatePeriod:
    def test_monthly_median_two_priors_insufficient(self):
        entries = _entries(["2021-08", "2022-08", "2023-08"], [10, 20, 999])
        cfg = baselines.BaselineConfig(method="monthly_median", grain="monthly")
        out = baselines.evaluate_period(entries, 2, cfg, metric="detection",
                                        scope="raw", exec_id="e1")
        assert out["reference_n"] == 2
        assert out["reference_median"] == 15.0  # computed but gated
        assert out["baseline_value"] is None  # n=2 < 3: no baseline issued
        assert out["eligibility"] == "insufficient-history"

    def test_monthly_median_three_priors(self):
        entries = _entries(["2020-08", "2021-08", "2022-08", "2023-08"],
                           [10, 20, 30, 999])
        cfg = baselines.BaselineConfig(method="monthly_median", grain="monthly")
        out = baselines.evaluate_period(entries, 3, cfg, metric="detection",
                                        scope="raw", exec_id="e1")
        assert out["baseline_value"] == 20.0
        assert out["reference_n"] == 3
        assert out["reference_start"] == "2020-08"
        assert out["eligibility"] == "provisional"  # median ok, z needs 7
        assert out["z_valid"] is False

    def test_trailing_record_fields(self):
        entries = _daily_entries([float(v) for v in range(40)])
        cfg = baselines.BaselineConfig(method="sensor_aware", grain="daily")
        out = baselines.evaluate_period(entries, 20, cfg, metric="detection",
                                        scope="modis", exec_id="e9")
        assert out["baseline_value"] == 9.5  # median(0..19), short history
        assert out["reference_n"] == 20
        assert out["eligibility"] == "eligible"
        assert out["z_valid"] is True
        assert out["config"]["trailing_k"] == 30
        assert out["algo_version"] == "baselines_v1"
        assert out["exec_id"] == "e9"

    def test_future_mutation_cannot_leak(self):
        entries = _daily_entries([10.0] * 20 + [999999.0] * 20)
        cfg = baselines.BaselineConfig(method="sensor_aware", grain="daily")
        before = baselines.evaluate_period(entries, 10, cfg, metric="detection",
                                           scope="raw")
        entries[30]["value"] = -5.0  # corrupt the future
        after = baselines.evaluate_period(entries, 10, cfg, metric="detection",
                                          scope="raw")
        assert before == after

    def test_determinism(self):
        entries = _daily_entries([float((i * 37) % 101) for i in range(60)])
        cfg = baselines.BaselineConfig(method="sensor_aware", grain="daily")
        first = [baselines.evaluate_period(entries, pos, cfg, metric="frp",
                                           scope="modis", exec_id="same")
                 for pos in range(len(entries))]
        second = [baselines.evaluate_period(entries, pos, cfg, metric="frp",
                                            scope="modis", exec_id="same")
                  for pos in range(len(entries))]
        assert first == second


class TestConfigValidation:
    def test_unknown_names_rejected(self):
        with pytest.raises(ValueError, match="unknown method"):
            baselines.BaselineConfig(method="nope", grain="daily").resolved_strategy()
        with pytest.raises(ValueError, match="unknown strategy"):
            baselines.BaselineConfig(method="sensor_aware", grain="daily",
                                     strategy="nope").resolved_strategy()
        with pytest.raises(ValueError, match="unknown grain"):
            baselines.BaselineConfig(method="sensor_aware",
                                     grain="hourly").resolved_k()
        with pytest.raises(ValueError, match="unsupported series"):
            baselines.extract_series({}, "daily", "detection", "NOAA-20")

    def test_run_all_rejects_unknown_method(self):
        with pytest.raises(ValueError, match="unknown method"):
            baselines.run_all_baselines({}, methods=("nope",))


def _baseline_record(**over):
    """Minimal baseline record for anomaly tests (eligible trailing)."""
    record = {
        "metric": "detection", "scope": "raw", "observed_period": "2023-08-15",
        "observed_value": 1800.0, "baseline_value": 1200.0,
        "baseline_method": "sensor_aware", "reference_strategy": "trailing",
        "reference_start": "2023-07-16", "reference_end": "2023-08-14",
        "reference_n": 30, "reference_median": 1200.0,
        "reference_mean": 1250.0, "reference_stdev": 200.0,
        "reference_coverage": 1.0, "geo_scope": "USA-pilot",
        "eligibility": "eligible", "eligibility_reasons": [], "z_valid": True,
        "algo_version": "baselines_v1", "config_version": "baseline_cfg_v1",
        "config": {"method": "sensor_aware", "grain": "daily",
                   "strategy": "trailing", "trailing_k": 30,
                   "median_min_n": 3, "zscore_min_n": 7},
        "exec_id": "e4b",
    }
    record.update(over)
    return record


class TestPctDeviation:
    def test_sign_and_magnitude(self):
        assert anomalies.pct_deviation(150.0, 100.0) == (50.0, "valid")
        assert anomalies.pct_deviation(75.0, 100.0) == (-25.0, "valid")
        assert anomalies.pct_deviation(0.0, 100.0) == (-100.0, "valid")

    def test_zero_and_missing_baseline(self):
        assert anomalies.pct_deviation(10.0, 0.0) == (None, "undefined-zero-baseline")
        assert anomalies.pct_deviation(0.0, 0.0) == (None, "undefined-zero-baseline")
        assert anomalies.pct_deviation(10.0, None) == (None, "missing-data")
        assert anomalies.pct_deviation(None, 100.0) == (None, "missing-data")


class TestZScore:
    def test_sign_and_value(self):
        assert anomalies.zscore(130.0, 100.0, 10.0, True) == (3.0, "valid")
        assert anomalies.zscore(85.0, 100.0, 10.0, True) == (-1.5, "valid")

    def test_gated_and_degenerate(self):
        assert anomalies.zscore(130.0, 100.0, 10.0, False) == (None, "unavailable")
        assert anomalies.zscore(130.0, 100.0, 0.0, True) == (None, "unstable-variance")
        assert anomalies.zscore(None, 100.0, 10.0, True) == (None, "missing-data")
        assert anomalies.zscore(130.0, None, 10.0, True) == (None, "missing-data")


class TestClassify:
    def _c(self, **over):
        kw = {"pct": 50.0, "pct_status": "valid", "z": 3.0, "z_status": "valid",
              "eligibility": "eligible", "observed": 150.0, "baseline": 100.0}
        kw.update(over)
        return anomalies.classify(**kw)

    def test_notable_and_typical(self):
        assert self._c()[0] == "notable-high"
        assert self._c(pct=-50.0, z=-3.0, observed=50.0)[0] == "notable-low"
        assert self._c(pct=10.0, z=1.0, observed=110.0)[0] == "typical"

    def test_conjunctive_rule(self):
        # Rare but tiny -> typical; big but common -> typical.
        assert self._c(pct=10.0, z=3.0, observed=110.0)[0] == "typical"
        assert self._c(pct=50.0, z=1.0)[0] == "typical"

    def test_exact_boundaries_count_as_notable(self):
        assert self._c(pct=25.0, z=2.0, observed=125.0)[0] == "notable-high"
        assert self._c(pct=24.9999, z=2.0, observed=125.0)[0] == "typical"

    def test_evidence_states(self):
        assert self._c(eligibility="insufficient-history")[0] == "insufficient-evidence"
        assert self._c(eligibility="unstable")[0] == "unstable"
        assert self._c(observed=None)[0] == "missing-data"
        cls, _ = self._c(pct=None, pct_status="undefined-zero-baseline",
                         z=None, z_status="unavailable",
                         eligibility="provisional", baseline=0.0)
        assert cls == "indeterminate"

    def test_provisional_and_z_only(self):
        cls, reasons = self._c(pct=60.0, z=None, z_status="unavailable",
                               eligibility="provisional")
        assert cls == "provisional-notable-high"
        assert any("median-only" in r for r in reasons)
        cls, _ = self._c(pct=10.0, z=None, z_status="unavailable",
                         eligibility="provisional", observed=110.0)
        assert cls == "provisional-typical"
        cls, reasons = self._c(pct=None, pct_status="undefined-zero-baseline",
                               z=2.5, observed=30.0, baseline=0.0)
        assert cls == "notable-high" and any("z-only" in r for r in reasons)


class TestDetectFromBaseline:
    def test_full_record_and_id(self):
        out = anomalies.detect_from_baseline(_baseline_record(), exec_id="e4c")
        assert out["anomaly_id"] == "anom-detection-raw-sensor_aware-daily-2023-08-15"
        assert out["exec_id"] == "e4c"
        assert out["pct_deviation"] == 50.0 and out["pct_status"] == "valid"
        assert out["z_score"] == 2.75 and out["z_status"] == "valid"
        assert out["classification"] == "notable-high"
        assert out["reference_n"] == 30
        assert out["algo_version"] == "anomalies_v1"
        assert any("trailing" in limit for limit in out["limitations"])

    def test_insufficient_baseline_propagates(self):
        record = _baseline_record(eligibility="insufficient-history",
                                  eligibility_reasons=["insufficient-history: n=2 < 3"],
                                  baseline_value=None, z_valid=False,
                                  reference_n=2)
        out = anomalies.detect_from_baseline(record)
        assert out["classification"] == "insufficient-evidence"
        assert out["pct_deviation"] is None and out["z_score"] is None

    def test_metric_independence(self):
        det = anomalies.detect_from_baseline(_baseline_record())
        clu = anomalies.detect_from_baseline(
            _baseline_record(metric="cluster", scope="count",
                             observed_value=10.0, baseline_value=400.0,
                             reference_mean=390.0, reference_stdev=20.0))
        assert det["classification"] == "notable-high"
        assert clu["classification"] == "notable-low"
        assert det["observed_value"] == 1800.0  # untouched by cluster inputs

    def test_unknown_scope_id_sanitized(self):
        out = anomalies.detect_from_baseline(
            _baseline_record(scope="terra/aqua"))
        assert "/" not in out["anomaly_id"]


def _anomaly(**over):
    """Anomaly record via the real detector (eligible trailing)."""
    base = _baseline_record()
    base.update(over.pop("baseline", {}))
    out = anomalies.detect_from_baseline(base, exec_id="e4c")
    out.update(over)
    return out


class TestConfidenceComponents:
    def test_sample_size_gates(self):
        full = confidence.components(_anomaly())
        assert (full["reference_n"], full["median_gate"], full["z_gate"],
                full["variance_state"]) == (30, "pass", "pass", "stable")
        small = confidence.components(
            _anomaly(reference_n=4, z_status="unavailable",
                     eligibility="provisional"))
        assert (small["median_gate"], small["z_gate"],
                small["variance_state"]) == ("pass", "fail", "not-enough-data")
        tiny = confidence.components(
            _anomaly(reference_n=1, z_status="unavailable",
                     eligibility="insufficient-history"))
        assert (tiny["median_gate"], tiny["z_gate"]) == ("fail", "fail")

    def test_unstable_and_missing_inputs(self):
        assert confidence.components(
            _anomaly(eligibility="unstable",
                     z_status="unstable-variance"))["variance_state"] == "unstable"
        bare = confidence.components({})
        assert bare["reference_n"] == 0 and bare["median_gate"] == "fail"
        assert bare["sensor_note"] == "scope note unavailable"

    def test_sensor_and_metric_notes(self):
        assert "no cross-sensor borrowing" in confidence.components(
            _anomaly(scope="modis"))["sensor_note"]
        assert "2E definition" in confidence.components(
            _anomaly(scope="harmonized"))["sensor_note"]
        assert "not total emitted energy" in confidence.components(
            _anomaly(metric="frp"))["metric_caveat"]
        assert "not individual wildfires" in confidence.components(
            _anomaly(metric="cluster"))["metric_caveat"]

    def test_composite_refuses_and_computes(self):
        comp = confidence.components(_anomaly())
        with pytest.raises(confidence.UncalibratedConfidence,
                           match="no approved weights"):
            confidence.composite_confidence(comp)
        with pytest.raises(confidence.UncalibratedConfidence, match="sum to 1"):
            confidence.composite_confidence(
                comp, weights={"w_n": 1.0, "w_cov": 1.0, "w_var": 1.0})
        out = confidence.composite_confidence(
            comp, weights={"w_n": 0.5, "w_cov": 0.25, "w_var": 0.25})
        assert out["composite"] == 1.0  # n=30, coverage=1, stable
        assert out["weights"] == {"w_n": 0.5, "w_cov": 0.25, "w_var": 0.25}


class TestExplain:
    def test_covers_required_points(self):
        text = confidence.explain(_anomaly())
        for needle in ("detection (raw)", "2023-08-15", "sensor_aware",
                       "trailing", "1,800.00", "1,200.00", "50.00%",
                       "2.75", "notable-high", "30 historical observations",
                       "single-cycle trailing reference"):
            assert needle in text, needle

    def test_provisional_and_invalid_stats(self):
        text = confidence.explain(_anomaly(
            baseline={"eligibility": "provisional", "z_valid": False,
                      "reference_n": 5, "reference_stdev": 150.0}))
        assert "Provisional result (provisional)" in text
        assert "Z-score: unavailable" in text

    def test_no_probability_language(self):
        text = confidence.explain(_anomaly()).lower()
        assert "not the probability of a wildfire" in text
        for banned in ("probability of wildfire occurring",
                       "fire danger level", "predicted fire",
                       "chance of ignition"):
            assert banned not in text, banned

    def test_deterministic(self):
        record = _anomaly()
        assert confidence.explain(record) == confidence.explain(record)
        assert confidence.components(record) == confidence.components(record)

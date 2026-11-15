"""2-day FRP pilot tests (isolated synthetic fixtures + read-only checks).

Covers: determinism, immutability, missing/zero/negative FRP, invalid
geometry/sensor/daynight, probability + expectation bounds, lookup
consistency, sensor separation, date filtering, no double counting,
no database modification. Real-data checks open calendar.duckdb
read-only and hash it before/after.
"""

import hashlib
import math
import statistics
from pathlib import Path

import duckdb
import pytest

from src.calendar.experimental.frp_operator_pilot import (
    coefficients, lookup, operator, results, validation)

REPO = Path(__file__).resolve().parent.parent


def _row(**over):
    base = {"detection_id": "d1", "product": "modis", "daynight": "D",
            "frp": 10.0, "scan": 1.0, "track": 1.0}
    base.update(over)
    return base


class TestDeterminism:
    def test_identical_outputs(self):
        first = operator.expected_frp(10.0, 1.5, "modis", "D")
        second = operator.expected_frp(10.0, 1.5, "modis", "D")
        assert first == second

    def test_lookup_deterministic(self):
        table = lookup.build_lookup()
        again = lookup.build_lookup()
        assert table["ratios"] == again["ratios"]
        assert lookup.lookup(table, "modis", "D", 1.1, 10.0) == \
            lookup.lookup(table, "modis", "D", 1.1, 10.0)


class TestInputs:
    def test_missing_frp_excluded_never_zero(self):
        for bad in (None, float("nan")):
            verdict = validation.validate_row(_row(frp=bad))
            assert verdict["eligible"] is False
            assert any("missing" in r for r in verdict["reasons"])
        with pytest.raises(ValueError, match="missing FRP"):
            operator.detection_probability(None, 1.0, "modis", "D")

    def test_zero_frp_maps_to_zero(self):
        assert operator.detection_probability(0.0, 1.0, "modis", "D") == 0.0
        assert operator.expected_frp(0.0, 1.0, "modis", "D")["E_ret"] == 0.0
        assert validation.validate_row(_row(frp=0.0))["eligible"] is True

    def test_negative_frp_invalid(self):
        assert validation.validate_row(
            _row(frp=-5.0))["eligible"] is False

    def test_invalid_geometry_excluded(self):
        for scan, track in [(None, 1.0), (1.0, None), (0.0, 1.0),
                            (-1.0, 1.0), (float("nan"), 1.0)]:
            verdict = validation.validate_row(_row(scan=scan, track=track))
            assert verdict["eligible"] is False, (scan, track)
        with pytest.raises(ValueError, match="pixel area"):
            operator.pixel_area_km2(0.0, 1.0)

    def test_missing_identity_excluded(self):
        assert validation.validate_row(
            _row(product="bogus"))["eligible"] is False
        assert validation.validate_row(
            _row(daynight="X"))["eligible"] is False
        with pytest.raises(ValueError, match="coefficients"):
            coefficients.get_params("bogus", "D")


class TestBounds:
    @pytest.mark.parametrize("product", ["modis", "viirs-snpp"])
    @pytest.mark.parametrize("daynight", ["D", "N"])
    def test_probability_bounds(self, product, daynight):
        for frp in (0.0, 0.1, 1.0, 10.0, 100.0, 1000.0):
            for area in (0.1, 0.3, 1.0, 2.0, 9.6):
                prob = operator.detection_probability(
                    frp, area, product, daynight)
                assert 0.0 <= prob <= 1.0

    def test_probability_monotone_in_frp(self):
        probs = [operator.detection_probability(f, 1.5, "modis", "D")
                 for f in (0.5, 5.0, 50.0, 500.0)]
        assert probs == sorted(probs)

    def test_expected_bounded_by_input(self):
        for frp in (0.5, 10.0, 500.0):
            out = operator.expected_frp(frp, 2.0, "viirs-snpp", "N")
            assert 0.0 <= out["E_ret"] <= frp
            assert out["Dp"] > 0 and out["r"] > 0

    def test_lookup_no_extrapolation(self):
        table = lookup.build_lookup()
        assert lookup.lookup(table, "modis", "D", 0.01, 10.0) is None
        assert lookup.lookup(table, "modis", "D", 1.1, 100000.0) is None
        assert lookup.lookup(table, "bogus", "D", 1.1, 10.0) is None
        assert lookup.lookup(table, "modis", "D", 1.1, 0.0) is None


class TestLookupConsistency:
    def test_lookup_matches_direct_computation(self):
        table = lookup.build_lookup()
        # At exact grid points the lookup equals p(P) (interpolation exact).
        for product, daynight, area, pref in [
                ("modis", "D", 1.1, 10.0), ("viirs-snpp", "N", 0.25, 5.0)]:
            edges = table["area_bins"][product]
            mid = None
            for i in range(len(edges) - 1):
                if edges[i] <= area < edges[i + 1]:
                    mid = (edges[i] + edges[i + 1]) / 2.0
            direct = operator.detection_probability(
                pref, mid, product, daynight)
            assert lookup.lookup(table, product, daynight, area,
                                 pref) == pytest.approx(direct, rel=1e-9)

    def test_table_metadata_complete(self):
        table = lookup.build_lookup()
        for key in ("version", "method", "source", "units", "assumptions",
                    "generated_utc", "validation_status", "area_bins",
                    "frp_grid", "ratios"):
            assert key in table
        assert table["validation_status"] == "exploratory-unvalidated"


class TestSeparation:
    def test_sensor_separation(self):
        modis = operator.detection_probability(10.0, 1.0, "modis", "D")
        viirs = operator.detection_probability(10.0, 1.0, "viirs-snpp", "D")
        assert modis != viirs  # distinct published coefficients

    def test_daynight_separation(self):
        day = operator.detection_probability(10.0, 1.0, "modis", "D")
        night = operator.detection_probability(10.0, 1.0, "modis", "N")
        assert day != night


class TestRealDataGuards:
    def test_two_day_scope_and_counts(self):
        con = duckdb.connect(str(REPO / "calendar.duckdb"), read_only=True)
        try:
            days = con.execute(
                "SELECT DISTINCT substr(acquisition_timestamp_utc,1,10) "
                "FROM detections WHERE substr(acquisition_timestamp_utc,1,10)"
                " IN ('2023-08-15','2023-08-16')").fetchall()
            n = con.execute(
                "SELECT COUNT(*) FROM detections WHERE "
                "substr(acquisition_timestamp_utc,1,10) IN "
                "('2023-08-15','2023-08-16')").fetchone()[0]
        finally:
            con.close()
        assert sorted(d[0] for d in days) == ["2023-08-15", "2023-08-16"]
        assert n == 4783

    def test_no_double_counting_keys_unique(self):
        con = duckdb.connect(str(REPO / "calendar.duckdb"), read_only=True)
        try:
            n, distinct = con.execute(
                "SELECT COUNT(*), COUNT(DISTINCT detection_id) FROM "
                "detections WHERE substr(acquisition_timestamp_utc,1,10) IN "
                "('2023-08-15','2023-08-16')").fetchone()
        finally:
            con.close()
        assert n == distinct == 4783

    def test_database_unmodified(self):
        target = REPO / "calendar.duckdb"
        before = hashlib.sha256(target.read_bytes()).hexdigest()
        con = duckdb.connect(str(target), read_only=True)
        try:
            con.execute("SELECT COUNT(*) FROM detections").fetchone()
        finally:
            con.close()
        after = hashlib.sha256(target.read_bytes()).hexdigest()
        assert before == after

    def test_geometry_distribution_anchors(self):
        con = duckdb.connect(str(REPO / "calendar.duckdb"), read_only=True)
        try:
            rows = con.execute(
                """SELECT product,
                 AVG(CASE WHEN product='modis' AND scan*track BETWEEN 1.0 AND 1.2 THEN 1.0
                          WHEN product='viirs-snpp' AND scan*track < 0.3 THEN 1.0 ELSE 0 END)
                 FROM detections
                 WHERE substr(acquisition_timestamp_utc,1,10) IN ('2023-08-15','2023-08-16')
                 GROUP BY 1""").fetchall()
        finally:
            con.close()
        # Subset-tolerant bands around published global anchors (0.50 / 0.90).
        fracs = dict(rows)
        assert 0.15 <= fracs["modis"] <= 0.65, fracs
        assert 0.50 <= fracs["viirs-snpp"] <= 0.95, fracs


class TestCrossMapping:
    def test_deterministic(self):
        first = operator.cross_map_expected(10.0, 1.5, "viirs-snpp",
                                            "modis", "D")
        second = operator.cross_map_expected(10.0, 1.5, "viirs-snpp",
                                             "modis", "D")
        assert first == second

    def test_labeled_return(self):
        out = operator.cross_map_expected(10.0, 1.5, "viirs-snpp",
                                          "modis", "D")
        assert out["method"] == "det-frp-oop-v1-xmap"
        assert (out["from_product"], out["to_product"]) == (
            "viirs-snpp", "modis")
        assert out["validation_status"] == "exploratory-unvalidated"
        joined = " ".join(out["assumptions"])
        assert "area transfer unvalidated" in joined
        assert "no diurnal correction" in joined

    def test_hand_computed_oracle(self):
        # MODIS-D, A=1.5: Dp=4.44*1.5+0.52=7.18; r=0.07+1.26/1.5=0.91;
        # z=(10-7.18)*0.91=2.5662; sigmoid=0.928654334904;
        # p=(0.928654334904-0.045)/0.955=0.925292497282.
        out = operator.cross_map_expected(10.0, 1.5, "viirs-snpp",
                                          "modis", "D")
        assert out["Dp"] == pytest.approx(7.18)
        assert out["r"] == pytest.approx(0.91)
        assert out["p"] == pytest.approx(0.925292497282)
        assert out["E_ret"] == pytest.approx(9.252924972820)

    def test_directions_differ(self):
        to_modis = operator.cross_map_expected(
            10.0, 1.5, "viirs-snpp", "modis", "D")["E_ret"]
        to_viirs = operator.cross_map_expected(
            10.0, 1.5, "modis", "viirs-snpp", "D")["E_ret"]
        assert to_modis != to_viirs  # sensor-specific operators

    def test_from_label_is_pinned_irrelevant(self):
        # Documented semantics: from_product labels direction only.
        first = operator.cross_map_expected(10.0, 1.5, "viirs-snpp",
                                            "modis", "D")
        second = operator.cross_map_expected(10.0, 1.5, "modis",
                                             "modis", "D")
        assert first["E_ret"] == second["E_ret"]
        assert first["p"] == second["p"]

    def test_bounds_and_monotonicity(self):
        prev = -1.0
        for pref in (0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 500.0):
            val = operator.cross_map_expected(
                pref, 1.5, "viirs-snpp", "modis", "D")["E_ret"]
            assert 0.0 <= val <= pref
            assert val >= prev
            prev = val

    def test_invalid_inputs_rejected(self):
        good = {"frp": 10.0, "area": 1.5, "daynight": "D"}
        with pytest.raises(ValueError, match="coefficients"):
            operator.cross_map_expected(10.0, 1.5, "bogus", "modis", "D")
        with pytest.raises(ValueError, match="coefficients"):
            operator.cross_map_expected(10.0, 1.5, "viirs-snpp",
                                        "bogus", "D")
        with pytest.raises(ValueError, match="coefficients"):
            operator.cross_map_expected(10.0, 1.5, "viirs-snpp",
                                        "modis", "X")
        for bad_frp, match in ((None, "missing FRP"),
                               (float("nan"), "missing FRP"),
                               (float("inf"), "non-finite"),
                               (-1.0, "negative FRP"),
                               ("10", "non-numeric")):
            with pytest.raises(ValueError, match=match):
                operator.cross_map_expected(bad_frp, good["area"],
                                            "viirs-snpp", "modis", "D")
        for bad_area in (None, 0.0, -1.0, float("nan"), float("inf"),
                         "1.5"):
            with pytest.raises(ValueError, match="area"):
                operator.cross_map_expected(good["frp"], bad_area,
                                            "viirs-snpp", "modis", "D")


class TestAODAttenuation:
    def test_monotone_and_labeled(self):
        import math

        out = operator.aod_attenuation(100.0, 0.5)
        assert out["P_attenuated"] == pytest.approx(100.0 * math.exp(-0.5))
        assert out["validation_status"] == "indicative-unvalidated"
        assert operator.aod_attenuation(100.0, 0.0)["P_attenuated"] == 100.0

    def test_invalid_inputs_rejected(self):
        with pytest.raises(ValueError, match="AOD"):
            operator.aod_attenuation(100.0, -0.1)
        with pytest.raises(ValueError, match="AOD"):
            operator.aod_attenuation(100.0, float("nan"))
        with pytest.raises(ValueError, match="FRP"):
            operator.aod_attenuation(None, 0.5)


class TestDetectionRegime:
    def test_above_limit_far_above(self):
        regime = operator.detection_regime(50.0, 1.0, "modis", "N")
        assert regime["regime"] == "above-limit"
        assert regime["margin_MW"] == pytest.approx(50.0 - 5.44)
        assert regime["Dp"] == pytest.approx(5.44)

    def test_below_limit_far_below(self):
        regime = operator.detection_regime(0.5, 1.0, "modis", "N")
        assert regime["regime"] == "below-limit"
        assert regime["margin_MW"] < 0

    def test_no_fire_for_exact_zero(self):
        regime = operator.detection_regime(0.0, 1.0, "modis", "N")
        assert regime["regime"] == "no-fire"

    def test_transition_near_limit(self):
        regime = operator.detection_regime(0.5, 0.2, "viirs-snpp", "N")
        assert regime["regime"] == "transition"
        assert regime["Dp"] == pytest.approx(0.626)
        assert regime["half_width_MW"] == pytest.approx(2.0 / 7.22)

    def test_boundary_pins_transition(self):
        # Exact band edges stay inside 'transition' (strict inequalities).
        dp = 4.43 * 1.0 + 1.01
        width = 2.0 / (0.11 + 1.06 / 1.0)
        for edge in (dp - width, dp + width):
            regime = operator.detection_regime(edge, 1.0, "modis", "N")
            assert regime["regime"] == "transition", edge

    def test_band_edge_probabilities_documented(self):
        # p(Dp±2/r) per coefficient row (convention meaning, pinned).
        cases = (("modis", "D", 1.5, 0.077699, 0.875180),
                 ("modis", "N", 1.5, 0.077699, 0.875180),
                 ("viirs-snpp", "D", 0.3, 0.072845, 0.874523),
                 ("viirs-snpp", "N", 0.3, 0.072845, 0.874523))
        for product, daynight, area, lo, hi in cases:
            dp = operator.detection_limit(area, product, daynight)
            width = 2.0 / operator.transition_rate(area, product,
                                                   daynight)
            assert operator.detection_probability(
                dp - width, area, product, daynight) == pytest.approx(
                    lo, abs=1e-6)
            assert operator.detection_probability(
                dp + width, area, product, daynight) == pytest.approx(
                    hi, abs=1e-6)

    def test_four_row_sweep(self):
        rows = (("modis", "D", 1.5), ("modis", "N", 2.0),
                ("viirs-snpp", "D", 0.3), ("viirs-snpp", "N", 0.2))
        order = ["no-fire", "below-limit", "transition", "above-limit"]
        for product, daynight, area in rows:
            assert operator.detection_limit(
                area, product, daynight) > 0
            assert 2.0 / operator.transition_rate(
                area, product, daynight) > 0
            seen = [operator.detection_regime(
                pref, area, product, daynight)["regime"]
                for pref in (0.0, 0.05, 2.0, 50.0, 2000.0)]
            assert seen[0] == "no-fire"
            ranks = [order.index(label) for label in seen[1:]]
            assert ranks == sorted(ranks), (product, daynight, seen)

    def test_assumptions_attached(self):
        regime = operator.detection_regime(50.0, 1.0, "modis", "N")
        assert "labeling convention" in " ".join(regime["assumptions"])

    def test_invalid_inputs_rejected(self):
        for bad_frp, match in ((None, "missing FRP"),
                               (float("nan"), "missing FRP"),
                               (float("inf"), "non-finite"),
                               (-1.0, "negative FRP")):
            with pytest.raises(ValueError, match=match):
                operator.detection_regime(bad_frp, 1.0, "modis", "N")
        for bad_area in (None, 0.0, -2.0, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="area"):
                operator.detection_regime(5.0, bad_area, "modis", "N")
        with pytest.raises(ValueError, match="coefficients"):
            operator.detection_regime(5.0, 1.0, "bogus", "N")


def _oor_row():
    return {"detection_id": "d1",
            "acquisition_timestamp_utc": "2023-08-15T06:00:00+00:00",
            "sensor": "MODIS", "satellite": "Aqua", "product": "modis",
            "latitude": 38.5, "longitude": -119.5, "frp": 10.0,
            "scan": 1.0, "track": 1.2, "daynight": "D"}


class TestResultRecords:
    def test_eligible_record_shape(self):
        row = _oor_row()
        area = operator.pixel_area_km2(row["scan"], row["track"])
        calc = operator.expected_frp(row["frp"], area, "modis", "D")
        regime = operator.detection_regime(row["frp"], area, "modis",
                                           "D")
        record = results.result_record(row, calc, regime, area_km2=area)
        assert record["calculation_status"] == "ok"
        assert record["observed_frp"] == 10.0
        assert record["observed_frp_units"] == "MW"
        assert record["observed_frp_status"] == "observed"
        assert record["expected_frp_status"] == "experimental"
        assert record["pixel_area_km2"] == area
        assert record["equation_version"].startswith("det-frp-oop-v1")
        assert record["result_version"] == "det-frp-result-v1"

    def test_unavailable_stays_null(self):
        row = _oor_row()
        area = operator.pixel_area_km2(row["scan"], row["track"])
        calc = operator.expected_frp(row["frp"], area, "modis", "D")
        regime = operator.detection_regime(row["frp"], area, "modis",
                                           "D")
        record = results.result_record(row, calc, regime, area_km2=area)
        assert record["aod_550"] is None
        assert record["aod_status"] == "unavailable"
        assert record["cloud_flag"] is None
        assert record["cloud_status"] == "unavailable"
        assert "aod_550" not in record["inputs_present"]

    def test_excluded_record_labeled(self):
        row = _oor_row()
        row["frp"] = None
        record = results.result_record(
            row, None, None, reasons=("missing FRP",))
        assert record["calculation_status"].startswith("excluded")
        assert record["expected_frp_status"] == "unavailable"
        assert record["quality_flags"] == ["missing FRP"]

    def test_l2_extract_plugs_in(self):
        row = _oor_row()
        area = operator.pixel_area_km2(row["scan"], row["track"])
        calc = operator.expected_frp(row["frp"], area, "modis", "D")
        regime = operator.detection_regime(row["frp"], area, "modis",
                                           "D")
        record = results.result_record(
            row, calc, regime, area_km2=area,
            l2={"aod_550": 0.3, "aod_source": "unit-test",
                "cloud_flag": "clear", "cloud_source": "unit-test"})
        assert record["aod_550"] == 0.3
        assert record["aod_status"] == "observed"
        assert "aod_550" in record["inputs_present"]

    def test_aggregate_record(self):
        row = _oor_row()
        area = operator.pixel_area_km2(row["scan"], row["track"])
        calc = operator.expected_frp(row["frp"], area, "modis", "D")
        regime = operator.detection_regime(row["frp"], area, "modis",
                                           "D")
        good = results.result_record(row, calc, regime, area_km2=area)
        bad = results.result_record(_oor_row(), None, None,
                                    reasons=("missing FRP",))
        agg = results.aggregate_record([good, bad])
        assert (agg["n_observations"], agg["n_eligible"],
                agg["n_excluded"]) == (2, 1, 0 + 1)
        assert agg["by_sensor"]["MODIS"]["n"] == 1
        assert agg["by_sensor"]["MODIS"]["observed_sum_MW"] == 10.0
        assert agg["l2_coverage"] == {"with_aod": 0, "with_cloud": 0}
        assert agg["status"] == "experimental"

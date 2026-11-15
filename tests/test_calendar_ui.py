"""Phase 3 UI tests: pure helpers + pinned-output integration.

Streamlit is never invoked here (AppTest coverage lands in 3F); these tests
exercise the framework-free query/filter/HTML layer only.
"""

from datetime import date, datetime

import pytest

from src.calendar.ui import (analytics_view, aoi, calendar_view, filters,
                             map_view, query)
from src.calendar.ui import events_view


class TestPeriodMath:
    def test_daily_range_is_single_day(self):
        assert filters.period_to_range("daily", "2023-08-15") == ("2023-08-15", "2023-08-15")

    def test_weekly_range_is_mon_sun(self):
        assert filters.period_to_range("weekly", "2023-W31") == ("2023-07-31", "2023-08-06")

    def test_weekly_cross_year_edge(self):
        # 2023-01-01 is a Sunday: it belongs to ISO week 2022-W52.
        assert filters.iso_week_label(date(2023, 1, 1)) == "2022-W52"
        assert filters.period_to_range("weekly", "2022-W52") == ("2022-12-26", "2023-01-01")

    def test_monthly_range_leap_and_common(self):
        assert filters.period_to_range("monthly", "2023-02") == ("2023-02-01", "2023-02-28")
        assert filters.period_to_range("monthly", "2024-02") == ("2024-02-01", "2024-02-29")

    def test_shift_period_boundaries(self):
        assert filters.shift_period("daily", "2023-01-01", -1) == "2022-12-31"
        assert filters.shift_period("monthly", "2023-01", -1) == "2022-12"
        assert filters.shift_period("monthly", "2023-12", 1) == "2024-01"
        assert filters.shift_period("weekly", "2023-W01", -1) == "2022-W52"
        assert filters.shift_period("weekly", "2022-W52", 1) == "2023-W01"

    def test_period_overlap_boundaries(self):
        assert filters.period_overlaps_range(
            "daily", "2023-08-15", "2023-08-15", "2023-08-15") is True
        assert filters.period_overlaps_range(
            "daily", "2023-08-16", "2023-08-15", "2023-08-15") is False
        assert filters.period_overlaps_range(
            "weekly", "2023-W33", "2023-08-15", "2023-08-15") is True
        assert filters.period_overlaps_range(
            "weekly", "2023-W32", "2023-08-15", "2023-08-15") is False
        assert filters.period_overlaps_range(
            "monthly", "2023-08", "2023-08-15", "2023-08-16") is True
        assert filters.period_overlaps_range(
            "monthly", "2023-07", "2023-08-15", "2023-08-16") is False
        # inverted + malformed inputs never raise
        assert filters.period_overlaps_range(
            "daily", "2023-08-15", "2023-08-16", "2023-08-15") is True
        assert filters.period_overlaps_range(
            "daily", "junk", "2023-08-15", "2023-08-16") is False
        assert filters.period_overlaps_range(
            "daily", "2023-08-15", "junk", "2023-08-16") is False

    def test_events_in_range_real_and_empty(self):
        import sqlite3

        conn = sqlite3.connect("file:calendar_events.sqlite?mode=ro",
                               uri=True)
        conn.row_factory = sqlite3.Row
        try:
            options, truncated = events_view._events_in_range(
                conn, "2023-08-15", "2023-08-15", limit=5)
            assert len(options) == 5 and truncated is True
            assert all(item["id"].startswith("ev-") for item in options)
            assert "→" in options[0]["label"]
            empty, _ = events_view._events_in_range(
                conn, "2024-01-01", "2024-01-02")
            assert empty == []
        finally:
            conn.close()

    def test_sync_inputs_from_range(self):
        state = {filters.KEY_RANGE_START: "2023-08-15",
                 filters.KEY_RANGE_END: "2023-08-16"}
        assert filters.sync_inputs_from_range(
            state, start_key="a", end_key="b",
            sentinel_key="s") is True
        assert (state["a"], state["b"]) == ("2023-08-15", "2023-08-16")
        assert filters.sync_inputs_from_range(
            state, start_key="a", end_key="b",
            sentinel_key="s") is False  # unchanged: no write
        state[filters.KEY_RANGE_START] = "2023-09-01"
        state[filters.KEY_RANGE_END] = "2023-09-02"
        assert filters.sync_inputs_from_range(
            state, start_key="a", end_key="b",
            sentinel_key="s") is True
        assert (state["a"], state["b"]) == ("2023-09-01", "2023-09-02")
        state[filters.KEY_RANGE_START] = "junk"
        assert filters.sync_inputs_from_range(
            state, start_key="a", end_key="b",
            sentinel_key="s") is False
        assert state["a"] == "2023-09-01"  # preserved on garbage

    def test_malformed_periods_raise(self):
        with pytest.raises(ValueError, match="bad daily"):
            filters.period_to_range("daily", "2023-13-40")
        with pytest.raises(ValueError, match="bad weekly"):
            filters.period_to_range("weekly", "2023-W54")
        with pytest.raises(ValueError, match="bad weekly"):
            filters.period_to_range("weekly", "2023-08-01")
        with pytest.raises(ValueError, match="bad monthly"):
            filters.period_to_range("monthly", "2023-13")
        with pytest.raises(ValueError, match="unknown grain"):
            filters.period_to_range("fortnightly", "x")

    def test_month_weeks_layout(self):
        weeks = filters.month_weeks(2023, 8)  # Aug 1 2023 is a Tuesday
        assert weeks[0][0] is None and weeks[0][1] == date(2023, 8, 1)
        assert all(len(w) == 7 for w in weeks)
        assert weeks[-1][-1] is None  # Aug 31 is a Thursday

    def test_validate_catches_all_fields(self):
        bad = filters.FilterState(grain="hourly", period="junk",
                                  date_start="2023-05-01", date_end="2023-04-01",
                                  products=("modis", "landsat"),
                                  satellites=("Terra", "NOAA-20"), mode="events")
        errors = filters.validate(bad)
        assert any("grain" in e for e in errors)
        assert any("mode" in e for e in errors)
        assert any("landsat" in e for e in errors)
        assert any("NOAA-20" in e for e in errors)
        assert any("after date_end" in e for e in errors)
        assert any("period" in e.lower() or "daily" in e for e in errors)
        assert filters.validate(filters.default_filters()) == []


class TestCoverage:
    SPANS = {"modis-2023-USA": {"min": "2023-01-01", "max": "2023-12-31"},
             "viirs-snpp-2023-USA": {"min": "2023-01-01", "max": "2023-12-31"}}

    def test_inside_span_complete(self):
        out = query.coverage_for("2023-08-15", self.SPANS)
        assert out["status"] == "complete"
        assert all(out["products"].values())

    def test_outside_span_unavailable(self):
        assert query.coverage_for("2024-01-01", self.SPANS)["status"] == "unavailable"
        assert query.coverage_for("2022-12-31", self.SPANS)["status"] == "unavailable"

    def test_partial_span(self):
        spans = {"a": {"min": "2023-01-01", "max": "2023-06-30"},
                 "b": {"min": "2023-01-01", "max": "2023-12-31"}}
        out = query.coverage_for("2023-08-15", spans)
        assert out["status"] == "partial"
        assert out["products"] == {"a": False, "b": True}

    def test_catalog_spans_extraction(self):
        catalog = {"datasets": [
            {"dataset_id": "d1", "versions": [
                {"coverage": {"temporal": {"min": "2023-01-01T00:00:00+00:00",
                                           "max": "2023-12-31T23:00:00+00:00"}}}]},
            {"dataset_id": "d2", "versions": []},
        ]}
        assert query.catalog_spans(catalog) == {
            "d1": {"min": "2023-01-01", "max": "2023-12-31"}}


class TestMetricsAccess:
    @pytest.fixture(scope="class")
    def metrics(self):
        return query.load_metrics()  # pinned 2D output (integration)

    def test_daily_row_shape(self, metrics):
        row = query.detection_row(metrics, "daily", "2023-08-15")
        assert row["raw_observations"] > 0
        assert set(row["by_product"]) == {"modis", "viirs-snpp"}
        harm = query.harmonized_row(metrics, "daily", "2023-08-15")
        assert harm["harmonized_observations"] <= row["raw_observations"]

    def test_missing_period_is_none_not_zero(self, metrics):
        assert query.detection_row(metrics, "daily", "2024-01-01") is None
        assert query.harmonized_row(metrics, "monthly", "2024-01") is None
        assert query.cluster_row(metrics, "daily", "1999-01-01") is None
        assert query.frp_rows(metrics, "daily", "2024-01-01") == []

    def test_frp_rows_per_product(self, metrics):
        rows = query.frp_rows(metrics, "daily", "2023-08-15")
        assert sorted(r["product"] for r in rows) == ["modis", "viirs-snpp"]
        assert all(r["unit"] == "MW" for r in rows)

    def test_yearly_totals_reconcile(self, metrics):
        det = query.detection_row(metrics, "yearly", "2023")
        harm = query.harmonized_row(metrics, "yearly", "2023")
        clu = query.cluster_row(metrics, "yearly", "2023")
        assert det["raw_observations"] == 482672
        assert harm["harmonized_observations"] == 346802
        assert clu["clusters"] == 137640

    def test_unknown_grain_rejected(self, metrics):
        with pytest.raises(ValueError, match="unknown grain"):
            query.detection_row(metrics, "hourly", "x")

    def test_series_values(self, metrics):
        raw = calendar_view.series_value(metrics, "daily", "2023-08-15", "Detections (raw)")
        modis = calendar_view.series_value(metrics, "daily", "2023-08-15", "Detections MODIS")
        viirs = calendar_view.series_value(metrics, "daily", "2023-08-15", "Detections VIIRS")
        assert raw == modis + viirs
        assert calendar_view.series_value(metrics, "daily", "2024-01-01", "Clusters") is None
        with pytest.raises(ValueError, match="unknown metric"):
            calendar_view.series_value(metrics, "daily", "2023-08-15", "Anomaly z")


class TestCalendarHtml:
    def test_month_grid_days_and_selection(self):
        values = {f"2023-08-{d:02d}": d * 10 for d in range(1, 32)}
        html = calendar_view.month_grid_html(2023, 8, values, selected="2023-08-15")
        assert html.count("<td") >= 31
        assert "2023-08-15" in html and "3px solid" in html  # selected marked
        assert "Mon" in html and "Sun" in html

    def test_unavailable_cells_not_zero(self):
        values = {"2023-08-01": 5, "2023-08-02": None}
        html = calendar_view.month_grid_html(2023, 8, values)
        assert "n/a" in html and "Outside verified pilot coverage" in html

    def test_week_strip_and_year_grid(self):
        monday = date(2023, 7, 31)
        values = {(monday.fromordinal(monday.toordinal() + i)).isoformat(): i
                  for i in range(7)}
        html = calendar_view.week_strip_html(monday, values, selected="2023-08-01")
        assert "2023-08-01" in html and "3px solid" in html
        yearly = {f"2023-{m:02d}": m for m in range(1, 13)}
        html = calendar_view.year_grid_html(2023, yearly, selected="2023-08")
        assert "2023-08" in html and "Aug" in html

    def test_intensity_scale_monotonic_bounded(self):
        colors = calendar_view.INTENSITY_STEPS
        assert calendar_view.intensity_color(0, 100) == colors[0]
        assert calendar_view.intensity_color(-5, 100) == colors[0]
        assert calendar_view.intensity_color(100, 100) == colors[-1]
        assert calendar_view.intensity_color(999, 100) == colors[-1]
        assert calendar_view.intensity_color(10, 0) == colors[0]
        seq = [calendar_view.intensity_color(v, 100) for v in (1, 25, 50, 75, 100)]
        assert [colors.index(c) for c in seq] == sorted(colors.index(c) for c in seq)

    def test_methodology_note_exact(self):
        assert calendar_view.METHODOLOGY_NOTE == (
            "Satellite observations and analytical hotspot clusters are not "
            "equivalent to confirmed individual wildfires.")


class TestAnalyticsNotes:
    def test_frp_scope_note_exact(self):
        assert analytics_view.FRP_SCOPE_NOTE == (
            "FRP scopes are per-sensor sums of reported values; MODIS and "
            "VIIRS FRP are not directly comparable.")

    def test_trailing_reference_note_exact(self):
        assert analytics_view.TRAILING_REFERENCE_NOTE == (
            "Trailing reference: strictly precedes the evaluation period "
            "(no lookahead).")


class TestCoerceDay:
    def test_accepts_date_datetime_str(self):
        assert filters.coerce_day(date(2023, 8, 15)) == "2023-08-15"
        assert filters.coerce_day(datetime(2023, 8, 15, 12, 30)) == "2023-08-15"
        assert filters.coerce_day(" 2023-08-15 ") == "2023-08-15"

    def test_rejects_garbage(self):
        with pytest.raises(ValueError, match="bad date"):
            filters.coerce_day("15/08/2023")
        with pytest.raises(ValueError, match="bad date"):
            filters.coerce_day("")


class TestMapQueries:
    DAY = "2023-08-15"

    @pytest.fixture(scope="class")
    def metrics(self):
        return query.load_metrics()

    def _day_row(self, metrics):
        return next(r for r in metrics["daily"]["detection_activity"]["rows"]
                    if r["period"] == self.DAY)

    def test_count_matches_pinned_metrics(self, metrics):
        day = self._day_row(metrics)
        assert query.count_detections(date_start=self.DAY, date_end=self.DAY) == \
            day["raw_observations"]

    def test_product_and_satellite_split(self, metrics):
        day = self._day_row(metrics)
        modis = query.count_detections(date_start=self.DAY, date_end=self.DAY,
                                       products=("modis",))
        assert modis == day["by_product"]["modis"]
        terra = query.count_detections(date_start=self.DAY, date_end=self.DAY,
                                       satellites=("Terra",))
        aqua = query.count_detections(date_start=self.DAY, date_end=self.DAY,
                                      satellites=("Aqua",))
        assert terra + aqua == modis
        snpp = query.count_detections(date_start=self.DAY, date_end=self.DAY,
                                      satellites=("N",))
        assert snpp == day["by_product"]["viirs-snpp"]

    def test_empty_selection_matches_nothing(self):
        assert query.count_detections(date_start=self.DAY, date_end=self.DAY,
                                      products=()) == 0
        assert query.count_detections(date_start=self.DAY, date_end=self.DAY,
                                      satellites=()) == 0

    def test_limit_and_order(self):
        df = query.query_detections(date_start=self.DAY, date_end=self.DAY, limit=10)
        assert len(df) == 10
        assert df["frp"].is_monotonic_decreasing
        with pytest.raises(ValueError, match="limit"):
            query.query_detections(date_start=self.DAY, date_end=self.DAY, limit=0)
        with pytest.raises(ValueError, match="unknown detection columns"):
            query.query_detections(date_start=self.DAY, date_end=self.DAY,
                                   columns=("detection_id", "anomaly_z"))

    def test_bad_dates_rejected(self):
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            query.count_detections(date_start="yesterday", date_end=self.DAY)
        with pytest.raises(ValueError, match="after date_end"):
            query.count_detections(date_start="2023-08-16", date_end=self.DAY)

    def test_bbox_narrows(self):
        full = query.count_detections(date_start=self.DAY, date_end=self.DAY)
        cal = query.count_detections(date_start=self.DAY, date_end=self.DAY,
                                     bbox=(-124.48, 32.53, -114.13, 42.01))
        assert 0 < cal < full

    def test_confidence_filters(self):
        modis_all = query.count_detections(date_start=self.DAY, date_end=self.DAY,
                                           products=("modis",))
        modis_hi = query.count_detections(date_start=self.DAY, date_end=self.DAY,
                                          products=("modis",), modis_conf_min=80)
        assert 0 < modis_hi < modis_all
        # MODIS threshold must not touch VIIRS rows.
        both = query.count_detections(date_start=self.DAY, date_end=self.DAY,
                                      modis_conf_min=80)
        viirs = query.count_detections(date_start=self.DAY, date_end=self.DAY,
                                       products=("viirs-snpp",))
        assert both == modis_hi + viirs
        h_only = query.count_detections(date_start=self.DAY, date_end=self.DAY,
                                        products=("viirs-snpp",),
                                        viirs_conf=("h",))
        assert 0 < h_only < viirs

    def test_detection_roundtrip_and_unknown(self):
        df = query.query_detections(date_start=self.DAY, date_end=self.DAY, limit=3)
        det = query.detection_by_id(df.iloc[0]["detection_id"])
        assert det["detection_id"] == df.iloc[0]["detection_id"]
        assert det["satellite"] in ("Terra", "Aqua", "N")
        assert query.detection_by_id("no-such-id") is None

    def test_member_and_group_lookup(self):
        df = query.query_detections(date_start=self.DAY, date_end=self.DAY, limit=50)
        member = query.member_by_id(df.iloc[0]["detection_id"])
        assert member["classification"] in ("matched", "ambiguous",
                                            "MODIS-only", "VIIRS-only")
        assert query.member_by_id("no-such-id") is None
        matched = query.members_of(df["detection_id"].tolist())
        assert len(matched) == len(df)
        with_match = matched[matched["match_id"].notna()]
        if len(with_match):
            group = query.match_group(with_match.iloc[0]["match_id"])
            assert group["classification"] in ("matched", "ambiguous")
        assert query.match_group("mg9999999") is None
        assert query.members_of([]).empty
        with pytest.raises(ValueError, match="bounded"):
            query.members_of([f"id-{i}" for i in range(10001)])

    def test_cluster_count_matches_metrics(self, metrics):
        clu = next(r for r in metrics["daily"]["cluster_activity"]["rows"]
                   if r["period"] == self.DAY)
        assert query.count_clusters(date_start=self.DAY, date_end=self.DAY) == \
            clu["clusters"]

    def test_cluster_roundtrip_and_unknown(self):
        df = query.query_clusters(date_start=self.DAY, date_end=self.DAY, limit=5)
        assert len(df) == 5
        assert df["n_detections"].is_monotonic_decreasing
        clu = query.cluster_by_id(df.iloc[0]["cluster_id"])
        assert clu["cluster_id"] == df.iloc[0]["cluster_id"]
        assert query.cluster_by_id("cl9999999") is None


class TestMapHtml:
    def test_detection_popup_fields(self):
        det = {"detection_id": "d1", "satellite": "Terra", "instrument": "MODIS",
               "acquisition_timestamp_utc": "2023-08-15T01:00:00+00:00",
               "latitude": 39.7, "longitude": -120.0, "frp": 12.5,
               "confidence": "80", "daynight": "D"}
        html = map_view.detection_popup_html(det)
        assert "d1" in html and "Terra" in html and "12.5" in html

    def test_missing_fields_say_unavailable(self):
        html = map_view.detection_popup_html({"detection_id": "d9"})
        assert html.count("unavailable") >= 5
        assert map_view.fmt(float("nan")) == "unavailable"
        assert map_view.fmt(None) == "unavailable"

    def test_cluster_popup_and_large_warning(self):
        small = {"cluster_id": "cl1", "n_detections": 3, "n_modis": 1,
                 "n_viirs": 2, "first_observation_utc": "a",
                 "last_observation_utc": "b", "observation_span_h": 2.0,
                 "frp_sum": 30.0}
        html = map_view.cluster_popup_html(small)
        assert "cl1" in html and "fire perimeter" not in html.lower()
        big = dict(small, cluster_id="cl9", n_detections=3872)
        html = map_view.cluster_popup_html(big)
        assert "not a single incident" in html
        assert "wildfire" not in html.lower()

    def test_frp_color_bounds(self):
        assert map_view.frp_color(100, 50, 75) == map_view.FRP_HIGH
        assert map_view.frp_color(60, 50, 75) == map_view.FRP_MID
        assert map_view.frp_color(10, 50, 75) == map_view.FRP_LOW
        assert map_view.frp_color(None, 50, 75) == "#95a5a6"
        assert map_view.frp_color(float("nan"), 50, 75) == "#95a5a6"

    def test_cap_notice(self):
        assert "all 100" in map_view.cap_notice(100, 100, "detections")
        assert "5,000 of 48,212" in map_view.cap_notice(48212, 5000, "detections")

    def test_bounds_and_zoom_helpers(self):
        assert map_view.bounds_to_bbox({"_southWest": {"lat": 32, "lng": -124},
                                        "_northEast": {"lat": 42, "lng": -114}}) == (-124, 32, -114, 42)
        assert map_view.bounds_to_bbox([[32, -124], [42, -114]]) == (-124, 32, -114, 42)
        assert map_view.bounds_to_bbox(None) is None
        assert map_view.bounds_to_bbox({"junk": 1}) is None
        assert map_view.zoom_to_grid(None) == 1.0
        assert map_view.zoom_to_grid(3) == 2.0
        assert map_view.zoom_to_grid(10) == 0.25

    def test_density_grid(self):
        import pandas as pd

        df = pd.DataFrame({"longitude": [0.1, 0.2, 5.0], "latitude": [0.1, 0.2, 5.0],
                           "frp": [10.0, 20.0, None]})
        grid = map_view.density_grid(df, 1.0)
        assert len(grid) == 2
        assert grid["frp_sum"].sum() == 30.0
        assert grid["n"].sum() == 3
        assert map_view.density_grid(df.iloc[0:0], 1.0).empty

    def test_legend_honesty(self):
        html = map_view.legend_html("clusters", "")
        assert "Not Confirmed Wildfire Perimeters" in html
        assert "not a fire perimeter" in html
        html = map_view.legend_html("detections", "frp", 1.0)
        assert "not energy" in html


class TestMapBuilders:
    @pytest.fixture(scope="class")
    def det_frame(self):
        return query.query_detections(date_start="2023-08-15", date_end="2023-08-15",
                                      limit=20)

    @pytest.fixture(scope="class")
    def clu_frame(self):
        return query.query_clusters(date_start="2023-08-15", date_end="2023-08-15",
                                    limit=20)

    def test_detection_map_layers(self, det_frame):
        param = map_view.build_detection_map(det_frame)
        html = param.get_root().render()
        assert "MODIS Terra" in html and "MODIS Aqua" in html
        assert "VIIRS Suomi-NPP" in html and "Boundaries" in html
        assert det_frame.iloc[0]["detection_id"] in html  # tooltip carried

    def test_detection_map_empty_and_density(self, det_frame):
        param = map_view.build_detection_map(det_frame.iloc[0:0])
        param.get_root().render()  # must not raise
        param = map_view.build_detection_map(det_frame, color_mode="frp",
                                             show_density=True, grid_deg=1.0)
        html = param.get_root().render()
        assert "FRP density" in html  # layer present (° is JSON-escaped)
        assert "Display grid" in html  # cell popups present

    def test_cluster_map_layers(self, clu_frame):
        param = map_view.build_cluster_map(clu_frame)
        html = param.get_root().render()
        assert "Analytical Hotspot Clusters" in html
        assert "Observation extents (bbox)" in html
        assert clu_frame.iloc[0]["cluster_id"] in html
        assert "wildfire" not in html.lower().replace("not confirmed wildfire", "")

    def test_member_id_list_bounded(self):
        import json as _json

        clu = {"detection_ids": _json.dumps([f"id-{i}" for i in range(100)])}
        preview, total = map_view.member_id_list(clu)
        assert total == 100 and len(preview) == 50
        assert map_view.member_id_list({}) == ([], 0)
        assert map_view.member_id_list({"detection_ids": "junk"}) == ([], 0)


class TestSelectionSummary:
    ROW = {"raw_observations": 100, "by_product": {"modis": 30, "viirs-snpp": 70},
           "by_satellite": {"Terra": 12, "Aqua": 18, "N": 70}}

    def test_full_selection(self):
        out = calendar_view.selection_summary(self.ROW, ("modis", "viirs-snpp"),
                                              ("Terra", "Aqua", "N"))
        assert out["raw_in_selection"] == 100
        assert out["by_satellite"] == {"Terra": 12, "Aqua": 18, "N": 70}

    def test_partial_and_empty_selection(self):
        out = calendar_view.selection_summary(self.ROW, ("modis",), ("Terra",))
        assert out["raw_in_selection"] == 30
        assert out["by_product"] == {"modis": 30}
        assert out["by_satellite"] == {"Terra": 12}
        out = calendar_view.selection_summary(self.ROW, (), ())
        assert out["raw_in_selection"] == 0

    def test_missing_keys_default_zero(self):
        out = calendar_view.selection_summary({}, ("modis",), ("Terra",))
        assert out == {"raw_in_selection": 0, "by_product": {"modis": 0},
                       "by_satellite": {"Terra": 0}}


class TestAoI:
    def test_validate_bbox_ok_and_errors(self):
        assert aoi.validate_bbox((-124.48, 32.53, -114.13, 42.01)) == \
            (-124.48, 32.53, -114.13, 42.01)
        with pytest.raises(aoi.AoIError, match="must be < E"):
            aoi.validate_bbox((10.0, 32.0, -10.0, 42.0))
        with pytest.raises(aoi.AoIError, match="must be < N"):
            aoi.validate_bbox((-10.0, 42.0, 10.0, 32.0))
        with pytest.raises(aoi.AoIError, match="outside"):
            aoi.validate_bbox((-200.0, 32.0, 10.0, 42.0))
        with pytest.raises(aoi.AoIError, match="four numbers"):
            aoi.validate_bbox(("a", 1.0, 2.0, 3.0))

    def test_save_load_delete_roundtrip(self, tmp_path):
        store = str(tmp_path / "aois.json")
        assert aoi.load_aois(store) == {}
        aoi.save_aoi(store, "home", (-120.0, 34.0, -119.0, 35.0))
        loaded = aoi.load_aois(store)
        assert loaded["home"]["bbox"] == [-120.0, 34.0, -119.0, 35.0]
        assert loaded["home"]["crs"] == "EPSG:4326"
        with pytest.raises(aoi.AoIError, match="already exists"):
            aoi.save_aoi(store, "home", (-120.0, 34.0, -119.0, 35.0))
        with pytest.raises(aoi.AoIError, match="must not be empty"):
            aoi.save_aoi(store, "  ", (-120.0, 34.0, -119.0, 35.0))
        remaining = aoi.delete_aoi(store, "home")
        assert remaining == {} and aoi.load_aois(store) == {}
        with pytest.raises(aoi.AoIError, match="unknown AOI"):
            aoi.delete_aoi(store, "home")

    def test_corrupt_store_raises_loudly(self, tmp_path):
        store = tmp_path / "aois.json"
        store.write_text("{not json")
        with pytest.raises(aoi.AoIError, match="not valid JSON"):
            aoi.load_aois(str(store))
        store.write_text("[1, 2]")
        with pytest.raises(aoi.AoIError, match="object keyed by name"):
            aoi.load_aois(str(store))

    def test_resolve_modes(self):
        assert aoi.resolve_bbox(mode="full") is None
        cal = aoi.resolve_bbox(mode="region", region_id="california")
        assert cal == (-124.48, 32.53, -114.13, 42.01)
        with pytest.raises(aoi.AoIError, match="unknown region"):
            aoi.resolve_bbox(mode="region", region_id="atlantis")
        assert aoi.resolve_bbox(mode="custom", custom=(-9.0, 1.0, 9.0, 2.0)) == \
            (-9.0, 1.0, 9.0, 2.0)
        with pytest.raises(aoi.AoIError, match="no bbox provided"):
            aoi.resolve_bbox(mode="custom", custom=None)
        assert aoi.resolve_bbox(mode="viewport", viewport=(-9.0, 1.0, 9.0, 2.0)) == \
            (-9.0, 1.0, 9.0, 2.0)
        with pytest.raises(aoi.AoIError, match="no map bounds"):
            aoi.resolve_bbox(mode="viewport", viewport=None)
        assert aoi.resolve_bbox(mode="saved", saved={"bbox": [-9, 1, 9, 2]}) == \
            (-9.0, 1.0, 9.0, 2.0)
        with pytest.raises(aoi.AoIError, match="none is loaded"):
            aoi.resolve_bbox(mode="saved", saved=None)
        with pytest.raises(aoi.AoIError, match="unknown AOI mode"):
            aoi.resolve_bbox(mode="polygon")

    def test_validate_point_ranges(self):
        assert aoi.validate_point(40.59, -122.39) == (40.59, -122.39)
        with pytest.raises(aoi.AoIError, match="latitude"):
            aoi.validate_point(91.0, 0.0)
        with pytest.raises(aoi.AoIError, match="longitude"):
            aoi.validate_point(0.0, 200.0)
        with pytest.raises(aoi.AoIError, match="numbers"):
            aoi.validate_point("a", "b")

    def test_radius_bbox_math_and_limits(self):
        w, s, e, n = aoi.radius_bbox(40.0, -120.0, 111.32)
        assert (w, s, e, n) == (
            pytest.approx(-120.0 - 111.32 / (111.32 * 0.7660444431),
                          abs=0.01),
            pytest.approx(39.0), pytest.approx(
                -120.0 + 111.32 / (111.32 * 0.7660444431), abs=0.01),
            pytest.approx(41.0))
        with pytest.raises(aoi.AoIError, match="radius"):
            aoi.radius_bbox(40.0, -120.0, 0.1)
        with pytest.raises(aoi.AoIError, match="radius"):
            aoi.radius_bbox(40.0, -120.0, 5000.0)
        # polar clamp stays valid
        box = aoi.radius_bbox(89.5, 0.0, 100.0)
        assert box[3] == 90.0 and box[1] < box[3]

    def test_geocode_stubbed(self):
        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return [{"lat": "40.5", "lon": "-122.4",
                         "display_name": "Redding, CA"}]

        seen = {}

        def _get(url, params=None, headers=None, timeout=None):
            seen["ua"] = (headers or {}).get("User-Agent", "")
            seen["q"] = (params or {}).get("q")
            return _Resp()

        out = aoi.geocode_nominatim("Redding", transport=_get)
        assert (out["latitude"], out["longitude"]) == (40.5, -122.4)
        assert "FireAgent" in seen["ua"] and seen["q"] == "Redding"

        def _boom(url, params=None, headers=None, timeout=None):
            raise ConnectionError("offline")

        with pytest.raises(aoi.AoIError, match="unavailable"):
            aoi.geocode_nominatim("Redding", transport=_boom)
        with pytest.raises(aoi.AoIError, match="empty"):
            aoi.geocode_nominatim("   ")


class TestConfidenceGuard:
    DAY = "2023-08-15"

    def test_empty_viirs_selection_keeps_modis(self):
        both = query.count_detections(date_start=self.DAY, date_end=self.DAY,
                                      viirs_conf=())
        modis = query.count_detections(date_start=self.DAY, date_end=self.DAY,
                                       products=("modis",))
        assert both == modis > 0

    def test_none_means_no_filter(self):
        plain = query.count_detections(date_start=self.DAY, date_end=self.DAY)
        assert query.count_detections(date_start=self.DAY, date_end=self.DAY,
                                      modis_conf_min=None,
                                      viirs_conf=None) == plain


class TestMonthCover:
    def test_single_month(self):
        assert query.month_cover("2023-08-15", "2023-08-15") == [(2023, "08")]

    def test_cross_month(self):
        assert query.month_cover("2023-01-31", "2023-02-02") == [
            (2023, "01"), (2023, "02")]

    def test_cross_year(self):
        assert query.month_cover("2022-12-30", "2023-01-02") == [
            (2022, "12"), (2023, "01")]

    def test_full_year(self):
        cover = query.month_cover("2023-01-01", "2023-12-31")
        assert len(cover) == 12 and cover[0] == (2023, "01")


class TestPruningEquivalence:
    @pytest.fixture(scope="class")
    def metrics(self):
        return query.load_metrics()

    def _daily_sum(self, metrics, days):
        rows = {r["period"]: r["raw_observations"]
                for r in metrics["daily"]["detection_activity"]["rows"]}
        return sum(rows[d] for d in days)

    def test_cross_shard_range_matches_metrics(self, metrics):
        days = ["2023-01-31", "2023-02-01", "2023-02-02"]
        assert query.count_detections(date_start=days[0], date_end=days[-1]) == \
            self._daily_sum(metrics, days)

    def test_full_year_matches_metrics(self, metrics):
        assert query.count_detections(date_start="2023-01-01",
                                      date_end="2023-12-31") == 482672


class TestFingerprint:
    def test_stable_hashable_and_identifying(self):
        first, second = query.fingerprint(), query.fingerprint()
        assert first == second
        hash(first)  # cache-key safe
        blob = repr(first)
        assert "bfef579a" in blob and "7caf5845" in blob and "9741f859" in blob
        assert "calendar.duckdb" in blob


class TestUTCAndFrpSemantics:
    def test_temporal_basis_label(self):
        assert "UTC" in calendar_view.TEMPORAL_BASIS_LABEL
        assert "acquisition" in calendar_view.TEMPORAL_BASIS_LABEL

    def test_frp_missing_counts_surfaced(self):
        metrics = query.load_metrics()
        rows = query.frp_rows(metrics, "daily", "2023-08-15")
        assert all({"frp_sum", "n_contributing", "n_missing_frp", "unit"} <= set(r)
                   for r in rows)


class TestDetailCompleteness:
    def test_detection_detail_fields(self):
        df = query.query_detections(date_start="2023-08-15", date_end="2023-08-15",
                                    limit=1)
        det = query.detection_by_id(df.iloc[0]["detection_id"])
        required = {"detection_id", "sensor", "satellite", "instrument",
                    "acquisition_timestamp_utc", "latitude", "longitude",
                    "confidence", "frp", "daynight", "product",
                    "product_version", "source_file", "source_sha256"}
        assert required <= set(det)

    def test_cluster_detail_fields(self):
        df = query.query_clusters(date_start="2023-08-15", date_end="2023-08-15",
                                  limit=1)
        clu = query.cluster_by_id(df.iloc[0]["cluster_id"])
        required = {"cluster_id", "n_detections", "n_modis", "n_viirs",
                    "satellites", "first_observation_utc",
                    "last_observation_utc", "observation_span_h",
                    "centroid_lon", "centroid_lat", "lon_min", "lon_max",
                    "lat_min", "lat_max", "frp_sum", "frp_by_sensor",
                    "match_groups", "algo_version", "params_version", "exec_id"}
        assert required <= set(clu)


class TestDualViews:
    def test_index_consistent_with_metrics(self):
        metrics = query.load_metrics()
        index = query.load_index()
        yearly_det = query.detection_row(metrics, "yearly", "2023")
        yearly_harm = query.harmonized_row(metrics, "yearly", "2023")
        sensor = index["sensor_view"]
        assert sensor["modis"]["raw_observations"] == \
            yearly_det["by_product"]["modis"]
        assert sensor["viirs-snpp"]["raw_observations"] == \
            yearly_det["by_product"]["viirs-snpp"]
        assert index["harmonized_view"]["harmonized_observations"] == \
            yearly_harm["harmonized_observations"]
        assert index["reconciliation"]["passed"] is True

    def test_provenance_chain_exists(self):
        import glob as _glob
        import os as _os

        chain = query.exec_chain()
        assert len(chain) == 5
        expected = {"2B": ["match_groups.parquet", "match_members.parquet"],
                    "2C": ["clusters.parquet", "cluster_members.parquet"],
                    "2D": ["metrics_2d.json"], "2E": ["index_2e.json"]}
        for stage, files in expected.items():
            for name in files:
                paths = _glob.glob(f"analytics_output/{chain[stage]}/{name}")
                assert len(paths) == 1 and _os.path.getsize(paths[0]) > 0


class TestAnalyticsHelpers:
    @pytest.fixture(scope="class")
    def records(self):
        return query.load_anomalies()

    def test_series_selection_and_order(self, records):
        rows = analytics_view.series_for(records, "detection", "raw",
                                         "sensor_aware", "daily")
        assert len(rows) == 365
        assert [r["eval_period"] for r in rows] == sorted(r["eval_period"] for r in rows)
        assert all(r["baseline_method"] == "sensor_aware" for r in rows)
        assert analytics_view.series_for(records, "detection", "raw",
                                         "sensor_aware", "weekly") != []

    def test_in_range_overlap(self, records):
        rows = analytics_view.series_for(records, "detection", "raw",
                                         "sensor_aware", "daily")
        assert len(analytics_view.in_range(rows, "daily", "2023-08-01", "2023-08-31")) == 31
        weeks = analytics_view.series_for(records, "detection", "raw",
                                          "sensor_aware", "weekly")
        # 2022-W52 (Dec 26–Jan 1) overlaps a January start.
        clipped = analytics_view.in_range(weeks, "weekly", "2023-01-01", "2023-01-31")
        assert clipped[0]["eval_period"] == "2022-W52"

    def test_notables_and_counts(self, records):
        rows = analytics_view.series_for(records, "detection", "raw",
                                         "sensor_aware", "daily")
        noted = analytics_view.notables(rows)
        assert noted and all(n["classification"].startswith(
            ("notable", "provisional-notable")) for n in noted)
        counts = analytics_view.eligibility_counts(rows)
        assert sum(counts.values()) == len(rows)
        assert counts.get("insufficient-evidence", 0) == 3  # days 1..3 (n<3)

    def test_status_text_and_note(self):
        assert analytics_view.status_text("eligible") == "Valid baseline"
        assert "Insufficient" in analytics_view.status_text("insufficient-history")
        assert analytics_view.METHODOLOGY_NOTE == (
            "Statistical anomalies indicate unusual satellite observation "
            "activity relative to an eligible historical reference. They do "
            "not establish wildfire occurrence, fire severity, or future "
            "fire danger.")

    def test_scope_coverage_matches_engine(self):
        from src.calendar.analytics import baselines

        covered = {(m, s) for m, scopes in analytics_view.SCOPES_BY_METRIC.items()
                   for s in scopes}
        assert covered == set(baselines.SERIES_SPECS)


class TestDataIntegrity:
    def test_raw_archive_untouched(self):
        from src.calendar import settings
        from src.calendar.storage import manifest

        report = manifest.verify_tree(settings.RAW_ROOT)
        assert report["total"] == 2 and len(report["ok"]) == 2
        assert report["mismatch"] == [] and report["missing"] == []

    def test_normalized_inputs_still_valid(self):
        from src.calendar.analytics import discovery

        report = discovery.validate_inputs("catalog.json", "calendar.duckdb")
        assert report["overall"] == "pass"

    def test_no_duplicate_duckdb_tables(self):
        import duckdb

        con = duckdb.connect("calendar.duckdb", read_only=True)
        try:
            names = con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema NOT IN ('information_schema', 'pg_catalog')"
            ).fetchall()
        finally:
            con.close()
        assert sorted(n[0] for n in names) == ["detections"]


class TestCitations:
    def test_citation_block_complete(self):
        assert len(calendar_view.CITATIONS) == 3
        joined = " ".join(calendar_view.CITATIONS)
        for token in ("FIRMS", "MODIS Collection 6.1", "VIIRS",
                      "482,672", "Sofiev", "gmd-19-10123-2026"):
            assert token in joined, token
        assert "research preview only" in joined  # never production FRP

    def test_zero_na_legend_exact(self):
        assert calendar_view.ZERO_NA_LEGEND == (
            "0 means zero recorded observations inside verified coverage; "
            "n/a means outside verified pilot coverage (no data, not zero "
            "fire activity).")


class TestDetectionsOfCluster:
    def test_unknown_cluster_is_empty(self):
        df = query.detections_of_cluster("cl-does-not-exist")
        assert len(df) == 0

    def test_pinned_cluster_matches_record(self):
        clu = query.cluster_by_id("cl0000161")
        df = query.detections_of_cluster("cl0000161")
        assert len(df) == clu["n_detections"] == 3872
        assert df["detection_id"].is_unique
        assert list(df.columns) == list(query.CLUSTER_TIMELINE_COLUMNS)
        import pandas as pd

        days = pd.to_datetime(df["acquisition_timestamp_utc"], utc=True)
        assert days.is_monotonic_increasing  # deterministic order
        assert int(df["frp"].isna().sum()) == clu["frp_null"]

    def test_unknown_columns_rejected(self):
        with pytest.raises(ValueError, match="unknown detection columns"):
            query.detections_of_cluster("cl0000161",
                                        columns=("detection_id", "bogus"))


def _timeline_frame():
    import pandas as pd

    return pd.DataFrame([
        {"detection_id": "a",
         "acquisition_timestamp_utc": "2023-08-15T06:00:00+00:00",
         "product": "modis", "frp": 10.0},
        {"detection_id": "b",
         "acquisition_timestamp_utc": "2023-08-15T18:00:00+00:00",
         "product": "viirs-snpp", "frp": 5.0},
        {"detection_id": "c",
         "acquisition_timestamp_utc": "2023-08-15T20:00:00+00:00",
         "product": "modis", "frp": None},
        {"detection_id": "d",
         "acquisition_timestamp_utc": "2023-08-16T06:00:00+00:00",
         "product": "viirs-snpp", "frp": 0.0},
        {"detection_id": "e",
         "acquisition_timestamp_utc": "2023-08-16T07:00:00+00:00",
         "product": "modis", "frp": 20.0},
        {"detection_id": "f", "acquisition_timestamp_utc": None,
         "product": "modis", "frp": 1.0},
    ])


class TestClusterTimeline:
    def test_day_grouping_and_order(self):
        tl = map_view.cluster_timeline(_timeline_frame())
        assert tl["day"].tolist() == ["2023-08-15", "2023-08-16"]
        assert tl["n"].tolist() == [3, 2]  # undated row excluded
        assert tl["n_modis"].tolist() == [2, 1]
        assert tl["n_viirs"].tolist() == [1, 1]

    def test_frp_missing_never_zero_filled(self):
        tl = map_view.cluster_timeline(_timeline_frame())
        row15 = tl[tl["day"] == "2023-08-15"].iloc[0]
        assert row15["frp_modis"] == 10.0  # None excluded, not zero
        assert row15["frp_viirs"] == 5.0
        assert row15["frp_missing"] == 1
        row16 = tl[tl["day"] == "2023-08-16"].iloc[0]
        assert row16["frp_viirs"] == 0.0  # real zero contributes
        assert row16["frp_missing"] == 0

    def test_empty_frame_is_empty(self):
        import pandas as pd

        tl = map_view.cluster_timeline(pd.DataFrame(
            columns=["acquisition_timestamp_utc", "product", "frp"]))
        assert len(tl) == 0


class TestPlaybackSlice:
    def test_exact_day_slice(self):
        part = map_view.playback_slice(_timeline_frame(), "2023-08-15")
        assert sorted(part["detection_id"]) == ["a", "b", "c"]

    def test_cumulative_slice(self):
        part = map_view.playback_slice(_timeline_frame(), "2023-08-16",
                                       cumulative=True)
        assert sorted(part["detection_id"]) == ["a", "b", "c", "d", "e"]

    def test_empty_day_is_empty(self):
        part = map_view.playback_slice(_timeline_frame(), "2023-08-17")
        assert len(part) == 0
        assert list(part.columns) == list(_timeline_frame().columns)


def _built_html(builder, frame):
    import pandas as pd

    if frame is None:
        frame = pd.DataFrame()
    return builder(frame).get_root().render()


class TestMapTilesAndGraticule:
    def test_default_providers_keyless_with_attribution(self):
        import pandas as pd

        html = map_view.build_detection_map(
            pd.DataFrame(columns=["latitude", "longitude", "frp",
                                  "detection_id", "satellite",
                                  "instrument"])).get_root().render()
        for name in ("OpenStreetMap", "Esri WorldImagery"):
            assert name in html, name
        for token in ("tile.openstreetmap.org",
                      "server.arcgisonline.com", "OpenStreetMap",
                      "Esri"):
            assert token in html, token
        # CARTO excluded: keyless endpoint returns API-KEY watermarks
        # (pixel-verified) and key presence alone proved insufficient.
        assert "basemaps.cartocdn.com" not in html
        assert "CartoDB Voyager" not in html

    def test_carto_gated_off_by_default(self):
        names = [n for n, _, _ in map_view.select_tile_providers()]
        assert names[0] == "OpenStreetMap"
        assert "CartoDB Voyager" not in names

    def test_carto_enabled_with_key(self, monkeypatch):
        from src.calendar import settings

        monkeypatch.setattr(settings, "CAL_CARTO_ENABLED", "true")
        monkeypatch.setattr(settings, "CARTO_API_KEY", "DUMMY-KEY")
        providers = map_view.select_tile_providers()
        carto = [p for p in providers if p[0] == "CartoDB Voyager"]
        assert len(carto) == 1
        assert "api_key=DUMMY-KEY" in carto[0][1]
        assert "CARTO" in carto[0][2]

    def test_carto_enabled_without_key_excluded(self, monkeypatch):
        from src.calendar import settings

        monkeypatch.setattr(settings, "CAL_CARTO_ENABLED", "true")
        monkeypatch.setattr(settings, "CARTO_API_KEY", "")
        names = [n for n, _, _ in map_view.select_tile_providers()]
        assert "CartoDB Voyager" not in names

    def test_tile_source_line_names_default(self):
        line = map_view.tile_source_line()
        assert line.startswith("Basemap: OpenStreetMap")
        assert "Esri WorldImagery" in line

    def test_graticule_line_counts(self):
        layer = map_view.graticule_layer((0.0, 0.0, 10.0, 5.0), 5.0)
        lines = [c for c in layer._children.values()
                 if c.__class__.__name__ == "PolyLine"]
        assert len(lines) == 5  # meridians 0/5/10 + parallels 0/5

    def test_graticule_step_adapts_to_span(self):
        assert map_view.graticule_step(120.0) == 10.0
        assert map_view.graticule_step(30.0) == 5.0
        assert map_view.graticule_step(8.0) == 2.0
        assert map_view.graticule_step(2.0) == 1.0

    def test_cluster_map_has_fallback_layers(self):
        import pandas as pd

        html = map_view.build_cluster_map(
            pd.DataFrame(columns=["centroid_lat", "centroid_lon",
                                  "lat_min", "lon_min", "lat_max",
                                  "lon_max", "cluster_id",
                                  "n_detections"])).get_root().render()
        assert "Graticule (offline reference grid)" in html
        assert "Boundaries (approx.)" in html

    def test_layer_names_unique_no_duplicates(self):
        import pandas as pd

        frame = pd.DataFrame(columns=["latitude", "longitude", "frp",
                                      "detection_id", "satellite",
                                      "instrument"])
        first = map_view.build_detection_map(frame).get_root().render()
        second = map_view.build_detection_map(frame).get_root().render()
        for html in (first, second):
            assert html.count("tile.openstreetmap.org") == 1
            assert html.count("server.arcgisonline.com") == 1
            assert html.count("Graticule (offline reference grid)") == 1

    def test_offline_note_exact(self):
        assert map_view.TILE_OFFLINE_NOTE == (
            "Basemap tiles need network access. If the background stays "
            "white, enable Graticule (offline reference grid) in the "
            "layer control; points and overlays render offline.")

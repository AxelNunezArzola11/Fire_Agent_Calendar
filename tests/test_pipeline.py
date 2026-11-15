"""Pipeline tests: real-data 2-day run, empty provider chain.

Read-only stores (ro sqlite conn); no network (fallback agents).
"""

import sqlite3

import pytest

from src.calendar.agents.providers import ProviderChain
from src.calendar.pipeline import exports_csv, runner

REPO = "calendar_events.sqlite"


@pytest.fixture(scope="module")
def ro_conn():
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    conn = sqlite3.connect(f"file:{root / REPO}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def two_day(ro_conn):
    seen = []
    result = runner.run_pipeline(
        "2023-08-15", "2023-08-16", events_conn=ro_conn,
        chain=ProviderChain([]),
        progress=lambda n, s, d: seen.append((n, s)))
    result.progress_seen = seen
    return result


class TestPipelineRun:
    def test_stages_complete(self, two_day):
        assert [s.name for s in two_day.stages] == [
            "acquisition", "detection", "events", "operator",
            "validation", "export", "metrics"]
        assert [s.status for s in two_day.stages] == ["ok"] * 7
        assert two_day.acquisition["n_detections"] == 4783

    def test_progress_covers_stages(self, two_day):
        names = {n for n, _ in two_day.progress_seen}
        assert {"acquisition", "detection", "events", "operator",
                "validation", "export", "metrics"} <= names

    def test_agent_runs_traced(self, two_day):
        assert len(two_day.agent_runs) == 7  # 3 det + 3 ev + 1 sci
        assert all(r["status"] == "fallback" for r in two_day.agent_runs)
        assert all("EVIDENCE" not in (r["summary"] or "")
                   for r in two_day.agent_runs)

    def test_operator_coverage(self, two_day):
        agg = two_day.operator["aggregate"]
        assert (agg["n_observations"], agg["n_eligible"]) == (4783, 4783)
        assert two_day.operator["truncated"] is False

    def test_validation_passes(self, two_day):
        assert two_day.validation["failed"] == 0
        assert two_day.validation["passed"] == 8

    def test_exports_built(self, two_day):
        assert set(two_day.csvs) == {"observations", "events",
                                     "agents", "metrics"}
        assert two_day.csvs["observations"].split("\n")[0].split(
            ",")[0] == "observation_id"
        assert len(two_day.csvs["observations"].splitlines()) == 4784
        assert "execution_id" in two_day.csvs["agents"]

    def test_execution_id_deterministic(self, two_day, ro_conn):
        again = runner.run_pipeline(
            "2023-08-15", "2023-08-16", events_conn=ro_conn,
            chain=ProviderChain([]))
        assert again.execution_id == two_day.execution_id

    def test_csv_reproducible_modulo_timestamp(self, two_day, ro_conn):
        again = runner.run_pipeline(
            "2023-08-15", "2023-08-16", events_conn=ro_conn,
            chain=ProviderChain([]))

        def _strip(text):
            import csv as _csv
            import io as _io

            rows = list(_csv.reader(_io.StringIO(text)))
            head = rows[0]
            if "processing_utc" in head:
                cut = head.index("processing_utc")
                rows = [r[:cut] + r[cut + 1:] for r in rows]
            return rows

        assert _strip(again.csvs["observations"]) == _strip(
            two_day.csvs["observations"])


class TestPipelineContracts:
    def test_inverted_range_rejected(self, ro_conn):
        with pytest.raises(ValueError, match="Start Date"):
            runner.run_pipeline("2023-08-16", "2023-08-15",
                                events_conn=ro_conn,
                                chain=ProviderChain([]))

    def test_bad_date_rejected(self, ro_conn):
        with pytest.raises(ValueError, match="invalid date range"):
            runner.run_pipeline("junk", "2023-08-15",
                                events_conn=ro_conn,
                                chain=ProviderChain([]))

    def test_bbox_restricts_run(self, ro_conn):
        result = runner.run_pipeline(
            "2023-08-15", "2023-08-16", events_conn=ro_conn,
            chain=ProviderChain([]),
            bbox=(-124.5, 32.5, -114.0, 42.0))
        assert result.config["bbox"] == [-124.5, 32.5, -114.0, 42.0]
        assert result.acquisition["n_detections"] < 4783
        assert result.acquisition["n_detections"] > 0
        assert all(s.status == "ok" for s in result.stages)

    def test_out_of_coverage_explained(self, ro_conn):
        result = runner.run_pipeline(
            "2024-01-01", "2024-01-02", events_conn=ro_conn,
            chain=ProviderChain([]))
        assert result.stages[0].status == "failed"
        assert "outside verified pilot coverage" in \
            result.stages[0].detail
        assert all(s.status == "skipped" for s in result.stages[1:])

    def test_schema_version(self):
        assert exports_csv.SCHEMA_VERSION == \
            "fireagent-pipeline-csv-v1"

    def test_csv_contracts(self, two_day):
        import csv as _csv
        import io as _io

        from src.calendar.pipeline import exports_csv

        assert exports_csv.SCHEMA_VERSION == \
            "fireagent-pipeline-csv-v1"
        obs = list(_csv.DictReader(
            _io.StringIO(two_day.csvs["observations"])))
        assert list(obs[0].keys()) == \
            list(exports_csv.OBSERVATION_COLUMNS)
        assert all(r["observed_frp_units"] == "MW" for r in obs)
        assert all(r["aod_550"] == "" for r in obs)  # nulls, not zeros
        assert all(r["equation_version"].startswith("det-frp-oop-v1")
                   for r in obs)
        assert all(r["execution_id"] == two_day.execution_id
                   for r in obs)
        assert "2B bfef579a" in obs[0]["provenance"]
        agents = list(_csv.DictReader(
            _io.StringIO(two_day.csvs["agents"])))
        assert list(agents[0].keys()) == \
            list(exports_csv.AGENT_COLUMNS)
        assert all(r["fallback"] == "True" for r in agents)
        events = list(_csv.DictReader(
            _io.StringIO(two_day.csvs["events"])))
        assert list(events[0].keys()) == \
            list(exports_csv.EVENT_COLUMNS)
        metrics = list(_csv.DictReader(
            _io.StringIO(two_day.csvs["metrics"])))
        names = {r["metric"] for r in metrics}
        assert {"total_latency_ms", "n_detections",
                "agent_fallbacks"} <= names

    def test_frontend_csv_parity(self, two_day):
        import csv as _csv
        import io as _io

        from src.calendar.ui import frames

        shown = frames.operator_sensor_frame(
            two_day.operator["aggregate"]["by_sensor"])
        assert frames.check_arrow_compatible(shown, "operator") == []
        rows = list(_csv.DictReader(
            _io.StringIO(two_day.csvs["observations"])))
        for _, ui_row in shown.iterrows():
            sensor = ui_row["sensor"]
            csv_rows = [r for r in rows
                        if r["sensor"] == sensor
                        and r["calculation_status"] == "ok"]
            assert ui_row["n"] == len(csv_rows)
            assert ui_row["observed_sum_MW"] == \
                pytest.approx(sum(float(r["observed_frp_MW"])
                                  for r in csv_rows))
            assert ui_row["expected_sum_MW"] == \
                pytest.approx(sum(float(r["expected_retrieved_frp_MW"])
                                  for r in csv_rows))

    def test_l2_matcher_filters_to_viirs(self):
        inventory = [{"granule_ur": "G1",
                      "time_start": "2023-08-15T06:00:00Z",
                      "time_end": "2023-08-15T06:06:00Z",
                      "bbox": [-121.0, 37.0, -119.0, 39.0]}]
        matcher = runner.build_l2_matcher(inventory)
        assert matcher({
            "product": "modis", "latitude": 38.0,
            "longitude": -120.0,
            "acquisition_timestamp_utc":
            "2023-08-15T06:03:00+00:00"}) == (None, "unmatched")
        assert matcher({
            "product": "viirs-snpp", "latitude": 38.0,
            "longitude": -120.0,
            "acquisition_timestamp_utc":
            "2023-08-15T06:03:00+00:00"}) == ("G1", "matched")

"""Phase 1D tests: normalization, regions, Parquet, DuckDB, catalog, quality.

Fixtures are tiny SYNTHETIC frames built inline (never real observations,
never written to project data dirs — tmp_path only).
"""

import pandas as pd
import pytest

from src.calendar.processing import catalog, duckdb_layer, normalize, parquet, quality, regions


@pytest.fixture()
def raw_frame():
    return pd.DataFrame([
        {"latitude": "39.7", "longitude": "-120.0", "acq_date": "2023-08-20",
         "acq_time": "0130", "satellite": "Terra", "instrument": "MODIS",
         "confidence": "85", "frp": "12.5", "daynight": "N", "scan": "1.0",
         "track": "1.0", "version": "6.1", "brightness": "310.2", "type": "0"},
        {"latitude": "91.0", "longitude": "-120.0", "acq_date": "2023-08-20",
         "acq_time": "0130", "satellite": "Terra", "instrument": "MODIS",
         "confidence": "85", "frp": "12.5", "daynight": "N", "scan": "1.0",
         "track": "1.0", "version": "6.1", "brightness": "310.2", "type": "0"},
        {"latitude": "34.0", "longitude": "-118.0", "acq_date": "not-a-date",
         "acq_time": "0130", "satellite": "Aqua", "instrument": "MODIS",
         "confidence": "h", "frp": "bad", "daynight": "D", "scan": "x",
         "track": "y", "version": "6.1", "brightness": "300", "type": "0"},
        {"latitude": "34.0", "longitude": "-118.2", "acq_date": "2023-08-21",
         "acq_time": "130", "satellite": "Aqua", "instrument": "MODIS",
         "confidence": "", "frp": "", "daynight": "D", "scan": "", "track": "",
         "version": "", "brightness": "", "type": ""},
    ])


def _norm(raw_frame):
    return normalize.normalize_frame(
        raw_frame, product="modis", sensor="terra_aqua",
        source_file="fixture.csv", source_sha256="ab" * 32,
        processing_version="test_v1")


class TestNormalize:
    def test_accept_reject_split_with_reasons(self, raw_frame):
        batch = _norm(raw_frame)
        assert batch.source_rows == 4
        assert len(batch.accepted) == 2  # rows 0 and 3
        assert len(batch.rejected) == 2
        reasons = " ".join(r["reason"] for r in batch.rejected)
        assert "out of range" in reasons and "notadate" in reasons.replace("-", "")

    def test_utc_and_originals_preserved(self, raw_frame):
        batch = _norm(raw_frame)
        first = batch.accepted.iloc[0]
        assert first["acquisition_timestamp_utc"] == "2023-08-20T01:30:00+00:00"
        assert first["original_acquisition_timestamp"] == "2023-08-20 0130"
        assert first["src_brightness"] == "310.2" and first["src_type"] == "0"
        assert first["confidence"] == "85" and first["frp"] == 12.5

    def test_short_time_padded_and_empties_nulled(self, raw_frame):
        batch = _norm(raw_frame)
        last = batch.accepted.iloc[1]
        assert last["acquisition_timestamp_utc"] == "2023-08-21T01:30:00+00:00"
        assert last["confidence"] == "" and last["product_version"] == ""
        assert pd.isna(last["frp"]) and pd.isna(last["scan"])

    def test_detection_id_stable_and_unique(self, raw_frame):
        first = _norm(raw_frame).accepted
        second = _norm(raw_frame).accepted
        assert first["detection_id"].tolist() == second["detection_id"].tolist()
        assert first["detection_id"].nunique() == 2
        assert all(len(v) == 32 for v in first["detection_id"])

    def test_missing_required_column_raises(self):
        with pytest.raises(ValueError, match="required columns"):
            normalize.normalize_frame(pd.DataFrame([{"latitude": "1"}]),
                                      product="p", sensor="s", source_file="f",
                                      source_sha256="x", processing_version="v")


class TestRegions:
    def test_california_filter_keeps_stats(self, raw_frame):
        kept, stats = regions.filter_region(
            _norm(raw_frame).accepted.assign(
                latitude=[39.7, 34.0], longitude=[-120.0, -118.2]), "california")
        assert stats["kept_rows"] == 2 and stats["region_version"] == "cal_bbox_v1"
        outside = pd.DataFrame({"latitude": [51.5], "longitude": [0.0]})
        kept, stats = regions.filter_region(outside, "california")
        assert stats["kept_rows"] == 0
        with pytest.raises(KeyError):
            regions.filter_region(outside, "atlantis")


class TestParquet:
    def test_roundtrip_and_idempotent_rewrite(self, tmp_path, raw_frame):
        df = _norm(raw_frame).accepted
        root = str(tmp_path / "pq")
        first = parquet.write_dataset(df, root=root, product="modis",
                                      sensor="terra_aqua", processing_version="v1")
        assert len(first["written"]) == 1 and first["written"][0].endswith(".parquet")
        second = parquet.write_dataset(df, root=root, product="modis",
                                       sensor="terra_aqua", processing_version="v1")
        assert second["written"] == [] and len(second["skipped"]) == 1
        back = parquet.read_dataset(root, product="modis", processing_version="v1")
        assert len(back) == len(df)
        assert set(back["detection_id"]) == set(df["detection_id"])

    def test_empty_frame_writes_nothing(self, tmp_path):
        rep = parquet.write_dataset(pd.DataFrame(), root=str(tmp_path),
                                    product="p", sensor="s", processing_version="v")
        assert rep == {"written": [], "skipped": [], "files": []}


class TestDuckDB:
    @pytest.fixture()
    def db(self, tmp_path, raw_frame):
        df = _norm(raw_frame).accepted
        root = str(tmp_path / "pq")
        parquet.write_dataset(df, root=root, product="modis", sensor="terra_aqua",
                              processing_version="v1")
        db_path = str(tmp_path / "cal.duckdb")
        duckdb_layer.init_db(db_path, root)
        return db_path

    def test_discovery_count_filters(self, db):
        assert duckdb_layer.datasets(db) == [
            {"product": "modis", "sensor": "terra_aqua", "rows": 2}]
        assert duckdb_layer.count(db) == 2
        assert duckdb_layer.count(db, sensor="nope") == 0
        assert duckdb_layer.count(db, start="2023-08-21T00:00:00+00:00") == 1
        assert duckdb_layer.count(db, bbox=(-121, 39, -119, 40)) == 1
        rows = duckdb_layer.fetch(db, limit=10)
        assert len(rows) == 2

    def test_init_idempotent_and_recoverable(self, tmp_path, db):
        import os

        duckdb_layer.init_db(db, str(tmp_path / "pq"))  # recreate over existing
        assert duckdb_layer.count(db) == 2
        os.remove(db)  # recovery = rebuild from Parquet (the system of record)
        duckdb_layer.init_db(db, str(tmp_path / "pq"))
        assert duckdb_layer.count(db) == 2

    def test_cross_product_select_star_with_divergent_columns(self, tmp_path):
        """Regression (Phase 2B): per-product src_* cols must not break SELECT *."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        base = tmp_path / "pq" / "v1"
        left = pd.DataFrame({
            "detection_id": ["a"], "acquisition_timestamp_utc": ["2023-01-01T00:00:00+00:00"],
            "latitude": [0.0], "longitude": [0.0], "satellite": ["Terra"],
            "instrument": ["MODIS"], "src_brightness": ["300.0"]})
        right = pd.DataFrame({
            "detection_id": ["b"], "acquisition_timestamp_utc": ["2023-01-01T00:00:00+00:00"],
            "latitude": [0.0], "longitude": [0.0], "satellite": ["N"],
            "instrument": ["VIIRS"], "src_bright_ti4": ["290.0"]})
        for product, frame in (("modis", left), ("viirs-snpp", right)):
            out_dir = base / f"product={product}" / "sensor=s" / "year=2023" / "month=01"
            out_dir.mkdir(parents=True)
            pq.write_table(pa.Table.from_pandas(frame, preserve_index=False),
                           out_dir / "part-x.parquet")
        db_path = str(tmp_path / "u.duckdb")
        duckdb_layer.init_db(db_path, str(tmp_path / "pq"))
        rows = duckdb_layer.fetch(db_path, limit=10)
        assert len(rows) == 2
        assert rows["src_brightness"].isna().sum() == 1  # NULL where absent
        assert rows["src_bright_ti4"].isna().sum() == 1


class TestCatalog:
    def test_lineage_and_version_history(self, tmp_path):
        cat = str(tmp_path / "catalog.json")
        catalog.record_dataset(
            cat, dataset_id="modis-2023-USA",
            raw_files=[{"path": "r.csv", "sha256": "a" * 64, "size_bytes": 10}],
            processing_version="v1", parquet_files=["p1.parquet"],
            validation_status="valid", record_counts={"accepted": 2},
            coverage={"dates": 1}, quality={"null_frp": 0})
        catalog.record_dataset(
            cat, dataset_id="modis-2023-USA",
            raw_files=[{"path": "r.csv", "sha256": "a" * 64, "size_bytes": 10}],
            processing_version="v2", parquet_files=["p2.parquet"],
            validation_status="valid", record_counts={"accepted": 2},
            coverage={"dates": 1}, quality={"null_frp": 0})
        latest = catalog.latest_version(cat, "modis-2023-USA")
        assert latest["processing_version"] == "v2"
        entry = catalog.load_catalog(cat)["datasets"][0]
        assert [v["processing_version"] for v in entry["versions"]] == ["v1", "v2"]
        assert catalog.latest_version(cat, "unknown") is None


class TestQuality:
    def test_counters_and_empty_handling(self, raw_frame):
        batch = _norm(raw_frame)
        rep = quality.quality_report(source_rows=batch.source_rows,
                                     accepted=batch.accepted, rejected=batch.rejected)
        assert (rep["source_rows"], rep["accepted_rows"], rep["rejected_rows"]) == (4, 2, 2)
        assert rep["null_frp"] == 1 and rep["null_confidence"] == 1
        assert rep["temporal_coverage"]["min"].startswith("2023-08-20")
        empty = quality.quality_report(source_rows=0, accepted=pd.DataFrame(), rejected=[])
        assert empty["accepted_rows"] == 0 and empty["temporal_coverage"] is None
        assert "note" in empty  # missing input reported, never zero-activity

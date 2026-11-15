"""Phase 2 tests: analytical engine. Synthetic fixtures in tmp only."""

import pandas as pd
import pytest

from src.calendar.analytics import clustering, discovery, index, matching, metrics, runs
from src.calendar.processing import catalog as catalog_mod
from src.calendar.processing import duckdb_layer, parquet


def _frame(rows, product="modis", sensor="terra_aqua"):
    base = {
        "detection_id": [], "sensor": [], "satellite": [], "instrument": [],
        "product": [], "product_version": [], "acquisition_timestamp_utc": [],
        "original_acquisition_timestamp": [], "latitude": [], "longitude": [],
        "confidence": [], "frp": [], "daynight": [], "scan": [], "track": [],
        "source_file": [], "source_sha256": [], "ingestion_timestamp_utc": [],
        "processing_version": [],
    }
    for i, r in enumerate(rows):
        base["detection_id"].append(r.get("detection_id", f"id-{i:04d}"))
        base["sensor"].append(sensor)
        base["satellite"].append(r.get("satellite", "Terra"))
        base["instrument"].append("MODIS" if product == "modis" else "VIIRS")
        base["product"].append(product)
        base["product_version"].append("6.1")
        base["acquisition_timestamp_utc"].append(r["ts"])
        base["original_acquisition_timestamp"].append(r["ts"])
        base["latitude"].append(r["lat"])
        base["longitude"].append(r["lon"])
        base["confidence"].append(r.get("confidence", "80"))
        base["frp"].append(r.get("frp", 10.0))
        base["daynight"].append(r.get("daynight", "D"))
        base["scan"].append(1.0)
        base["track"].append(1.0)
        base["source_file"].append("fixture.csv")
        base["source_sha256"].append("ab" * 32)
        base["ingestion_timestamp_utc"].append("2026-01-01T00:00:00+00:00")
        base["processing_version"].append("test_v1")
    return pd.DataFrame(base)


@pytest.fixture()
def mini_store(tmp_path):
    """Two tiny synthetic datasets (modis + viirs) with catalog + DuckDB."""
    root, cat, db = (str(tmp_path / "pq"), str(tmp_path / "catalog.json"),
                     str(tmp_path / "cal.duckdb"))
    frames = {
        ("modis", "terra_aqua"): _frame([
            {"ts": "2023-08-20T01:30:00+00:00", "lat": 39.7, "lon": -120.0},
            {"ts": "2023-08-20T02:30:00+00:00", "lat": 39.8, "lon": -120.1},
        ]),
        ("viirs-snpp", "snpp"): _frame([
            {"ts": "2023-08-20T01:35:00+00:00", "lat": 39.71, "lon": -120.01,
             "satellite": "N"},
        ], product="viirs-snpp", sensor="snpp"),
    }
    for (product, sensor), df in frames.items():
        written = parquet.write_dataset(df, root=root, product=product,
                                        sensor=sensor, processing_version="test_v1")
        catalog_mod.record_dataset(
            cat, dataset_id=f"{product}-test",
            raw_files=[{"path": str(tmp_path / "raw.csv"), "sha256": "ab" * 32,
                        "size_bytes": 10}],
            processing_version="test_v1", parquet_files=written["files"],
            validation_status="valid",
            record_counts={"source": len(df), "accepted": len(df), "rejected": 0},
            coverage={}, quality={})
    (tmp_path / "raw.csv").write_text("fixture\n")  # lineage target exists
    duckdb_layer.init_db(db, root)
    return {"root": root, "catalog": cat, "db": db}


class TestRuns:
    def test_open_complete_load_get(self, tmp_path):
        runs_path = str(tmp_path / "runs.jsonl")
        exec_id = runs.new_execution(runs_path, stage="2test",
                                     inputs={"a": 1}, params={"p": 2},
                                     region="california",
                                     date_range=("2023-01-01", "2023-12-31"))
        runs.complete_execution(runs_path, exec_id, {"out": [1]})
        records = runs.load_runs(runs_path)
        assert len(records) == 2  # open + completion; history never rewritten
        assert records[0]["inputs"] == {"a": 1}
        assert records[1]["outputs"] == {"out": [1]}
        assert len(runs.get_run(runs_path, exec_id)) == 2
        assert runs.load_runs(str(tmp_path / "missing.jsonl")) == []


class TestDiscovery2A:
    def test_discovers_both_sensors_with_agreement(self, mini_store):
        datasets = discovery.discover_inputs(mini_store["catalog"], mini_store["db"])
        assert sorted(d["dataset_id"] for d in datasets) == ["modis-test", "viirs-snpp-test"]
        modis = next(d for d in datasets if d["product"] == "modis")
        assert modis["sensor"] == "terra_aqua" and modis["live_rows"] == 2
        assert modis["counts_agree"] is True and modis["raw_file_present"] is True

    def test_validation_passes_clean(self, mini_store):
        report = discovery.validate_inputs(mini_store["catalog"], mini_store["db"])
        assert report["overall"] == "pass"
        assert report["invalid_examples"] == []
        assert any(c["check"].endswith(":lineage") and c["status"] == "pass"
                   for c in report["checks"])

    def test_validation_flags_problems(self, tmp_path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        root, cat, db = (str(tmp_path / "pq"), str(tmp_path / "c.json"),
                         str(tmp_path / "c.duckdb"))
        df = _frame([
            {"ts": "2023-08-20T01:30:00+00:00", "lat": 39.7, "lon": -120.0},
            {"ts": "2023-08-20T01:30:00+00:00", "lat": 999.0, "lon": -120.0},  # bad coords
        ])
        written = parquet.write_dataset(df, root=root, product="modis",
                                        sensor="terra_aqua", processing_version="v1")
        # Append a bad-timestamp row bypassing the writer (writer requires valid ts).
        bad = df.iloc[[0]].copy()
        bad["acquisition_timestamp_utc"] = "not-a-time"
        bad["detection_id"] = "id-badts"
        extra = str(tmp_path / "pq" / "v1" / "product=modis" / "sensor=terra_aqua"
                    / "year=2023" / "month=08" / "part-badts.parquet")
        pq.write_table(pa.Table.from_pandas(bad.drop(columns=["product", "sensor"]),
                                            preserve_index=False), extra)
        catalog_mod.record_dataset(
            cat, dataset_id="modis-test",
            raw_files=[{"path": str(tmp_path / "gone.csv"), "sha256": "ab" * 32,
                        "size_bytes": 10}],  # lineage target MISSING
            processing_version="v1", parquet_files=written["files"],
            validation_status="valid",
            record_counts={"source": 3, "accepted": 3, "rejected": 0},
            coverage={}, quality={})
        duckdb_layer.init_db(db, root)
        report = discovery.validate_inputs(cat, db)
        by_check = {c["check"]: c for c in report["checks"]}
        assert by_check["modis-test:coords"]["status"] == "fail"
        assert by_check["modis-test:timestamps"]["status"] == "fail"
        assert by_check["modis-test:lineage"]["status"] == "fail"
        assert report["overall"] == "fail"
        assert len(report["invalid_examples"]) == 2  # reported, never deleted


def _match_frame(rows, instrument):
    """Synthetic matcher input: (id, lat, lon, ts, daynight, scan, track)."""
    return pd.DataFrame([{
        "detection_id": r[0], "latitude": r[1], "longitude": r[2],
        "acquisition_timestamp_utc": r[3], "satellite": "T" if instrument == "MODIS" else "N",
        "instrument": instrument, "daynight": r[4] if len(r) > 4 else "D",
        "scan": r[5] if len(r) > 5 else 1.0, "track": r[6] if len(r) > 6 else 1.0,
    } for r in rows])


class TestMatching2B:
    def test_clear_match_high_evidence(self):
        modis = _match_frame([("m1", 39.7, -120.0, "2023-08-20T01:30:00+00:00")], "MODIS")
        viirs = _match_frame([("v1", 39.7005, -120.0005, "2023-08-20T02:00:00+00:00",
                               "D", 0.375, 0.375)], "VIIRS")
        groups, members, stats = matching.match_frames(modis, viirs, exec_id="e1")
        assert len(groups) == 1 and groups.iloc[0]["classification"] == "matched"
        assert groups.iloc[0]["evidence"] == "high"
        assert set(members["classification"]) == {"matched"}
        assert groups.iloc[0]["params_version"] == "match_params_v1"
        assert groups.iloc[0]["exec_id"] == "e1"

    def test_close_space_far_time_singletons(self):
        modis = _match_frame([("m1", 39.7, -120.0, "2023-08-20T01:30:00+00:00")], "MODIS")
        viirs = _match_frame([("v1", 39.7, -120.0, "2023-08-25T01:30:00+00:00")], "VIIRS")
        groups, members, _ = matching.match_frames(modis, viirs)
        assert len(groups) == 0  # no group at all: pure singletons
        assert sorted(members["classification"]) == ["MODIS-only", "VIIRS-only"]

    def test_close_time_far_space_singletons(self):
        modis = _match_frame([("m1", 39.7, -120.0, "2023-08-20T01:30:00+00:00")], "MODIS")
        viirs = _match_frame([("v1", 45.0, -110.0, "2023-08-20T01:35:00+00:00")], "VIIRS")
        groups, members, _ = matching.match_frames(modis, viirs)
        assert len(groups) == 0
        assert sorted(members["classification"]) == ["MODIS-only", "VIIRS-only"]

    def test_one_to_many_attach(self):
        modis = _match_frame([("m1", 0.0, 0.0, "2023-06-01T12:00:00+00:00")], "MODIS")
        viirs = _match_frame([
            ("v1", 0.0005, 0.0, "2023-06-01T12:30:00+00:00"),
            ("v2", 0.005, 0.0, "2023-06-01T13:00:00+00:00"),
        ], "VIIRS")
        groups, members, _ = matching.match_frames(modis, viirs)
        assert len(groups) == 1 and groups.iloc[0]["classification"] == "matched"
        import json as _json

        assert _json.loads(groups.iloc[0]["viirs_ids"]) == ["v1", "v2"]
        assert "secondary-attach" in groups.iloc[0]["notes"]

    def test_many_to_one_attach(self):
        modis = _match_frame([
            ("m1", 0.0, 0.0, "2023-06-01T12:00:00+00:00"),
            ("m2", 0.005, 0.0, "2023-06-01T12:10:00+00:00"),
        ], "MODIS")
        viirs = _match_frame([("v1", 0.0005, 0.0, "2023-06-01T12:30:00+00:00")], "VIIRS")
        groups, members, _ = matching.match_frames(modis, viirs)
        assert len(groups) == 1 and groups.iloc[0]["classification"] == "matched"
        import json as _json

        assert _json.loads(groups.iloc[0]["modis_ids"]) == ["m1", "m2"]

    def test_ambiguous_component_not_forced(self):
        # m1<->v1 mutual-best matched; v2 attaches (uncontested best m1,
        # group-coherent). m2's only core candidate v2 is matched, but m2 is
        # incoherent with v1 (2.17 km) -> refused -> ambiguous {m2} kept.
        modis = _match_frame([
            ("m1", 0.0, 0.0, "2023-06-01T12:00:00+00:00"),
            ("m2", 0.02, 0.0, "2023-06-01T12:05:00+00:00"),
        ], "MODIS")
        viirs = _match_frame([
            ("v1", 0.0005, 0.0, "2023-06-01T12:30:00+00:00"),
            ("v2", 0.008, 0.0, "2023-06-01T12:35:00+00:00"),
        ], "VIIRS")
        groups, members, _ = matching.match_frames(modis, viirs)
        by_class = members.groupby("classification").size().to_dict()
        assert by_class == {"matched": 3, "ambiguous": 1}
        amb = groups[groups["classification"] == "ambiguous"].iloc[0]
        import json as _json

        assert sorted(_json.loads(amb["modis_ids"]) + _json.loads(amb["viirs_ids"])) == ["m2"]
        assert "partners-matched-elsewhere:1" in amb["notes"]
        assert _json.loads(amb["candidates"])[0]["partner"] == "v2"
        assert set(members["detection_id"]) == {"m1", "m2", "v1", "v2"}  # all kept

    def test_ambiguous_multi_member_component(self):
        # m1<->v1 mutual-best matched. v2's best is the matched m1 but its
        # 2nd-best (m2) falls inside the 15% compete margin -> v2 defers;
        # m2 sees only v2 -> both defer -> one ambiguous {m2,v2} group.
        modis = _match_frame([
            ("m1", 0.0, 0.0, "2023-06-01T12:00:00+00:00"),
            ("m2", 0.0166, 0.0, "2023-06-01T12:33:00+00:00"),
        ], "MODIS")
        viirs = _match_frame([
            ("v1", 0.0005, 0.0, "2023-06-01T12:30:00+00:00"),
            ("v2", 0.008, 0.0, "2023-06-01T12:35:00+00:00"),
        ], "VIIRS")
        groups, members, _ = matching.match_frames(modis, viirs)
        by_class = members.groupby("classification").size().to_dict()
        assert by_class == {"matched": 2, "ambiguous": 2}
        amb = groups[groups["classification"] == "ambiguous"]
        assert len(amb) == 1
        import json as _json

        got = sorted(_json.loads(amb.iloc[0]["modis_ids"])
                     + _json.loads(amb.iloc[0]["viirs_ids"]))
        assert got == ["m2", "v2"]
        assert set(members["detection_id"]) == {"m1", "m2", "v1", "v2"}

    def test_expanded_window_only_is_ambiguous_weak(self):
        modis = _match_frame([("m1", 0.0, 0.0, "2023-06-01T12:00:00+00:00")], "MODIS")
        viirs = _match_frame([("v1", 0.02, 0.0, "2023-06-01T12:30:00+00:00")], "VIIRS")
        groups, members, _ = matching.match_frames(modis, viirs)
        assert len(groups) == 1 and groups.iloc[0]["classification"] == "ambiguous"
        assert "mutual-expanded-best-weak" in groups.iloc[0]["notes"]
        import json as _json

        pairs = _json.loads(groups.iloc[0]["pairs"])
        assert len(pairs) == 1 and pairs[0]["window"] == "expanded"

    def test_missing_geometry_still_matched_medium(self):
        import math

        modis = _match_frame([("m1", 39.7, -120.0, "2023-08-20T01:30:00+00:00",
                               "D", math.nan, math.nan)], "MODIS")
        viirs = _match_frame([("v1", 39.7005, -120.0005, "2023-08-20T02:00:00+00:00")], "VIIRS")
        groups, _, _ = matching.match_frames(modis, viirs)
        assert groups.iloc[0]["classification"] == "matched"
        assert groups.iloc[0]["evidence"] == "medium"  # no geometry bonus, no veto

    def test_epoch_scale_regression_13h_apart_not_core(self):
        # Regression: pandas-3 microsecond resolution once made epoch_s 1000x
        # too small, collapsing all time differences. 13 h apart on the same
        # day must NOT core-match (12 h tolerance), only expanded-window.
        modis = _match_frame([("m1", 39.7, -120.0, "2023-08-20T00:00:00+00:00")], "MODIS")
        viirs = _match_frame([("v1", 39.7, -120.0, "2023-08-20T13:00:00+00:00")], "VIIRS")
        groups, members, _ = matching.match_frames(modis, viirs)
        assert set(members["classification"]) == {"ambiguous"}
        assert groups.iloc[0]["classification"] == "ambiguous"

    def test_missing_timestamp_raises_loudly(self):
        modis = _match_frame([("m1", 39.7, -120.0, "not-a-time")], "MODIS")
        viirs = _match_frame([("v1", 39.7, -120.0, "2023-08-20T01:30:00+00:00")], "VIIRS")
        with pytest.raises(ValueError, match="unparseable timestamps"):
            matching.match_frames(modis, viirs)

    def test_duplicate_ids_rejected(self):
        modis = _match_frame([
            ("m1", 39.7, -120.0, "2023-08-20T01:30:00+00:00"),
            ("m1", 39.8, -120.1, "2023-08-20T02:30:00+00:00"),
        ], "MODIS")
        viirs = _match_frame([("v1", 39.7, -120.0, "2023-08-20T01:30:00+00:00")], "VIIRS")
        with pytest.raises(ValueError, match="duplicate detection_id"):
            matching.match_frames(modis, viirs)

    def test_repeated_execution_identical(self):
        modis = _match_frame([
            ("m1", 0.0, 0.0, "2023-06-01T12:00:00+00:00"),
            ("m2", 0.008, 0.0, "2023-06-01T12:05:00+00:00"),
        ], "MODIS")
        viirs = _match_frame([
            ("v1", 0.0005, 0.0, "2023-06-01T12:30:00+00:00"),
            ("v2", 0.0085, 0.0, "2023-06-01T12:35:00+00:00"),
        ], "VIIRS")
        first = matching.match_frames(modis, viirs, exec_id="same")
        second = matching.match_frames(modis, viirs, exec_id="same")
        pd.testing.assert_frame_equal(first[0], second[0])
        pd.testing.assert_frame_equal(first[1], second[1])
        assert first[2] == second[2]

    def test_repeated_attach_keeps_ids_unique_and_linked(self):
        # Regression: two attaches to the same group must not collide ids;
        # every member link must be bidirectional (member->group->member).
        import json as _json

        modis = _match_frame([("m1", 0.0, 0.0, "2023-06-01T12:00:00+00:00")], "MODIS")
        viirs = _match_frame([
            ("v1", 0.0005, 0.0, "2023-06-01T12:30:00+00:00"),
            ("v2", 0.001, 0.0, "2023-06-01T12:35:00+00:00"),
            ("v3", 0.0015, 0.0, "2023-06-01T12:40:00+00:00"),
        ], "VIIRS")
        groups, members, _ = matching.match_frames(modis, viirs)
        assert len(groups) == 1 and groups["match_id"].nunique() == 1
        listed = sorted(_json.loads(groups.iloc[0]["modis_ids"])
                        + _json.loads(groups.iloc[0]["viirs_ids"]))
        assert listed == ["m1", "v1", "v2", "v3"]
        assert set(members["match_id"]) == {groups.iloc[0]["match_id"]}
        assert set(members["detection_id"]) == set(listed)

    def test_accounting_invariant_no_double_count(self):
        modis = _match_frame([
            ("m1", 0.0, 0.0, "2023-06-01T12:00:00+00:00"),
            ("m9", 50.0, 50.0, "2023-07-01T12:00:00+00:00"),
        ], "MODIS")
        viirs = _match_frame([
            ("v1", 0.0005, 0.0, "2023-06-01T12:30:00+00:00"),
            ("v9", -50.0, -50.0, "2023-07-01T12:00:00+00:00"),
        ], "VIIRS")
        groups, members, _ = matching.match_frames(modis, viirs)
        assert len(members) == 4 and members["detection_id"].nunique() == 4
        counts = members["classification"].value_counts().to_dict()
        assert counts == {"matched": 2, "MODIS-only": 1, "VIIRS-only": 1}


def _cluster_frame(rows):
    """Synthetic clustering input: (id, lat, lon, ts, satellite, instrument, frp)."""
    return pd.DataFrame([{
        "detection_id": r[0], "latitude": r[1], "longitude": r[2],
        "acquisition_timestamp_utc": r[3], "satellite": r[4], "instrument": r[5],
        "product": "modis" if r[5] == "MODIS" else "viirs-snpp",
        "frp": r[6] if len(r) > 6 else 10.0,
    } for r in rows])


def _match_groups(pairs):
    """Synthetic matched-groups frame from [(match_id, [modis], [viirs])]."""
    import json as _json

    return pd.DataFrame([{
        "match_id": mid, "classification": "matched",
        "modis_ids": _json.dumps(m), "viirs_ids": _json.dumps(v),
    } for mid, m, v in pairs])


class TestClustering2C:
    def test_chain_links_but_far_stays_separate(self):
        df = _cluster_frame([
            ("a", 0.0, 0.0, "2023-06-01T12:00:00+00:00", "Terra", "MODIS"),
            ("b", 0.01, 0.0, "2023-06-01T13:00:00+00:00", "Terra", "MODIS"),
            ("c", 5.0, 5.0, "2023-06-01T13:00:00+00:00", "Terra", "MODIS"),
        ])
        clusters, membership, _ = clustering.cluster_frames(
            df, _match_groups([]), exec_id="e")
        assert len(clusters) == 2  # {a,b} chained (1.1 km), {c} alone
        assert set(membership["detection_id"]) == {"a", "b", "c"}
        assert membership["cluster_id"].nunique() == 2

    def test_temporal_gap_boundary(self):
        df = _cluster_frame([
            ("a", 0.0, 0.0, "2023-06-01T12:00:00+00:00", "Terra", "MODIS"),
            ("b", 0.005, 0.0, "2023-06-02T11:00:00+00:00", "Terra", "MODIS"),
            ("c", 0.005, 0.0, "2023-06-03T12:00:00+00:00", "Terra", "MODIS"),
        ])
        clusters, membership, _ = clustering.cluster_frames(
            df, _match_groups([]), exec_id="e")
        # a-b gap 23 h links; b-c gap 25 h breaks -> {a,b}, {c}.
        assert len(clusters) == 2
        same = membership[membership["detection_id"].isin(["a", "b"])]
        assert same["cluster_id"].nunique() == 1

    def test_cross_sensor_links_only_via_matches(self):
        df = _cluster_frame([
            ("m1", 0.0, 0.0, "2023-06-01T12:00:00+00:00", "Terra", "MODIS"),
            ("v1", 0.001, 0.0, "2023-06-01T12:10:00+00:00", "N", "VIIRS"),
            ("v2", 0.0015, 0.0, "2023-06-01T12:15:00+00:00", "N", "VIIRS"),
        ])
        # v1+v2 link same-sensor; m1 joins only through the matched group.
        clusters, membership, _ = clustering.cluster_frames(
            df, _match_groups([("mg1", ["m1"], ["v1"])]), exec_id="e")
        assert len(clusters) == 1
        assert set(membership["detection_id"]) == {"m1", "v1", "v2"}
        # Without the match group, m1 stays separate from {v1,v2}.
        clusters2, _, _ = clustering.cluster_frames(df, _match_groups([]), exec_id="e")
        assert len(clusters2) == 2

    def test_ambiguous_groups_create_no_links(self):
        import json as _json

        df = _cluster_frame([
            ("m1", 0.0, 0.0, "2023-06-01T12:00:00+00:00", "Terra", "MODIS"),
            ("v1", 0.001, 0.0, "2023-06-01T12:10:00+00:00", "N", "VIIRS"),
        ])
        amb = pd.DataFrame([{"match_id": "mg1", "classification": "ambiguous",
                             "modis_ids": _json.dumps(["m1"]),
                             "viirs_ids": _json.dumps(["v1"])}])
        clusters, membership, _ = clustering.cluster_frames(df, amb, exec_id="e")
        assert len(clusters) == 2  # no link from ambiguous verdicts
        assert membership["cluster_id"].nunique() == 2

    def test_outputs_frp_sensors_span(self):
        import json as _json

        df = _cluster_frame([
            ("m1", 0.0, 0.0, "2023-06-01T12:00:00+00:00", "Terra", "MODIS", 5.0),
            ("v1", 0.001, 0.0, "2023-06-01T14:00:00+00:00", "N", "VIIRS", 15.0),
            ("v2", 0.002, 0.0, "2023-06-01T15:00:00+00:00", "N", "VIIRS", float("nan")),
        ])
        clusters, membership, _ = clustering.cluster_frames(
            df, _match_groups([("mg1", ["m1"], ["v1"])]), exec_id="e")
        assert len(clusters) == 1
        row = clusters.iloc[0]
        assert (row["n_detections"], row["n_modis"], row["n_viirs"]) == (3, 1, 2)
        assert sorted(_json.loads(row["satellites"])) == ["N", "Terra"]
        assert row["observation_span_h"] == 3.0
        assert "wildfire" not in str(row.to_dict()).lower()
        assert row["frp_sum"] == 20.0 and row["frp_n"] == 2 and row["frp_null"] == 1
        assert row["frp_max"] == 15.0
        assert _json.loads(row["match_groups"]) == ["mg1"]
        assert row["completeness_dates"] == 1
        assert set(membership["detection_id"]) == {"m1", "v1", "v2"}  # traceable

    def test_repeated_execution_identical(self):
        df = _cluster_frame([
            ("m1", 0.0, 0.0, "2023-06-01T12:00:00+00:00", "Terra", "MODIS"),
            ("v1", 0.001, 0.0, "2023-06-01T12:10:00+00:00", "N", "VIIRS"),
        ])
        first = clustering.cluster_frames(df, _match_groups([("mg1", ["m1"], ["v1"])]),
                                          exec_id="same")
        second = clustering.cluster_frames(df, _match_groups([("mg1", ["m1"], ["v1"])]),
                                           exec_id="same")
        pd.testing.assert_frame_equal(first[0], second[0])
        pd.testing.assert_frame_equal(first[1], second[1])


def _metric_frames():
    det = pd.DataFrame([
        {"detection_id": "m1", "acquisition_timestamp_utc": "2023-08-20T01:00:00+00:00",
         "satellite": "Terra", "instrument": "MODIS", "product": "modis", "frp": 10.0},
        {"detection_id": "v1", "acquisition_timestamp_utc": "2023-08-20T02:00:00+00:00",
         "satellite": "N", "instrument": "VIIRS", "product": "viirs-snpp", "frp": 20.0},
        {"detection_id": "v2", "acquisition_timestamp_utc": "2023-08-21T02:00:00+00:00",
         "satellite": "N", "instrument": "VIIRS", "product": "viirs-snpp",
         "frp": float("nan")},
    ])
    mem = pd.DataFrame([
        {"detection_id": "m1", "match_id": "mg1", "classification": "matched"},
        {"detection_id": "v1", "match_id": "mg1", "classification": "matched"},
        {"detection_id": "v2", "match_id": None, "classification": "VIIRS-only"},
    ])
    clu = pd.DataFrame([
        {"cluster_id": "cl1", "first_observation_utc": "2023-08-20T01:00:00+00:00",
         "n_detections": 2, "n_modis": 1, "n_viirs": 1, "observation_span_h": 1.0},
        {"cluster_id": "cl2", "first_observation_utc": "2023-08-21T02:00:00+00:00",
         "n_detections": 1, "n_modis": 0, "n_viirs": 1, "observation_span_h": 0.0},
    ])
    return det, mem, clu


class TestMetrics2D:
    def test_detection_activity_raw_and_harmonized(self):
        det, mem, _ = _metric_frames()
        out = metrics.detection_activity(
            det, mem, grain="daily", scope={"region": "test"},
            inputs={"d": 1}, exec_id="e")
        assert out["metric"] == "detection_activity" and out["exec_id"] == "e"
        by_period = {r["period"]: r for r in out["rows"]}
        assert by_period["2023-08-20"]["raw_observations"] == 2
        assert by_period["2023-08-20"]["by_product"] == {"modis": 1, "viirs-snpp": 1}
        harm = {r["period"]: r for r in out["harmonized_rows"]}
        # mg1 (m1+v1) collapses to 1 on 08-20; v2 singleton on 08-21.
        assert harm["2023-08-20"]["harmonized_observations"] == 1
        assert harm["2023-08-20"]["matched_groups_attributed"] == 1
        assert harm["2023-08-21"]["harmonized_observations"] == 1
        assert harm["2023-08-21"]["singletons"] == 1
        # Raw total reconciles; harmonized < raw (no double counting).
        assert sum(r["raw_observations"] for r in out["rows"]) == 3
        assert sum(r["harmonized_observations"] for r in out["harmonized_rows"]) == 2

    def test_cluster_activity_composition(self):
        _, _, clu = _metric_frames()
        out = metrics.cluster_activity(
            clu, grain="daily", scope={"region": "test"}, inputs={"d": 1}, exec_id="e")
        assert out["metric"] == "cluster_activity"
        assert [r["clusters"] for r in out["rows"]] == [1, 1]
        assert out["rows"][0]["modis_detections"] == 1
        assert "never individual wildfires" in out["definition"].lower()
        sub = metrics.cluster_activity(
            clu, grain="daily", scope={}, inputs={}, exec_id="e",
            only_cluster_ids={"cl2"})
        assert len(sub["rows"]) == 1 and sub["rows"][0]["clusters"] == 1

    def test_frp_per_sensor_and_missing(self):
        det, _, _ = _metric_frames()
        out = metrics.frp_summary(det, grain="daily", scope={"region": "test"},
                                  inputs={"d": 1}, exec_id="e")
        by_key = {(r["period"], r["product"]): r for r in out["rows"]}
        assert by_key[("2023-08-20", "modis")]["frp_sum"] == 10.0
        assert by_key[("2023-08-20", "viirs-snpp")]["frp_sum"] == 20.0
        assert by_key[("2023-08-21", "viirs-snpp")]["n_missing_frp"] == 1
        assert by_key[("2023-08-21", "viirs-snpp")]["n_contributing"] == 0
        assert "combined" not in out  # default: no mixed total
        combo = metrics.frp_summary(det, grain="daily", scope={}, inputs={},
                                    exec_id="e", combined=True)
        assert "warning" in combo["combined"]
        assert combo["combined"]["rows"][0]["sensor_composition"] == {
            "modis": 1, "viirs-snpp": 1}

    def test_grains_and_invalid_grain(self):
        det, mem, _ = _metric_frames()
        monthly = metrics.detection_activity(
            det, mem, grain="monthly", scope={}, inputs={}, exec_id="e")
        assert [r["period"] for r in monthly["rows"]] == ["2023-08"]
        assert monthly["rows"][0]["raw_observations"] == 3
        weekly = metrics.detection_activity(
            det, mem, grain="weekly", scope={}, inputs={}, exec_id="e")
        assert weekly["rows"][0]["period"].startswith("2023-W")
        with pytest.raises(ValueError, match="unknown grain"):
            metrics.detection_activity(
                det, mem, grain="fortnightly", scope={}, inputs={}, exec_id="e")


def _index_frames():
    det = pd.DataFrame([
        {"detection_id": "m1", "frp": 10.0},
        {"detection_id": "v1", "frp": 20.0},
        {"detection_id": "v2", "frp": float("nan")},
    ])
    mem = pd.DataFrame([
        {"detection_id": "m1", "product": "modis", "match_id": "mg1",
         "classification": "matched"},
        {"detection_id": "v1", "product": "viirs-snpp", "match_id": "mg1",
         "classification": "matched"},
        {"detection_id": "v2", "product": "viirs-snpp", "match_id": None,
         "classification": "VIIRS-only"},
    ])
    groups = pd.DataFrame([{"match_id": "mg1", "classification": "matched"}])
    clusters = pd.DataFrame([
        {"cluster_id": "cl1", "n_detections": 2},
        {"cluster_id": "cl2", "n_detections": 1},
    ])
    membership = pd.DataFrame([
        {"detection_id": "m1", "cluster_id": "cl1"},
        {"detection_id": "v1", "cluster_id": "cl1"},
        {"detection_id": "v2", "cluster_id": "cl2"},
    ])
    return det, mem, groups, clusters, membership


class TestIndex2E:
    def test_sensor_view_breakdown(self):
        det, mem, _, _, _ = _index_frames()
        view = index.sensor_view(mem, det)
        assert view["modis"]["raw_observations"] == 1
        assert view["viirs-snpp"]["frp_sum"] == 20.0
        assert view["viirs-snpp"]["frp_missing"] == 1
        assert view["modis"]["matched_members"] == 1
        assert view["viirs-snpp"]["singletons"] == 1

    def test_harmonized_view_no_double_count(self):
        _, mem, groups, clusters, _ = _index_frames()
        view = index.harmonized_view(mem, groups, clusters)
        # 1 matched group + 0 ambiguous + 1 singleton = 2, never 3.
        assert view["harmonized_observations"] == 2
        assert view["matched_groups"] == 1 and view["singletons"] == 1
        assert view["clusters"] == 2

    def test_reconcile_pass_and_fail(self):
        det, mem, groups, clusters, membership = _index_frames()
        ok = index.reconcile(n_input=3, members_df=mem, match_groups_df=groups,
                             clusters_df=clusters, membership_df=membership)
        assert ok["passed"] is True
        bad = index.reconcile(n_input=99, members_df=mem, match_groups_df=groups,
                              clusters_df=clusters, membership_df=membership)
        assert bad["passed"] is False
        assert any(not c["passed"] for c in bad["checks"])

    def test_composite_refuses_unset_weights(self):
        with pytest.raises(index.UncalibratedComposite, match="no approved weights"):
            index.composite_index({"det": 1, "clu": 1, "frp": 1},
                                  {"det": 1, "clu": 1, "frp": 1})
        with pytest.raises(index.UncalibratedComposite, match="sum to 1"):
            index.composite_index({"det": 2, "clu": 2, "frp": 2},
                                  {"det": 1, "clu": 1, "frp": 1},
                                  weights={"w_det": 1.0, "w_clu": 1.0, "w_frp": 1.0})
        out = index.composite_index(
            {"det": 2.0, "clu": 4.0, "frp": 6.0}, {"det": 1.0, "clu": 2.0, "frp": 3.0},
            weights={"w_det": 0.5, "w_clu": 0.25, "w_frp": 0.25})
        assert out["index"] == 2.0  # 0.5*2 + 0.25*2 + 0.25*2
        assert out["weights"]["w_det"] == 0.5  # recorded, never silent

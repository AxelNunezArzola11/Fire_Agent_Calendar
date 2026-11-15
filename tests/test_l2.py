"""L2 subsystem tests (stubbed HTTP; tmp files; no network)."""

import json

import pytest

from src.calendar.l2 import (batches, budget, cmr, download,
                             earthdata, inputs, integrity, manifest,
                             match, products, readers)


class TestProducts:
    def test_get_and_unknown(self):
        assert products.get("VNP14IMG")["format"] == "NetCDF4"
        with pytest.raises(ValueError, match="unknown L2 product"):
            products.get("BOGUS")

    def test_needed_for_equation(self):
        assert "VNP14IMG" in products.needed_for_equation("Eq.7")
        assert "VNP03IMG" in products.needed_for_equation("Eq.1")
        assert products.needed_for_equation("Eq.12") == ["MOD04_L2",
                                                         "MYD04_L2"]


class _StubHTTP:
    def __init__(self, payload=None, exc=None):
        self._payload = payload if payload is not None else {}
        self._exc = exc
        self.calls = []

    def __call__(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params))
        if self._exc:
            raise self._exc

        class _Resp:
            def raise_for_status(self):
                pass

            def json(inner_self):
                return self._payload

        return _Resp()


class TestCMR:
    def test_granule_query_and_bbox(self):
        params = cmr.granule_query(
            "C1", bbox=(-125.0, 32.0, -114.0, 42.0),
            date_start="2023-08-15", date_end="2023-08-16")
        assert params["bounding_box"] == "-125.0,32.0,-114.0,42.0"
        assert params["temporal"].startswith("2023-08-15T")
        with pytest.raises(ValueError, match="invalid bbox"):
            cmr.granule_query("C1", bbox=(1.0, 1.0, 0.0, 0.0),
                              date_start="2023-08-15",
                              date_end="2023-08-16")

    def test_resolve_collection(self):
        stub = _StubHTTP({"feed": {"entry": [
            {"id": "C123", "title": "VNP14IMG v002"}]}})
        out = cmr.resolve_collection("VNP14IMG", "002",
                                     transport=stub)
        assert out["concept_id"] == "C123"
        assert "collections.json" in stub.calls[0][0]

    def test_resolve_empty(self):
        stub = _StubHTTP({"feed": {"entry": []}})
        with pytest.raises(Exception, match="no CMR collection"):
            cmr.resolve_collection("VNP14IMG", "002", transport=stub)

    def test_search_granules(self):
        stub = _StubHTTP({"feed": {"hits": 2, "entry": [
            {"title": "G1", "producer_granule_id": "P1",
             "time_start": "2023-08-15T00:00:00Z",
             "time_end": "2023-08-15T00:05:00Z",
             "granule_size": "12.5",
             "links": [{"rel": "data#",
                        "href": "https://x/G1.nc"}],
             "checksum": {"value": "ab", "algorithm": "SHA256"}},
            {"title": "G2", "links": []}]}})
        out_fed = cmr.search_granules(
            "C1", bbox=(-125.0, 32.0, -114.0, 42.0),
            date_start="2023-08-15", date_end="2023-08-16",
            transport=_StubHTTP({"feed": {"hits": 1, "entry": [
                {"title": "G3", "links": [
                    {"rel": "http://esipfed.org/ns/fedsearch/1.1/"
                            "data#",
                     "href": "https://x/G3.nc"},
                    {"rel": "http://esipfed.org/ns/fedsearch/1.1/s3#",
                     "href": "s3://x/G3.nc"}]}]}}))
        out = cmr.search_granules(
            "C1", bbox=(-125.0, 32.0, -114.0, 42.0),
            date_start="2023-08-15", date_end="2023-08-16",
            transport=stub)
        assert out["hits"] == 2
        assert out["granules"][0]["size_mb"] == 12.5
        assert out["granules"][0]["data_urls"] == ["https://x/G1.nc"]
        assert out["granules"][1]["size_mb"] is None
        assert out_fed["granules"][0]["data_urls"] == ["https://x/G3.nc"]

    def test_search_failure_mapped(self):
        stub = _StubHTTP(exc=ConnectionError("offline"))
        with pytest.raises(Exception, match="granule search failed"):
            cmr.search_granules(
                "C1", bbox=(0.0, 0.0, 1.0, 1.0),
                date_start="2023-08-15", date_end="2023-08-16",
                transport=stub)

    def test_bbox_from_boxes_and_polygons(self):
        stub = _StubHTTP({"feed": {"hits": 2, "entry": [
            {"title": "G1", "boxes": ["-120.0 35.0 -118.0 37.0"]},
            {"title": "G2",
             "polygons": [["35.0 -120.0 37.0 -120.0 37.0 -118.0 "
                           "35.0 -118.0 35.0 -120.0"]]}]}})
        out = cmr.search_granules(
            "C1", bbox=(-125.0, 32.0, -114.0, 42.0),
            date_start="2023-08-15", date_end="2023-08-16",
            transport=stub)
        assert out["granules"][0]["bbox"] == [-120.0, 35.0, -118.0,
                                              37.0]
        assert out["granules"][1]["bbox"] == [-120.0, 35.0, -118.0,
                                              37.0]


class TestEarthdata:
    def test_status_without_credentials(self, monkeypatch):
        from src.calendar import settings

        monkeypatch.setattr(settings, "EARTHDATA_USERNAME", "")
        monkeypatch.setattr(settings, "EARTHDATA_PASSWORD", "")
        status = earthdata.status()
        assert status["available"] is False
        assert "EARTHDATA_USERNAME" in status["reason"]

    def test_token_exchange_and_failure(self, monkeypatch):
        from src.calendar import settings

        monkeypatch.setattr(settings, "EARTHDATA_USERNAME", "u")
        monkeypatch.setattr(settings, "EARTHDATA_PASSWORD", "p")

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"access_token": "TOK"}

        seen = {}

        def _post(url, auth=None, timeout=None):
            seen["url"] = url
            seen["auth"] = auth
            return _Resp()

        assert earthdata.bearer_token(post=_post) == "TOK"
        assert seen["url"].endswith("/api/users/token")
        assert seen["auth"] == ("u", "p")

        def _bad(url, auth=None, timeout=None):
            raise ConnectionError("offline")

        with pytest.raises(Exception, match="token exchange failed"):
            earthdata.bearer_token(post=_bad)

    def test_transient_retry_then_success(self, monkeypatch):
        from src.calendar import settings

        monkeypatch.setattr(settings, "EARTHDATA_USERNAME", "u")
        monkeypatch.setattr(settings, "EARTHDATA_PASSWORD", "p")
        calls = []

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"access_token": "TOK2"}

        def _flaky(url, auth=None, timeout=None):
            calls.append(url)
            if len(calls) < 2:
                raise TimeoutError("slow")
            return _Resp()

        assert earthdata.bearer_token(post=_flaky) == "TOK2"
        assert len(calls) == 2

    def test_quota_reuses_existing_token(self, monkeypatch):
        from src.calendar import settings

        monkeypatch.setattr(settings, "EARTHDATA_USERNAME", "u")
        monkeypatch.setattr(settings, "EARTHDATA_PASSWORD", "p")

        class _Quota:
            status_code = 403

            def json(self):
                return {"error": "max_token_limit"}

        class _QuotaError(Exception):
            def __init__(self):
                super().__init__("403 Client Error")
                self.response = _Quota()

        def _post(url, auth=None, timeout=None):
            raise _QuotaError()

        def _get(url, auth=None, timeout=None):
            class _Resp:
                def raise_for_status(self):
                    pass

                def json(self):
                    return [{"access_token": "REUSED"}]

            return _Resp()

        assert earthdata.bearer_token(post=_post, get=_get) == \
            "REUSED"

    def test_list_and_revoke_tokens(self, monkeypatch):
        from src.calendar import settings

        monkeypatch.setattr(settings, "EARTHDATA_USERNAME", "u")
        monkeypatch.setattr(settings, "EARTHDATA_PASSWORD", "p")

        class _Resp:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                pass

            def json(self):
                return self._payload

        seen = {}

        def _get(url, auth=None, timeout=None):
            seen["url"] = url
            return _Resp([{"access_token": "AA",
                           "created_at": "2026-01-01"}])

        assert earthdata.list_tokens(get=_get)[0]["access_token"] == \
            "AA"
        assert seen["url"].endswith("/api/users/tokens")

        def _delete(url, params=None, timeout=None):
            seen["revoke"] = (url, params)
            return _Resp({})

        assert earthdata.revoke_token("AA", delete=_delete) is True
        assert seen["revoke"][1] == {"token": "AA"}


class TestManifest:
    def test_append_load_pending(self, tmp_path):
        manifest.append(tmp_path, {"granule_ur": "G1",
                                   "status": "complete"})
        manifest.append(tmp_path, {"granule_ur": "G2",
                                   "status": "failed"})
        manifest.append(tmp_path, {"granule_ur": "G2",
                                   "status": "complete"})
        assert manifest.pending_urs(manifest.load(tmp_path)) == []
        manifest.append(tmp_path, {"granule_ur": "G3",
                                   "status": "inventoried"})
        assert manifest.pending_urs(manifest.load(tmp_path)) == ["G3"]

    def test_missing_is_empty(self, tmp_path):
        assert manifest.load(tmp_path / "nope") == []

    def test_complete_never_regresses(self, tmp_path):
        manifest.append(tmp_path, {"granule_ur": "G1",
                                   "status": "complete"})
        manifest.append(tmp_path, {"granule_ur": "G1",
                                   "status": "inventoried"})
        assert manifest.latest_by_ur(tmp_path)["G1"]["status"] == \
            "complete"
        assert manifest.pending_urs(manifest.load(tmp_path)) == []


class TestBudget:
    def test_estimate_and_decide(self):
        est = budget.estimate_mb([{"size_mb": 10.0}, {}])
        assert est["total_mb"] == 60.0 and est["unknown_count"] == 1
        go = budget.decide(100.0, free_mb=5000.0, budget_mb=2048.0)
        assert go["proceed"] is True
        stop = budget.decide(3000.0, free_mb=5000.0,
                             budget_mb=2048.0)
        assert stop["proceed"] is False
        assert "budget" in stop["reasons"][0]
        tight = budget.decide(100.0, free_mb=500.0, budget_mb=2048.0,
                              keep_free_mb=1000.0)
        assert tight["proceed"] is False


class TestIntegrity:
    def test_verify_roundtrip(self, tmp_path):
        target = tmp_path / "f.bin"
        target.write_bytes(b"abc" * 100)
        digest = integrity.file_digest(str(target))
        assert integrity.verify(str(target), digest)["match"] is True
        assert integrity.verify(str(target), "00")["match"] is False
        assert integrity.verify(str(target), None)["checked"] is False


class _FakeSession:
    def __init__(self, chunks, status_code=200):
        self._chunks = chunks
        self._status = status_code
        self.seen_headers = []

    def get(self, url, headers=None, stream=None, timeout=None):
        self.seen_headers.append(headers or {})

        class _Resp:
            status_code = self._status

            def raise_for_status(inner_self):
                if self._status >= 400:
                    raise IOError(f"HTTP {self._status}")

            def iter_content(inner_self, chunk_size=None):
                return iter(self._chunks)

        return _Resp()


class TestDownload:
    def test_stream_and_resume(self, tmp_path):
        session = _FakeSession([b"ab", b"cd"])
        dest = tmp_path / "g.nc"
        first = download.download_granule(session, "https://x/g",
                                          dest)
        assert first["bytes"] == 4 and first["resumed"] is False
        session2 = _FakeSession([b"ef"])
        second = download.download_granule(session2, "https://x/g",
                                           dest)
        # server ignored Range (200): restarted cleanly, no dup bytes
        assert dest.read_bytes() == b"ef"
        assert second["resumed"] is False

    def test_checksum_mismatch_raises(self, tmp_path):
        session = _FakeSession([b"data"])
        with pytest.raises(Exception, match="mismatch"):
            download.download_granule(
                session, "https://x/g", tmp_path / "g.nc",
                expected_digest="00" * 32)

    def test_local_path_is_safe(self, tmp_path):
        dest = download.local_path(tmp_path, "VNP14IMG",
                                   "../../evil.nc")
        assert dest.parent.parent == tmp_path
        assert dest.name == "evil.nc"  # no doubled suffix
        assert download.local_path(
            tmp_path, "P", "G123", suffix=".nc").name == "G123.nc"
        assert download.local_path(
            tmp_path, "P", "G123.nc", suffix=".nc").name == "G123.nc"


class TestReaders:
    def test_detect_formats(self):
        import shutil

        assert readers.detect_format("g.nc")["readable"] is True
        hdf = readers.detect_format("g.hdf")
        assert hdf["readable"] is (shutil.which("gdalinfo") is not None)
        assert readers.detect_format("g.txt")["readable"] is False

    def test_unsupported_raises(self):
        with pytest.raises(Exception, match="unsupported suffix"):
            readers.subdatasets("g.txt")

    def test_gdal_catalog_parsing(self, monkeypatch):
        import json as _json

        payload = {"metadata": {"SUBDATASETS": {
            "SUBDATASET_1_NAME": "HDF4_SDS:X:'f.hdf':0",
            "SUBDATASET_1_DESC": "[100x100] fire mask",
            "SUBDATASET_2_NAME": "HDF4_SDS:X:'f.hdf':1",
            "SUBDATASET_2_DESC": "[100x100] QA"}}}

        class _Proc:
            returncode = 0
            stdout = _json.dumps(payload)
            stderr = ""

        monkeypatch.setattr(readers.subprocess, "run",
                            lambda *a, **k: _Proc())
        monkeypatch.setattr(readers, "_gdal_cli",
                            lambda *a: "/usr/bin/gdalinfo")
        pairs = readers._gdal_catalog("f.hdf")
        assert pairs[0][0].endswith(":0")
        assert "fire mask" in pairs[0][1]
        assert readers.subdatasets("f.hdf") == [p[0] for p in pairs]

    def test_gdal_missing_reported(self, monkeypatch):
        monkeypatch.setattr(readers, "_gdal_cli",
                            lambda *a: None)
        assert readers.detect_format("g.hdf")["readable"] is False
        with pytest.raises(Exception, match="no gdalinfo"):
            readers.subdatasets("g.hdf")

    def test_read_variable_with_fake_rasterio(self, monkeypatch):
        import numpy as np

        class _FakeDS:
            subdatasets = ["netcdf:g.nc:fire_mask"]

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, band):
                return np.zeros((2, 2), dtype=np.uint8)

            profile = {"driver": "netCDF"}

        import rasterio

        monkeypatch.setattr(rasterio, "open", lambda *a, **k: _FakeDS())
        out = readers.read_variable("g.nc", "fire_mask")
        assert out["values"].shape == (2, 2)
        with pytest.raises(Exception, match="not in granule"):
            readers.read_variable("g.nc", "nope")


class TestMatch:
    def _granules(self):
        return [{"granule_ur": "G1",
                 "time_start": "2023-08-15T06:00:00Z",
                 "time_end": "2023-08-15T06:06:00Z",
                 "bbox": [-121.0, 37.0, -119.0, 39.0]}]

    def test_matched(self):
        out = match.match_observation(
            {"latitude": 38.0, "longitude": -120.0,
             "acquisition_utc": "2023-08-15T06:03:00+00:00"},
            self._granules())
        assert out["status"] == "matched"
        assert out["granule_ur"] == "G1"
        assert "not pixel identity" in out["confidence"]

    def test_unmatched_far(self):
        out = match.match_observation(
            {"latitude": 0.0, "longitude": 0.0,
             "acquisition_utc": "2023-08-15T06:03:00+00:00"},
            self._granules())
        assert out["status"] == "unmatched"

    def test_bad_observation(self):
        out = match.match_observation({"latitude": None},
                                      self._granules())
        assert out["status"] == "unmatched"


class TestBatches:
    def test_splits(self):
        assert batches.split_range("2023-08-15", "2023-08-21") == [
            ("2023-08-15", "2023-08-21")]
        assert batches.split_range("2023-08-15", "2023-08-22") == [
            ("2023-08-15", "2023-08-21"),
            ("2023-08-22", "2023-08-22")]
        assert len(batches.split_range("2023-08-01", "2023-08-30")) == 5
        assert batches.split_range("2023-01-28", "2023-02-03") == [
            ("2023-01-28", "2023-02-03")]
        assert batches.split_range("2023-12-30", "2024-01-02") == [
            ("2023-12-30", "2024-01-02")]
        with pytest.raises(ValueError, match="start must not"):
            batches.split_range("2023-08-16", "2023-08-15")

    def test_run_sequential_success_and_resume(self, tmp_path):
        seen = []

        def _fetch(block):
            return {"granules": [
                {"granule_ur": f"G-{block[0]}", "size_mb": 1.0}]}

        def _download(ur, granule):
            seen.append(ur)
            manifest.append(tmp_path, {"granule_ur": ur,
                                       "status": "complete"})
            return {"status": "complete"}

        blocks = batches.split_range("2023-08-15", "2023-08-22")
        first = batches.run_sequential(
            tmp_path, "VNP14IMG", blocks, budget_mb=2048.0,
            fetch_inventory=_fetch, download_one=_download)
        assert (first["complete"], first["failed"]) == (2, 0)
        assert first["granules_complete"] == 2
        second = batches.run_sequential(
            tmp_path, "VNP14IMG", blocks, budget_mb=2048.0,
            fetch_inventory=_fetch, download_one=_download)
        assert second["complete"] == 2
        assert len(seen) == 2  # resumed: no re-downloads

    def test_failure_isolation_and_budget_stop(self, tmp_path):
        def _fetch(block):
            if block[0] == "2023-08-22":
                raise ConnectionError("offline")
            return {"granules": [
                {"granule_ur": f"G-{block[0]}", "size_mb": 1.0}]}

        def _download(ur, granule):
            if ur.endswith("2023-08-15"):
                raise IOError("disk hiccup")
            manifest.append(tmp_path, {"granule_ur": ur,
                                       "status": "complete"})
            return {"status": "complete"}

        blocks = batches.split_range("2023-08-15", "2023-08-22")
        out = batches.run_sequential(
            tmp_path, "VNP14IMG", blocks, budget_mb=2048.0,
            fetch_inventory=_fetch, download_one=_download)
        assert out["failed"] == 2  # one inventory + one download
        assert out["granules_failed"] == 1
        assert any("offline" in b["detail"] for b in out["blocks"])
        over = batches.run_sequential(
            tmp_path, "VNP14IMG", blocks, budget_mb=0.001,
            fetch_inventory=_fetch, download_one=_download)
        assert over["skipped_budget"] >= 1


class TestInputs:
    def test_unmatched_all_none(self):
        out = inputs.assemble(None, None)
        assert out["match_status"] == "unmatched"
        assert out["aod_550"] is None

    def test_values_pass_through(self):
        out = inputs.assemble(
            "G1", {"aod_550": 0.3, "aod_source": "s",
                   "cloud_flag": "clear", "cloud_source": "s2",
                   "pixel_area_km2": 0.21, "area_source": "s3",
                   "qa_flags": [1], "qa_source": "s4",
                   "provenance": "unit"})
        assert out["aod_550"] == 0.3
        assert out["match_status"] == "matched"

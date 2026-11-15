"""Phase 1A tests: acquisition layer. No network (all HTTP mocked)."""

from datetime import date, timedelta

import pytest

from src.calendar.acquisition import archive, nrt, sensors, transport


class FakeResp:
    def __init__(self, status=200, content=b"a,b\n1,2\n", text=None, headers=None):
        self.status_code = status
        self.content = content
        self.text = text if text is not None else content.decode("utf-8", "replace")
        self.headers = headers or {}


# --- sensors ----------------------------------------------------------------

class TestSensorRegistry:
    def test_five_distinct_identities(self):
        assert sorted(sensors.SENSORS) == ["aqua", "noaa20", "noaa21", "snpp", "terra"]
        sats = [s.satellite for s in sensors.SENSORS.values()]
        assert len(set(sats)) == 5

    def test_modis_shares_nrt_source_but_not_identity(self):
        assert sensors.TERRA.nrt_source == sensors.AQUA.nrt_source == "MODIS_NRT"
        assert sensors.TERRA.satellite != sensors.AQUA.satellite

    def test_archive_url_and_validation(self):
        url = sensors.archive_url("modis", 2023, "United_States")
        assert url == ("https://firms.modaps.eosdis.nasa.gov/data/country/modis/2023/"
                       "modis_2023_United_States.csv")
        with pytest.raises(KeyError):
            sensors.archive_url("viirs-noaa20", 2023, "United_States")
        with pytest.raises(ValueError):
            sensors.archive_url("viirs-snpp", 2005, "Italy")
        with pytest.raises(ValueError):
            sensors.archive_url("modis", 2023, "a/b")


# --- transport ---------------------------------------------------------------

class TestTransport:
    def test_success_first_try(self, monkeypatch):
        monkeypatch.setattr("requests.get", lambda *a, **k: FakeResp())
        res = transport.fetch_bytes("http://x", timeout_s=5, secrets=("K",))
        assert (res.status, res.attempts) == (200, 1)

    def test_retries_then_succeeds_with_backoff(self, monkeypatch):
        import requests as rq

        calls, sleeps = [], []
        def flaky(*a, **k):
            calls.append(1)
            if len(calls) < 3:
                raise rq.Timeout("slow")
            return FakeResp()
        monkeypatch.setattr("requests.get", flaky)
        monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
        res = transport.fetch_bytes("http://x", timeout_s=5, max_retries=3,
                                    backoff_s=(2.0, 4.0, 8.0))
        assert res.attempts == 3
        assert sleeps == [2.0, 4.0]

    def test_exhaustion_raises_bounded(self, monkeypatch):
        import requests as rq

        calls = []
        def always_timeout(*a, **k):
            calls.append(1)
            raise rq.Timeout("slow")
        monkeypatch.setattr("requests.get", always_timeout)
        monkeypatch.setattr("time.sleep", lambda s: None)
        with pytest.raises(transport.FetchTimeout):
            transport.fetch_bytes("http://x", timeout_s=1, max_retries=2)
        assert len(calls) == 1 + 2  # initial + bounded retries, never infinite

    def test_auth_errors_and_redaction(self, monkeypatch):
        monkeypatch.setattr("requests.get", lambda *a, **k: FakeResp(401, b"no"))
        with pytest.raises(transport.AuthError):
            transport.fetch_bytes("http://x/SECRET-KEY/data", timeout_s=1, secrets=("SECRET-KEY",))
        monkeypatch.setattr("requests.get",
                            lambda *a, **k: FakeResp(400, b"Invalid MAP_KEY."))
        try:
            transport.fetch_bytes("http://x/SECRET-KEY/data", timeout_s=1, secrets=("SECRET-KEY",))
            raise AssertionError("expected AuthError")
        except transport.AuthError as exc:
            assert "SECRET-KEY" not in str(exc)

    def test_404_not_retried(self, monkeypatch):
        calls = []
        def nf(*a, **k):
            calls.append(1)
            return FakeResp(404, b"{}")
        monkeypatch.setattr("requests.get", nf)
        with pytest.raises(transport.HttpError):
            transport.fetch_bytes("http://x", timeout_s=1)
        assert len(calls) == 1

    def test_429_retried(self, monkeypatch):
        calls = []
        def rate(*a, **k):
            calls.append(1)
            return FakeResp(429, b"slow down") if len(calls) < 2 else FakeResp()
        monkeypatch.setattr("requests.get", rate)
        monkeypatch.setattr("time.sleep", lambda s: None)
        res = transport.fetch_bytes("http://x", timeout_s=1)
        assert (res.status, res.attempts) == (200, 2)


# --- nrt ---------------------------------------------------------------------

class TestNrtUrl:
    def test_undated_and_dated_forms(self):
        r = nrt.AreaRequest("VIIRS_SNPP_NRT", "-122.5,37.0,-121.5,38.0", 1)
        assert nrt.build_area_url("B", "K", r) == "B/K/VIIRS_SNPP_NRT/-122.5,37.0,-121.5,38.0/1"
        r2 = nrt.AreaRequest("MODIS_SP", "-10,30,10,40", 5, date(2023, 6, 1))
        assert nrt.build_area_url("B", "K", r2).endswith("/5/2023-06-01")

    def test_validation(self):
        with pytest.raises(ValueError):
            nrt.build_area_url("B", "K", nrt.AreaRequest("NOPE", "0,0,1,1", 1))
        with pytest.raises(ValueError):
            nrt.build_area_url("B", "K", nrt.AreaRequest("MODIS_NRT", "0,0,1,1", 6))
        with pytest.raises(ValueError):
            nrt.build_area_url("B", "K", nrt.AreaRequest("MODIS_NRT", "1,1,0,0", 1))
        with pytest.raises(transport.AuthError):
            nrt.build_area_url("B", "", nrt.AreaRequest("MODIS_NRT", "0,0,1,1", 1))

    def test_fingerprint_stable_and_specific(self):
        a = nrt.AreaRequest("MODIS_NRT", "0,0,1,1", 2, date(2023, 1, 1))
        b = nrt.AreaRequest("MODIS_NRT", "0,0,1,1", 2, date(2023, 1, 1))
        c = nrt.AreaRequest("MODIS_NRT", "0,0,1,1", 2, date(2023, 1, 2))
        assert nrt.request_fingerprint(a) == nrt.request_fingerprint(b)
        assert nrt.request_fingerprint(a) != nrt.request_fingerprint(c)

    def test_plan_chunks_verified_layout(self):
        chunks = nrt.plan_chunks("MODIS_NRT", "0,0,1,1", 7, date(2023, 1, 10))
        assert [(c.days, c.start_date) for c in chunks] == [
            (5, date(2023, 1, 6)), (2, date(2023, 1, 4))]
        # full coverage, no overlap, no gap
        covered = set()
        for c in chunks:
            for i in range(c.days):
                d = c.start_date + timedelta(days=i)
                assert d not in covered
                covered.add(d)
        assert len(covered) == 7

    def test_fetch_window_statuses(self, monkeypatch):
        def fake_fetch(url, **kw):
            return transport.FetchResult(b"h1,h2\n", 200, 1, 0.1, "redacted")

        monkeypatch.setattr(nrt.transport, "fetch_bytes", fake_fetch)
        res = nrt.fetch_window(nrt.AreaRequest("MODIS_NRT", "0,0,1,1", 1), key="K")
        assert res.status == "empty"  # header-only: valid, zero rows
        assert len(res.sha256) == 64


# --- archive -----------------------------------------------------------------

class TestArchive:
    def test_probe_published_and_missing(self, monkeypatch):
        monkeypatch.setattr("requests.head",
                            lambda *a, **k: FakeResp(headers={"Content-Length": "42"}))
        p = archive.probe_open_archive("modis", 2023, "Italy")
        assert (p.published, p.size_bytes) == (True, 42)
        monkeypatch.setattr("requests.head", lambda *a, **k: FakeResp(404, b"{}"))
        p = archive.probe_open_archive("modis", 1999 + 1, "Italy")
        assert p.published is False

    def test_fetch_success_and_guards(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            archive.transport, "fetch_bytes",
            lambda *a, **k: transport.FetchResult(b"a,b\n1,2\n", 200, 1, 0.1, "u"))
        staged = archive.fetch_open_archive("modis", 2023, "Italy", str(tmp_path))
        assert staged.filename == "modis_2023_Italy.csv"
        assert staged.size_bytes == 8 and len(staged.sha256) == 64
        with pytest.raises(transport.AcquisitionError):  # byte cap
            archive.fetch_open_archive("modis", 2023, "Italy", str(tmp_path), max_bytes=4)
        with pytest.raises(transport.AcquisitionError):  # overwrite refusal
            archive.fetch_open_archive("modis", 2023, "Italy", str(tmp_path))

    def test_fetch_404_maps_to_not_published(self, monkeypatch, tmp_path):
        def nf(*a, **k):
            raise transport.HttpError(404, "{}")
        monkeypatch.setattr(archive.transport, "fetch_bytes", nf)
        with pytest.raises(archive.ArchiveNotPublished):
            archive.fetch_open_archive("modis", 2023, "Italy", str(tmp_path))

    def test_request_spec(self):
        spec = archive.ArchiveRequestSpec(("MCD14ML",), ("terra",), "0,0,1,1",
                                          "2023-01-01", "2023-12-31")
        assert len(spec.spec_id) == 16
        assert "MANUAL" in spec.render_instructions()
        with pytest.raises(ValueError):
            archive.ArchiveRequestSpec(("MCD14ML",), ("pluto",), "0,0,1,1", "2023-01-01", "2023-12-31")

    def test_stage_file(self, tmp_path):
        src = tmp_path / "DL_FIRE_M-C61_1.csv"
        src.write_bytes(b"lat,lon\n1,2\n")
        inbox = tmp_path / "inbox"
        staged = archive.stage_file(str(src), product="MCD14ML", sensor_hint="modis",
                                    dest_dir=str(inbox))
        assert staged.filename == src.name and (inbox / src.name).read_bytes() == b"lat,lon\n1,2\n"
        with pytest.raises(FileNotFoundError):
            archive.stage_file(str(tmp_path / "missing.csv"))

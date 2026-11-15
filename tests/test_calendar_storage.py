"""Phase 1B tests: storage, manifest, integrity, manager. Tmp dirs only."""

import json
import os
import time

import pytest

from src.calendar.storage import integrity, layout, manager, manifest


@pytest.fixture()
def staged(tmp_path):
    src = tmp_path / "staging" / "modis_2023_United_States.csv"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"latitude,longitude\n39.7,-120.0\n")
    return src


class TestLayout:
    def test_paths_and_traversal_rejection(self, tmp_path):
        p = layout.raw_file_path(str(tmp_path), "modis", "terra", "b1", "f.csv")
        assert str(p).endswith("modis/terra/b1/f.csv")
        with pytest.raises(ValueError):
            layout.raw_file_path(str(tmp_path), "..", "s", "b", "f.csv")
        with pytest.raises(ValueError):
            layout.raw_file_path(str(tmp_path), "p", "s", "b", "a/b.csv")


class TestIntegrity:
    def test_hash_verify_sidecar_roundtrip(self, tmp_path):
        f = tmp_path / "x.bin"
        f.write_bytes(b"abc" * 1000)
        digest = integrity.sha256_file(f)
        assert integrity.verify_file(f, digest) is True
        assert integrity.verify_file(f, "0" * 64) is False
        sidecar = integrity.write_sidecar(f)
        assert sidecar.is_file()
        assert integrity.read_sidecar(f) == digest
        assert integrity.read_sidecar(tmp_path / "nope.bin") is None


class TestManifest:
    def test_register_and_idempotent_duplicate(self, tmp_path, staged):
        raw = str(tmp_path / "raw")
        first = manifest.register_staged(
            str(staged), raw_root=raw, product="modis", sensor="terra",
            batch_id="b1", provenance={"url": "http://x"})
        assert first.status == "registered"
        assert first.path.endswith("modis/terra/b1/modis_2023_United_States.csv")
        assert (tmp_path / "raw" / "modis" / "terra" / "b1" /
                "modis_2023_United_States.csv.sha256").is_file()
        again = manifest.register_staged(
            str(staged), raw_root=raw, product="modis", sensor="terra", batch_id="b1")
        assert again.status == "duplicate_of"
        assert again.path == first.path
        assert manifest.find_by_sha(raw, first.sha256) == [first.path]

    def test_occupied_by_different_bytes_refused(self, tmp_path, staged):
        raw = str(tmp_path / "raw")
        manifest.register_staged(str(staged), raw_root=raw, product="p", sensor="s",
                                 batch_id="b")
        staged.write_bytes(b"different bytes")
        with pytest.raises(manifest.RegistrationError):
            manifest.register_staged(str(staged), raw_root=raw, product="p", sensor="s",
                                     batch_id="b")

    def test_expected_sha_mismatch_refused(self, tmp_path, staged):
        with pytest.raises(manifest.RegistrationError):
            manifest.register_staged(str(staged), raw_root=str(tmp_path / "raw"),
                                     product="p", sensor="s", batch_id="b",
                                     expected_sha256="f" * 64)

    def test_quarantine_never_trusted(self, tmp_path, staged):
        raw = str(tmp_path / "raw")
        qpath = manifest.quarantine_file(str(staged), raw_root=raw,
                                         quarantine_dir="quarantine",
                                         reasons=["bad header", "short rows"])
        assert "quarantine" in qpath
        assert manifest.registered_files(raw) == {}
        events = manifest.load_events(raw)
        assert events[-1]["event"] == "quarantined"
        assert events[-1]["reasons"] == ["bad header", "short rows"]

    def test_verify_tree_ok_mismatch_missing(self, tmp_path, staged):
        raw = str(tmp_path / "raw")
        reg = manifest.register_staged(str(staged), raw_root=raw, product="p",
                                       sensor="s", batch_id="b")
        assert manifest.verify_tree(raw)["mismatch"] == []
        with open(reg.path, "ab") as handle:  # corrupt the TEST copy only
            handle.write(b"tamper")
        assert manifest.verify_tree(raw)["mismatch"] == [reg.path]
        os.remove(reg.path)
        assert manifest.verify_tree(raw)["missing"] == [reg.path]

    def test_manifest_is_jsonl_with_provenance(self, tmp_path, staged):
        raw = str(tmp_path / "raw")
        manifest.register_staged(str(staged), raw_root=raw, product="p", sensor="s",
                                 batch_id="b", provenance={"url": "http://x"})
        lines = (tmp_path / "raw" / "manifest.jsonl").read_text().strip().splitlines()
        event = json.loads(lines[-1])
        assert event["event"] == "registered" and event["provenance"]["url"] == "http://x"


class TestManager:
    @pytest.mark.parametrize("frac,expected", [
        (0.0, "Normal"), (0.69, "Normal"), (0.70, "Warning"), (0.84, "Warning"),
        (0.85, "High"), (0.94, "High"), (0.95, "Critical"), (1.0, "Critical"),
    ])
    def test_bands(self, frac, expected):
        assert manager.band(frac) == expected

    def test_ensure_capacity_pass_and_block(self, tmp_path, monkeypatch):
        from collections import namedtuple

        Usage = namedtuple("Usage", "total used free")
        monkeypatch.setattr("shutil.disk_usage",
                            lambda p: Usage(1000, 900, 100))
        ok = manager.ensure_capacity(str(tmp_path), 10, reserve_bytes=50, what="t")
        assert ok["free_bytes"] == 100
        with pytest.raises(manager.CapacityBlocked, match="blocked t"):
            manager.ensure_capacity(str(tmp_path), 60, reserve_bytes=50, what="t")

    def test_backup_roundtrip_idempotent(self, tmp_path, staged):
        raw, backup = str(tmp_path / "raw"), str(tmp_path / "mirror")
        manifest.register_staged(str(staged), raw_root=raw, product="p", sensor="s",
                                 batch_id="b")
        first = manager.backup_new_files(raw, backup)
        assert len(first["copied"]) == 1 and first["failed"] == []
        second = manager.backup_new_files(raw, backup)
        assert second["copied"] == [] and len(second["skipped"]) == 1

    def test_storage_report_shape(self, tmp_path, staged):
        raw, backup = str(tmp_path / "raw"), str(tmp_path / "mirror")
        manifest.register_staged(str(staged), raw_root=raw, product="p", sensor="s",
                                 batch_id="b")
        rep = manager.storage_report(raw, backup)
        assert rep["registered_files"] == 1
        assert rep["backup_missing_files"] == 1
        assert rep["disk"]["band"] in ("Normal", "Warning", "High", "Critical")
        assert rep["dataset_bytes"] == {"p/s": staged.stat().st_size}

    def test_registered_bytes_immutable_under_ops(self, tmp_path, staged):
        raw, backup = str(tmp_path / "raw"), str(tmp_path / "mirror")
        reg = manifest.register_staged(str(staged), raw_root=raw, product="p",
                                       sensor="s", batch_id="b")
        before = (open(reg.path, "rb").read(), os.stat(reg.path).st_mtime_ns)
        time.sleep(0.01)
        manifest.verify_tree(raw)
        manager.storage_report(raw, backup)
        manager.backup_new_files(raw, backup)
        after = (open(reg.path, "rb").read(), os.stat(reg.path).st_mtime_ns)
        assert before == after  # verify/report/backup never touch raw bytes

"""
tests/test_sar.py
=================
Pytest tests for src/sar/change_detection.py.
All tests use synthetic data; no network access required.
The last test monkeypatches the socket module to guarantee no network is used.
"""

import os, sys, socket, tempfile
import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

# Allow imports from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.sar.change_detection import valid_mask, multilook, delta_db, load_cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_uniform(h, w, value, dtype=np.float32):
    return np.full((h, w), value, dtype=dtype)


def all_valid(h, w):
    return np.ones((h, w), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Test multilook: known mean in LINEAR power
# ---------------------------------------------------------------------------

class TestMultilook:
    def test_uniform_array_mean_linear(self):
        """multilook of a uniform array should equal that constant."""
        k = 4
        h, w = 8, 8
        value = 0.04  # linear σ⁰
        power = make_uniform(h, w, value)
        valid = all_valid(h, w)
        ml = multilook(power, valid, k)
        assert ml.shape == (h // k, w // k)
        np.testing.assert_allclose(ml, value, rtol=1e-5)

    def test_linear_mean_differs_from_db_mean(self):
        """Demonstrate that averaging in linear power ≠ averaging in dB.

        Block contains alternating 0.01 and 0.09 (linear power).
        Linear mean = 0.05; dB mean = 10*log10(0.05) = -13.01 dB.
        Mean of individual dBs: (10*log10(0.01) + 10*log10(0.09)) / 2 = (-20 + -10.46) / 2 = -15.23 dB.
        These two values are different.
        """
        # 2×2 block, k=2
        power = np.array([[0.01, 0.09],
                           [0.09, 0.01]], dtype=np.float32)
        valid = all_valid(2, 2)
        ml = multilook(power, valid, 2)
        assert ml.shape == (1, 1)

        linear_mean = 0.05
        db_of_linear_mean     = 10 * np.log10(linear_mean)  # -13.01
        mean_of_db_values     = 0.5 * (10*np.log10(0.01) + 10*np.log10(0.09))  # -15.23

        np.testing.assert_allclose(ml[0, 0], linear_mean, rtol=1e-5,
                                   err_msg="multilook should return linear mean, not dB mean")
        # Confirm the two dB numbers differ
        assert abs(db_of_linear_mean - mean_of_db_values) > 1.0, (
            f"Expected a difference > 1 dB between linear-mean-then-dB "
            f"({db_of_linear_mean:.3f}) and mean-of-dBs ({mean_of_db_values:.3f})"
        )

    def test_block_lt50pct_valid_becomes_nan(self):
        """Blocks with < 50 % valid pixels → NaN (state rule)."""
        k = 4  # 4×4 = 16 pixels; need < 8 valid to trigger NaN
        power = make_uniform(4, 4, 0.05)
        valid = np.zeros((4, 4), dtype=np.uint8)
        # Mark only 7 of 16 pixels valid (43.75 % < 50 %)
        valid[0, :4] = 1  # 4 valid
        valid[1, :3] = 1  # 3 valid => total 7/16
        ml = multilook(power, valid, k)
        assert ml.shape == (1, 1)
        assert np.isnan(ml[0, 0]), f"Expected NaN, got {ml[0, 0]}"

    def test_block_exactly50pct_valid_is_not_nan(self):
        """Block with exactly 50 % valid pixels should NOT be NaN."""
        k = 4  # 16 pixels; 8 valid = 50 %
        power = make_uniform(4, 4, 0.05)
        valid = np.zeros((4, 4), dtype=np.uint8)
        valid[:2, :4] = 1  # first two rows = 8 valid
        ml = multilook(power, valid, k)
        assert ml.shape == (1, 1)
        assert np.isfinite(ml[0, 0]), f"Expected finite, got {ml[0, 0]}"
        np.testing.assert_allclose(ml[0, 0], 0.05, rtol=1e-5)

    def test_nan_power_treated_as_invalid(self):
        """NaN pixels in power are excluded even if valid mask says 1."""
        k = 2
        power = np.array([[np.nan, 0.04],
                           [0.04,  0.04]], dtype=np.float32)
        valid = all_valid(2, 2)
        ml = multilook(power, valid, k)
        # 3 of 4 pixels have finite power => 75% valid
        np.testing.assert_allclose(ml[0, 0], 0.04, rtol=1e-5)

    def test_mask0_excluded_from_mean(self):
        """mask==0 pixels must not contribute to the block mean."""
        k = 2
        # Block: two valid pixels (0.01) and two invalid (0.09, mask=0)
        power = np.array([[0.09, 0.01],
                           [0.01, 0.09]], dtype=np.float32)
        valid = np.array([[0, 1],
                          [1, 0]], dtype=np.uint8)
        ml = multilook(power, valid, k)
        # Mean should be from the two valid pixels: (0.01 + 0.01) / 2 = 0.01
        np.testing.assert_allclose(ml[0, 0], 0.01, rtol=1e-5,
                                   err_msg="mask=0 pixels must not affect mean")


# ---------------------------------------------------------------------------
# Test delta_db
# ---------------------------------------------------------------------------

class TestDeltaDb:
    def test_identical_arrays_give_zero(self):
        """delta_db of identical PRE and POST → 0 dB everywhere."""
        arr = np.array([[0.05, 0.10], [0.02, 0.08]], dtype=np.float32)
        ddb = delta_db(arr, arr)
        np.testing.assert_allclose(ddb, 0.0, atol=1e-6)

    def test_sign_correct_for_decrease(self):
        """POST < PRE ⟹ delta_dB < 0 (decrease in backscatter)."""
        pre  = np.array([[1.0]], dtype=np.float32)
        post = np.array([[0.1]], dtype=np.float32)  # 10 dB decrease
        ddb  = delta_db(pre, post)
        np.testing.assert_allclose(ddb[0, 0], -10.0, atol=1e-4)

    def test_sign_correct_for_increase(self):
        """POST > PRE ⟹ delta_dB > 0."""
        pre  = np.array([[0.1]], dtype=np.float32)
        post = np.array([[1.0]], dtype=np.float32)  # +10 dB
        ddb  = delta_db(pre, post)
        np.testing.assert_allclose(ddb[0, 0], +10.0, atol=1e-4)

    def test_known_values(self):
        """Hand-computed check: post/pre = 0.5 → -3.01 dB."""
        pre  = np.array([[2.0]], dtype=np.float32)
        post = np.array([[1.0]], dtype=np.float32)
        ddb  = delta_db(pre, post)
        expected = 10.0 * np.log10(0.5)  # -3.0103 dB
        np.testing.assert_allclose(ddb[0, 0], expected, atol=1e-4)

    def test_nan_propagates(self):
        """NaN in either input → NaN in output."""
        pre  = np.array([[np.nan, 0.1]], dtype=np.float32)
        post = np.array([[0.1,    np.nan]], dtype=np.float32)
        ddb  = delta_db(pre, post)
        assert np.isnan(ddb[0, 0]), "NaN pre → NaN output"
        assert np.isnan(ddb[0, 1]), "NaN post → NaN output"

    def test_zero_or_negative_gives_nan(self):
        """Zero or negative power ⟹ NaN (log undefined)."""
        pre  = np.array([[0.0, -0.1, 0.1]], dtype=np.float32)
        post = np.array([[0.1,  0.1, 0.0]], dtype=np.float32)
        ddb  = delta_db(pre, post)
        assert np.isnan(ddb[0, 0]), "zero pre → NaN"
        assert np.isnan(ddb[0, 1]), "negative pre → NaN"
        assert np.isnan(ddb[0, 2]), "zero post → NaN"

    def test_output_dtype_is_float32(self):
        arr = np.array([[0.1, 0.2]], dtype=np.float32)
        ddb = delta_db(arr, arr)
        assert ddb.dtype == np.float32


# ---------------------------------------------------------------------------
# Test valid_mask
# ---------------------------------------------------------------------------

class TestValidMask:
    def test_all_valid(self):
        ones = np.ones((3, 3), dtype=np.uint8)
        pwr  = make_uniform(3, 3, 0.05)
        vm   = valid_mask(ones, ones, pwr, pwr)
        assert vm.all()

    def test_mask0_excluded(self):
        mask_pre  = np.ones((2, 2), dtype=np.uint8)
        mask_post = np.ones((2, 2), dtype=np.uint8)
        mask_pre[0, 0] = 0   # invalid in PRE
        pwr = make_uniform(2, 2, 0.05)
        vm  = valid_mask(mask_pre, mask_post, pwr, pwr)
        assert not vm[0, 0], "mask_pre=0 should make pixel invalid"
        assert vm[1, 1], "other pixels should still be valid"

    def test_zero_power_excluded(self):
        m = np.ones((2, 2), dtype=np.uint8)
        pwr_pre  = make_uniform(2, 2, 0.05)
        pwr_post = make_uniform(2, 2, 0.05)
        pwr_pre[0, 0]  = 0.0  # zero
        pwr_post[1, 1] = 0.0
        vm = valid_mask(m, m, pwr_pre, pwr_post)
        assert not vm[0, 0]
        assert not vm[1, 1]
        assert vm[0, 1] and vm[1, 0]

    def test_nan_power_excluded(self):
        m = np.ones((2, 2), dtype=np.uint8)
        pwr = make_uniform(2, 2, 0.05)
        pwr[0, 0] = np.nan
        vm = valid_mask(m, m, pwr, pwr)
        assert not vm[0, 0]

    def test_negative_power_excluded(self):
        m = np.ones((2, 2), dtype=np.uint8)
        pwr = make_uniform(2, 2, 0.05)
        pwr_post = pwr.copy()
        pwr_post[0, 0] = -0.01
        vm = valid_mask(m, m, pwr, pwr_post)
        assert not vm[0, 0]


# ---------------------------------------------------------------------------
# Test GeoTIFF round-trip
# ---------------------------------------------------------------------------

class TestGeoTIFFRoundTrip:
    def test_round_trip_preserves_transform_and_crs(self, tmp_path):
        """Write a float32 GeoTIFF and read it back; check transform and CRS."""
        arr = np.array([[1.0, 2.0, np.nan],
                        [4.0, np.nan, 6.0]], dtype=np.float32)
        transform = from_origin(west=234885.0, north=4434475.0,
                                xsize=100.0, ysize=100.0)
        # Build CRS from WKT to avoid PROJ database version conflicts in some envs
        _WKT_32611 = (
            'PROJCS["WGS 84 / UTM zone 11N",'
            'GEOGCS["WGS 84",DATUM["WGS_1984",'
            'SPHEROID["WGS 84",6378137,298.257223563]],'
            'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
            'PROJECTION["Transverse_Mercator"],'
            'PARAMETER["latitude_of_origin",0],'
            'PARAMETER["central_meridian",-117],'
            'PARAMETER["scale_factor",0.9996],'
            'PARAMETER["false_easting",500000],'
            'PARAMETER["false_northing",0],'
            'UNIT["metre",1]]'
        )
        try:
            crs = CRS.from_epsg(32611)
        except Exception:
            crs = CRS.from_wkt(_WKT_32611)
        out_path = str(tmp_path / "test.tif")

        with rasterio.open(
            out_path, "w",
            driver="GTiff", height=2, width=3,
            count=1, dtype="float32",
            crs=crs, transform=transform, nodata=np.nan,
        ) as dst:
            dst.write(arr, 1)

        with rasterio.open(out_path) as src:
            arr_back = src.read(1)
            # Check the CRS was round-tripped by comparing EPSG authority if available,
            # falling back to substring match on the WKT projection name.
            read_wkt = src.crs.to_wkt()
            assert ("UTM zone 11N" in read_wkt or "UTM_zone_11N" in read_wkt
                    or "32611" in read_wkt), f"Unexpected CRS WKT: {read_wkt[:120]}"
            assert src.transform == transform
            assert src.nodata != src.nodata  # NaN != NaN

        # Finite pixels match; NaN pixels are NaN
        finite = np.isfinite(arr)
        np.testing.assert_allclose(arr_back[finite], arr[finite], rtol=1e-6)
        assert np.all(np.isnan(arr_back[~finite]))


# ---------------------------------------------------------------------------
# Test: module imports and runs with no network
# ---------------------------------------------------------------------------

class TestNoNetwork:
    def test_module_importable_no_network(self, monkeypatch):
        """The module must import and run multilook/delta_db with no network."""

        def _no_socket(*args, **kwargs):
            raise OSError("Network access is blocked in this test")

        monkeypatch.setattr(socket, "socket", _no_socket)

        # Re-import should succeed (no top-level network calls)
        import importlib
        import src.sar.change_detection as mod
        importlib.reload(mod)

        # Core functions must run without network
        pwr = make_uniform(4, 4, 0.05)
        vm  = all_valid(4, 4)
        ml  = mod.multilook(pwr, vm, 2)
        ddb = mod.delta_db(ml, ml)
        np.testing.assert_allclose(ddb, 0.0, atol=1e-6)

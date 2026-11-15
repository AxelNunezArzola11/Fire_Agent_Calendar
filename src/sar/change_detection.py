"""
src/sar/change_detection.py
===========================
Pure functions for SAR change-detection built on the local NISAR cache.

No global state. No network access. No side effects beyond what is
explicitly returned.

Grid convention (from data/cache/meta.json)
-------------------------------------------
  origin   : (X0=234885.0 m, Y_TOP=4434475.0 m)  EPSG:32611
  pixel     : 10 m × 10 m  (positive X east, negative Y north)
  array size: 4096 rows × 4096 columns

Backscatter arrays
------------------
  All NISAR GCOV arrays are stored as linear POWER (not amplitude, not dB).
  The NISAR convention for diagonal terms (HHHH, HVHV) in GCOV is that each
  pixel value represents the calibrated backscatter cross-section per unit
  area σ⁰ in linear power units.
  Units: dimensionless (σ⁰ linear)

Mask convention
---------------
  mask == 1  ⟹  pixel is valid (confirmed in Task 3c Step 2)
  mask == 0  ⟹  pixel is invalid / no-data
"""

from __future__ import annotations

import json
import os
from typing import Tuple

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import from_origin

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Array2D = np.ndarray   # 2-D float32 or uint8 ndarray


# ---------------------------------------------------------------------------
# load_cache
# ---------------------------------------------------------------------------

def load_cache(
    cache_dir: str,
) -> Tuple[Array2D, Array2D, Array2D, Array2D, Array2D, Array2D, dict, object, CRS]:
    """Load all NISAR arrays from the local .npy cache.

    Parameters
    ----------
    cache_dir : str
        Path to directory containing ``pre_hhhh.npy``, ``pre_hvhv.npy``,
        ``pre_mask.npy``, ``post_hhhh.npy``, ``post_hvhv.npy``,
        ``post_mask.npy``, and ``meta.json``.

    Returns
    -------
    pre_hhhh, pre_hvhv, pre_mask, post_hhhh, post_hvhv, post_mask : Array2D
        ``pre/post_hhhh`` and ``pre/post_hvhv`` are float32 arrays of linear
        SAR backscatter power (σ⁰, dimensionless).  Values are positive-finite
        where valid and NaN where the sensor reported no data.
        ``pre/post_mask`` are uint8 arrays (0 = invalid, 1 = valid).
    meta : dict
        Parsed ``meta.json`` (geotransform, crs, row_slice, col_slice).
    transform : rasterio.Affine
        Affine geotransform derived from ``meta["geotransform"]``.
        ``transform * (col, row)`` gives the (easting, northing) of the
        upper-left corner of that pixel.
    crs : rasterio.crs.CRS
        Coordinate reference system (EPSG:32611).
    """
    meta_path = os.path.join(cache_dir, "meta.json")
    with open(meta_path) as fh:
        meta = json.load(fh)

    gt = meta["geotransform"]           # [x0, dx, 0, y_top, 0, -dy]
    transform = from_origin(
        west=gt[0],
        north=gt[3],
        xsize=gt[1],
        ysize=abs(gt[5]),
    )
    crs = CRS.from_string(meta["crs"])  # EPSG:32611

    def _load(name: str, dtype) -> Array2D:
        arr = np.load(os.path.join(cache_dir, name), mmap_mode="r")
        return arr.astype(dtype, copy=False)

    pre_hhhh  = _load("pre_hhhh.npy",  np.float32)
    pre_hvhv  = _load("pre_hvhv.npy",  np.float32)
    pre_mask  = _load("pre_mask.npy",  np.uint8)
    post_hhhh = _load("post_hhhh.npy", np.float32)
    post_hvhv = _load("post_hvhv.npy", np.float32)
    post_mask = _load("post_mask.npy", np.uint8)

    return pre_hhhh, pre_hvhv, pre_mask, post_hhhh, post_hvhv, post_mask, meta, transform, crs


# ---------------------------------------------------------------------------
# valid_mask
# ---------------------------------------------------------------------------

def valid_mask(
    mask_pre: Array2D,
    mask_post: Array2D,
    pwr_pre: Array2D,
    pwr_post: Array2D,
) -> Array2D:
    """Compute a boolean validity mask for a single polarisation channel.

    A pixel is valid (True) when ALL of the following hold:

    1. ``mask_pre == 1`` AND ``mask_post == 1``
       (NISAR sensor quality flag; 1 = valid).
    2. ``pwr_pre > 0`` AND ``pwr_post > 0``
       (exclude zero or negative power: physically impossible for σ⁰).
    3. ``np.isfinite(pwr_pre)`` AND ``np.isfinite(pwr_post)``
       (exclude NaN and ±inf stored in the array).

    Parameters
    ----------
    mask_pre, mask_post : Array2D  (uint8, shape H×W)
        Sensor quality flags for the PRE and POST acquisitions.
        Value 1 = valid, 0 = invalid.
    pwr_pre, pwr_post : Array2D  (float32, shape H×W)
        Linear SAR backscatter power for the PRE and POST acquisitions.
        Units: dimensionless σ⁰ linear.

    Returns
    -------
    valid : Array2D  (bool, shape H×W)
        True where the pixel is usable; False otherwise.
        NaN pixels in the input power arrays always produce False.
    """
    valid = (
        (mask_pre == 1)
        & (mask_post == 1)
        & np.isfinite(pwr_pre)
        & np.isfinite(pwr_post)
        & (pwr_pre > 0)
        & (pwr_post > 0)
    )
    return valid


# ---------------------------------------------------------------------------
# multilook
# ---------------------------------------------------------------------------

def multilook(
    power: Array2D,
    valid: Array2D,
    k: int,
) -> Array2D:
    """Block-average SAR linear power over a k×k window using only valid pixels.

    Processing is done in-plane without allocating a full-resolution working
    copy: for each non-overlapping k×k block the mean of valid pixels is
    computed.  Blocks with fewer than 50 % valid pixels are set to NaN (the
    "< 50% valid" rule ensures that the multi-looked value is not dominated
    by a sparse sample).

    This implements multilooking in LINEAR power, as required.  Converting to
    dB after averaging is NOT the same as averaging in dB: the difference
    can be several tenths of a dB for heterogeneous targets.

    Parameters
    ----------
    power : Array2D  (float32, shape H×W)
        Linear SAR backscatter power, in units of σ⁰ (dimensionless).
        NaN pixels are treated as invalid regardless of *valid*.
    valid : Array2D  (bool, shape H×W)
        Pixel-level validity mask.  Pixels where ``valid == False`` are
        excluded from the block mean even if *power* is finite there.
    k : int
        Side length of the square multilook window in pixels.
        E.g. k=10 produces a 10×10 multilook ⟹ output pixel covers 100 m²
        when the input pixel spacing is 10 m.

    Returns
    -------
    ml : Array2D  (float32, shape H//k × W//k)
        Multi-looked linear power array.  Each output pixel is the mean of
        valid linear-power values in the corresponding k×k input block.
        Output is NaN where the block has < 50 % valid pixels.
        Pixels that are NaN in *power* OR where ``valid == False`` are both
        excluded from the mean.
    """
    H, W = power.shape
    nr = H // k
    nc = W // k

    # Crop to exact multiple of k
    power_c = power[:nr * k, :nc * k]
    valid_c = valid[:nr * k, :nc * k]

    # Reshape to (nr, k, nc, k) for vectorised block stats
    p = power_c.reshape(nr, k, nc, k)
    v = valid_c.reshape(nr, k, nc, k)

    # Mask: valid AND finite in power
    use = v & np.isfinite(p)

    # Sum of valid power values and count per block
    p_masked = np.where(use, p.astype(np.float64), 0.0)
    sum_pwr  = p_masked.sum(axis=(1, 3))   # shape (nr, nc)
    cnt_use  = use.sum(axis=(1, 3))         # number of valid pixels per block

    total_k2 = float(k * k)
    frac_valid = cnt_use / total_k2

    with np.errstate(invalid="ignore", divide="ignore"):
        ml = (sum_pwr / cnt_use).astype(np.float32)

    # Blocks with < 50 % valid pixels → NaN
    ml[frac_valid < 0.5] = np.nan

    return ml


# ---------------------------------------------------------------------------
# delta_db
# ---------------------------------------------------------------------------

def delta_db(
    pre_ml: Array2D,
    post_ml: Array2D,
) -> Array2D:
    """Compute delta backscatter in dB: 10·log10(post_ml / pre_ml).

    A negative value means the POST backscatter decreased relative to PRE
    (e.g. fire-affected area).  A positive value means an increase.

    Parameters
    ----------
    pre_ml, post_ml : Array2D  (float32, shape H×W)
        Multi-looked linear backscatter power for the PRE and POST
        acquisitions.  Both must be on the same grid (same shape).
        NaN pixels produce NaN in the output.
        Zero or negative pixels produce NaN (log is undefined there).
        Units: dimensionless σ⁰ linear (output is in dB).

    Returns
    -------
    ddb : Array2D  (float32, shape H×W)
        delta backscatter in dB.  NaN where:
          - either input is NaN,
          - either input is ≤ 0, or
          - either input is non-finite.
        Otherwise: ``10 * log10(post_ml / pre_ml)`` cast to float32.
    """
    pre_f  = pre_ml.astype(np.float64)
    post_f = post_ml.astype(np.float64)

    # Identify usable pixels
    good = (
        np.isfinite(pre_f)
        & np.isfinite(post_f)
        & (pre_f > 0)
        & (post_f > 0)
    )

    out = np.full(pre_ml.shape, np.nan, dtype=np.float64)
    out[good] = 10.0 * np.log10(post_f[good] / pre_f[good])

    return out.astype(np.float32)

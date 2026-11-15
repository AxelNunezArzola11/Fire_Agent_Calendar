"""Granule readers: format-aware variable access (honest limits).

NetCDF4 granules (VIIRS) open via rasterio/GDAL subdatasets. HDF4
(MODIS MxD) opens via the system GDAL CLI (gdalinfo/gdal_translate),
which ships HDF4 support: rasterio's bundled GDAL lacks the driver
and installs are out of scope, so the CLI bridge is the reader.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

NETCDF_SUFFIXES = (".nc", ".nc4")
HDF4_SUFFIXES = (".hdf", ".h4", ".hdf4")
HDF5_SUFFIXES = (".h5", ".he5", ".hdf5")


def _gdal_cli(*names: str):
    """First available system GDAL binary (None when absent)."""
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


def detect_format(path: str | Path) -> dict:
    """{format, readable, reason} from suffix + tool availability."""
    suffix = Path(str(path)).suffix.lower()
    if suffix in NETCDF_SUFFIXES:
        return {"format": "NetCDF4", "readable": True,
                "reason": "rasterio netCDF driver available"}
    if suffix in HDF4_SUFFIXES:
        tool = _gdal_cli("gdalinfo")
        return {"format": "HDF4", "readable": tool is not None,
                "reason": "system gdalinfo HDF4 bridge"
                if tool else "no gdalinfo binary found"}
    if suffix in HDF5_SUFFIXES:
        tool = _gdal_cli("gdalinfo")
        return {"format": "HDF5", "readable": tool is not None,
                "reason": "system gdalinfo HDF5 bridge"
                if tool else "no gdalinfo binary found"}
    return {"format": "unknown", "readable": False,
            "reason": f"unsupported suffix {suffix!r}"}


def _gdal_catalog(path: str) -> list[tuple[str, str]]:
    """(subdataset name, description) pairs via gdalinfo -json."""
    import json as _json

    tool = _gdal_cli("gdalinfo")
    if tool is None:
        raise _ReaderError("no gdalinfo binary found")
    try:
        proc = subprocess.run(
            [tool, "-json", str(path)], capture_output=True, text=True,
            timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        raise _ReaderError(f"gdalinfo failed: {type(exc).__name__}")
    if proc.returncode != 0:
        raise _ReaderError(f"gdalinfo failed: {proc.stderr[:200]}")
    try:
        info = _json.loads(proc.stdout or "{}")
    except ValueError:
        raise _ReaderError("gdalinfo returned invalid JSON")
    flat = (info.get("metadata") or {}).get("SUBDATASETS") or {}
    pairs: list[tuple[str, str]] = []
    if isinstance(flat, dict):
        stems = {key[:-5] for key in flat if key.endswith("_NAME")}
        pairs = [(str(flat.get(stem + "_NAME") or ""),
                  str(flat.get(stem + "_DESC") or ""))
                 for stem in sorted(stems)]
        pairs = [(name, desc) for name, desc in pairs if name]
    if pairs:
        return pairs
    listed = info.get("subdatasets") or []
    return [(str(sub["name"]), str(sub.get("description") or ""))
            for sub in listed
            if isinstance(sub, dict) and sub.get("name")]


def _gdal_subdatasets(path: str) -> list[str]:
    return [name for name, _ in _gdal_catalog(path)]


def subdatasets(path: str | Path) -> list[str]:
    """Subdataset inventory of a readable granule (else _ReaderError)."""
    import rasterio

    info = detect_format(path)
    if not info["readable"]:
        raise _ReaderError(info["reason"])
    if info["format"] in ("HDF4", "HDF5"):
        return _gdal_subdatasets(str(path))
    with rasterio.open(path) as dataset:
        return list(dataset.subdatasets or [])


def read_variable(path: str | Path, variable: str):
    """One variable as a NumPy array + profile (metadata dict).

    Returns {values, profile, subdataset}. Raises _ReaderError when
    the format is unsupported or the variable is absent. HDF reads
    go through gdal_translate into a temporary GeoTIFF (cleaned up).
    """
    import rasterio

    info = detect_format(path)
    if not info["readable"]:
        raise _ReaderError(info["reason"])
    if info["format"] in ("HDF4", "HDF5"):
        for name, desc in _gdal_catalog(str(path)):
            hay = f"{name} {desc}".lower()
            if variable.lower() in hay:
                return _gdal_read_variable(name)
        raise _ReaderError(f"variable {variable!r} not in granule")
    target = None
    for sub in subdatasets(path):
        if sub.endswith(f":{variable}") or sub.endswith(f"/{variable}"):
            target = sub
            break
    if target is None:
        raise _ReaderError(f"variable {variable!r} not in granule")
    with rasterio.open(target) as dataset:
        return {"values": dataset.read(1),
                "profile": dict(dataset.profile),
                "subdataset": target}


def _gdal_read_variable(subdataset: str) -> dict:
    import tempfile

    import rasterio

    tool = _gdal_cli("gdal_translate")
    if tool is None:
        raise _ReaderError("no gdal_translate binary found")
    with tempfile.TemporaryDirectory(prefix="fireagent-hdf-") as work:
        dest = str(Path(work) / "band.tif")
        try:
            proc = subprocess.run(
                [tool, "-of", "GTiff", subdataset, dest],
                capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.SubprocessError) as exc:
            raise _ReaderError(
                f"gdal_translate failed: {type(exc).__name__}")
        if proc.returncode != 0:
            raise _ReaderError(
                f"gdal_translate failed: {proc.stderr[:200]}")
        with rasterio.open(dest) as dataset:
            return {"values": dataset.read(1),
                    "profile": dict(dataset.profile),
                    "subdataset": subdataset}


class _ReaderError(Exception):
    """Granule read failure (format, missing file, or absent variable)."""

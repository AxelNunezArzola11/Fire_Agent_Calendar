"""L2 product catalog: definitions for OOp input products.

Concept IDs are discovered live via CMR (never hardcoded): versions
pin the science, CMR resolves the current collection concept id.
"""

from __future__ import annotations

PRODUCTS = {
    # MODIS Collection 6.1 (HDF4; not readable in this runtime).
    "MOD14": {"platform": "Terra", "version": "061",
              "provides": ("fire mask", "FRP", "QA"),
              "format": "HDF4", "readable_here": True,
              "equations": ("Eq.5", "Eq.6", "Eq.7")},
    "MYD14": {"platform": "Aqua", "version": "061",
              "provides": ("fire mask", "FRP", "QA"),
              "format": "HDF4", "readable_here": True,
              "equations": ("Eq.5", "Eq.6", "Eq.7")},
    "MOD35": {"platform": "Terra", "version": "061",
              "provides": ("cloud mask",),
              "format": "HDF4", "readable_here": True,
              "equations": ("Eq.10", "Eq.11")},
    "MYD35": {"platform": "Aqua", "version": "061",
              "provides": ("cloud mask",),
              "format": "HDF4", "readable_here": True,
              "equations": ("Eq.10", "Eq.11")},
    "MOD03": {"platform": "Terra", "version": "061",
              "provides": ("geolocation", "scan geometry"),
              "format": "HDF4", "readable_here": True,
              "equations": ("Eq.1", "Eq.2")},
    "MYD03": {"platform": "Aqua", "version": "061",
              "provides": ("geolocation", "scan geometry"),
              "format": "HDF4", "readable_here": True,
              "equations": ("Eq.1", "Eq.2")},
    "MOD04_L2": {"platform": "Terra", "version": "061",
                 "provides": ("aerosol optical depth",),
                 "format": "HDF4", "readable_here": True,
                 "equations": ("Eq.12",)},
    "MYD04_L2": {"platform": "Aqua", "version": "061",
                 "provides": ("aerosol optical depth",),
                 "format": "HDF4", "readable_here": True,
                 "equations": ("Eq.12",)},
    # VIIRS SNPP Collection 2 (NetCDF4; readable via rasterio/GDAL).
    "VNP14IMG": {"platform": "SNPP", "version": "002",
                 "provides": ("fire mask", "FRP", "QA"),
                 "format": "NetCDF4", "readable_here": True,
                 "equations": ("Eq.5", "Eq.6", "Eq.7")},
    "VNP03IMG": {"platform": "SNPP", "version": "2",
                 "provides": ("geolocation",),
                 "format": "NetCDF4", "readable_here": True,
                 "equations": ("Eq.1", "Eq.2")},
}


def get(short_name: str) -> dict:
    """Product definition (ValueError on unknown short name)."""
    try:
        return dict(PRODUCTS[short_name])
    except KeyError:
        raise ValueError(f"unknown L2 product {short_name!r}; want "
                         f"{sorted(PRODUCTS)}")


def needed_for_equation(equation: str) -> list[str]:
    """Short names whose variables feed one paper equation."""
    return sorted(short for short, spec in PRODUCTS.items()
                  if equation in spec["equations"])

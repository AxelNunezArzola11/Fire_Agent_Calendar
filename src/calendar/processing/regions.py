"""Region registry + analytical geographic filtering (raw files untouched).

All regions are explicit WGS84 (`EPSG:4326`) approximate bounding boxes with
documented provenance and version identifiers. Preset regions cover global
demonstration areas (California, Brazil, Spain, Australia, and more) alongside
arbitrary user-defined bounding boxes and point+radius areas.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Region:
    id: str
    bbox: tuple[float, float, float, float]  # W, S, E, N (degrees, EPSG:4326)
    crs: str
    source: str  # boundary provenance
    version: str
    label: str = ""
    country: str = ""


CALIFORNIA = Region(
    id="california",
    bbox=(-124.48, 32.53, -114.13, 42.01),
    crs="EPSG:4326",
    source=("Phase 1D approximate state extent (land+water bbox, includes "
            "Channel Islands; NOT an official boundary polygon)"),
    version="cal_bbox_v1",
    label="California (USA)",
    country="United States",
)

BRAZIL = Region(
    id="brazil",
    bbox=(-73.99, -33.75, -34.79, 5.27),
    crs="EPSG:4326",
    source=("Natural Earth / UN national envelope approximate bbox for Brazil "
            "(NOT an official administrative boundary polygon)"),
    version="bra_bbox_v1",
    label="Brazil",
    country="Brazil",
)

SPAIN = Region(
    id="spain",
    bbox=(-9.30, 35.95, 3.33, 43.79),
    crs="EPSG:4326",
    source=("Natural Earth / IGN mainland & Balearic Islands approximate bbox for Spain "
            "(NOT an official administrative boundary polygon)"),
    version="esp_bbox_v1",
    label="Spain",
    country="Spain",
)

AUSTRALIA = Region(
    id="australia",
    bbox=(112.92, -43.64, 153.64, -10.68),
    crs="EPSG:4326",
    source=("Natural Earth / Geoscience Australia mainland & Tasmania approximate bbox "
            "(NOT an official administrative boundary polygon)"),
    version="aus_bbox_v1",
    label="Australia",
    country="Australia",
)

CONUS = Region(
    id="conus",
    bbox=(-125.00, 24.00, -66.50, 49.50),
    crs="EPSG:4326",
    source=("Approximate contiguous United States bbox "
            "(NOT an official administrative boundary polygon)"),
    version="conus_bbox_v1",
    label="Contiguous United States",
    country="United States",
)

PORTUGAL = Region(
    id="portugal",
    bbox=(-9.55, 36.95, -6.18, 42.16),
    crs="EPSG:4326",
    source=("Natural Earth mainland Portugal approximate bbox "
            "(NOT an official administrative boundary polygon)"),
    version="prt_bbox_v1",
    label="Portugal",
    country="Portugal",
)

CANADA = Region(
    id="canada",
    bbox=(-141.00, 41.70, -52.60, 70.00),
    crs="EPSG:4326",
    source=("Natural Earth approximate envelope for Canada "
            "(NOT an official administrative boundary polygon)"),
    version="can_bbox_v1",
    label="Canada",
    country="Canada",
)

CHILE = Region(
    id="chile",
    bbox=(-75.60, -55.90, -66.40, -17.50),
    crs="EPSG:4326",
    source=("Natural Earth approximate envelope for Chile "
            "(NOT an official administrative boundary polygon)"),
    version="chl_bbox_v1",
    label="Chile",
    country="Chile",
)

MEXICO = Region(
    id="mexico",
    bbox=(-117.12, 14.53, -86.71, 32.72),
    crs="EPSG:4326",
    source=("Natural Earth / INEGI approximate mainland & Baja envelope for Mexico "
            "(NOT an official administrative boundary polygon)"),
    version="mex_bbox_v1",
    label="Mexico",
    country="Mexico",
)

MADRID = Region(
    id="madrid",
    bbox=(-4.89, 39.52, -2.51, 41.32),
    crs="EPSG:4326",
    source=("Equirectangular ~100 km envelope centered on Madrid (40.42 N, 3.70 W) "
            "(NOT an official administrative boundary polygon)"),
    version="mad_bbox_v1",
    label="Madrid (Spain · 100 km)",
    country="Spain",
)

REGIONS: dict[str, Region] = {
    r.id: r
    for r in (
        CALIFORNIA,
        MADRID,
        MEXICO,
        BRAZIL,
        SPAIN,
        AUSTRALIA,
        CONUS,
        PORTUGAL,
        CANADA,
        CHILE,
    )
}


def region_label(region_id: str) -> str:
    """Human-friendly label for a preset region id."""
    reg = REGIONS.get(region_id)
    if reg is None:
        return str(region_id)
    return reg.label or reg.id


def filter_region(df: pd.DataFrame, region_id: str) -> tuple[pd.DataFrame, dict]:
    """Return (filtered_copy, stats). Pure: input untouched, coords unchanged."""
    region = REGIONS[region_id]  # KeyError for unknown regions
    w, s, e, n = region.bbox
    mask = ((df["longitude"] >= w) & (df["longitude"] <= e)
            & (df["latitude"] >= s) & (df["latitude"] <= n))
    kept = df.loc[mask].reset_index(drop=True)
    stats = {"region": region.id, "region_version": region.version,
             "crs": region.crs, "input_rows": len(df), "kept_rows": len(kept)}
    return kept, stats

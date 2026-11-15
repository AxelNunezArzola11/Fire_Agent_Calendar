"""Region registry + analytical geographic filtering (raw files untouched).

California (initial region) is an approximate state-extent bounding box —
documented as such, versioned, replaceable. Future regions extend REGIONS;
no code path hardcodes California.
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


CALIFORNIA = Region(
    id="california",
    bbox=(-124.48, 32.53, -114.13, 42.01),
    crs="EPSG:4326",
    source=("Phase 1D approximate state extent (land+water bbox, includes "
            "Channel Islands; NOT an official boundary polygon)"),
    version="cal_bbox_v1",
)

REGIONS: dict[str, Region] = {CALIFORNIA.id: CALIFORNIA}


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

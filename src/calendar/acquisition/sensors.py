"""Sensor / product registry. Five identities, never merged.

NRT and Standard (SP) area-API sources verified 2026-11-11 against the live
FIRMS API (MODIS_NRT, VIIRS_SNPP_NRT, MODIS_SP) and the reference client docs
for the remaining per-satellite sources. Open per-country archives verified
for ``modis`` and ``viirs-snpp`` (USA 2023); NOAA-20/21 have no per-country
archives (also: no Standard products for J1/J2 per FIRMS Readme).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Sensor:
    """One physical sensor identity."""

    id: str  # 'terra' | 'aqua' | 'snpp' | 'noaa20' | 'noaa21'
    satellite: str  # as FIRMS writes it in the CSV ('Terra', 'Aqua', 'NPP', 'NOAA-20', ...)
    instrument: str  # 'MODIS' | 'VIIRS'
    nrt_source: str  # area-API NRT source (MODIS_NRT covers Terra+Aqua)
    sp_source: str | None  # area-API Standard source; None if not published
    first_year: int  # first full year of Standard archive availability


TERRA = Sensor("terra", "Terra", "MODIS", "MODIS_NRT", "MODIS_SP", 2000)
AQUA = Sensor("aqua", "Aqua", "MODIS", "MODIS_NRT", "MODIS_SP", 2002)
SNPP = Sensor("snpp", "NPP", "VIIRS", "VIIRS_SNPP_NRT", "VIIRS_SNPP_SP", 2012)
NOAA20 = Sensor("noaa20", "NOAA-20", "VIIRS", "VIIRS_NOAA20_NRT", "VIIRS_NOAA20_SP", 2018)
NOAA21 = Sensor("noaa21", "NOAA-21", "VIIRS", "VIIRS_NOAA21_NRT", "VIIRS_NOAA21_SP", 2023)

SENSORS: dict[str, Sensor] = {s.id: s for s in (TERRA, AQUA, SNPP, NOAA20, NOAA21)}

NRT_SOURCES = ("MODIS_NRT", "VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT")
SP_SOURCES = ("MODIS_SP", "VIIRS_SNPP_SP", "VIIRS_NOAA20_SP", "VIIRS_NOAA21_SP")


@dataclass(frozen=True)
class ArchiveProduct:
    """A product published as open per-country yearly CSVs."""

    slug: str
    first_year: int


ARCHIVE_MODIS = ArchiveProduct("modis", 2000)
ARCHIVE_VIIRS_SNPP = ArchiveProduct("viirs-snpp", 2012)
ARCHIVE_PRODUCTS: dict[str, ArchiveProduct] = {
    p.slug: p for p in (ARCHIVE_MODIS, ARCHIVE_VIIRS_SNPP)
}

ARCHIVE_BASE = "https://firms.modaps.eosdis.nasa.gov/data/country"


def archive_filename(product_slug: str, year: int, country: str) -> str:
    """``viirs-snpp_2023_United_States.csv`` (country as FIRMS writes it)."""
    return f"{product_slug}_{year}_{country}.csv"


def archive_url(product_slug: str, year: int, country: str) -> str:
    """Build the open-archive URL. Raises KeyError/ValueError on bad input."""
    product = ARCHIVE_PRODUCTS[product_slug]  # KeyError if unknown slug
    if year < product.first_year:
        raise ValueError(f"{product_slug} archive starts at {product.first_year}, got {year}")
    if not country or "/" in country or "\\" in country:
        raise ValueError(f"invalid country slug: {country!r}")
    return f"{ARCHIVE_BASE}/{product_slug}/{year}/{archive_filename(product_slug, year, country)}"

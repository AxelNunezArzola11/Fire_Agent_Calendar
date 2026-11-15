"""Alternative C: versioned deterministic lookup of expected ratios.

ratio(sensor, daynight, area_bin, P) = E_ret(P)/P, precomputed on a
fixed reference grid. Out-of-domain queries return None (no
extrapolation). Missing/invalid inputs never reach the table.
"""

from __future__ import annotations

from datetime import datetime, timezone

from . import coefficients, operator

VERSION = "det-frp-lookup-v1"

# Fixed, documented bin edges (km²) and FRP grid (MW).
AREA_BINS = {
    "modis": [1.0, 1.2, 2.0, 4.0, 6.0, 10.0],
    "viirs-snpp": [0.1, 0.2, 0.3, 0.5, 1.0],
}
FRP_GRID = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]


def _bin_index(edges: list[float], area: float) -> int | None:
    for i in range(len(edges) - 1):
        if edges[i] <= area < edges[i + 1]:
            return i
    if area == edges[-1]:
        return len(edges) - 2
    return None


def build_lookup() -> dict:
    """Precompute the full ratio table (deterministic)."""
    table: dict[str, dict] = {}
    for product, edges in AREA_BINS.items():
        for daynight in ("D", "N"):
            for b in range(len(edges) - 1):
                mid = (edges[b] + edges[b + 1]) / 2.0
                for pref in FRP_GRID:
                    prob = operator.detection_probability(
                        pref, mid, product, daynight)
                    table[f"{product}|{daynight}|{b}|{pref:g}"] = prob
    return {
        "version": VERSION,
        "method": "E_ret(P)/P from det-frp-oop-v1 (Alternative A)",
        "source": coefficients.SOURCE,
        "units": coefficients.UNITS,
        "assumptions": [
            "clear-sky (no QA/cloud terms; FIRMS rows lack them)",
            "pixel area = scan*track (distributionally validated)",
            "FIRMS D/N mapped to paper day/night (twilight unresolved)",
            "retrieved FRP used as model input (exploratory; see report)",
        ],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "validation_status": "exploratory-unvalidated",
        "area_bins": AREA_BINS,
        "frp_grid": FRP_GRID,
        "ratios": table,
    }


def lookup(table: dict, product: str, daynight: str, area: float,
           frp: float) -> float | None:
    """Expected-ratio lookup with linear FRP interpolation in-bin.

    Returns None when out of domain (unknown series, area outside
    bins, FRP outside grid, non-positive FRP).
    """
    edges = table["area_bins"].get(product)
    if daynight not in ("D", "N") or edges is None:
        return None
    if frp is None or frp <= 0:
        return None
    b = _bin_index(edges, area)
    grid = table["frp_grid"]
    if b is None or not (grid[0] <= frp <= grid[-1]):
        return None
    ratios = table["ratios"]
    key = f"{product}|{daynight}|{b}|"
    for i in range(len(grid) - 1):
        lo, hi = grid[i], grid[i + 1]
        if lo <= frp <= hi:
            r_lo = ratios[key + f"{lo:g}"]
            r_hi = ratios[key + f"{hi:g}"]
            if hi == lo:
                return r_lo
            frac = (frp - lo) / (hi - lo)
            return r_lo + frac * (r_hi - r_lo)
    return None

"""Row validation for the pilot (eligibility rules, no silent coercion)."""

from __future__ import annotations

import math

from . import operator as op

VALID_PRODUCTS = ("modis", "viirs-snpp")
VALID_DAYNIGHT = ("D", "N")


def validate_row(row: dict) -> dict:
    """Return {eligible, reasons, area} for one detection record.

    Eligible requires: known sensor, D/N flag, finite FRP ≥ 0
    (None/NaN excluded, never zero-filled), positive finite
    scan/track. Negative FRP is invalid (unphysical).
    """
    reasons: list[str] = []
    product = row.get("product")
    if product not in VALID_PRODUCTS:
        reasons.append(f"unknown product {product!r}")
    daynight = row.get("daynight")
    if daynight not in VALID_DAYNIGHT:
        reasons.append(f"missing/invalid daynight {daynight!r}")
    frp = row.get("frp")
    if frp is None or (isinstance(frp, float) and math.isnan(frp)):
        reasons.append("missing FRP (excluded, not zero)")
    elif frp < 0:
        reasons.append(f"negative FRP {frp!r} (unphysical)")
    area = None
    try:
        area = op.pixel_area_km2(row.get("scan"), row.get("track"))
    except ValueError as exc:
        reasons.append(str(exc))
    return {"eligible": not reasons, "reasons": reasons, "area": area}

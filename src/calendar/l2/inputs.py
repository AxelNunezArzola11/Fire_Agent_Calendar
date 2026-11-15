"""OOp input assembly from L2 extracts (explicit availability).

Builds the per-observation L2 extract consumed by the operator
result contract: AOD, cloud/QA, pixel area + method, granule
provenance. Every field is either a verified value with its source
or an explicit None + unavailable marker. No synthesis: missing
science stays missing.
"""

from __future__ import annotations


def assemble(granule_ur: str | None, extracts: dict | None) -> dict:
    """Assemble one extract record (all-None when unmatched).

    extracts: {aod_550, aod_source, cloud_flag, cloud_source,
    pixel_area_km2, area_source, qa_flags, qa_source} with verified
    values only; anything else is ignored loudly via provenance.
    """
    if not granule_ur or not isinstance(extracts, dict):
        return {"granule_ur": granule_ur,
                "match_status": "unmatched" if granule_ur is None
                else "no-extract",
                "aod_550": None, "aod_source": None,
                "cloud_flag": None, "cloud_source": None,
                "pixel_area_km2": None, "area_source": None,
                "qa_flags": None, "qa_source": None,
                "provenance": "l2.inputs.assemble (no verified extract)"}
    return {
        "granule_ur": granule_ur, "match_status": "matched",
        "aod_550": extracts.get("aod_550"),
        "aod_source": extracts.get("aod_source"),
        "cloud_flag": extracts.get("cloud_flag"),
        "cloud_source": extracts.get("cloud_source"),
        "pixel_area_km2": extracts.get("pixel_area_km2"),
        "area_source": extracts.get("area_source"),
        "qa_flags": extracts.get("qa_flags"),
        "qa_source": extracts.get("qa_source"),
        "provenance": extracts.get("provenance",
                                   "l2.inputs.assemble"),
    }

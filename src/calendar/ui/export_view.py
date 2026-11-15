"""Data Export view: bounded CSV downloads over pinned analytical outputs.

Pure builders (unit-tested) plus a thin Streamlit render wrapper. Reads
only: detections via DuckDB, match/cluster membership via pinned 2B/2C
parquet. Nothing is recomputed; missing values stay missing (empty
fields, never zero); every export carries its filter + provenance
record in a README sidecar.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone

from . import filters, query
from .calendar_view import CITATIONS

CATEGORIES = {
    "modis": "MODIS observations (sensor-specific)",
    "viirs": "VIIRS observations (sensor-specific)",
    "combined": "Combined sensor-specific observations",
    "harmonized": "Harmonized observations + matching classifications",
    "clusters": "Cluster-level analytical records",
}

DETECTION_COLUMNS = (
    "detection_id", "acquisition_timestamp_utc", "latitude", "longitude",
    "sensor", "satellite", "instrument", "product", "product_version",
    "confidence", "frp", "daynight", "scan", "track",
)

# Hard safety rails (displayed whenever they engage; never silent).
EXPORT_ROW_CAP = 100000
EXPORT_BYTE_BUDGET = 64 * 1024 * 1024
PAGE_SIZE = 5000

KEY_EX_CATEGORY = "cal_ex_category"
KEY_EX_PAYLOAD = "cal_ex_payload"

CRS_NOTE = "WGS84 latitude/longitude in decimal degrees"
TEMPORAL_NOTE = "UTC acquisition timestamps (satellite collection time)"
FRP_NOTE = "Fire Radiative Power in megawatts (MW); empty means missing, not zero"


def _cell(value) -> str:
    """Exact-value cell encoding (missing -> empty, never zero-filled)."""
    if value is None:
        return ""
    try:
        import math

        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return ""
    except TypeError:
        pass
    if value != value:  # NaN / NaT / NA
        return ""
    if value is True or value is False:
        return str(value)
    return str(value)


def rows_to_csv(rows: list[dict], columns: list[str]) -> str:
    """Deterministic CSV text (header + rows, exact values, UTF-8 str)."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([_cell(row.get(col)) for col in columns])
    return buffer.getvalue()


def encode_csv(text: str) -> bytes:
    """UTF-8 with BOM (Excel-compatible) CSV payload."""
    return ("\ufeff" + text).encode("utf-8")


def export_filename(category: str, start: str, end: str, ext: str = "csv") -> str:
    """Descriptive ASCII filename (dates are validated YYYY-MM-DD)."""
    safe = "".join(c if (c.isalnum() or c in "-_") else "-" for c in category)
    return f"fireagent-calendar-{safe}-{start}_{end}.{ext}"


def _partners_for(match_id: str | None, own_id: str,
                  groups: dict) -> str:
    """Pipe-separated partner ids within one match group (bounded)."""
    if not match_id or match_id not in groups:
        return ""
    group = groups[match_id]
    try:
        members = json.loads(group.get("modis_ids") or "[]") + json.loads(
            group.get("viirs_ids") or "[]")
    except (ValueError, TypeError):
        return ""
    partners = sorted({m for m in members if m != own_id})
    return "|".join(partners[:50])


def enrich_detections(frame, *, harmonized: bool) -> tuple[list[dict], list[str]]:
    """Observation rows + match/cluster columns (each observation once).

    Membership lookups run in bounded batches. Partner ids are included
    only for the harmonized category; sensor-specific exports carry the
    match classification without implying correspondence.
    """
    import pandas as pd

    ids = frame["detection_id"].tolist() if len(frame) else []
    members = {}
    if ids:
        mem = query.members_of(ids)
        if len(mem):
            members = mem.set_index("detection_id").to_dict("index")
    clusters = {}
    if ids:
        clu = query.cluster_of(ids)
        if len(clu):
            clusters = clu.set_index("detection_id")["cluster_id"].to_dict()
    groups = {}
    if harmonized:
        mids = sorted({str(m.get("match_id")) for m in members.values()
                       if m.get("match_id")})
        if mids:
            grp = query.match_groups_of(mids)
            if len(grp):
                groups = grp.set_index("match_id").to_dict("index")
    columns = list(DETECTION_COLUMNS) + ["cluster_id", "match_id",
                                         "match_classification",
                                         "match_best_score"]
    if harmonized:
        columns += ["partner_ids", "source_file", "source_sha256"]
    else:
        columns += ["source_file", "source_sha256"]
    rows = []
    for record in frame.to_dict("records"):
        did = record["detection_id"]
        member = members.get(did) or {}
        rows.append({
            **{col: record.get(col) for col in DETECTION_COLUMNS},
            "cluster_id": clusters.get(did, ""),
            "match_id": member.get("match_id") or "",
            "match_classification": member.get("classification") or "",
            "match_best_score": member.get("best_score")
            if member.get("best_score") is not None else "",
            **({"partner_ids": _partners_for(member.get("match_id"), did, groups)}
               if harmonized else {}),
            "source_file": record.get("source_file") or "",
            "source_sha256": record.get("source_sha256") or "",
        })
    _ = pd  # frame typing only
    return rows, columns


def readme_text(*, category: str, filters_used: dict, total: int,
                exported: int, truncated: str | None) -> str:
    """Plain-text sidecar: filters, counts, lineage, glossary."""
    chain = query.exec_chain()
    chain4 = query.phase4_chain()
    lines = [
        "FireAgent Calendar data export (README sidecar)",
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"Category: {category} — {CATEGORIES[category]}",
        f"Filters: {json.dumps(filters_used, sort_keys=True, default=str)}",
        f"Rows matching: {total:,}; rows exported: {exported:,}.",
    ]
    if truncated:
        lines.append(f"TRUNCATED: {truncated}")
    lines += [
        "",
        "Provenance:",
        f"  Phase 2A validation: exec {(chain['2A'] or 'unconfigured')[:8]}",
        f"  Phase 2B matching (matcher_v1): exec {(chain['2B'] or 'unconfigured')[:8]}",
        f"  Phase 2C clustering (cluster_v1): exec {(chain['2C'] or 'unconfigured')[:8]}",
        f"  Normalized inputs: Phase 1D cal_norm_v1 (see catalog.json).",
        f"  Baselines/anomalies (context only): 4B {(chain4['4B'] or 'unconfigured')[:8]} / "
        f"4C {(chain4['4C'] or 'unconfigured')[:8]}.",
        "",
        f"Coordinates: {CRS_NOTE}.",
        f"Temporal basis: {TEMPORAL_NOTE}.",
        f"FRP: {FRP_NOTE}.",
        "",
        "Column glossary:",
        "  detection_id: stable observation identifier (use for joins).",
        "  acquisition_timestamp_utc: satellite collection time (UTC).",
        "  sensor/satellite/instrument/product/product_version: source identity.",
        "  confidence: MODIS 0-100 score or VIIRS l/n/h class (not comparable).",
        "  frp: reported per-observation FRP (MW); repeated observations "
        "included, never total emitted energy.",
        "  cluster_id: analytical hotspot-cluster membership (observation "
        "group, not a wildfire or perimeter).",
        "  match_id/match_classification/match_best_score: cross-sensor "
        "correspondence verdict (matched/ambiguous/singleton).",
        "  partner_ids (harmonized only): fellow member ids, | separated.",
        "  source_file/source_sha256: immutable raw lineage.",
        "",
        "Harmonized counting (derive, don't re-sum): matched groups count "
        "once each (distinct match_id where classification=matched); "
        "ambiguous members and singletons count individually. Counting "
        "rows directly double-counts matched observations.",
        "",
        "Satellite observations are not confirmed wildfires; clusters are "
        "not perimeters or incidents.",
        "",
        "How to cite:",
        *(f"  - {citation}" for citation in CITATIONS),
    ]
    return "\n".join(lines) + "\n"


def _selection() -> dict:
    """Shared sidebar filters as one dict (single source of truth)."""
    import streamlit as st

    return {
        "date_start": filters.coerce_day(
            st.session_state.get(filters.KEY_RANGE_START, "2023-01-01")),
        "date_end": filters.coerce_day(
            st.session_state.get(filters.KEY_RANGE_END, "2023-12-31")),
        "products": tuple(st.session_state.get(filters.KEY_PRODUCTS,
                                               list(query.PRODUCTS))),
        "satellites": tuple(st.session_state.get(filters.KEY_SATELLITES,
                                                 list(query.SATELLITES))),
    }


def _scoped_products(category: str, selection: dict) -> tuple:
    """Category products intersected with the shared product filter."""
    wanted = {
        "modis": ("modis",),
        "viirs": ("viirs-snpp", "viirs-noaa20", "viirs-noaa21"),
    }.get(category, selection["products"])
    if category in ("combined", "harmonized"):
        return selection["products"]
    return tuple(p for p in selection["products"] if p in wanted)


def _scoped_satellites(category: str, selection: dict) -> tuple:
    """Satellites intersected with the category's valid set."""
    satellites = selection["satellites"]
    if category == "modis":
        return tuple(s for s in satellites if s in ("Terra", "Aqua"))
    if category == "viirs":
        return tuple(s for s in satellites if s in ("N", "NPP", "NOAA-20", "N20", "NOAA-21", "N21"))
    return satellites


def export_source_status(category: str) -> dict:
    """Verify all required scientific data sources before counting or exporting."""
    if category not in CATEGORIES:
        raise ValueError(f"unknown export category {category!r}")
    if category == "clusters":
        sources = [query.artifact_status("2C", "clusters.parquet")]
    else:
        sources = [
            query.duckdb_status(),
            query.artifact_status("2B", "match_members.parquet"),
            query.artifact_status("2C", "cluster_members.parquet"),
        ]
        if category == "harmonized":
            sources.append(query.artifact_status("2B", "match_groups.parquet"))
    missing = [s for s in sources if not s["available"]]
    return {
        "available": len(missing) == 0,
        "category": category,
        "sources": sources,
        "missing": missing,
        "reason": "; ".join(m["reason"] for m in missing)
        if missing else "All required export sources verified.",
    }


def _count_for(category: str, selection: dict, bbox) -> int:
    if category == "clusters":
        return query.count_clusters(date_start=selection["date_start"],
                                    date_end=selection["date_end"], bbox=bbox)
    return query.count_detections(
        date_start=selection["date_start"], date_end=selection["date_end"],
        products=_scoped_products(category, selection),
        satellites=_scoped_satellites(category, selection), bbox=bbox)


def _fetch_detection_page(category: str, selection: dict, bbox,
                          limit: int, offset: int):
    """One page of full-column detection rows (deterministic order)."""
    import pandas as pd

    frame = query.query_detections(
        date_start=selection["date_start"], date_end=selection["date_end"],
        products=_scoped_products(category, selection),
        satellites=_scoped_satellites(category, selection), bbox=bbox,
        limit=limit + offset, columns=query.DETECTION_COLUMNS)
    return frame.iloc[offset:offset + limit].reset_index(drop=True)


def build_export(category: str, selection: dict, bbox) -> dict:
    """Generate one export payload (bounded rows + bytes).

    Returns {csv_text, readme, total, exported, truncated, filename}.
    Raises ValueError on unknown category or invalid selection, and
    DataUnavailableError when any required data source is unavailable.
    """
    if category not in CATEGORIES:
        raise ValueError(f"unknown export category {category!r}")
    query._day_bounds(selection["date_start"], selection["date_end"])
    src_st = export_source_status(category)
    if not src_st["available"]:
        raise query.DataUnavailableError(
            f"Export refused for category {category!r}: required data source "
            f"unavailable ({src_st['reason']})",
            status=src_st,
        )
    total = _count_for(category, selection, bbox)
    if total == 0:
        columns = (list(query.CLUSTER_MAP_COLUMNS)
                   if category == "clusters"
                   else list(DETECTION_COLUMNS) + ["cluster_id", "match_id",
                                                  "match_classification",
                                                  "match_best_score",
                                                  "source_file",
                                                  "source_sha256"])
        if category == "harmonized":
            columns.insert(-2, "partner_ids")
        return {"csv_text": rows_to_csv([], columns),
                "readme": readme_text(
                    category=category, filters_used={**selection, "bbox": bbox,
                                                    "aoi": "shared-filters"},
                    total=0, exported=0, truncated=None),
                "total": 0, "exported": 0, "truncated": None,
                "filename": export_filename(category, selection["date_start"],
                                            selection["date_end"])}
    chunks: list[str] = []
    columns: list[str] | None = None
    exported = 0
    truncated: str | None = None
    remaining_rows = min(total, EXPORT_ROW_CAP)
    if total > EXPORT_ROW_CAP:
        truncated = (f"row cap: first {EXPORT_ROW_CAP:,} of {total:,} rows "
                     f"(narrow filters for full coverage)")
    offset = 0
    csv_bytes = 0
    while exported < remaining_rows:
        take = min(PAGE_SIZE, remaining_rows - exported)
        if category == "clusters":
            frame = query.query_clusters(
                date_start=selection["date_start"],
                date_end=selection["date_end"], bbox=bbox,
                limit=take + offset).iloc[offset:offset + take]
            offset += take
            page_rows = frame.to_dict("records")
            page_columns = list(frame.columns)
        else:
            frame = _fetch_detection_page(category, selection, bbox, take,
                                          offset)
            offset += take
            page_rows, page_columns = enrich_detections(
                frame, harmonized=(category == "harmonized"))
        if columns is None:
            columns = page_columns
            header = rows_to_csv([], columns)
            chunks.append(header)
            csv_bytes += len(header.encode("utf-8"))
        body = rows_to_csv(page_rows, columns)
        # Strip the repeated header (keep data lines only).
        body = "\n".join(body.split("\n")[1:])
        if body:
            body += "\n"
        encoded = len(body.encode("utf-8"))
        if csv_bytes + encoded > EXPORT_BYTE_BUDGET:
            truncated = (truncated + "; " if truncated else "") + \
                (f"byte budget: stopped at {exported:,} rows "
                 f"({EXPORT_BYTE_BUDGET // (1024 * 1024)} MB cap)")
            break
        chunks.append(body)
        csv_bytes += encoded
        exported += len(page_rows)
        if len(page_rows) < take:
            break
    csv_text = "".join(chunks)
    return {"csv_text": csv_text,
            "readme": readme_text(
                category=category,
                filters_used={**selection, "bbox": bbox,
                              "aoi": "shared-filters"},
                total=total, exported=exported, truncated=truncated),
            "total": total, "exported": exported, "truncated": truncated,
            "filename": export_filename(category, selection["date_start"],
                                        selection["date_end"])}


def render_export_tab() -> None:
    """Full Data Export tab (count preview + prepare + downloads)."""
    import streamlit as st

    from .. import settings
    from . import aoi as aoi_mod
    from . import map_view

    st.header("Data export")
    st.caption("Bounded CSV downloads from pinned analytical outputs. "
               "Rows show exact source values; missing stays missing.")
    category = st.selectbox(
        "Export category", options=list(CATEGORIES), key=KEY_EX_CATEGORY,
        format_func=lambda c: CATEGORIES[c])
    try:
        selection = _selection()
        resolved_ex = aoi_mod.resolve_session_aoi(
            st.session_state,
            aoi_path=settings.CAL_AOI_PATH,
            include_session_fallback=True,
        )
        bbox = resolved_ex.bbox if resolved_ex is not None else None
    except ValueError as exc:
        st.error(f"Invalid filter: {exc}")
        return
    except aoi_mod.AoIError as exc:
        st.error(f"Area of interest: {exc}")
        return
    st.caption(f"Shared filters: {selection['date_start']} to "
               f"{selection['date_end']} · "
               f"{', '.join(selection['products']) or 'no products'} · "
               f"{', '.join(selection['satellites']) or 'no satellites'}"
               + (f" · bbox {bbox}" if bbox else " · full pilot"))
    src_st = export_source_status(category)
    if not src_st["available"]:
        st.session_state[KEY_EX_PAYLOAD] = None
        missing_lines = "\n".join(
            f"- `{m.get('path') or m.get('filename') or m.get('env_var')}` "
            f"(`{m.get('env_var')}` · status `{m.get('status')}`): {m.get('reason')}"
            for m in src_st["missing"]
        )
        st.warning(
            f"Export unavailable for '{CATEGORIES[category]}': required scientific "
            "data source is not present in this deployment."
        )
        st.info(
            "Missing export source(s):\n"
            f"{missing_lines}\n\n"
            "Configure the required path/execution settings via Streamlit Secrets or "
            "environment variables. Export refuses to generate empty or fabricated CSV "
            "files when source data is unavailable."
        )
        return
    try:
        total = _count_for(category, selection, bbox)
    except FileNotFoundError as exc:
        st.session_state[KEY_EX_PAYLOAD] = None
        st.warning(f"Export unavailable for '{CATEGORIES[category]}': {exc}")
        return
    except ValueError as exc:
        st.error(f"Invalid filter: {exc}")
        return
    st.metric("Rows matching", f"{total:,}")
    if total == 0:
        st.info("No rows match these filters: nothing to export.")
        st.session_state[KEY_EX_PAYLOAD] = None
        return
    if total > EXPORT_ROW_CAP:
        st.warning(f"Large result: exports are capped at the first "
                   f"{EXPORT_ROW_CAP:,} rows ({total:,} match). Narrow the "
                   f"filters for full coverage.")
    if st.button("Prepare CSV", key="cal_ex_prepare",
                 help="Builds the CSV in bounded pages (nothing is recomputed)."):
        with st.spinner("Building export…"):
            try:
                payload = build_export(category, selection, bbox)
            except FileNotFoundError as exc:
                st.session_state[KEY_EX_PAYLOAD] = None
                st.warning(f"Export refused — data source unavailable: {exc}")
                return
            except ValueError as exc:
                st.error(f"Export failed: {exc}")
                return
        st.session_state[KEY_EX_PAYLOAD] = payload
    payload = st.session_state.get(KEY_EX_PAYLOAD)
    if not payload:
        st.caption("Press Prepare CSV, then download the files below.")
        return
    if payload["truncated"]:
        st.warning(f"Truncated export: {payload['truncated']}")
    st.caption(f"Prepared {payload['exported']:,} of {payload['total']:,} rows.")
    st.download_button("Download CSV", data=encode_csv(payload["csv_text"]),
                       file_name=payload["filename"], mime="text/csv",
                       key="cal_ex_csv", width="stretch")
    st.download_button("Download README sidecar",
                       data=payload["readme"].encode("utf-8"),
                       file_name=payload["filename"].replace(".csv", "-README.txt"),
                       mime="text/plain", key="cal_ex_readme",
                       width="stretch")

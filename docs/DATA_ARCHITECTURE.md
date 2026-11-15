# FireAgent Calendar — Data & Storage Architecture

FireAgent Calendar separates its **pinned Stage 4C USA baseline archive** from **independent regional historical archives** and **operational event records**, ensuring zero cross-contamination between geographic domains while supporting both offline reproducibility and live multi-year NASA FIRMS expansion.

---

## 1. Storage Tiers Overview

| Tier | Default Path | Setting Key | Write Policy | Purpose |
|---|---|---|---|---|
| **Pinned USA Parquet Store** | `normalized_parquet/cal_norm_v1/` | `CAL_PARQUET_ROOT` | Read-only in UI | Partitioned 2023 USA MODIS & VIIRS S-NPP detections (482,672 accepted rows). |
| **Pinned USA DuckDB View** | `calendar.duckdb` | `CAL_DUCKDB_PATH` | Read-only in UI | DuckDB view (`detections`) reading Hive-partitioned Parquet (`year=*/month=*`). |
| **Independent Regional DuckDB** | `regional_archives/regional.duckdb` | `CAL_REGIONAL_DUCKDB_PATH` | Read/write per `archive_id` | Persistent independent storage for Madrid, Mexico, Brazil, Australia, custom bboxes, and multi-year queries. |
| **Analytical Artifacts & Ledger** | `analytics_output/`, `analytics_runs.jsonl` | `CAL_ANALYTICS_ROOT`, `CAL_RUNS_PATH` | Append-only runs | Execution-ID-keyed Stage 2A–4C JSON/Parquet outputs (`CAL_EXEC_*`). |
| **Event & Alert SQLite DB** | `calendar_events.sqlite` | `CAL_EVENTS_DB_PATH` | Transactional SQLite | Stage 5A–5C wildfire events, state transitions, alerts, subscriptions, and agent reports. |
| **Saved AOIs** | `calendar_aois.json` | `CAL_AOI_PATH` | User JSON store | Named user bounding boxes. |

All storage keys resolve centrally through `src/calendar/settings.py:get_setting` (precedence: explicit runtime `monkeypatch`/attribute override > OS environment variable > `st.secrets` > documented default). Paths use `settings.resolve_repo_path(...)`. On read-only filesystems (such as read-only container mounts), `regional.resolve_regional_db_path(...)` automatically falls back to `$TMPDIR/fireagent_regional_archives/regional.duckdb` and **refuses** to point to `calendar.duckdb`.

---

## 2. Canonical Detection Schema (`cal_norm_v1`)

Defined in `src/calendar/processing/normalize.py`:

| Column | Type | Description |
|---|---|---|
| `detection_id` | VARCHAR | Deterministic hash identifier of the raw detection row. |
| `sensor` | VARCHAR | `terra_aqua`, `snpp`, `noaa20`, `noaa21`. |
| `satellite` | VARCHAR | Canonical satellite (`Terra`, `Aqua`, `N`, `NOAA-20`, `NOAA-21`). |
| `instrument` | VARCHAR | `MODIS` or `VIIRS`. |
| `product` | VARCHAR | `modis`, `viirs-snpp`, `viirs-noaa20`, `viirs-noaa21`. |
| `product_version` | VARCHAR | FIRMS product version string (e.g. `6.1NRT`, `2.0`). |
| `acquisition_timestamp_utc` | VARCHAR | ISO-8601 UTC timestamp synthesized from `acq_date` + `acq_time`. |
| `original_acquisition_timestamp` | VARCHAR | Raw `YYYY-MM-DD HHMM` acquisition string. |
| `latitude` | DOUBLE | WGS84 latitude $[-90, 90]$. |
| `longitude` | DOUBLE | WGS84 longitude $[-180, 180]$. |
| `confidence` | VARCHAR | Numeric string for MODIS (`0`–`100`), categorical (`l`/`n`/`h`) for VIIRS. |
| `frp` | DOUBLE | Fire Radiative Power in MW (NULL preserved when missing; never zero-filled). |
| `daynight` | VARCHAR | Overpass flag (`D` or `N`). |
| `scan` | DOUBLE | Cross-track pixel dimension (km). |
| `track` | DOUBLE | Along-track pixel dimension (km). |
| `source_file` | VARCHAR | Raw CSV filename or live FIRMS request key. |
| `source_sha256` | VARCHAR | SHA-256 digest of raw upstream CSV payload. |
| `ingestion_timestamp_utc` | VARCHAR | UTC normalization timestamp. |
| `processing_version` | VARCHAR | `cal_norm_v1` / `cal_norm_v1_live`. |

---

## 3. Independent Regional DuckDB Schema (`regional.duckdb`)

Managed by `src/calendar/pipeline/regional.py`:

### 3.1 Deterministic `archive_id` Generation (`compute_archive_id`)
- Named preset region (e.g., `mexico`, `brazil`, `australia`, `madrid`): `reg_<slug>`
- Point + radius (e.g., `40.42, -3.70, 100 km`): `pt_40.42_-3.70_100.0km`
- Custom bounding box `(W, S, E, N)`: `bbox_<sha256[:10]>`

### 3.2 Tables

1. **`regional_archives`** (one row per independent archive)
   - `archive_id` (PK), `region_id`, `region_label`, `aoi_source`, `crs` (`EPSG:4326`)
   - `lon_min`, `lat_min`, `lon_max`, `lat_max`
   - `center_lat`, `center_lon`, `radius_km` (populated for point+radius AOIs; nullable otherwise)
   - `created_utc`, `updated_utc`

2. **`regional_acquisitions`** (execution & coverage ledger)
   - `exec_id` (PK), `archive_id`, `date_start`, `date_end`
   - `products` (JSON list), `satellites` (JSON list or NULL), `firms_sources` (JSON list)
   - `n_detections`, `rejected_count`, `data_origin` (`firms_live_api`)
   - `status` (`complete` | `empty` | `partial` | `failed`)
   - `retrieved_utc`, `errors` (JSON), `limitations` (JSON), `chunks` (JSON chunk digests)

3. **`regional_detections`** (isolated regional observations)
   - All `cal_norm_v1` columns plus `archive_id`, `acq_date` (`YYYY-MM-DD`), `firms_source`, `data_origin`, and `exec_id`.

### 3.3 Deduplication & Isolation Rules
- Before inserting into `regional_detections`, `store_regional_dataset` compares candidate rows against existing rows in the same `archive_id` using both:
  1. `detection_id`
  2. Composite tuple `(product, satellite, acquisition_timestamp_utc, ROUND(latitude, 5), ROUND(longitude, 5))`
- Repeated or overlapping fetches (for example extending `2023-01-01..2023-01-10` to `2023-01-05..2023-02-10`) skip duplicates cleanly without inflating counts or FRP totals.
- `aoi.is_external_aoi` checks `region_id`, `point_radius`, and `bbox` so bordering areas such as **Mexico** or **Canada** never pull US observations from `calendar.duckdb`.

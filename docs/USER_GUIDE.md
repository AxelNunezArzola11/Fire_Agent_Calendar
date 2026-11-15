# FireAgent Calendar — User Guide

This guide explains how to operate **FireAgent Calendar** (`calendar_app.py`) locally and on Streamlit Community Cloud, including multi-year date selection, global Geographic Areas of Interest (AOI), independent regional DuckDB historical archives, multi-agent execution with IBM Granite / Qwen / deterministic fallback, and CSV/Parquet data export.

---

## 1. Quick Start

Run the primary application locally:

```bash
.venv/bin/streamlit run calendar_app.py
```

When the application launches, the **left sidebar** holds the shared temporal, sensor, confidence, and geographic filters that synchronize across all tabs (`Fire Calendar`, `Activity Map`, `Historical Analytics`, `Events & Alerts`, `Data Export`, and `About`).

---

## 2. Navigating the Application Tabs

### 2.1 Fire Calendar (`src/calendar/ui/calendar_view.py`)
- Displays daily, weekly, and monthly summaries from Stage 2D (`metrics_2d.json`) and dual-view Stage 2E (`index_2e.json`).
- **Temporal basis**: all calendars use **satellite acquisition date (UTC)**, not NRT product ingestion time.
- **Sensor distinction**: harmonized counts reconcile matched cross-sensor observations without merging MODIS and VIIRS radiative power (`FRP`) into a single scale.

### 2.2 Activity Map (`src/calendar/ui/map_view.py`)
- Switch between **Detection Mode** (individual thermal anomalies) and **Cluster Mode** (Phase 2C spatiotemporal hotspot clusters).
- **Color modes**: `sensor` (MODIS vs VIIRS) or `frp` (relative within-view quartiles).
- **Live NASA FIRMS Retrieval**: when a non-USA region, multi-year date range, or NOAA-20 / NOAA-21 sensor is selected, click **Retrieve live NASA FIRMS** to fetch live FIRMS Area API observations (`FIRMS_MAP_KEY` required), normalize them into `cal_norm_v1`, persist them in `regional_archives/regional.duckdb`, and share them automatically with Historical Analytics and Events & Alerts.

### 2.3 Historical Analytics (`src/calendar/ui/analytics_view.py`)
Historical Analytics supports two clearly separated modes:

1. **Pinned Stage 4C Baseline Archive (2023 USA · MODIS + VIIRS S-NPP)**:
   - Reads the pinned `anomalies_4c.json` execution (`CAL_EXEC_4C`) and 2023 USA DuckDB archive (`calendar.duckdb`).
   - Supports baseline methods: `monthly_median`, `seasonal`, and `sensor_aware` (trailing lookback $K \in \{30\text{d}, 12\text{w}, 6\text{m}\}$).
2. **Dynamic Multi-Year / Regional (DuckDB + NASA FIRMS)**:
   - Automatically activates when any external region, point+radius AOI, multi-year date window, or NOAA-20 / NOAA-21 sensor is active.
   - **Automatic Regional Archive Population**:
     1. Computes a deterministic `archive_id` (for example `pt_40.42_-3.70_100.0km`, `reg_mexico`, `reg_brazil`, `reg_australia`).
     2. Checks `regional_archives/regional.duckdb` for complete day-by-day and sensor coverage.
     3. Automatically retrieves missing observations via NASA FIRMS Area API (`FIRMS_MAP_KEY`), deduplicates and persists them in `regional.duckdb`, and renders:
        - **Observed Activity & Historical Baseline** (`st.line_chart` / `st.bar_chart`).
        - **Statistical Deviation (Z-Score)** & Notable anomaly inspector.
        - **Satellite & Sensor Distribution** (counts and FRP breakdown by satellite and product).
        - **FRP Intensity Distribution** (`0-5`, `5-20`, `20-50`, `50-100`, `100-500`, `500+` MW separated between MODIS and VIIRS).
   - Click **Refresh Regional Archive from NASA FIRMS** to force re-retrieval for the active window.

### 2.4 Events & Alerts (`src/calendar/ui/events_view.py`)
- Inspect persistent Stage 5 events (`calendar_events.sqlite`) and interactive **FireAgent Multi-Agent Pipeline** executions.
- Select the AI provider mode in the UI / environment (`CAL_AI_PROVIDER`: `auto`, `granite`, `qwen`, `deterministic`):
  - **IBM Granite** (`GraniteProvider` via IBM watsonx.ai IAM/REST or OpenAI-compatible/Ollama v1 endpoints).
  - **Qwen** (`QwenChatProvider` via OpenAI-compatible API).
  - **Deterministic Fallback** (`DeterministicFallbackProvider` — always available offline with zero external LLM dependencies).
- Run the end-to-end 6-stage pipeline directly against the active sidebar/analytics AOI without manually entering a raw bounding box (`aoi.resolve_session_aoi` resolves preset regions, point+radius, custom bbox, or saved AOIs automatically).

### 2.5 Data Export (`src/calendar/ui/export_view.py`)
- Generates filtered CSV/Parquet-derived extracts paired with provenance headers, methodology limitations, and execution IDs (`CAL_EXEC_*`).

---

## 3. Selecting a Geographic Area of Interest (AOI)

Open **Geographic area of interest** in the left sidebar (`calendar_app.py`) and choose one of six modes:

| AOI Mode | How to Configure | Archive / Pipeline Behavior |
|---|---|---|
| **Full archive extent (USA / North America)** | Default (`mode="full"`) | Queries pinned 2023 USA Stage 4C baseline (`calendar.duckdb`); live pipeline requires an explicit AOI. |
| **Preset country / region** | Pick California, Madrid, Mexico, Brazil, Spain, Australia, CONUS, Portugal, Canada, or Chile | Non-USA presets resolve to independent `reg_<id>` DuckDB archives (`CAL_REGIONAL_DUCKDB_PATH`). |
| **Point + radius (lat/lon + km)** | Use quick presets (**Madrid 40.42, -3.70 · 100 km**, **CDMX**, **Brasilia**, **Sydney**), Nominatim place search, or coordinates (`0.5`–`2000.0` km) | Generates an equirectangular WGS84 bbox via `aoi.radius_bbox` and an isolated `pt_<lat>_<lon>_<r>km` regional archive. |
| **Custom bounding box / lat-lon** | Enter `(West, South, East, North)` in EPSG:4326 or click quick presets (`CA`, `MX`, `MAD`, `BR`, `ES`, `AU`) | Validated by `aoi.validate_bbox` ($W < E, S < N$, finite coordinates) and stored under `bbox_<hash10>`. |
| **Current map view** | Pan/zoom on the Activity Map and capture the map geometry / bbox | Uses `cal_map_geometry` / `cal_map_bbox`. |
| **Saved area** | Save/load named bboxes in `CAL_AOI_PATH` (`calendar_aois.json`) | Reusable named geographic masks. |

> **Important:** Regions outside the USA (including **Madrid**, **Mexico**, **Brazil**, **Australia**, **Canada**, **Spain**, **Portugal**, **Chile**, and non-USA point+radius areas) **never** backfill from `calendar.duckdb` (USA 2023 baseline). Even when a bounding box borders US territory (e.g., northern Mexico), `aoi.is_external_aoi` enforces strict regional isolation.

---

## 4. Satellite & Sensor Controls

In **Shared filters**:
- **Products**: `modis`, `viirs-snpp`, `viirs-noaa20`, `viirs-noaa21`
- **Satellites**:
  - `Terra` (`modis`, 2000-11-01 to present)
  - `Aqua` (`modis`, 2002-07-04 to present)
  - `N` (`viirs-snpp` / Suomi-NPP, 2012-01-20 to present)
  - `NOAA-20` (`viirs-noaa20`, 2018-04-01 to present — live FIRMS / regional archive)
  - `NOAA-21` (`viirs-noaa21`, 2023-03-01 to present — live FIRMS / regional archive)
- **Confidence filters**:
  - **MODIS**: continuous numeric score `0–100`.
  - **VIIRS**: categorical classes `l` (low), `n` (nominal), `h` (high).
  - Both scales are filtered independently because they are physically non-equivalent.

---

## 5. Troubleshooting Common Messages

- **"Regional DuckDB archive has no recorded observations... FIRMS_MAP_KEY is not configured"**: add `FIRMS_MAP_KEY=...` to `.env` (local) or `.streamlit/secrets.toml` (Streamlit Cloud) and rerun; the archive populates automatically.
- **"No NASA FIRMS sources cover the selected date range and satellites"**: check sensor launch/archive start dates (e.g., VIIRS NOAA-21 begins `2023-03-01`; VIIRS NOAA-20 begins `2018-04-01`).
- **Missing Stage 4C artifact in custom deployment**: verify `CAL_ANALYTICS_ROOT`, `CAL_RUNS_PATH`, and `CAL_EXEC_4C` match an actual completed execution in `analytics_runs.jsonl`.

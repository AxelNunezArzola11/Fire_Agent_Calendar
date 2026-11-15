# FireAgent Calendar — Multi-Agent & Pipeline Architecture

FireAgent separates **deterministic Earth-observation computation** from **bounded natural-language synthesis**. Agents never modify observations, recompute FRP/baselines, alter event lifecycles, or fabricate data.

---

## 1. Six-/Seven-Stage End-to-End Pipeline (`src/calendar/pipeline/runner.py`)

`run_pipeline(...)` executes seven sequential stages with explicit status (`ok`, `partial`, `failed`, `skipped`), execution timing (`latency_ms`), and honest short-circuiting when required inputs or AOIs are absent:

1. **`acquisition`**:
   - Resolves the session AOI automatically via `aoi.resolve_aoi` / `aoi.resolve_session_aoi` (explicit bbox -> preset region -> point+radius -> saved AOI -> map geometry).
   - Queries either local DuckDB (`calendar.duckdb`) or live NASA FIRMS Area API (`regional.fetch_firms_regional`, optionally persisting to `regional_archives/regional.duckdb`).
   - Fails cleanly without silently defaulting to California/CONUS if no AOI is selected for live FIRMS retrieval.
2. **`detection`**:
   - Runs Phase 2B cross-sensor matching (`matching.py`), Phase 2C hotspot clustering (`clustering.py`), and invokes `CoordinatorAgent` with `kind="detections"`.
3. **`events`**:
   - Associates clusters 1:1 into deterministic wildfire event candidates (`association.py`), computes event severity/lifecycle rollups (`lifecycle.py`, `events_view.py`), and invokes `CoordinatorAgent` with `kind="event"`.
4. **`operator`**:
   - Evaluates the experimental Sofiev & Kouznetsov (2026) FRP Observation Operator (`operator.py`, `validation.py`, `results.py`) over eligible MODIS and VIIRS S-NPP rows (up to `OPERATOR_ROW_CAP = 50,000`), optionally enriched via NASA CMR `VNP14IMG` Level-2 granule colocation (`L2_MAX_DAYS = 7`), and invokes `CoordinatorAgent` with `kind="science"`.
5. **`validation`**:
   - Runs structural, grounding, and scientific integrity assertions (`_validate_result`) across all stages; propagates pass/fail flags to event rollups.
6. **`export`**:
   - Serializes verified `observations`, `events`, and `agents` CSV deliverables (`exports_csv.py`) stamped with execution ID, processing timestamp, and provenance headers.
7. **`metrics`**:
   - Aggregates stage latencies, provider outcomes, fallback counts, and validation scores into `metrics` CSV (`exports_csv.metrics_csv`).

---

## 2. Specialist Agents & Coordinator (`src/calendar/agents/`)

| Component | Module | Responsibility |
|---|---|---|
| **Coordinator** | `coordinator.py` (`analyze`) | Routes `AgentRequest(kind, target_id, payload)` to specialist workers (`detections`, `event`, `science`), passes outputs through validation gates, aggregates `CoordinatorResult`, and writes audit rows to SQLite `qwen_log`. |
| **Detection Intelligence Agent** | `detection_intel.py` | Interprets cluster density, cross-sensor matching classes, FRP distributions (sensor-separated), and MODIS/VIIRS confidence structure from deterministic payloads only. |
| **Event Analysis Agent** | `event_analysis.py` | Synthesizes lifecycle state transitions, spatial extents, persistence duration, and alert context from Stage 5 event payloads. |
| **Science Observation Agent** | `science_observer.py` | Explains Observation Operator eligibility, detectability regimes, excluded NOAA-20/21 sensors, and unvalidated exploratory limits (`payloads.SCIENCE_MISSING_INPUTS`). |
| **Evidence Validation Agent** | `evidence_validator.py` | Enforces fail-closed JSON schema checks (`check_llm_json`), banned-claim lexical filters, `evidence_used` ID grounding against `payload`, numeric grounding, and structural `check_findings`. |

---

## 3. AI Provider Chain & Fallback (`providers.py`, `granite.py`, `events/qwen.py`)

Agents execute against an ordered `ProviderChain` constructed by `default_chain()` according to `CAL_AI_PROVIDER` (`auto`, `granite`, `qwen`, `deterministic`):

1. **`GraniteProvider` (`src/calendar/agents/granite.py`)**:
   - **IBM watsonx.ai REST mode**: Exchanges `WATSONX_API_KEY` (or `CAL_GRANITE_API_KEY`) for an IAM bearer token (`https://iam.cloud.ibm.com/identity/token`) and calls `/ml/v1/text/generation?version=2024-05-01` with `project_id` and model (default `ibm/granite-4-h-small`).
   - **OpenAI-compatible mode** (auto-detected when URL ends with `/v1` or `/chat/completions` without `project_id`, or `CAL_GRANITE_ENDPOINT_STYLE=openai`): calls `/chat/completions` for hosted or local Ollama/vLLM Granite endpoints.
   - Retries bounded transient HTTP statuses `(429, 502, 503, 504)` up to `MAX_ATTEMPTS_CAP = 3`, redacts all secrets (`redact()`), and raises clean `_GraniteError` on transport/auth/schema failure.
2. **`QwenChatProvider` (`src/calendar/agents/providers.py`)**:
   - Connects to OpenAI-compatible Qwen endpoints (`CAL_QWEN_ENABLED=true`, `CAL_QWEN_BASE_URL`, `CAL_QWEN_API_KEY` or `OPENROUTER_API_KEY`, `CAL_QWEN_MODEL`).
3. **`DeterministicFallbackProvider` (`providers.py` / `qwen.explain`)**:
   - Pure rule-based synthesis built from verified payload fields (`findings_for`, rule templates, explicit caveats).
   - Always available offline; engages automatically with `status="fallback"` and traceable `fallback_reason` whenever IBM Granite or Qwen is unconfigured, unreachable, returns malformed JSON, or fails `evidence_validator.check_llm_json`.

---

## 4. Guardrails & Provenance

- **Zero hallucinated physics**: LLMs never receive write access to DuckDB, SQLite, Parquet, or computed metrics; every cited ID and number in an LLM response is verified against the input evidence dictionary before acceptance.
- **No silent degradation**: Every `TraceStep` records `agent`, `status` (`ok` | `fallback` | `error`), `provider`, `model`, `latency_ms`, `validation`, `tokens_used`, and `fallback_reason`.

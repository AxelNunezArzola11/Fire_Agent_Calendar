"""src/calendar/settings.py — FireAgent Calendar configuration.

Standalone settings module (reads env directly) so the existing root
``config.py`` stays byte-identical (Preserve Before Modify). All names use
the ``CAL_`` prefix. Secrets are never logged.
"""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv

    load_dotenv(override=False)
except ImportError:  # pragma: no cover
    pass


def _clean(value: str | None) -> str:
    if not value:
        return ""
    value = value.strip().strip('"').strip("'")
    for prefix in ("your_", "change_me", "<", "TODO"):
        if value.lower().startswith(prefix.lower()):
            return ""
    return value


# --- FIRMS credentials (shared keys, own accessors; never logged) ---
FIRMS_MAP_KEY: str = _clean(os.getenv("FIRMS_MAP_KEY"))
EARTHDATA_USERNAME: str = _clean(os.getenv("EARTHDATA_USERNAME"))
EARTHDATA_PASSWORD: str = _clean(os.getenv("EARTHDATA_PASSWORD"))

# --- Acquisition transport ---
FIRMS_AREA_BASE_URL: str = os.getenv(
    "CAL_FIRMS_AREA_BASE_URL", "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
)
ACQUIRE_TIMEOUT_S: float = float(os.getenv("CAL_ACQUIRE_TIMEOUT_S", "120"))
ACQUIRE_MAX_RETRIES: int = int(os.getenv("CAL_ACQUIRE_MAX_RETRIES", "3"))
ACQUIRE_BACKOFF_S: tuple = (2.0, 4.0, 8.0)  # exponential backoff between retries
FIRMS_MAX_DAYS_PER_REQUEST: int = 5  # Area API cap (verified in ingestor.py)

# --- Future scheduling (configured, NOT activated in Phase 1) ---
NRT_POLL_MINUTES: int = int(os.getenv("CAL_NRT_POLL_MINUTES", "15"))
STANDARD_SYNC_CRON: str = os.getenv("CAL_STANDARD_SYNC_CRON", "daily")
RECOVERY_CRON: str = os.getenv("CAL_RECOVERY_CRON", "daily")

# --- Storage layout (relative to repo root unless absolute) ---
RAW_ROOT: str = os.getenv("CAL_RAW_ROOT", "data_raw_firms")
PARQUET_ROOT: str = os.getenv("CAL_PARQUET_ROOT", "normalized_parquet")
CATALOG_PATH: str = os.getenv("CAL_CATALOG_PATH", "catalog.json")
OPS_DB_PATH: str = os.getenv("CAL_OPS_DB_PATH", "ops.sqlite")
BACKUP_ROOT: str = os.getenv("CAL_BACKUP_ROOT", "backup_mirror")
QUARANTINE_DIRNAME: str = "quarantine"

# --- Storage Manager thresholds (fraction of disk used) ---
STORAGE_WARN: float = 0.70
STORAGE_HIGH: float = 0.85
STORAGE_CRITICAL: float = 0.95

# --- Processing ---
PROCESSING_VERSION: str = "cal_norm_v1"

# --- FireAgent Calendar Phase 5: events, alerts, notifications, Qwen ---
# Additive pins and configuration. Earlier sections untouched. Secrets are
# never logged; unset secrets mean the corresponding integration is
# unavailable (fail-open), never silently degraded.
CAL_EVENTS_DB_PATH: str = os.getenv("CAL_EVENTS_DB_PATH", "calendar_events.sqlite")
CAL_EXEC_5A: str = os.getenv("CAL_EXEC_5A", "270adf518d7647b98660b4b2bdfd833d")
CAL_EXEC_5B: str = os.getenv("CAL_EXEC_5B", "7b4bc145e03e483498711fbfa672a0ce")
CAL_EXEC_5C: str = os.getenv("CAL_EXEC_5C", "e78f50c530384c1a9d84b93430d5873e")
CAL_APP_URL: str = os.getenv("CAL_APP_URL", "")
# SMTP (fail-open when incomplete; same STARTTLS pattern as email_alerts.py).
CAL_SMTP_ENABLED: str = os.getenv("CAL_SMTP_ENABLED", "false")
CAL_SMTP_HOST: str = os.getenv("CAL_SMTP_HOST", "smtp.gmail.com")
CAL_SMTP_PORT: int = int(os.getenv("CAL_SMTP_PORT", "587"))
CAL_SMTP_USER: str = _clean(os.getenv("CAL_SMTP_USER"))
CAL_SMTP_PASSWORD: str = _clean(os.getenv("CAL_SMTP_PASSWORD"))
CAL_SMTP_FROM: str = _clean(os.getenv("CAL_SMTP_FROM")) or CAL_SMTP_USER
CAL_DIGEST_DEFAULT: str = os.getenv("CAL_DIGEST_DEFAULT", "daily")
# Qwen-compatible interpretation endpoint (OpenAI-style chat completions).
# Disabled unless explicitly enabled AND fully configured.
CAL_QWEN_ENABLED: str = os.getenv("CAL_QWEN_ENABLED", "false")
CAL_QWEN_BASE_URL: str = os.getenv("CAL_QWEN_BASE_URL", "")
# Reuses the existing OpenRouter credential when no dedicated Qwen key is
# set (local demo activation; explicit CAL_QWEN_API_KEY always wins). The
# value is never logged, printed, or copied anywhere by this module.
CAL_QWEN_API_KEY: str = _clean(os.getenv("CAL_QWEN_API_KEY")
                               or os.getenv("OPENROUTER_API_KEY"))
CAL_QWEN_MODEL: str = os.getenv("CAL_QWEN_MODEL", "")
CAL_QWEN_TIMEOUT_S: float = float(os.getenv("CAL_QWEN_TIMEOUT_S", "20"))

# --- FireAgent Calendar UI (Phase 3; additive — earlier sections untouched) ---
# Canonical analytical outputs pinned by execution id. Override via env only
# to point the UI at a verified re-run (e.g. 2B 176428fa, 2D c55a66ca).
CAL_DUCKDB_PATH: str = os.getenv("CAL_DUCKDB_PATH", "calendar.duckdb")
CAL_ANALYTICS_ROOT: str = os.getenv("CAL_ANALYTICS_ROOT", "analytics_output")
CAL_EXEC_2A: str = os.getenv("CAL_EXEC_2A", "7ce0d0f7fb1f4e43a348703dc8017a0d")
CAL_EXEC_2B: str = os.getenv("CAL_EXEC_2B", "bfef579ade4b4e368eef12732573137a")
CAL_EXEC_2C: str = os.getenv("CAL_EXEC_2C", "7caf584581684169820505f8b73672aa")
CAL_EXEC_2D: str = os.getenv("CAL_EXEC_2D", "9741f859517c418fa15ff6b6cbe078e2")
CAL_EXEC_2E: str = os.getenv("CAL_EXEC_2E", "15bee7bd8c7947319f5a98d069f952c3")
CAL_EXEC_4B: str = os.getenv("CAL_EXEC_4B", "61722b7636194828899dbeac7085b70e")
CAL_EXEC_4C: str = os.getenv("CAL_EXEC_4C", "461ecbc98ca14f11910e8bcda7254027")
CAL_MAP_RENDER_LIMIT: int = int(os.getenv("CAL_MAP_RENDER_LIMIT", "5000"))
CAL_AOI_PATH: str = os.getenv("CAL_AOI_PATH", "calendar_aois.json")

# --- Map basemap providers (additive — earlier untouched) ---
# CARTO's keyless endpoint returns "API KEY REQUIRED" watermark tiles
# (verified by pixel inspection); a key must also be verified by the
# operator, since key presence alone does not prove tile access.
CARTO_API_KEY: str = _clean(os.getenv("CARTO_API_KEY"))
CAL_CARTO_ENABLED: str = os.getenv("CAL_CARTO_ENABLED", "false")

# --- Level-2 acquisition (additive — earlier untouched) ---
CAL_L2_ROOT: str = os.getenv("CAL_L2_ROOT", "l2_data")
CAL_L2_BUDGET_MB: float = float(os.getenv("CAL_L2_BUDGET_MB", "2048"))
CAL_L2_PAGE_SIZE: int = int(os.getenv("CAL_L2_PAGE_SIZE", "50"))

# --- IBM Granite fallback (watsonx.ai REST; additive — earlier untouched) ---
# Shared keys with the legacy dashboard (own accessors, never logged).
# Granite is enabled by credential presence; absence fails open to the
# deterministic fallback, never silently degraded.
WATSONX_API_KEY: str = _clean(os.getenv("WATSONX_API_KEY"))
WATSONX_PROJECT_ID: str = _clean(os.getenv("WATSONX_PROJECT_ID"))
WATSONX_URL: str = _clean(os.getenv("WATSONX_URL"))
GRANITE_MODEL_ID: str = os.getenv("WATSONX_MODEL_ID",
                                  "ibm/granite-4-h-small").strip()
GRANITE_TIMEOUT_S: float = float(os.getenv("CAL_GRANITE_TIMEOUT_S", "30"))
GRANITE_MAX_ATTEMPTS: int = int(os.getenv("CAL_GRANITE_MAX_ATTEMPTS", "2"))
GRANITE_IAM_URL: str = os.getenv(
    "CAL_GRANITE_IAM_URL", "https://iam.cloud.ibm.com/identity/token")

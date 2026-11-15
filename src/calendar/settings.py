"""src/calendar/settings.py — FireAgent Calendar configuration.

Standalone settings module (reads env directly) so the existing root
``config.py`` stays byte-identical (Preserve Before Modify). All names use
the ``CAL_`` prefix. Secrets are never logged.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import sys

try:
    from dotenv import load_dotenv

    load_dotenv(override=False)
except ImportError:  # pragma: no cover
    pass

REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent


def _clean(value: str | None) -> str:
    if not value:
        return ""
    value = value.strip().strip('"').strip("'")
    for prefix in ("your_", "change_me", "<", "TODO"):
        if value.lower().startswith(prefix.lower()):
            return ""
    return value


def _from_streamlit_secrets(name: str) -> str | None:
    """Read a scalar setting from loaded st.secrets without importing streamlit."""
    st_mod = sys.modules.get("streamlit")
    if st_mod is None:
        return None
    secrets = getattr(st_mod, "secrets", None)
    if secrets is None:
        return None
    try:
        if name in secrets:
            val = secrets[name]
            if isinstance(val, (str, int, float)) and not isinstance(val, bool):
                return str(val).strip()
        if "calendar" in secrets:
            nested = secrets["calendar"]
            if isinstance(nested, Mapping) and name in nested:
                val = nested[name]
                if isinstance(val, (str, int, float)) and not isinstance(val, bool):
                    return str(val).strip()
    except Exception:
        return None
    return None


def _env(name: str, default: str = "") -> str:
    """Resolve configuration via env var -> st.secrets -> default."""
    val = os.getenv(name)
    if val is not None:
        return val.strip()
    sec = _from_streamlit_secrets(name)
    if sec is not None:
        return sec
    return default


def resolve_repo_path(raw: str | Path) -> Path:
    """Resolve relative paths against CWD first, then REPO_ROOT if present."""
    text = str(raw).strip() if raw is not None else ""
    if not text:
        return Path("")
    p = Path(text)
    if p.is_absolute() or p.exists():
        return p
    repo_candidate = REPO_ROOT / p
    if repo_candidate.exists():
        return repo_candidate
    return p


# --- FIRMS credentials (shared keys, own accessors; never logged) ---
FIRMS_MAP_KEY: str = _clean(_env("FIRMS_MAP_KEY"))
EARTHDATA_USERNAME: str = _clean(_env("EARTHDATA_USERNAME"))
EARTHDATA_PASSWORD: str = _clean(_env("EARTHDATA_PASSWORD"))

# --- Acquisition transport ---
FIRMS_AREA_BASE_URL: str = _env(
    "CAL_FIRMS_AREA_BASE_URL", "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
)
ACQUIRE_TIMEOUT_S: float = float(_env("CAL_ACQUIRE_TIMEOUT_S", "120"))
ACQUIRE_MAX_RETRIES: int = int(_env("CAL_ACQUIRE_MAX_RETRIES", "3"))
ACQUIRE_BACKOFF_S: tuple = (2.0, 4.0, 8.0)  # exponential backoff between retries
FIRMS_MAX_DAYS_PER_REQUEST: int = 5  # Area API cap (verified in ingestor.py)

# --- Future scheduling (configured, NOT activated in Phase 1) ---
NRT_POLL_MINUTES: int = int(_env("CAL_NRT_POLL_MINUTES", "15"))
STANDARD_SYNC_CRON: str = _env("CAL_STANDARD_SYNC_CRON", "daily")
RECOVERY_CRON: str = _env("CAL_RECOVERY_CRON", "daily")

# --- Storage layout (relative to repo root unless absolute) ---
RAW_ROOT: str = _env("CAL_RAW_ROOT", "data_raw_firms")
PARQUET_ROOT: str = _env("CAL_PARQUET_ROOT", "normalized_parquet")
CATALOG_PATH: str = _env("CAL_CATALOG_PATH", "catalog.json")
OPS_DB_PATH: str = _env("CAL_OPS_DB_PATH", "ops.sqlite")
BACKUP_ROOT: str = _env("CAL_BACKUP_ROOT", "backup_mirror")
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
CAL_EVENTS_DB_PATH: str = _env("CAL_EVENTS_DB_PATH", "calendar_events.sqlite")
CAL_EXEC_5A: str = _env("CAL_EXEC_5A", "270adf518d7647b98660b4b2bdfd833d")
CAL_EXEC_5B: str = _env("CAL_EXEC_5B", "7b4bc145e03e483498711fbfa672a0ce")
CAL_EXEC_5C: str = _env("CAL_EXEC_5C", "e78f50c530384c1a9d84b93430d5873e")
CAL_APP_URL: str = _env("CAL_APP_URL", "")
# SMTP (fail-open when incomplete; same STARTTLS pattern as email_alerts.py).
CAL_SMTP_ENABLED: str = _env("CAL_SMTP_ENABLED", "false")
CAL_SMTP_HOST: str = _env("CAL_SMTP_HOST", "smtp.gmail.com")
CAL_SMTP_PORT: int = int(_env("CAL_SMTP_PORT", "587"))
CAL_SMTP_USER: str = _clean(_env("CAL_SMTP_USER"))
CAL_SMTP_PASSWORD: str = _clean(_env("CAL_SMTP_PASSWORD"))
CAL_SMTP_FROM: str = _clean(_env("CAL_SMTP_FROM")) or CAL_SMTP_USER
CAL_DIGEST_DEFAULT: str = _env("CAL_DIGEST_DEFAULT", "daily")
# Qwen-compatible interpretation endpoint (OpenAI-style chat completions).
# Disabled unless explicitly enabled AND fully configured.
CAL_QWEN_ENABLED: str = _env("CAL_QWEN_ENABLED", "false")
CAL_QWEN_BASE_URL: str = _env("CAL_QWEN_BASE_URL", "")
# Reuses the existing OpenRouter credential when no dedicated Qwen key is
# set (local demo activation; explicit CAL_QWEN_API_KEY always wins). The
# value is never logged, printed, or copied anywhere by this module.
CAL_QWEN_API_KEY: str = _clean(_env("CAL_QWEN_API_KEY")
                               or _env("OPENROUTER_API_KEY"))
CAL_QWEN_MODEL: str = _env("CAL_QWEN_MODEL", "")
CAL_QWEN_TIMEOUT_S: float = float(_env("CAL_QWEN_TIMEOUT_S", "20"))

# --- FireAgent Calendar UI (Phase 3; additive — earlier sections untouched) ---
# Canonical analytical outputs pinned by execution id. Override via env only
# to point the UI at a verified re-run (e.g. 2B 176428fa, 2D c55a66ca).
CAL_DUCKDB_PATH: str = _env("CAL_DUCKDB_PATH", "calendar.duckdb")
CAL_REGIONAL_ROOT: str = _env("CAL_REGIONAL_ROOT", "regional_archives")
CAL_REGIONAL_DUCKDB_PATH: str = _env(
    "CAL_REGIONAL_DUCKDB_PATH", "regional_archives/regional.duckdb"
)
CAL_ANALYTICS_ROOT: str = _env("CAL_ANALYTICS_ROOT", "analytics_output")
CAL_RUNS_PATH: str = _env("CAL_RUNS_PATH", "analytics_runs.jsonl")
CAL_EXEC_2A: str = _env("CAL_EXEC_2A", "7ce0d0f7fb1f4e43a348703dc8017a0d")
CAL_EXEC_2B: str = _env("CAL_EXEC_2B", "bfef579ade4b4e368eef12732573137a")
CAL_EXEC_2C: str = _env("CAL_EXEC_2C", "7caf584581684169820505f8b73672aa")
CAL_EXEC_2D: str = _env("CAL_EXEC_2D", "9741f859517c418fa15ff6b6cbe078e2")
CAL_EXEC_2E: str = _env("CAL_EXEC_2E", "15bee7bd8c7947319f5a98d069f952c3")
CAL_EXEC_4B: str = _env("CAL_EXEC_4B", "61722b7636194828899dbeac7085b70e")
CAL_EXEC_4C: str = _env("CAL_EXEC_4C", "461ecbc98ca14f11910e8bcda7254027")
CAL_MAP_RENDER_LIMIT: int = int(_env("CAL_MAP_RENDER_LIMIT", "5000"))
CAL_AOI_PATH: str = _env("CAL_AOI_PATH", "calendar_aois.json")

# --- Map basemap providers (additive — earlier untouched) ---
# CARTO's keyless endpoint returns "API KEY REQUIRED" watermark tiles
# (verified by pixel inspection); a key must also be verified by the
# operator, since key presence alone does not prove tile access.
CARTO_API_KEY: str = _clean(_env("CARTO_API_KEY"))
CAL_CARTO_ENABLED: str = _env("CAL_CARTO_ENABLED", "false")

# --- Level-2 acquisition (additive — earlier untouched) ---
CAL_L2_ROOT: str = _env("CAL_L2_ROOT", "l2_data")
CAL_L2_BUDGET_MB: float = float(_env("CAL_L2_BUDGET_MB", "2048"))
CAL_L2_PAGE_SIZE: int = int(_env("CAL_L2_PAGE_SIZE", "50"))

# --- IBM Granite fallback (watsonx.ai REST or OpenAI-compatible; additive) ---
# Shared keys with the legacy dashboard (own accessors, never logged).
# Granite is enabled by credential presence; absence fails open to the
# deterministic fallback, never silently degraded.
CAL_AI_PROVIDER: str = _env("CAL_AI_PROVIDER", "auto").lower() or "auto"
WATSONX_API_KEY: str = _clean(_env("WATSONX_API_KEY"))
WATSONX_PROJECT_ID: str = _clean(_env("WATSONX_PROJECT_ID"))
WATSONX_URL: str = _clean(_env("WATSONX_URL"))
GRANITE_MODEL_ID: str = _env("WATSONX_MODEL_ID", "ibm/granite-4-h-small")
CAL_GRANITE_API_KEY: str = _clean(_env("CAL_GRANITE_API_KEY") or WATSONX_API_KEY)
CAL_GRANITE_PROJECT_ID: str = _clean(_env("CAL_GRANITE_PROJECT_ID") or WATSONX_PROJECT_ID)
CAL_GRANITE_BASE_URL: str = _clean(_env("CAL_GRANITE_BASE_URL") or WATSONX_URL)
CAL_GRANITE_MODEL: str = _env("CAL_GRANITE_MODEL") or GRANITE_MODEL_ID
CAL_GRANITE_ENDPOINT_STYLE: str = _env("CAL_GRANITE_ENDPOINT_STYLE", "")
GRANITE_TIMEOUT_S: float = float(_env("CAL_GRANITE_TIMEOUT_S", "30"))
GRANITE_MAX_ATTEMPTS: int = int(_env("CAL_GRANITE_MAX_ATTEMPTS", "2"))
GRANITE_IAM_URL: str = _env(
    "CAL_GRANITE_IAM_URL", "https://iam.cloud.ibm.com/identity/token")

_STORAGE_KEYS: tuple[str, ...] = (
    "RAW_ROOT",
    "PARQUET_ROOT",
    "CATALOG_PATH",
    "OPS_DB_PATH",
    "BACKUP_ROOT",
    "CAL_EVENTS_DB_PATH",
    "CAL_DUCKDB_PATH",
    "CAL_REGIONAL_ROOT",
    "CAL_REGIONAL_DUCKDB_PATH",
    "CAL_ANALYTICS_ROOT",
    "CAL_RUNS_PATH",
    "CAL_AOI_PATH",
    "CAL_EXEC_2A",
    "CAL_EXEC_2B",
    "CAL_EXEC_2C",
    "CAL_EXEC_2D",
    "CAL_EXEC_2E",
    "CAL_EXEC_4B",
    "CAL_EXEC_4C",
    "CAL_EXEC_5A",
    "CAL_EXEC_5B",
    "CAL_EXEC_5C",
    "FIRMS_MAP_KEY",
    "CAL_AI_PROVIDER",
    "CAL_QWEN_ENABLED",
    "CAL_QWEN_BASE_URL",
    "CAL_QWEN_API_KEY",
    "CAL_QWEN_MODEL",
    "WATSONX_API_KEY",
    "WATSONX_PROJECT_ID",
    "WATSONX_URL",
    "GRANITE_MODEL_ID",
    "CAL_GRANITE_API_KEY",
    "CAL_GRANITE_PROJECT_ID",
    "CAL_GRANITE_BASE_URL",
    "CAL_GRANITE_MODEL",
    "CAL_GRANITE_ENDPOINT_STYLE",
)

_ENV_FOR_ATTR: dict[str, str] = {
    "RAW_ROOT": "CAL_RAW_ROOT",
    "PARQUET_ROOT": "CAL_PARQUET_ROOT",
    "CATALOG_PATH": "CAL_CATALOG_PATH",
    "OPS_DB_PATH": "CAL_OPS_DB_PATH",
    "BACKUP_ROOT": "CAL_BACKUP_ROOT",
    "GRANITE_MODEL_ID": "WATSONX_MODEL_ID",
}

_SETTINGS_SNAPSHOT: dict[str, tuple[str, str | None]] = {
    name: (str(globals()[name]), os.getenv(_ENV_FOR_ATTR.get(name, name)))
    for name in _STORAGE_KEYS
}


def get_setting(name: str, default: str = "") -> str:
    """Resolve a configuration attribute dynamically (supports monkeypatch, env, secrets)."""
    env_name = _ENV_FOR_ATTR.get(name, name)
    snap = _SETTINGS_SNAPSHOT.get(name)
    cur_attr = globals().get(name)
    cur_env = os.getenv(env_name)
    if snap is not None:
        orig_attr, orig_env = snap
        cur_attr_str = "" if cur_attr is None else str(cur_attr).strip()
        if cur_attr_str != orig_attr:
            return cur_attr_str
        if cur_env != orig_env:
            if cur_env is not None:
                return cur_env.strip()
            sec = _from_streamlit_secrets(env_name)
            return sec if sec is not None else default
        if orig_env is not None:
            return cur_attr_str
        sec = _from_streamlit_secrets(env_name)
        if sec is not None:
            return sec
        return cur_attr_str
    if cur_env is not None:
        return cur_env.strip()
    sec = _from_streamlit_secrets(env_name)
    if sec is not None:
        return sec
    if cur_attr is not None:
        return str(cur_attr).strip()
    return default

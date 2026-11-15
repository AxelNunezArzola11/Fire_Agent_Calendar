"""OpenRouter Qwen model discovery (bounded, cached, approval-gated).

Discovers chat-compatible Qwen models available to the configured API
key WITHOUT sending chat completions (catalog + key-info GETs only —
no completion charges possible). Selection policy:

1. Free Qwen chat model (zero prompt AND completion price).
2. Cheapest accessible Qwen chat model — paid use requires explicit
   operator approval recorded in the store; never auto-selected.
3. Anything else (including generic `openrouter/free` routing, which
   is not necessarily Qwen) is never presented as Qwen.

Results persist in store `meta` (`qwen_discovery`) with a 24 h TTL;
a manual refresh always re-queries. Every failure mode degrades to
the deterministic fallback with an explicit reason — discovery can
never break the app or trigger paid inference silently.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

MODELS_URL = "https://openrouter.ai/api/v1/models"
KEY_URL = "https://openrouter.ai/api/v1/auth/key"
CACHE_TTL_S = 24 * 3600
META_KEY = "qwen_discovery"
APPROVAL_KEY = "qwen_paid_approval"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _price(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def catalog_entry(raw: dict) -> dict | None:
    """Normalize one catalog record (None when unusable for chat).

    Requires text-in/text-out modality declarations; extracts pricing
    and Qwen-family membership. Defensive: unknown shapes are skipped,
    never guessed.
    """
    model_id = raw.get("id") or ""
    if not model_id or not isinstance(model_id, str):
        return None
    arch = raw.get("architecture") or {}
    inputs = arch.get("input_modalities") or []
    outputs = arch.get("output_modalities") or []
    chat_ok = ("text" in inputs) and ("text" in outputs)
    pricing = raw.get("pricing") or {}
    prompt = _price(pricing.get("prompt"))
    completion = _price(pricing.get("completion"))
    priced = prompt is not None and completion is not None
    return {
        "id": model_id,
        "chat": bool(chat_ok),
        "qwen": "qwen" in model_id.lower(),
        "free": bool(priced) and prompt == 0.0 and completion == 0.0,
        "priced": bool(priced),
        "combined": (prompt + completion) if priced else None,
        "context_length": raw.get("context_length"),
    }


def fetch_json(url: str, *, api_key: str = "", timeout_s: float = 20.0,
               transport=None) -> dict:
    """GET JSON (only ever the catalog or key-info endpoints).

    `transport` is an injectable `requests.get`-compatible callable
    (tests stub it; production passes nothing). Raises on HTTP,
    network, timeout, or JSON errors — callers convert to reasons.
    """
    import requests

    getter = transport or requests.get
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = getter(url, headers=headers, timeout=timeout_s)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("catalog response is not a JSON object")
    return data


def check_key(api_key: str, *, timeout_s: float = 20.0,
              transport=None) -> dict:
    """Verify the API key authorizes successfully (free key-info call).

    Returns {authorized, label}. Distinguishes key authorization from
    mere public-catalog reachability.
    """
    if not api_key:
        return {"authorized": False, "label": "no API key configured"}
    try:
        data = fetch_json(KEY_URL, api_key=api_key, timeout_s=timeout_s,
                          transport=transport)
    except Exception as exc:  # noqa: BLE001 - converted to a reason
        return {"authorized": False,
                "label": f"authorization check failed: {exc}"[:200]}
    payload = data.get("data") or {}
    return {"authorized": True,
            "label": str(payload.get("label") or "key authorized")[:80]}


def discover_models(api_key: str = "", *, timeout_s: float = 20.0,
                    transport=None) -> dict:
    """Fetch + normalize the model catalog (no completions sent).

    Returns {models, fetched_at, key_authorized, key_label, error}.
    On any failure models == [] and error names the cause.
    """
    started = time.monotonic()
    try:
        data = fetch_json(MODELS_URL, api_key=api_key, timeout_s=timeout_s,
                          transport=transport)
    except Exception as exc:  # noqa: BLE001 - converted to a reason
        return {"models": [], "fetched_at": _utcnow(),
                "key_authorized": False,
                "key_label": "catalog unreachable",
                "error": f"catalog fetch failed: {exc}"[:200],
                "latency_ms": round((time.monotonic() - started) * 1000, 1)}
    raw_models = data.get("data") or []
    models = []
    for raw in raw_models:
        if isinstance(raw, dict):
            entry = catalog_entry(raw)
            if entry is not None:
                models.append(entry)
    key = check_key(api_key, timeout_s=timeout_s, transport=transport)
    return {"models": models, "fetched_at": _utcnow(),
            "key_authorized": key["authorized"], "key_label": key["label"],
            "error": "",
            "latency_ms": round((time.monotonic() - started) * 1000, 1)}


def select_model(models: list[dict], *, allow_paid: bool = False) -> dict:
    """Apply the selection policy (pure function).

    Returns {model_id, price_class, reason, needs_approval}. price_class
    is one of free|paid|none. A paid Qwen is reported with
    needs_approval=True unless allow_paid was explicitly granted — it
    is never silently selected.
    """
    candidates = [m for m in models if m["chat"] and m["qwen"]]
    free = sorted([m for m in candidates if m["free"]],
                  key=lambda m: m["id"])
    if free:
        return {"model_id": free[0]["id"], "price_class": "free",
                "needs_approval": False,
                "reason": f"free Qwen chat model ({len(free)} eligible)"}
    priced = sorted([m for m in candidates if m["priced"]],
                    key=lambda m: (m["combined"], m["id"]))
    if priced:
        best = priced[0]
        if allow_paid:
            return {"model_id": best["id"], "price_class": "paid",
                    "needs_approval": False,
                    "reason": f"cheapest Qwen chat model ({best['combined']}"
                    " combined), operator-approved"}
        return {"model_id": best["id"], "price_class": "paid",
                "needs_approval": True,
                "reason": f"cheapest Qwen chat model ({best['id']}) is paid; "
                "operator approval required before use"}
    if candidates:
        return {"model_id": None, "price_class": "none",
                "needs_approval": False,
                "reason": "Qwen chat models listed but none carry usable "
                "pricing; deterministic fallback stays active"}
    return {"model_id": None, "price_class": "none",
            "needs_approval": False,
            "reason": "no Qwen chat model in the catalog; deterministic "
            "fallback stays active"}


def validate_explicit_model(qwen_models: list[dict],
                            model_id: str) -> dict:
    """Pricing verdict for an explicitly configured model id.

    Returns {verdict, reason}; verdict is free|paid|unknown. Only an
    explicit zero/zero price is free; missing, unknown, malformed, or
    non-zero pricing is never free, and names never imply freedom.
    """
    match = next((m for m in qwen_models if m.get("id") == model_id),
                 None)
    if match is None:
        return {"verdict": "unknown",
                "reason": f"{model_id} is not in the discovered Qwen "
                          "catalog; pricing unverified"}
    if match.get("free"):
        return {"verdict": "free",
                "reason": f"{model_id} verified free (0/0 pricing)"}
    if match.get("priced"):
        return {"verdict": "paid",
                "reason": f"{model_id} is paid "
                          f"(combined {match.get('combined')}); operator "
                          "approval required before use"}
    return {"verdict": "unknown",
            "reason": f"{model_id} carries no usable pricing; treated "
                      "as not free"}


def refresh_discovery(conn, *, api_key: str = "",
                      timeout_s: float = 20.0, transport=None) -> dict:
    """Re-query the catalog and persist the selection (explicit action).

    Preserves any recorded paid-model approval. Returns the stored
    record {fetched_at, catalog_count, key_authorized, key_label,
    selection, qwen_models, error, latency_ms}. qwen_models is the
    bounded Qwen-chat subset used to validate explicit configuration.
    """
    from . import store as store_mod

    previous = store_mod.meta_get(conn, META_KEY) or {}
    approved = store_mod.meta_get(conn, APPROVAL_KEY) or {}
    found = discover_models(api_key, timeout_s=timeout_s, transport=transport)
    selection = select_model(
        found["models"],
        allow_paid=bool(approved.get("model_id")) and
        approved.get("model_id") == (select_model(
            found["models"], allow_paid=True).get("model_id")))
    # Re-resolve approval against the FRESH catalog: approval names a
    # model id; it only applies while that id is still the cheapest.
    record = {"fetched_at": found["fetched_at"],
              "catalog_count": len(found["models"]),
              "key_authorized": found["key_authorized"],
              "key_label": found["key_label"],
              "selection": selection, "error": found["error"],
              "latency_ms": found["latency_ms"],
              "qwen_models": [m for m in found["models"]
                              if m["chat"] and m["qwen"]]}
    store_mod.meta_set(conn, META_KEY, record)
    if previous.get("selection", {}).get("model_id") != \
            selection.get("model_id"):
        logger.info("qwen discovery: selection %s (%s)",
                    selection.get("model_id"), selection.get("price_class"))
    return record


def approve_paid_model(conn, model_id: str) -> dict:
    """Record explicit operator approval for one paid model id."""
    from . import store as store_mod

    record = {"model_id": model_id, "approved_at": _utcnow()}
    store_mod.meta_set(conn, APPROVAL_KEY, record)
    return record


def load_discovery(conn, *, now_ts: float | None = None) -> dict | None:
    """Load the cached discovery record (None when absent or stale)."""
    from . import store as store_mod

    record = store_mod.meta_get(conn, META_KEY)
    if not record:
        return None
    try:
        fetched = datetime.fromisoformat(record["fetched_at"])
        age = (now_ts if now_ts is not None else time.time()) - \
            fetched.timestamp()
    except (ValueError, TypeError, KeyError):
        return None
    if age > CACHE_TTL_S:
        return None
    return record


def effective_model(conn, config) -> tuple[str | None, str]:
    """Resolve the model id Explain should use (selection or config).

    Returns (model_id_or_None, human-readable reason). Strict
    free-model policy: a discovered free selection wins; an explicit
    CAL_QWEN_MODEL is used only after its discovered pricing
    validates as free (or names an approved paid id); anything else
    resolves to None with an explanatory reason so callers fail over
    to another provider or deterministic fallback. Paid endpoints are
    never silently substituted.
    """
    from . import store as store_mod

    record = load_discovery(conn)
    if record:
        selection = record.get("selection") or {}
        if selection.get("model_id") and not selection.get("needs_approval"):
            return (selection["model_id"],
                    f"discovered {selection.get('price_class')} model "
                    f"{selection['model_id']}")
    if config.model:
        if not record:
            return None, (f"explicit {config.model} has no discovery "
                          "record; pricing unverified, Qwen unavailable")
        verdict = validate_explicit_model(record.get("qwen_models") or [],
                                          config.model)
        if verdict["verdict"] == "free":
            return config.model, ("explicit CAL_QWEN_MODEL " +
                                  verdict["reason"])
        approved = store_mod.meta_get(conn, APPROVAL_KEY) or {}
        if verdict["verdict"] == "paid" and \
                approved.get("model_id") == config.model:
            return config.model, ("explicit CAL_QWEN_MODEL " +
                                  verdict["reason"] +
                                  " (operator-approved)")
        return None, ("explicit CAL_QWEN_MODEL blocked: " +
                      verdict["reason"])
    if record:
        selection = record.get("selection") or {}
        if selection.get("needs_approval"):
            return None, (f"Qwen unavailable: {selection.get('reason')} "
                          "(no explicit model configured)")
        return None, selection.get(
            "reason", "no model discovered or configured")
    return None, "no model discovered or configured"

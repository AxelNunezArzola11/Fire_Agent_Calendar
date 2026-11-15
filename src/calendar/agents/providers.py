"""LLM provider abstraction: Qwen first, Granite fallback, then local.

Providers return text; validation happens in evidence_validator, never
here. Secrets (API keys, tokens) are never logged, stored, or echoed:
availability reports name missing fields only, and error strings pass
through redact().
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from .. import settings
from ..events import qwen as qwen_mod

MAX_TOKENS = 600
TEMPERATURE = 0.0

_SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|apikey|access_token|token|secret|"
               r"password|project_id)\s*['\"]?\s*[:=]\s*['\"]?\S+?['\"]?"
               r"(?=[\s,};&]|$)"),
    re.compile(r"['\"](?:api[_-]?key|apikey|access_token|token|secret|"
               r"password)['\"]\s*:\s*['\"][^'\"]+['\"]",
               re.IGNORECASE),
    re.compile(r"Bearer\s+\S+", re.IGNORECASE),
)


def redact(text: str) -> str:
    """Scrub credential-shaped substrings (defensive; callers send none).

    Covers bare (key=value), quoted ('key': '...'), and JSON-quoted
    ("apikey": "...") shapes plus Bearer tokens. Truncates to 500
    chars; call sites keep full lengths out of audit records.
    """
    scrubbed = str(text)
    for pattern in _SECRET_PATTERNS:
        scrubbed = pattern.sub("[redacted]", scrubbed)
    return scrubbed[:500]


@dataclass(frozen=True)
class Attempt:
    """One provider attempt outcome (auditable, secret-free)."""

    provider: str
    model: str
    ok: bool
    latency_ms: float
    text: str = ""
    error: str = ""
    response_chars: int = 0
    retries: int = 0
    input_tokens: int | None = None
    generated_tokens: int | None = None


class Provider:
    """One completion backend (name + availability + text completion)."""

    name = "base"

    def available(self) -> tuple[bool, str]:
        """(usable, reason). Reason never contains secret values."""
        raise NotImplementedError

    def complete(self, prompt: str, system: str) -> Attempt:
        """Run one bounded completion (no retries; the chain retries)."""
        raise NotImplementedError


class QwenProvider(Provider):
    """Qwen via OpenAI-style chat completions (requests, bounded)."""

    name = "qwen"

    def __init__(self, config=None, transport=None) -> None:
        self._config = config or qwen_mod.QwenConfig.from_settings()
        self._transport = transport or _qwen_transport

    def available(self) -> tuple[bool, str]:
        status = self._config.status()
        return status["available"], redact(status["reason"])

    @property
    def model(self) -> str:
        return self._config.model or ""

    def complete(self, prompt: str, system: str) -> Attempt:
        usable, reason = self.available()
        if not usable:
            return Attempt(self.name, self.model, False, 0.0,
                           error=f"unavailable: {reason}")
        started = time.monotonic()
        try:
            text, usage = self._transport(prompt, system, self._config)
        except Exception as exc:  # noqa: BLE001 - provider failure -> chain
            latency = (time.monotonic() - started) * 1000.0
            return Attempt(self.name, self.model, False, latency,
                           error=redact(f"{type(exc).__name__}: {exc}"))
        latency = (time.monotonic() - started) * 1000.0
        if not isinstance(text, str) or not text.strip():
            return Attempt(self.name, self.model, False, latency,
                           error="empty response")
        return Attempt(self.name, self.model, True, latency, text=text,
                       response_chars=len(text),
                       input_tokens=(usage or {}).get("input_tokens"),
                       generated_tokens=(usage or {}).get(
                           "generated_tokens"))


RESPONSE_BYTE_CAP = 256 * 1024


def _qwen_transport(prompt: str, system: str, config) -> str:
    """Role-specific chat completion (mirrors qwen.http_transport)."""
    import requests

    timeout = getattr(config, "timeout_s", 0)
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("qwen timeout_s must be > 0")
    response = requests.post(
        f"{config.base_url}/chat/completions",
        headers={"Authorization": f"Bearer {config.api_key}",
                 "Content-Type": "application/json"},
        json={"model": config.model,
              "messages": [{"role": "system", "content": system},
                           {"role": "user", "content": prompt}],
              "temperature": TEMPERATURE,
              "max_tokens": MAX_TOKENS},
        timeout=timeout)
    response.raise_for_status()
    body = getattr(response, "content", b"") or b""
    if len(body) > RESPONSE_BYTE_CAP:
        raise ValueError("qwen response exceeds size cap")
    try:
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        use = data.get("usage") or {}
        usage = {"input_tokens": use.get("prompt_tokens"),
                 "generated_tokens": use.get("completion_tokens")}
        return text, usage
    except (ValueError, KeyError, IndexError, TypeError,
            AttributeError) as exc:
        raise ValueError(f"malformed qwen response: {type(exc).__name__}")


class ProviderChain:
    """Ordered failover across providers (first success wins)."""

    def __init__(self, providers) -> None:
        self._providers = tuple(providers)

    @property
    def providers(self):
        return self._providers

    def complete(self, prompt: str, system: str) -> tuple[Attempt, ...]:
        """Try each provider once, in order; returns all attempts.

        Transport-level failover only: stops at the first transport
        success. Validity-level failover (invalid JSON -> next
        provider) lives in runner, which iterates providers itself.
        Latency is per-attempt; callers sum across attempts.
        """
        attempts = []
        for provider in self._providers:
            try:
                attempt = provider.complete(prompt, system)
            except Exception as exc:  # noqa: BLE001 - faulty provider
                name = getattr(provider, "name", "?")
                attempt = Attempt(name, "", False, 0.0,
                                  error=redact(f"{type(exc).__name__}: {exc}"))
            attempts.append(attempt)
            if attempt.ok:
                break
        return tuple(attempts)


class DeterministicProvider(Provider):
    """Explicit deterministic-rules backend (never calls an external LLM).

    Signals the runner to skip external inference cleanly and produce an
    auditable rule-based response with deterministic classifications.
    """

    name = "deterministic"

    def available(self) -> tuple[bool, str]:
        return True, "deterministic rules enabled"

    @property
    def model(self) -> str:
        return "deterministic-rules-v1"

    def complete(self, prompt: str, system: str) -> Attempt:
        _ = (prompt, system)
        return Attempt(self.name, self.model, False, 0.0,
                       error="deterministic mode selected (no external LLM)")


def resolve_ai_provider_mode(mode: str | None = None) -> str:
    """Resolve requested AI provider mode ('auto'|'granite'|'qwen'|'deterministic')."""
    raw = (
        mode
        if mode is not None
        else settings.get_setting("CAL_AI_PROVIDER", "auto")
    )
    cleaned = str(raw or "auto").strip().lower()
    if cleaned in ("auto", "granite", "qwen", "deterministic"):
        return cleaned
    return "auto"


def _load_granite_provider():
    from . import granite as granite_mod

    return granite_mod.GraniteProvider()


def provider_status(mode: str | None = None, *, qwen_config=None) -> dict:
    """Structured availability status for AI providers (never exposes secrets)."""
    import logging as _logging

    selected = resolve_ai_provider_mode(mode)
    qwen_provider = QwenProvider(config=qwen_config)
    qwen_ok, qwen_reason = qwen_provider.available()

    granite_ok, granite_reason, granite_model = False, "unavailable", ""
    try:
        gp = _load_granite_provider()
        granite_ok, granite_reason = gp.available()
        granite_model = gp.model
    except Exception as exc:  # noqa: BLE001
        _logging.getLogger(__name__).warning(
            "granite provider unavailable: %s", type(exc).__name__)
        granite_reason = f"load failed: {type(exc).__name__}"

    if selected == "deterministic":
        effective = "deterministic"
    elif selected == "granite":
        effective = "granite" if granite_ok else "deterministic"
    elif selected == "qwen":
        effective = "qwen" if qwen_ok else "deterministic"
    else:
        if qwen_ok:
            effective = "qwen"
        elif granite_ok:
            effective = "granite"
        else:
            effective = "deterministic"

    return {
        "configured_mode": selected,
        "effective_mode": effective,
        "deterministic": {"available": True, "model": "deterministic-rules-v1"},
        "qwen": {
            "available": qwen_ok,
            "reason": qwen_reason,
            "model": qwen_provider.model,
        },
        "granite": {
            "available": granite_ok,
            "reason": granite_reason,
            "model": granite_model,
        },
    }


def default_chain(mode: str | None = None, *, qwen_config=None) -> ProviderChain:
    """Configured provider chain honoring CAL_AI_PROVIDER.

    - ``auto`` (default): Qwen first, IBM Granite fallback (Qwen-only if
      the Granite module cannot load, with the reason logged).
    - ``granite``: IBM Granite first.
    - ``qwen``: Qwen first.
    - ``deterministic``: no external LLM providers (immediate deterministic rules).
    """
    import logging as _logging

    selected = resolve_ai_provider_mode(mode)
    if selected == "deterministic":
        return ProviderChain([])

    providers_list: list = []
    if selected in ("auto", "qwen"):
        providers_list.append(QwenProvider(config=qwen_config))
    if selected in ("auto", "granite"):
        try:
            providers_list.append(_load_granite_provider())
        except Exception as exc:  # noqa: BLE001 - optional provider
            _logging.getLogger(__name__).warning(
                "granite provider unavailable: %s", type(exc).__name__)
    return ProviderChain(providers_list)

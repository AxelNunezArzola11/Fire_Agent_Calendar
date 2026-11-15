"""
src/sar/qwen_gateway.py
=======================
QwenGateway — drop-in LLM backend for SAR change-detection summaries.

Backends
--------
ollama (default):
    model: qwen3:1.7b
    think: false  (disables chain-of-thought tokens)
    num_predict: capped (default 300)
    temperature: 0.2
    timeout: 180 s

openrouter:
    model: set via OPENROUTER_MODEL env var (REQUIRED — no default; raises if unset)
    key: from env OPENROUTER_API_KEY (never logged/printed)
    reasoning: {"max_tokens": 200} to cap chain-of-thought token consumption
        (see https://openrouter.ai/docs/use-cases/reasoning-models)
        "max_tokens" within the reasoning dict caps the tokens used for reasoning;
        remaining tokens (out of the top-level max_tokens budget) go to content.
        Using effort:"low" alone does NOT reliably prevent content=null for all
        reasoning-capable models — the model may still fill the budget with reasoning.
        The max_tokens cap in the reasoning dict is the more reliable mechanism.
    reasoning text is returned in choices[0].message.reasoning (a separate field
        from choices[0].message.content) — content=null is still possible if
        reasoning_max_tokens >= total max_tokens.
    max_tokens raised to 600 (from 300) to give enough space for content after
        the 200-token reasoning cap.
    same response interface as ollama
    retries with exponential backoff on HTTP 429
    max 3 retries; delays: 2, 4, 8 s

OpenRouter reasoning docs:
  URL: https://openrouter.ai/docs/use-cases/reasoning-models
  The page could not be fetched in Task 5h (no network access during this task).
  The implementation follows the same reasoning-cap pattern established in Task 5g,
  based on the documented "max_tokens" field inside the "reasoning" parameter dict.

Selection
---------
  env var QWEN_BACKEND: "ollama" (default) or "openrouter"
  If the backend is unreachable, raises BackendUnavailableError — no silent fallback.

Dependencies: requests (stdlib-like), python-dotenv (already in project).
"""

import json
import logging
import os
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load env from .env if present
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass  # python-dotenv not installed; env vars must be set externally

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class BackendUnavailableError(RuntimeError):
    """Raised when the configured backend cannot be reached or returns errors."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _get_openrouter_model() -> str:
    """Return OPENROUTER_MODEL from environment. Raises if unset."""
    model = os.getenv("OPENROUTER_MODEL")
    if not model:
        raise BackendUnavailableError(
            "OPENROUTER_MODEL is not set in the environment. "
            "Set OPENROUTER_MODEL to the desired model slug (e.g. "
            "'qwen/qwen3-8b:free') before using the openrouter backend."
        )
    return model


_OLLAMA_MODEL = "qwen3:1.7b"
# Raised from 300 to 600 to reduce content=null when reasoning tokens are generated.
# (https://openrouter.ai/docs/use-cases/reasoning-models: max_tokens covers reasoning
#  + completion combined; too low a budget causes content=null.)
_OPENROUTER_DEFAULT_MAX_TOKENS = 600

_RETRY_DELAYS = (2, 4, 8)   # seconds; only on HTTP 429


# ---------------------------------------------------------------------------
# Internal HTTP helper (no external HTTP library required beyond requests)
# ---------------------------------------------------------------------------

def _post_json(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    """POST JSON payload; return parsed JSON response. Raises on HTTP errors."""
    import requests   # imported lazily so the module is importable without requests
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    if resp.status_code >= 400:
        raise requests.HTTPError(
            f"HTTP {resp.status_code}: {resp.text[:300]}", response=resp
        )
    return resp.json()


def _call_with_backoff(
    url: str,
    payload: dict,
    headers: dict,
    timeout: float,
    retry_delays: tuple = _RETRY_DELAYS,
) -> dict:
    """
    POST JSON with exponential backoff on HTTP 429.

    Raises BackendUnavailableError if all retries are exhausted or if a
    non-429 HTTP error occurs.  Never silently falls back to another backend.
    """
    import requests

    last_exc: Optional[Exception] = None
    attempts = list(retry_delays) + [None]

    for delay in attempts:
        try:
            return _post_json(url, payload, headers, timeout)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            if status == 429:
                last_exc = exc
                if delay is not None:
                    logger.warning("HTTP 429 — retrying in %ds (%s)", delay, exc)
                    time.sleep(delay)
                    continue
            raise BackendUnavailableError(str(exc)) from exc
        except (requests.ConnectionError, requests.Timeout) as exc:
            raise BackendUnavailableError(
                f"Backend unreachable at {url}: {exc}"
            ) from exc

    raise BackendUnavailableError(
        f"All retries exhausted (HTTP 429) for {url}"
    ) from last_exc


# ---------------------------------------------------------------------------
# QwenGateway
# ---------------------------------------------------------------------------


class QwenGateway:
    """
    LLM gateway backed by Qwen3 via Ollama (local) or OpenRouter (cloud).

    Parameters
    ----------
    backend : str | None
        "ollama" or "openrouter".  Defaults to env var QWEN_BACKEND or "ollama".
    num_predict : int
        Max tokens to generate (Ollama) / max_tokens (OpenRouter).
        For OpenRouter, defaults to _OPENROUTER_DEFAULT_MAX_TOKENS (600) to
        reduce the risk of content=null when reasoning tokens are generated.
    timeout : float
        HTTP timeout in seconds. Default 180.
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        num_predict: int = 0,   # 0 = use backend-specific default
        timeout: float = 180.0,
    ) -> None:
        self.backend = (backend or os.getenv("QWEN_BACKEND", "ollama")).lower()
        self.timeout = timeout

        if self.backend == "ollama":
            self._base_url = _OLLAMA_BASE_URL
            # Use explicit value if given, otherwise Ollama default 300
            self.num_predict = num_predict if num_predict > 0 else 300
            logger.info("QwenGateway: backend=ollama model=%s url=%s",
                        _OLLAMA_MODEL, self._base_url)
        elif self.backend == "openrouter":
            self._api_key = os.getenv("OPENROUTER_API_KEY", "")
            if not self._api_key:
                raise BackendUnavailableError(
                    "OPENROUTER_API_KEY not set in environment / .env file."
                )
            # Resolve model at construction time; raises if OPENROUTER_MODEL unset.
            self._openrouter_model = _get_openrouter_model()
            # Use explicit value if given, otherwise raised default 600
            self.num_predict = num_predict if num_predict > 0 else _OPENROUTER_DEFAULT_MAX_TOKENS
            # Key is NEVER printed, logged, or included in any output.
            logger.info("QwenGateway: backend=openrouter model=%s max_tokens=%d",
                        self._openrouter_model, self.num_predict)
        else:
            raise BackendUnavailableError(
                f"Unknown QWEN_BACKEND '{self.backend}'. "
                "Set QWEN_BACKEND to 'ollama' or 'openrouter'."
            )

    # ------------------------------------------------------------------
    # Core generate
    # ------------------------------------------------------------------

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """
        Send a prompt and return the assistant's response text.

        Parameters
        ----------
        prompt : str
            User message content.
        system : str | None
            Optional system message.  If None, no system message is sent.

        Returns
        -------
        str
            The model's response text.

        Raises
        ------
        BackendUnavailableError
            If the backend cannot be reached or returns an unrecoverable error.
        """
        if self.backend == "ollama":
            return self._generate_ollama(prompt, system)
        return self._generate_openrouter(prompt, system)

    # ------------------------------------------------------------------
    # Ollama
    # ------------------------------------------------------------------

    def _generate_ollama(self, prompt: str, system: Optional[str]) -> str:
        """Call Ollama /api/chat endpoint."""
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": _OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": self.num_predict,
            },
            "think": False,   # disable chain-of-thought in qwen3
        }
        url = f"{self._base_url}/api/chat"
        logger.debug("QwenGateway/ollama POST %s think=false num_predict=%d",
                     url, self.num_predict)

        try:
            resp = _call_with_backoff(url, payload, headers={}, timeout=self.timeout)
        except BackendUnavailableError:
            raise
        except Exception as exc:
            raise BackendUnavailableError(f"Ollama request failed: {exc}") from exc

        # Parse response
        try:
            content = resp["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise BackendUnavailableError(
                f"Ollama returned unexpected response shape: {resp!r:.200}"
            ) from exc
        # Guard: content=null or empty/whitespace must never become "None" text
        if content is None or not str(content).strip():
            raise BackendUnavailableError(
                "empty content from backend"
            )
        return str(content)

    # ------------------------------------------------------------------
    # OpenRouter
    # ------------------------------------------------------------------

    def _generate_openrouter(self, prompt: str, system: Optional[str]) -> str:
        """Call OpenRouter /chat/completions endpoint."""
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Cap reasoning budget only when max_tokens > 200;
        # the critic uses max_tokens=40, so omit reasoning for short calls.
        _reasoning_cap = 200
        payload = {
            "model": self._openrouter_model,
            "messages": messages,
            "max_tokens": self.num_predict,
            "temperature": 0.2,
        }
        if self.num_predict > _reasoning_cap:
            # Cap reasoning tokens so content tokens are not starved.
            # reasoning.max_tokens limits thinking; remaining tokens go to content.
            # (https://openrouter.ai/docs/use-cases/reasoning-models)
            payload["reasoning"] = {"max_tokens": _reasoning_cap}
        url = f"{_OPENROUTER_BASE_URL}/chat/completions"
        # API key goes in header — never in payload, logs, or response parsing
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        logger.debug("QwenGateway/openrouter POST %s max_tokens=%d",
                     url, self.num_predict)

        try:
            resp = _call_with_backoff(url, payload, headers, self.timeout)
        except BackendUnavailableError:
            raise
        except Exception as exc:
            raise BackendUnavailableError(
                f"OpenRouter request failed: {exc}"
            ) from exc

        # Parse OpenAI-compatible response
        try:
            content = resp["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendUnavailableError(
                f"OpenRouter returned unexpected response: {str(resp)[:200]}"
            ) from exc
        # Guard: content=null (reasoning consumed all tokens) or empty/whitespace
        # must never become "None" text that reaches layer 1 as if it were a summary.
        if content is None or not str(content).strip():
            raise BackendUnavailableError(
                "empty content from backend"
            )
        return str(content)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_qwen_gateway(**kwargs) -> QwenGateway:
    """Return a configured QwenGateway (backend from QWEN_BACKEND env var)."""
    return QwenGateway(**kwargs)

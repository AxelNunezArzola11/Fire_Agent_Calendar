"""IBM Granite provider via watsonx.ai REST (requests, no extra SDK).

Second in the provider chain after Qwen. Uses the shared WATSONX_*
credentials (own accessors in settings). IAM token + text generation
over HTTPS with timeouts and one bounded retry on retryable statuses.
Secrets never leave this module: availability reasons name missing
fields only, and all errors pass through redact().
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .. import settings
from .providers import MAX_TOKENS, Attempt, Provider, redact

IAM_URL = "https://iam.cloud.ibm.com/identity/token"
API_VERSION = "2024-05-01"
RETRYABLE = (429, 502, 503, 504)
RETRY_WAIT_S = 2.0
MAX_ATTEMPTS_CAP = 3
MAX_INPUT_CHARS = 100000
GRANITE_MAX_TOKENS = MAX_TOKENS  # independent tuning point (see providers)

import time as _time


def _sleep(seconds: float) -> None:
    _time.sleep(seconds)


class _GraniteError(Exception):
    """Provider failure (message must never contain secrets)."""


@dataclass(frozen=True)
class GraniteConfig:
    """Effective Granite configuration (resolved from settings)."""

    api_key: str = ""
    project_id: str = ""
    url: str = ""
    model: str = "ibm/granite-4-h-small"
    timeout_s: float = 30.0
    max_attempts: int = 2
    iam_url: str = IAM_URL

    @classmethod
    def from_settings(cls) -> "GraniteConfig":
        def _str(name: str, default: str = "") -> str:
            value = getattr(settings, name, default)
            return value if isinstance(value, str) else default

        def _num(name: str, default):
            value = getattr(settings, name, default)
            return value if isinstance(value, (int, float)) else default

        return cls(
            api_key=_str("WATSONX_API_KEY"),
            project_id=_str("WATSONX_PROJECT_ID"),
            url=_str("WATSONX_URL").rstrip("/"),
            model=_str("GRANITE_MODEL_ID", "ibm/granite-4-h-small"),
            timeout_s=_num("GRANITE_TIMEOUT_S", 30.0),
            max_attempts=min(int(_num("GRANITE_MAX_ATTEMPTS", 2) or 2),
                             MAX_ATTEMPTS_CAP),
            iam_url=_str("GRANITE_IAM_URL", IAM_URL) or IAM_URL,
        )

    def status(self) -> dict:
        missing = [name for name, value in
                   (("WATSONX_API_KEY", self.api_key),
                    ("WATSONX_PROJECT_ID", self.project_id),
                    ("WATSONX_URL", self.url)) if not value]
        if missing:
            return {"available": False,
                    "reason": f"missing configuration: "
                              f"{', '.join(missing)}"}
        if not self.model:
            return {"available": False,
                    "reason": "missing configuration: WATSONX_MODEL_ID"}
        if not isinstance(self.timeout_s, (int, float)) \
                or self.timeout_s <= 0:
            return {"available": False,
                    "reason": "invalid configuration: "
                              "CAL_GRANITE_TIMEOUT_S must be > 0"}
        if not isinstance(self.max_attempts, int) \
                or not 1 <= self.max_attempts <= MAX_ATTEMPTS_CAP:
            return {"available": False,
                    "reason": "invalid configuration: "
                              "CAL_GRANITE_MAX_ATTEMPTS out of range"}
        return {"available": True, "reason": f"model {self.model}",
                "model": self.model}


def _requests_post(url, *, headers, json_data=None, form_data=None,
                   timeout):
    import requests

    return requests.post(url, headers=headers, json=json_data,
                         data=form_data, timeout=timeout)


class GraniteProvider(Provider):
    """Granite text generation on watsonx.ai (credential-gated)."""

    name = "granite"

    def __init__(self, config=None, post=None) -> None:
        self._config = config or GraniteConfig.from_settings()
        self._post = post or _requests_post

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
        if len(prompt) + len(system) > MAX_INPUT_CHARS:
            return Attempt(self.name, self.model, False, 0.0,
                           error="prompt exceeds size budget "
                                 f"({MAX_INPUT_CHARS} chars)")
        started = time.monotonic()
        try:
            token = self._iam_token()
            text, meta = self._generate(prompt, system, token)
        except _GraniteError as exc:
            latency = (time.monotonic() - started) * 1000.0
            return Attempt(self.name, self.model, False, latency,
                           error=redact(str(exc)))
        except Exception as exc:  # noqa: BLE001 - transport failure
            latency = (time.monotonic() - started) * 1000.0
            return Attempt(self.name, self.model, False, latency,
                           error=redact(f"{type(exc).__name__}: {exc}"))
        latency = (time.monotonic() - started) * 1000.0
        if not text.strip():
            return Attempt(self.name, self.model, False, latency,
                           error="empty response",
                           retries=meta["retries"])
        return Attempt(self.name, self.model, True, latency, text=text,
                       response_chars=len(text), retries=meta["retries"],
                       input_tokens=meta["input_tokens"],
                       generated_tokens=meta["generated_tokens"])

    def _iam_token(self) -> str:
        try:
            response = self._post(
                self._config.iam_url,
                headers={"Content-Type":
                         "application/x-www-form-urlencoded"},
                form_data={"grant_type":
                           "urn:ibm:params:oauth:grant-type:apikey",
                           "apikey": self._config.api_key},
                timeout=self._config.timeout_s)
        except Exception as exc:
            raise _GraniteError(f"iam request failed: {type(exc).__name__}")
        if response.status_code != 200:
            raise _GraniteError(
                f"iam token rejected (HTTP {response.status_code})")
        try:
            body = response.json()
            token = body.get("access_token", "") if isinstance(
                body, dict) else ""
        except (ValueError, AttributeError):
            raise _GraniteError("iam token response is not JSON")
        if not token:
            raise _GraniteError("iam token response lacks access_token")
        return token

    def _generate(self, prompt: str, system: str, token: str) -> str:
        # Chat completions (verified live): the text/generation endpoint
        # returns degenerate output for instruct prompts on this model.
        body = {"model_id": self._config.model,
                "project_id": self._config.project_id,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": prompt}],
                "parameters": {"decoding_method": "greedy",
                               "max_new_tokens": GRANITE_MAX_TOKENS}}
        url = (f"{self._config.url}/ml/v1/text/chat"
               f"?version={API_VERSION}")
        real_headers = {"Authorization": f"Bearer {token}",
                        "Content-Type": "application/json"}
        attempts = max(1, min(self._config.max_attempts,
                              MAX_ATTEMPTS_CAP))
        last_error = "no attempts ran"
        retries = 0
        for attempt in range(attempts):
            try:
                response = self._post(
                    url, headers=real_headers, json_data=body,
                    timeout=self._config.timeout_s)
            except Exception as exc:
                last_error = f"generation request failed: {type(exc).__name__}"
                break  # transport faults are not retried blindly
            if response.status_code in RETRYABLE \
                    and attempt + 1 < attempts:
                last_error = (f"generation retryable "
                              f"(HTTP {response.status_code})")
                retries += 1
                _sleep(RETRY_WAIT_S)
                continue
            if response.status_code != 200:
                raise _GraniteError(
                    f"generation rejected (HTTP {response.status_code})")
            try:
                data = response.json()
                message = data["choices"][0]["message"]
                text = message.get("content", "")
                if not isinstance(text, str):
                    raise TypeError("content is not a string")
                use = data.get("usage") or {}
                usage = {
                    "input_tokens": use.get(
                        "prompt_tokens",
                        use.get("input_token_count")),
                    "generated_tokens": use.get(
                        "completion_tokens",
                        use.get("generated_token_count"))}
            except (ValueError, KeyError, IndexError, AttributeError,
                    TypeError):
                raise _GraniteError("chat response has no choices")
            if not text.strip() and attempt + 1 < attempts:
                # Empty 200s are transient model flakes: one bounded
                # retry inside the same attempt budget, then fail loud.
                last_error = "empty response (retrying once)"
                retries += 1
                _sleep(RETRY_WAIT_S)
                continue
            return text, {"retries": retries,
                          "input_tokens": usage["input_tokens"],
                          "generated_tokens": usage["generated_tokens"]}
        raise _GraniteError(last_error)

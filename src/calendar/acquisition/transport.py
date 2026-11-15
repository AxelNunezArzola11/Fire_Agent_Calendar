"""Bounded, observable HTTP transport. Keys never appear in logs or errors."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class AcquisitionError(RuntimeError):
    """Base class for acquisition failures (safe to log: no secrets)."""


class AuthError(AcquisitionError):
    """Credentials rejected (HTTP 401, or 400 mentioning the map key)."""


class FetchTimeout(AcquisitionError, TimeoutError):
    """Connection or read timeout after bounded retries."""


class HttpError(AcquisitionError):
    """Non-retryable HTTP failure."""

    def __init__(self, status: int, body_head: str = ""):
        self.status = status
        super().__init__(f"HTTP {status}: {body_head[:200]}")


class NetworkError(AcquisitionError):
    """DNS / connection / TLS failure after bounded retries."""


def redact(text: str, secrets: tuple[str, ...]) -> str:
    """Replace every non-empty secret in *text* with '***'."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return text


@dataclass
class FetchResult:
    content: bytes
    status: int
    attempts: int
    elapsed_s: float
    url_redacted: str


def fetch_bytes(
    url: str,
    *,
    timeout_s: float,
    max_retries: int = 3,
    backoff_s: tuple[float, ...] = (2.0, 4.0, 8.0),
    secrets: tuple[str, ...] = (),
    label: str = "fetch",
) -> FetchResult:
    """GET *url* with bounded retries and exponential backoff.

    Retries: timeouts, connection errors, HTTP 429 and 5xx only.
    Raises AuthError (401 / bad-key 400), HttpError (other 4xx),
    FetchTimeout / NetworkError when retries are exhausted.
    """
    import requests

    redacted = redact(url, secrets)
    last_exc: Exception | None = None
    t_start = time.monotonic()
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = requests.get(url, timeout=timeout_s)
        except requests.Timeout as exc:
            last_exc = FetchTimeout(f"{label} timed out (attempt {attempt}): {redacted}")
        except requests.RequestException as exc:
            last_exc = NetworkError(f"{label} network error (attempt {attempt}): {exc}")
        else:
            status = resp.status_code
            if status == 200:
                logger.info("%s OK %s attempts=%d bytes=%d", label, redacted,
                            attempt, len(resp.content))
                return FetchResult(resp.content, status, attempt,
                                   time.monotonic() - t_start, redacted)
            body = redact(resp.text[:200], secrets)
            if status == 401 or (status == 400 and "MAP_KEY" in body.upper()):
                raise AuthError(f"{label} rejected credentials (HTTP {status}): {body}")
            if status == 429 or status >= 500:
                last_exc = HttpError(status, body)
            else:
                raise HttpError(status, body)
        if attempt > max_retries:
            logger.error("%s exhausted %d attempts: %s", label, attempt, last_exc)
            assert last_exc is not None
            raise last_exc
        delay = backoff_s[min(attempt - 1, len(backoff_s) - 1)]
        logger.warning("%s attempt %d failed (%s); retrying in %.0fs",
                       label, attempt, last_exc, delay)
        time.sleep(delay)

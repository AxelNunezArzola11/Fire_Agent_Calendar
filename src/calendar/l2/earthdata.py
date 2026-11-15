"""Earthdata authentication (URS login session, credential-safe).

Presence-gated: missing credentials report unavailable with field
names only. Sessions carry HTTPBasicAuth through the URS redirect
flow; tokens and passwords are never logged or persisted.
"""

from __future__ import annotations

URS_HOST = "urs.earthdata.nasa.gov"
TOKEN_URL = f"https://{URS_HOST}/api/users/token"
TOKENS_URL = f"https://{URS_HOST}/api/users/tokens"
REVOKE_URL = f"https://{URS_HOST}/api/users/revoke_token"


def status() -> dict:
    """Credential presence (names only, values never exposed)."""
    from .. import settings

    user = settings.EARTHDATA_USERNAME
    password = settings.EARTHDATA_PASSWORD
    missing = [name for name, value in
               (("EARTHDATA_USERNAME", user),
                ("EARTHDATA_PASSWORD", password)) if not value]
    if missing:
        return {"available": False,
                "reason": f"missing configuration: "
                          f"{', '.join(missing)}"}
    return {"available": True, "reason": "Earthdata credentials set"}


def _is_quota_error(exc: Exception) -> bool:
    """True when a token-exchange failure means quota exhaustion."""
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            body = response.json()
        except Exception:  # noqa: BLE001 - malformed body
            body = {}
        if isinstance(body, dict) and body.get("error") == \
                "max_token_limit":
            return True
        if getattr(response, "status_code", None) == 403:
            # URS uses 401 for bad credentials; 403 here is quota.
            return True
    return "max_token" in str(exc)


def bearer_token(*, timeout_s: float = 60.0, post=None,
                 get=None) -> str:
    """URS login token via basic-auth exchange (raises _AuthError).

    `post`/`get` inject requests callables for tests. On quota
    exhaustion, reuses an existing account token instead of failing.
    The token is returned to the caller only; this module never logs
    or persists it.
    """
    import logging as _logging
    import requests

    from .. import settings

    usable = status()
    if not usable["available"]:
        raise _AuthError(usable["reason"])
    sender = post or requests.post
    last_error: Exception | None = None
    quota_hit = False
    for _ in range(3):
        try:
            response = sender(
                TOKEN_URL,
                auth=(settings.EARTHDATA_USERNAME,
                      settings.EARTHDATA_PASSWORD),
                timeout=timeout_s)
            response.raise_for_status()
            token = response.json().get("access_token", "")
        except Exception as exc:  # noqa: BLE001 - classified below
            last_error = exc
            if "401" in str(getattr(exc, "response", "") or "") or \
                    "401" in str(exc):
                break  # bad credentials: retrying is pointless
            if _is_quota_error(exc):
                quota_hit = True
                break  # retrying cannot help: reuse instead
            continue  # transient transport fault: retry
        if not token:
            raise _AuthError("token exchange returned no token")
        return token
    if quota_hit:
        for candidate in list_tokens(timeout_s=timeout_s, get=get):
            if isinstance(candidate, dict) and \
                    candidate.get("access_token"):
                _logging.getLogger(__name__).info(
                    "earthdata: reusing existing account token "
                    "(quota full)")
                return candidate["access_token"]
    raise _AuthError(f"token exchange failed: "
                     f"{type(last_error).__name__}")


def session():
    """Authenticated requests session (raises _AuthError if unusable)."""
    import requests

    token = bearer_token()
    handle = requests.Session()
    handle.headers["Authorization"] = f"Bearer {token}"
    return handle


def list_tokens(*, timeout_s: float = 60.0, get=None) -> list:
    """Token metadata owned by the user (no secret values returned)."""
    import requests

    from .. import settings

    usable = status()
    if not usable["available"]:
        raise _AuthError(usable["reason"])
    getter = get or requests.get
    try:
        response = getter(
            TOKENS_URL,
            auth=(settings.EARTHDATA_USERNAME,
                  settings.EARTHDATA_PASSWORD),
            timeout=timeout_s)
        response.raise_for_status()
        tokens = response.json()
    except Exception as exc:  # noqa: BLE001 - converted to reason
        raise _AuthError(f"token listing failed: {type(exc).__name__}")
    return tokens if isinstance(tokens, list) else []


def revoke_token(token: str, *, timeout_s: float = 60.0,
                 delete=None) -> bool:
    """Revoke exactly the token held (zero blast radius).

    Returns True when the API confirms revocation. Callers revoke
    after use so the account token quota never fills.
    """
    import requests

    sender = delete or requests.delete
    try:
        response = sender(REVOKE_URL, params={"token": token},
                          timeout=timeout_s)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - converted to reason
        raise _AuthError(f"token revocation failed: {type(exc).__name__}")
    return True


class _AuthError(Exception):
    """Authentication unavailable (message carries no secrets)."""

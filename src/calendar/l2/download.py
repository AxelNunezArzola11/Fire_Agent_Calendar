"""Download manager: streaming, resumable, checksummed, budgeted.

One granule per call; callers loop with budget.decide() between
granules so the stop condition is re-checked every step. Partial
files resume via HTTP Range; completed files are never re-fetched.
"""

from __future__ import annotations

from pathlib import Path

CHUNK = 1 << 20


def local_path(root: str | Path, product: str, granule_ur: str,
                 suffix: str = "") -> Path:
    """Deterministic on-disk path (no traversal: basename only).

    Appends `suffix` only when given and missing (producer granule
    ids often already include the extension; callers that build
    names themselves pass suffix="" behaviour by default).
    """
    safe = Path(str(granule_ur)).name
    if suffix and not safe.lower().endswith(suffix.lower()):
        safe += suffix
    return Path(root) / product / safe


def download_granule(session, url: str, dest: Path, *,
                     expected_digest: str | None = None,
                     digest_algorithm: str = "sha256",
                     timeout_s: float = 120.0,
                     progress=None) -> dict:
    """Stream one URL to dest (resume partial; verify; report).

    Returns {path, bytes, resumed, checksum, status}. Raises
    _DownloadError on HTTP/network failure or digest mismatch.
    """
    from . import integrity as integrity_mod

    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.is_file() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    try:
        response = session.get(url, headers=headers, stream=True,
                               timeout=timeout_s)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - converted to reason
        raise _DownloadError(f"fetch failed: {type(exc).__name__}: "
                             f"{exc}")
    resumed = have > 0 and response.status_code == 206
    mode = "ab" if resumed else "wb"
    if have and not resumed:
        have = 0  # server ignored Range: restart cleanly
    total = have
    with open(dest, mode) as handle:
        for chunk in response.iter_content(chunk_size=CHUNK):
            if not chunk:
                continue
            handle.write(chunk)
            total += len(chunk)
            if progress:
                progress(total)
    check = integrity_mod.verify(str(dest), expected_digest,
                                 digest_algorithm)
    if check["checked"] and not check["match"]:
        raise _DownloadError(check["reason"])
    return {"path": str(dest), "bytes": total, "resumed": resumed,
            "checksum": check, "status": "complete"}


class _DownloadError(Exception):
    """Download failure (message carries URLs, never credentials)."""

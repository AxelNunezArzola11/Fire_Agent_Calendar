"""Checksum helpers (sha256/md5 file verification)."""

from __future__ import annotations

import hashlib


def file_digest(path: str, algorithm: str = "sha256") -> str:
    """Hex digest of a file (streamed, 1 MiB chunks)."""
    digest = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: str, expected: str | None,
           algorithm: str = "sha256") -> dict:
    """Compare a file against an expected digest (or report none)."""
    if not expected:
        return {"checked": False, "match": None,
                "reason": "no expected digest published"}
    actual = file_digest(path, algorithm)
    match = actual.lower() == str(expected).lower()
    return {"checked": True, "match": match,
            "reason": "match" if match else
            f"digest mismatch (got {actual[:16]}…)"}

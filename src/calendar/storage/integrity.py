"""SHA-256 integrity primitives. Read-only w.r.t. the hashed files."""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK = 65536


def sha256_file(path: str | Path) -> str:
    """Stream-hash a file (constant memory)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_file(path: str | Path, expected_sha256: str) -> bool:
    """True iff the file's current hash matches. Never modifies the file."""
    return sha256_file(path) == expected_sha256


def write_sidecar(path: str | Path) -> Path:
    """Write ``<name>.sha256`` containing ``<hash>  <name>``. Returns its path."""
    from .layout import sha_sidecar

    target = Path(path)
    digest = sha256_file(target)
    sidecar = sha_sidecar(target)
    sidecar.write_text(f"{digest}  {target.name}\n", encoding="utf-8")
    return sidecar


def read_sidecar(path: str | Path) -> str | None:
    """Read the hash from a ``.sha256`` sidecar, or None if absent/unparseable."""
    from .layout import sha_sidecar

    sidecar = sha_sidecar(Path(path))
    if not sidecar.is_file():
        return None
    try:
        head = sidecar.read_text(encoding="utf-8").strip().split()[0]
    except (IndexError, OSError):
        return None
    return head if len(head) == 64 else None

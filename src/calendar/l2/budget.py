"""Storage budget: estimates vs free space with stop conditions.

Pure decision logic (unit-tested); the only I/O is a free-space
probe. Downloads must consult decide() before every granule.
"""

from __future__ import annotations

import shutil

DEFAULT_BUDGET_MB = 2048.0


def free_mb(path: str = ".") -> float:
    """Free bytes at path, in MiB."""
    return shutil.disk_usage(path).free / (1024.0 * 1024.0)


def estimate_mb(granules: list[dict],
                default_mb: float = 50.0) -> dict:
    """Total estimate; unknown sizes use default_mb (flagged)."""
    known = sum(g.get("size_mb") or 0.0 for g in granules
                if g.get("size_mb") is not None)
    unknown = sum(1 for g in granules if g.get("size_mb") is None)
    return {"total_mb": known + unknown * default_mb,
            "known_mb": known, "unknown_count": unknown,
            "default_mb": default_mb, "n": len(granules)}


def decide(required_mb: float, *, free_mb: float,
           budget_mb: float = DEFAULT_BUDGET_MB,
           keep_free_mb: float = 1024.0) -> dict:
    """Go/no-go for a download batch (explicit stop conditions).

    Proceeds only when the batch fits BOTH the configured budget and
    free space minus a safety reserve. Never authorizes deletions.
    """
    reasons = []
    if required_mb > budget_mb:
        reasons.append(f"required {required_mb:.1f} MB exceeds budget "
                       f"{budget_mb:.1f} MB")
    if required_mb > free_mb - keep_free_mb:
        reasons.append(f"required {required_mb:.1f} MB exceeds free "
                       f"space {free_mb:.1f} MB minus "
                       f"{keep_free_mb:.1f} MB reserve")
    return {"proceed": not reasons, "reasons": reasons,
            "required_mb": required_mb, "free_mb": free_mb,
            "budget_mb": budget_mb}

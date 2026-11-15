"""Sequential L2 batch processing for ranges beyond one inventory.

Splits [start, end] into consecutive blocks (default 7 days), then
runs inventory -> budget -> download -> verify -> manifest per block,
strictly sequentially. Resume-safe (completed blocks/granules are
skipped), failure-isolated (failed blocks recorded explicitly, never
claimed as processed), and storage-budgeted (stop before overrun).
"""

from __future__ import annotations

from datetime import date, timedelta

DEFAULT_BLOCK_DAYS = 7


def split_range(date_start: str, date_end: str,
                block_days: int = DEFAULT_BLOCK_DAYS):
    """Consecutive (start, end) ISO blocks covering the range."""
    if block_days < 1:
        raise ValueError("block_days must be >= 1")
    start = date.fromisoformat(date_start)
    end = date.fromisoformat(date_end)
    if start > end:
        raise ValueError("start must not be after end")
    blocks = []
    cursor = start
    while cursor <= end:
        stop = min(cursor + timedelta(days=block_days - 1), end)
        blocks.append((cursor.isoformat(), stop.isoformat()))
        cursor = stop + timedelta(days=1)
    return blocks


def run_sequential(root, product: str, blocks, *,
                   budget_mb: float, fetch_inventory,
                   download_one, progress=None) -> dict:
    """Run blocks in order; returns a completion ledger.

    fetch_inventory(block) -> {granules: [...]} (may raise).
    download_one(granule_ur, granule) -> {status, ...} with status
    complete|failed|skipped (may raise -> recorded failed).
    Ledger: {blocks: [{block, status, detail}], complete, failed,
    skipped_budget, granules_complete, granules_failed}.
    """
    from . import budget as budget_mod
    from . import manifest as manifest_mod

    ledger = {"blocks": [], "complete": 0, "failed": 0,
              "skipped_budget": 0, "granules_complete": 0,
              "granules_failed": 0}
    known = manifest_mod.latest_by_ur(root)
    for start, end in blocks:
        label = f"{start}..{end}"
        try:
            inventory = fetch_inventory((start, end))
            granules = list(inventory.get("granules") or [])
        except Exception as exc:  # noqa: BLE001 - block isolated
            ledger["blocks"].append(
                {"block": label, "status": "failed",
                 "detail": f"inventory failed: {exc}"})
            ledger["failed"] += 1
            if progress:
                progress(label, "failed", "inventory failed")
            continue
        todo = [g for g in granules
                if (known.get(g.get("granule_ur") or "") or {}).get(
                    "status") != "complete"
                and g.get("granule_ur")]
        if not todo:
            ledger["blocks"].append(
                {"block": label, "status": "complete",
                 "detail": f"{len(granules)} granule(s) already complete"})
            ledger["complete"] += 1
            if progress:
                progress(label, "complete", "resumed")
            continue
        est = budget_mod.estimate_mb(todo)
        decision = budget_mod.decide(
            est["total_mb"], free_mb=budget_mod.free_mb("."),
            budget_mb=budget_mb)
        if not decision["proceed"]:
            ledger["blocks"].append(
                {"block": label, "status": "skipped_budget",
                 "detail": "; ".join(decision["reasons"])})
            ledger["skipped_budget"] += 1
            if progress:
                progress(label, "skipped_budget", "over budget")
            continue
        block_failed = []
        for granule in todo:
            ur = granule["granule_ur"]
            try:
                outcome = download_one(ur, granule)
                status = outcome.get("status", "failed")
            except Exception as exc:  # noqa: BLE001 - granule isolated
                status = "failed"
                manifest_mod.append(root, {
                    "granule_ur": ur, "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}"[:200]})
            if status == "complete":
                ledger["granules_complete"] += 1
            else:
                ledger["granules_failed"] += 1
                block_failed.append(ur)
        if block_failed:
            ledger["blocks"].append(
                {"block": label, "status": "failed",
                 "detail": f"{len(block_failed)} granule(s) failed: "
                           f"{', '.join(block_failed[:5])}"})
            ledger["failed"] += 1
        else:
            ledger["blocks"].append(
                {"block": label, "status": "complete",
                 "detail": f"{len(todo)} granule(s) complete"})
            ledger["complete"] += 1
        if progress:
            progress(label, ledger["blocks"][-1]["status"],
                     ledger["blocks"][-1]["detail"])
    return ledger

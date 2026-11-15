"""Historical archive acquisition.

Two paths (both real, neither invented):

1. **Open per-country yearly CSVs** (verified 2026-11-11: USA MODIS 2023 =
   7,572,283 bytes HTTP 200; VIIRS S-NPP 2023 = 30,962,191 bytes via HEAD;
   slug ``United_States``; 404 otherwise). Products: ``modis`` (2000+),
   ``viirs-snpp`` (2012+). Standard processing, no key required.
2. **Async email-link flow** (FIRMS download portal): cannot be automated
   blindly. :class:`ArchiveRequestSpec` captures the request parameters for a
   manual portal submission, and :func:`stage_file` registers a
   human-downloaded file into the pipeline with full provenance.

Registration into the raw archive + manifest is Phase 1B's job
(``storage.register``); this module stages bytes and computes identity.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import sensors, transport
from .. import settings

logger = logging.getLogger(__name__)


class ArchiveNotPublished(FileNotFoundError):
    """The yearly archive does not (yet) exist (HTTP 404). Not a failure."""


@dataclass
class StagedFile:
    """A downloaded file awaiting registration (Phase 1B)."""

    path: str  # local staged path (temp or inbox)
    filename: str  # original source filename, preserved
    sha256: str
    size_bytes: int
    product: str  # 'modis' | 'viirs-snpp' | NRT/SP source | 'manual'
    sensor_hint: str  # e.g. 'modis-terra+aqua'; refined at normalization
    provenance: dict = field(default_factory=dict)


@dataclass
class ArchiveProbe:
    product_slug: str
    year: int
    country: str
    url: str
    published: bool
    size_bytes: int | None = None
    last_modified: str | None = None
    etag: str | None = None


def probe_open_archive(product_slug: str, year: int, country: str) -> ArchiveProbe:
    """HEAD-probe a yearly archive (no download). Raises on transport errors."""
    import requests

    url = sensors.archive_url(product_slug, year, country)
    try:
        resp = requests.head(url, timeout=settings.ACQUIRE_TIMEOUT_S)
    except requests.Timeout as exc:
        raise transport.FetchTimeout(f"archive probe timed out: {url}") from exc
    except requests.RequestException as exc:
        raise transport.NetworkError(f"archive probe failed: {exc}") from exc
    if resp.status_code == 404:
        return ArchiveProbe(product_slug, year, country, url, False)
    if resp.status_code != 200:
        raise transport.HttpError(resp.status_code, resp.text[:200])
    size = resp.headers.get("Content-Length")
    return ArchiveProbe(
        product_slug, year, country, url, True,
        size_bytes=int(size) if size and size.isdigit() else None,
        last_modified=resp.headers.get("Last-Modified"),
        etag=resp.headers.get("ETag"),
    )


def fetch_open_archive(product_slug: str, year: int, country: str, dest_dir: str,
                       *, max_bytes: int | None = None) -> StagedFile:
    """Download a yearly archive CSV to *dest_dir* (original filename kept).

    Raises ArchiveNotPublished on 404. Refuses to write when the response
    exceeds *max_bytes* (capacity safeguard hook for Phase 1B/1C).
    """
    url = sensors.archive_url(product_slug, year, country)
    try:
        res = transport.fetch_bytes(
            url, timeout_s=settings.ACQUIRE_TIMEOUT_S,
            max_retries=settings.ACQUIRE_MAX_RETRIES,
            backoff_s=settings.ACQUIRE_BACKOFF_S,
            label=f"archive:{product_slug}/{year}/{country}",
        )
    except transport.HttpError as exc:
        if exc.status == 404:
            raise ArchiveNotPublished(f"not published (yet): {url}") from None
        raise
    if max_bytes is not None and len(res.content) > max_bytes:
        raise transport.AcquisitionError(
            f"refusing {len(res.content)}-byte download above {max_bytes}-byte cap: {url}")
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    filename = sensors.archive_filename(product_slug, year, country)
    out = dest / filename
    if out.exists():
        raise transport.AcquisitionError(f"refusing to overwrite staged file: {out}")
    out.write_bytes(res.content)
    digest = hashlib.sha256(res.content).hexdigest()
    logger.info("staged %s (%d bytes, sha256=%s…)", filename, len(res.content), digest[:16])
    return StagedFile(
        path=str(out), filename=filename, sha256=digest, size_bytes=len(res.content),
        product=product_slug, sensor_hint=("modis-terra+aqua" if product_slug == "modis"
                                           else "viirs-snpp"),
        provenance={"url": url, "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "attempts": res.attempts, "processing": "SP-standard"},
    )


@dataclass
class ArchiveRequestSpec:
    """Parameters for a manual FIRMS portal (email-link) request.

    The portal flow is interactive; this spec makes the request reproducible
    and reviewable instead of ad-hoc.
    """

    products: tuple[str, ...]  # e.g. ('MCD14ML', 'VNP14IMGTML')
    sensors: tuple[str, ...]  # sensor ids from sensors.SENSORS
    bbox: str
    start_date: str  # YYYY-MM-DD
    end_date: str
    format: str = "csv"
    spec_id: str = ""

    def __post_init__(self):
        if not self.spec_id:
            raw = f"{self.products}|{self.sensors}|{self.bbox}|{self.start_date}|{self.end_date}"
            self.spec_id = hashlib.sha256(raw.encode()).hexdigest()[:16]
        for sensor_id in self.sensors:
            if sensor_id not in sensors.SENSORS:
                raise ValueError(f"unknown sensor {sensor_id!r}")

    def render_instructions(self) -> str:
        return (
            f"Archive request {self.spec_id} (MANUAL portal flow):\n"
            f"  1. Log in at https://firms.modaps.eosdis.nasa.gov/download/ "
            f"(Earthdata or email code).\n"
            f"  2. Request products {', '.join(self.products)} for "
            f"{self.start_date}..{self.end_date}, bbox {self.bbox}, format {self.format}.\n"
            f"  3. Save every delivered zip/CSV unmodified, then register with:\n"
            f"     stage_file(<path>, spec_id='{self.spec_id}')\n"
            f"  4. Register into the raw archive via storage.register (Phase 1B).\n"
        )


def stage_file(path: str, *, product: str = "manual", sensor_hint: str = "unknown",
               provenance: dict | None = None, dest_dir: str | None = None) -> StagedFile:
    """Stage an already-downloaded file (e.g. from the email-link flow).

    Copies byte-identically to *dest_dir* (default: alongside, ``.staged/`` is
    NOT created — caller passes the inbox dir). Never modifies content.
    """
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"nothing to stage: {path}")
    data = src.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    target = Path(dest_dir) / src.name if dest_dir else src
    if dest_dir:
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise transport.AcquisitionError(f"refusing to overwrite staged file: {target}")
        shutil.copy2(src, target)
    logger.info("staged %s (%d bytes, sha256=%s…)", src.name, len(data), digest[:16])
    return StagedFile(path=str(target), filename=src.name, sha256=digest,
                      size_bytes=len(data), product=product, sensor_hint=sensor_hint,
                      provenance=provenance or {})

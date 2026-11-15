"""Level-2 acquisition and processing subsystem (modular, bounded).

CMR discovery (no auth) -> Earthdata-authenticated download -> checksum
-> format-aware readers -> FIRMS colocation -> OOp input assembly.
Storage-budget gated; manifest-resumable; secrets never logged.
"""

from . import batches, budget, cmr, download, earthdata, inputs
from . import integrity, manifest, match, products, readers

__all__ = ["batches", "budget", "cmr", "download", "earthdata",
           "inputs", "integrity", "manifest", "match", "products",
           "readers"]

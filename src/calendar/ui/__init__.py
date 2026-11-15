"""FireAgent Calendar UI helpers (Phase 3).

Framework-free data access, filter state, and HTML builders live here so
they are unit-testable without a browser. Streamlit `render_*` functions are
thin wrappers over these pure helpers.
"""

from . import (analytics_view, aoi, calendar_view, events_view,  # noqa: F401
               export_view, filters, map_view, query)

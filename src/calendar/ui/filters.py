"""Shared filter state + period math (Phase 3B; extended in 3D).

Framework-free: plain dataclasses and pure functions over date labels, so
the calendar, the map, and the tests share exactly one implementation of
period boundaries, navigation, and validation.
"""

from __future__ import annotations

import calendar as _calendar
from dataclasses import dataclass
from datetime import date, datetime

GRAINS = ("daily", "weekly", "monthly")
PRODUCTS = ("modis", "viirs-snpp")
SATELLITES = ("Terra", "Aqua", "N")
MODES = ("detections", "clusters")

# Session-state keys shared by the app and the views (single source).
KEY_GRAIN = "cal_grain"
KEY_PERIOD = "cal_period"
KEY_RANGE_START = "cal_range_start"
KEY_RANGE_END = "cal_range_end"
KEY_PRODUCTS = "cal_products"
KEY_SATELLITES = "cal_satellites"
KEY_MODE = "cal_mode"
KEY_MAP_TAB_HINT = "cal_open_map_for_period"
KEY_MODIS_CONF = "cal_modis_conf_min"
KEY_VIIRS_CONF = "cal_viirs_conf"
KEY_AOI_MODE = "cal_aoi_mode"
KEY_AOI_REGION = "cal_aoi_region"
KEY_AOI_W = "cal_aoi_w"
KEY_AOI_S = "cal_aoi_s"
KEY_AOI_E = "cal_aoi_e"
KEY_AOI_N = "cal_aoi_n"
KEY_AOI_SAVED = "cal_aoi_saved"

# App-open initial selected date (single source of truth). The calendar
# period, the shared date range, and therefore the default map day all
# derive from this: the map is a daily visualization and must not open
# on the full historical archive.
INITIAL_DAY = "2023-08-01"

AOI_MODES = ("full", "region", "custom", "viewport", "saved")


@dataclass(frozen=True)
class FilterState:
    """Shared calendar/map filter context (temporal basis: acquisition UTC)."""
    grain: str = "daily"
    period: str = "2023-08-01"
    date_start: str = "2023-08-01"
    date_end: str = "2023-08-01"
    products: tuple[str, ...] = PRODUCTS
    satellites: tuple[str, ...] = SATELLITES
    mode: str = "detections"
    region_id: str | None = None


def default_filters() -> FilterState:
    """Default context: full pilot year, both sensors, detection mode."""
    return FilterState()


def coerce_day(value) -> str:
    """Accept a date, datetime, or YYYY-MM-DD string -> ISO date string."""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        raise ValueError(f"bad date {value!r} (want YYYY-MM-DD)")


def validate(state: FilterState) -> list[str]:
    """Human-readable problems ([] when valid). Never raises."""
    errors = []
    if state.grain not in GRAINS:
        errors.append(f"unknown grain {state.grain!r}")
    if state.mode not in MODES:
        errors.append(f"unknown mode {state.mode!r}")
    for product in state.products:
        if product not in PRODUCTS:
            errors.append(f"unknown product {product!r}")
    for satellite in state.satellites:
        if satellite not in SATELLITES:
            errors.append(f"unknown satellite {satellite!r}")
    dates_ok = True
    for label in (state.date_start, state.date_end):
        try:
            coerce_day(label)
        except ValueError as exc:
            errors.append(str(exc))
            dates_ok = False
    if dates_ok and coerce_day(state.date_start) > coerce_day(state.date_end):
        errors.append("date_start is after date_end")
    try:
        period_to_range(state.grain, state.period)
    except ValueError as exc:
        errors.append(str(exc))
    return errors


def _parse_day(label: str) -> date:
    try:
        return date.fromisoformat(label)
    except ValueError:
        raise ValueError(f"bad daily period {label!r} (want YYYY-MM-DD)")


def _parse_week(label: str) -> tuple[int, int]:
    try:
        year_s, week_s = label.split("-W")
        year, week = int(year_s), int(week_s)
        start = date.fromisocalendar(year, week, 1)
    except (ValueError, TypeError):
        raise ValueError(f"bad weekly period {label!r} (want YYYY-Www)")
    if start.isocalendar()[0] != year or not 1 <= week <= 53:
        raise ValueError(f"bad weekly period {label!r} (want YYYY-Www)")
    return year, week


def _parse_month(label: str) -> tuple[int, int]:
    try:
        year_s, month_s = label.split("-")
        year, month = int(year_s), int(month_s)
        date(year, month, 1)
    except (ValueError, TypeError):
        raise ValueError(f"bad monthly period {label!r} (want YYYY-MM)")
    if not 1 <= month <= 12:
        raise ValueError(f"bad monthly period {label!r} (want YYYY-MM)")
    return year, month


def period_to_range(grain: str, period: str) -> tuple[str, str]:
    """Inclusive (start, end) UTC dates for a period label (ISO weeks)."""
    if grain == "daily":
        day = _parse_day(period).isoformat()
        return day, day
    if grain == "weekly":
        year, week = _parse_week(period)
        start = date.fromisocalendar(year, week, 1)
        end = date.fromisocalendar(year, week, 7)
        return start.isoformat(), end.isoformat()
    if grain == "monthly":
        year, month = _parse_month(period)
        last = _calendar.monthrange(year, month)[1]
        return f"{year:04d}-{month:02d}-01", f"{year:04d}-{month:02d}-{last:02d}"
    raise ValueError(f"unknown grain {grain!r}; want one of {GRAINS}")


def period_overlaps_range(grain: str, period: str, start: str,
                          end: str) -> bool:
    """True when a calendar period overlaps [start, end] (inclusive).

    Malformed inputs return False (never raise): callers treat that
    as out-of-sync and display the shared range instead.
    """
    try:
        lo, hi = period_to_range(grain, period)
        start_day, end_day = _parse_day(start), _parse_day(end)
    except ValueError:
        return False
    if start_day > end_day:
        start_day, end_day = end_day, start_day
    return lo <= end_day.isoformat() and hi >= start_day.isoformat()


def sync_inputs_from_range(state: dict, *, start_key: str,
                           end_key: str, sentinel_key: str) -> bool:
    """Copy the shared sidebar range into target inputs on change.

    state is a plain dict (st.session_state in production). Writes
    exactly when the shared range differs from the last synced
    value; ignores unparseable ranges without touching inputs.
    Callers must invoke this BEFORE instantiating the widgets.
    """
    start = state.get(KEY_RANGE_START)
    end = state.get(KEY_RANGE_END)
    if state.get(sentinel_key) == (start, end):
        return False
    try:
        _parse_day(start)
        _parse_day(end)
    except (ValueError, TypeError):
        return False
    state[start_key] = start
    state[end_key] = end
    state[sentinel_key] = (start, end)
    return True


def iso_week_label(day: date) -> str:
    """Metrics-compatible weekly label (e.g. 2023-01-01 -> '2022-W52')."""
    year, week, _ = day.isocalendar()
    return f"{year:04d}-W{week:02d}"


def shift_period(grain: str, period: str, delta: int) -> str:
    """Previous/next period label (delta = ±1; larger steps allowed)."""
    if grain == "daily":
        from datetime import timedelta

        return (_parse_day(period) + timedelta(days=delta)).isoformat()
    if grain == "weekly":
        from datetime import timedelta

        year, week = _parse_week(period)
        monday = date.fromisocalendar(year, week, 1) + timedelta(weeks=delta)
        return iso_week_label(monday)
    if grain == "monthly":
        year, month = _parse_month(period)
        total = (year * 12 + (month - 1)) + delta
        return f"{total // 12:04d}-{total % 12 + 1:02d}"
    raise ValueError(f"unknown grain {grain!r}; want one of {GRAINS}")


def month_weeks(year: int, month: int) -> list[list[date | None]]:
    """Monday-first grid weeks (None pads days outside the month)."""
    weeks: list[list[date | None]] = []
    week: list[date | None] = [None] * date(year, month, 1).weekday()
    for day in range(1, _calendar.monthrange(year, month)[1] + 1):
        week.append(date(year, month, day))
        if len(week) == 7:
            weeks.append(week)
            week = []
    if week:
        week.extend([None] * (7 - len(week)))
        weeks.append(week)
    return weeks


WEEKDAY_HEADERS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

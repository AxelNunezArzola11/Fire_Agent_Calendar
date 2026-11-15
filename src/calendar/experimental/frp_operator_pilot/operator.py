"""Deterministic expected-observation FRP mapping (Alternative A).

E_ret(P) = p(P) * P with p(P) the published sigmoid detection
function. Pure Python + math only; no randomness, no I/O.

Input interpretation (exploratory, see pilot report §5): FIRMS FRP
is an already-retrieved observation, while the paper's P is the
underlying FRP supplied to the operator. Feeding retrieved FRP in
computes a detection-discounted diagnostic (how much observed FRP
sits in low-detectability regimes), NOT a correction and NOT
recovered true FRP. Undetected fires are absent from the input, so
mapped sums are conditional on detection, never totals.
"""

from __future__ import annotations

import math

from . import coefficients

def aod_attenuation(frp: float, tau_4um: float) -> dict:
    """Thin-cloud/smoke attenuation (paper Eq. 12, INDICATIVE ONLY).

    Returns P·exp(−τ₄) with explicit labeling. The paper marks this
    term indicative, not validated; no AOD product is wired into the
    pipeline, so this function has no production callers. Provided
    for documented experimentation with verified AOD inputs only.
    """
    value = _require_frp_allowing_zero(frp)
    if isinstance(tau_4um, bool) or not isinstance(tau_4um,
                                                  (int, float)):
        raise ValueError(f"non-numeric AOD: {tau_4um!r}")
    tau = float(tau_4um)
    if not math.isfinite(tau) or tau < 0:
        raise ValueError(f"invalid AOD {tau_4um!r} (want finite ≥ 0)")
    return {"P_attenuated": value * math.exp(-tau),
            "attenuation": math.exp(-tau), "tau_4um": tau,
            "method": "det-frp-oop-v1-aod",
            "validation_status": "indicative-unvalidated"}


CROSS_MAP_METHOD = "det-frp-oop-v1-xmap"
CROSS_MAP_ASSUMPTIONS = (
    "clear-sky assumed (no QA/cloud terms)",
    "retrieved FRP fed as model P (exploratory)",
    "source-indicated area evaluated in target operator "
    "(area transfer unvalidated; sensors have disjoint native regimes)",
    "no overpass coincidence handling",
    "no diurnal correction (no universal multiplier applied)",
)
REGIME_ASSUMPTIONS = (
    "clear-sky assumed (no QA/cloud terms)",
    "transition band ±2/r is a labeling convention, not paper physics",
)


def _require_area(area_km2) -> float:
    """Shared area guard: finite int/float > 0, else ValueError."""
    if isinstance(area_km2, bool) or not isinstance(area_km2,
                                                   (int, float)):
        raise ValueError(f"non-numeric pixel area: {area_km2!r}")
    area = float(area_km2)
    if not math.isfinite(area) or area <= 0:
        raise ValueError(f"invalid pixel area: {area_km2!r}")
    return area


def _require_frp_allowing_zero(frp) -> float:
    """FRP guard: finite int/float ≥ 0, else ValueError (never coerced)."""
    if frp is None:
        raise ValueError("missing FRP (never zero-filled)")
    if isinstance(frp, bool) or not isinstance(frp, (int, float)):
        raise ValueError(f"non-numeric FRP: {frp!r}")
    try:
        missing = math.isnan(frp)
    except TypeError:
        raise ValueError(f"non-numeric FRP: {frp!r}")
    if missing:
        raise ValueError("missing FRP (never zero-filled)")
    value = float(frp)
    if not math.isfinite(value):
        raise ValueError(f"non-finite FRP: {frp!r}")
    if value < 0:
        raise ValueError(f"negative FRP {frp!r} (unphysical)")
    return value


def pixel_area_km2(scan, track) -> float:
    """Pixel area proxy from FIRMS scan×track dimensions (km²).

    Raises ValueError on missing/non-positive/non-finite inputs.
    Validated distributionally against published anchors (see pilot
    report §7); per-pixel equivalence to Eq. 1-2 geometry is an
    assumption, not a proof (no scan angle in FIRMS rows).
    """
    try:
        area = float(scan) * float(track)
    except (TypeError, ValueError):
        raise ValueError(f"non-numeric scan/track: {scan!r}/{track!r}")
    if not math.isfinite(area) or area <= 0:
        raise ValueError(f"invalid pixel area from scan/track: {scan!r}/{track!r}")
    return area


def detection_limit(area_km2: float, product: str, daynight: str) -> float:
    """Dp = λ·A + μ (paper Eq. 5; clear-sky assumed, no QA terms)."""
    area = _require_area(area_km2)
    params = coefficients.get_params(product, daynight)
    return params["lambda"] * area + params["mu"]


def transition_rate(area_km2: float, product: str, daynight: str) -> float:
    """r = r0 + r1/(A − A0) (paper Eq. 6; A0 = 0)."""
    area = _require_area(area_km2)
    params = coefficients.get_params(product, daynight)
    return params["r0"] + params["r1"] / area


def detection_probability(frp: float, area_km2: float, product: str,
                          daynight: str) -> float:
    """p(P) = [sigmoid((P−Dp)·r) − c_o] / (1 − c_o), clamped to [0,1].

    Exact 0.0 yields 0.0 (no fire → no retrieval) without consulting
    the sigmoid. Missing/non-finite/negative FRP and invalid areas
    raise ValueError (aligned with validation.validate_row).
    Exponent clamped to ±60 for numerical safety.
    """
    value = _require_frp_allowing_zero(frp)
    area = _require_area(area_km2)
    if value == 0.0:
        return 0.0
    params = coefficients.get_params(product, daynight)
    dp = params["lambda"] * area + params["mu"]
    rate = params["r0"] + params["r1"] / area
    z = max(-60.0, min(60.0, (value - dp) * rate))
    sigmoid = 1.0 / (1.0 + math.exp(-z))
    prob = (sigmoid - params["c_o"]) / (1.0 - params["c_o"])
    return max(0.0, min(1.0, prob))


def expected_frp(frp: float, area_km2: float, product: str,
                 daynight: str) -> dict:
    """E_ret(P) = p(P)·P plus diagnostic components (all deterministic)."""
    value = _require_frp_allowing_zero(frp)
    area = _require_area(area_km2)
    params = coefficients.get_params(product, daynight)
    dp = params["lambda"] * area + params["mu"]
    rate = params["r0"] + params["r1"] / area
    prob = detection_probability(value, area, product, daynight)
    return {"E_ret": prob * value if value > 0 else 0.0,
            "p": prob, "Dp": dp, "r": rate, "c_o": params["c_o"]}


def cross_map_expected(frp: float, area_km2: float, from_product: str,
                       to_product: str, daynight: str) -> dict:
    """Per-detection cross-sensor mapping (paper Eq. 13, exploratory form).

    Evaluates the TARGET sensor's pixel operator at the SOURCE
    detection's FRP and pixel area. The area transfer is UNVALIDATED:
    sensors have disjoint native area regimes, and the paper's Eq. 13
    consistency used overlapping granules with each sensor's own
    geometry plus a refit diurnal correction -- none of which exist
    here. from_product is direction-label only: source detectability
    does not enter the mathematics, so this is NOT Eq. 13
    cross-calibration, NOT a corrected measurement, and NOT
    commensurate FRP. Labeled dict return (method + assumptions) so
    the value can never travel without its caveats.
    """
    value = _require_frp_allowing_zero(frp)
    area = _require_area(area_km2)
    coefficients.get_params(from_product, daynight)  # validates identity
    calc = expected_frp(value, area, to_product, daynight)
    return {"E_ret": calc["E_ret"], "p": calc["p"], "Dp": calc["Dp"],
            "r": calc["r"], "method": CROSS_MAP_METHOD,
            "from_product": from_product, "to_product": to_product,
            "daynight": daynight, "area_km2": area,
            "assumptions": list(CROSS_MAP_ASSUMPTIONS),
            "validation_status": "exploratory-unvalidated"}


def detection_regime(frp: float, area_km2: float, product: str,
                     daynight: str) -> dict:
    """Detectability regime of one detection (diagnostic convention).

    margin = P − Dp; transition band is |margin| ≤ 2/r (half-width
    2/r). The ±2/r band is a documented diagnostic convention for
    labeling, not paper physics. Labels: 'no-fire' (exact 0.0),
    'below-limit', 'transition', 'above-limit'. Missing/non-finite/
    negative FRP and invalid areas raise ValueError (never placed).
    """
    value = _require_frp_allowing_zero(frp)
    area = _require_area(area_km2)
    params = coefficients.get_params(product, daynight)
    dp = params["lambda"] * area + params["mu"]
    rate = params["r0"] + params["r1"] / area
    half_width = 2.0 / rate
    margin = value - dp
    if value == 0.0:
        regime = "no-fire"
    elif margin < -half_width:
        regime = "below-limit"
    elif margin > half_width:
        regime = "above-limit"
    else:
        regime = "transition"
    return {"regime": regime, "margin_MW": margin,
            "half_width_MW": half_width, "Dp": dp,
            "assumptions": list(REGIME_ASSUMPTIONS)}

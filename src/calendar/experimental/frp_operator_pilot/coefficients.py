"""Published OOp coefficients (transcribed values with provenance).

Source: Sofiev & Kouznetsov (2026), Tables 1-2 (XLSX assets fetched
from the GMD article page; values cross-checked against the
FRP_OOp_pixel.py implementation, which matches exactly).
DOI: https://doi.org/10.5194/gmd-19-10123-2026

Equations (paper notation):
  Dp(Ap) = λ·Ap + μ            (Eq. 5; Ap in km², Dp in MW)
  r(Ap)  = r0 + r1 / (Ap-A0)   (Eq. 6; A0 = 0; r in MW⁻¹)
  p(P)   = [sigmoid((P-Dp)·r) − c_o] / (1 − c_o)

Day/night follows solar zenith (φ≥90° = night) in the paper; the
pilot maps FIRMS daynight D/N flags onto it (documented assumption;
twilight unresolvable in FIRMS rows).

Note: the paper's headline nadir limits (5.7/3.7 MW MODIS day/night)
are empirical minima, not evaluations of the fitted lines; the
operator itself uses the table fits below, exactly as implemented
in the authors' code.
"""

SOURCE = ("Sofiev & Kouznetsov (2026), GMD 19, 10123-10142, Tables 1-2; "
          "cross-checked vs FRP_OOp_pixel.py coefficient block")
UNITS = {"area": "km2", "frp": "MW", "rate": "per-MW"}

# (sensor, daynight) -> {lambda, mu, r0, r1, c_o}; daynight in {"D","N"}.
TABLES = {
    ("modis", "D"): {"lambda": 4.44, "mu": 0.52,
                     "r0": 0.07, "r1": 1.26, "c_o": 0.045},
    ("modis", "N"): {"lambda": 4.43, "mu": 1.01,
                     "r0": 0.11, "r1": 1.06, "c_o": 0.045},
    ("viirs-snpp", "D"): {"lambda": 6.17, "mu": 1.44,
                          "r0": 0.61, "r1": 0.21, "c_o": 0.05},
    ("viirs-snpp", "N"): {"lambda": 1.38, "mu": 0.35,
                          "r0": 2.12, "r1": 1.02, "c_o": 0.05},
}

A0 = 0.0


def get_params(product: str, daynight: str) -> dict:
    """Coefficient row for one product/daynight (KeyError if unknown)."""
    try:
        return dict(TABLES[(product, daynight)])
    except KeyError:
        raise ValueError(
            f"no published coefficients for product={product!r} "
            f"daynight={daynight!r}; want products "
            f"{sorted({p for p, _ in TABLES})} and daynight D/N")

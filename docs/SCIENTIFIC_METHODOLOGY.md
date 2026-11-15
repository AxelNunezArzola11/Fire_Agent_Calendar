# FireAgent Calendar — Scientific & Mathematical Methodology

FireAgent Calendar enforces strict separation between **satellite thermal observations**, **statistical analytical constructs**, **experimental forward-operator diagnostics**, and **confirmed ground wildfires**. Every scientific computation runs deterministically in Python/DuckDB with full version provenance.

---

## 1. Sensor Harmonization & Matching (Phase 2B — `matching.py`)

### 1.1 Physical Distinction Between Sensors
FireAgent ingests MODIS (`Terra`, `Aqua`) and VIIRS (`N` / Suomi-NPP, `NOAA-20`, `NOAA-21`). These instruments differ fundamentally in pixel geometry and confidence scales:
- **MODIS**: nadir pixel footprint $\approx 1.0\text{ km}$, bow-tie growth off-nadir, continuous numeric confidence $C \in [0, 100]$.
- **VIIRS I-band (375 m)**: nadir pixel footprint $\approx 0.375\text{ km}$, categorical confidence $C \in \{\text{l}, \text{n}, \text{h}\}$.

**Rule:** Radiative power ($\text{FRP}$ in MW) is **never** summed directly across MODIS and VIIRS into a single physical emission total without sensor stratification.

### 1.2 Adaptive Spatiotemporal Cross-Sensor Matching (`matcher_v1`, `MatchParams`)
Cross-sensor correspondence links MODIS detections to VIIRS detections using great-circle Haversine distance (`EARTH_KM = 6371.0`) and overpass temporal separation:

$$d_{\text{hav}}((\phi_1,\lambda_1), (\phi_2,\lambda_2)) = 2 R_\oplus \arcsin\!\sqrt{\sin^2\!\left(\frac{\Delta\phi}{2}\right) + \cos\phi_1 \cos\phi_2 \sin^2\!\left(\frac{\Delta\lambda}{2}\right)}$$

- **Core match window**: radius $\text{mean\_footprint} \times \text{core\_factor} \approx 1.38\text{ km}$ (for MODIS $1.0\text{ km}$ and VIIRS $0.375\text{ km}$, `core_factor = 2.0`), temporal tolerance $\Delta t \le 12\text{ h}$.
- **Expanded weak-evidence window**: `expanded_factor = 4.0`, $\Delta t \le 24\text{ h}$ (logged as weak/ambiguous evidence, never forced into core matches).
- **Contention resolution**: `compete_margin = 0.15` (mutual-best pairs and uncontested attachments form `matched` groups; multi-candidate ties remain explicit `ambiguous` groups).
- **Harmonized observation count** (`index_2e.json`): each matched cross-sensor group counts once, while ambiguous members and singletons stay individually counted so repeated satellite passes do not masquerade as distinct fires.

---

## 2. Spatiotemporal Hotspot Clustering (Phase 2C — `clustering.py`)

Hotspot clusters (`cluster_v1`, `ClusterParams`) are deterministic **union-find** spatiotemporal groups of satellite detections — never confirmed fire perimeters or official incidents:

- **Cross-sensor links**: inherited exclusively from Phase 2B `matched` group co-membership (ambiguous groups create no links).
- **Same-sensor links**: pairs sharing the same instrument within `spatial_tol_km = 2.0` km ($\approx 2\times$ MODIS nadir footprint) and `temporal_gap_h = 24.0` h inside $0.1^\circ$ spatial geocells.
- Every input detection lands in exactly one cluster (singletons preserved explicitly).

---

## 3. Historical Baselines & Anomaly Detection (Phase 4B/4C — `baselines.py`, `anomalies.py`)

### 3.1 Leakage-Free Trailing & Stratified References (`baselines_v1`)
For any evaluation period $t$ with metric value $Y_t$ (daily, weekly, or monthly grain), the reference set $\mathcal{R}_t$ contains **only strictly preceding periods** ($s < t$, zero look-ahead):

1. `sensor_aware` (default `trailing` strategy, applicable to single-year and multi-year archives including regional DuckDB):
   $$\mathcal{R}_t = \{ Y_s : t - K \le s \le t - 1 \}, \quad K = \begin{cases} 30 & \text{daily} \\ 12 & \text{weekly} \\ 6 & \text{monthly} \end{cases}$$
2. `monthly_median` (`same_month` strategy across prior years).
3. `seasonal` (`same_week` ISO-week strategy across prior years).

**Baseline Statistics:**
$$\text{Baseline}_t = \operatorname{median}(\mathcal{R}_t), \qquad \mu_t = \frac{1}{n_t}\sum_{s \in \mathcal{R}_t} Y_s, \qquad s_t = \sqrt{\frac{1}{n_t - 1}\sum_{s \in \mathcal{R}_t} (Y_s - \mu_t)^2}$$

where $n_t = |\mathcal{R}_t|$.

### 3.2 Eligibility Floors & Variance Stability
- Median baseline requires $n_t \ge \text{MEDIAN\_MIN\_N} = 3$; otherwise status is `insufficient-history` (never zero-filled or interpolated).
- Parametric $Z$-score requires $n_t \ge \text{ZSCORE\_MIN\_N} = 7$ ($\ge 6$ degrees of freedom) and non-degenerate reference variance ($s_t > 0$ above relative floor `VARIANCE_RTOL = 1e-9`).

### 3.3 Anomaly Classification (`anomalies_v1`, `classify_v1`)
$$\text{PctDev}_t = \frac{Y_t - \text{Baseline}_t}{\text{Baseline}_t} \times 100\%, \qquad Z_t = \frac{Y_t - \mu_t}{s_t}$$

- Notable threshold gate (`Z_NOTABLE = 2.0`, `PCT_NOTABLE = 25.0`):
  - If both $Z_t$ and $\text{PctDev}_t$ are valid: classified `notable-high` or `notable-low` iff $|Z_t| \ge 2.0$ **and** $|\text{PctDev}_t| \ge 25\%$; otherwise `typical`.
  - If $n_t \in [3, 6]$ (median valid, $Z$ unavailable): `provisional-notable-high` / `provisional-notable-low` / `provisional-typical`.
  - If $\text{Baseline}_t = 0$: `% deviation` is explicitly `undefined-zero-baseline`; $Z$-only evidence applies when $Z_t$ is valid.

---

## 4. Sofiev & Kouznetsov (2026) FRP Observation Operator (Experimental Pilot)

Implemented in `src/calendar/experimental/frp_operator_pilot/` following Sofiev & Kouznetsov (2026), *Geoscientific Model Development* **19**, 10123–10142 ([DOI: 10.5194/gmd-19-10123-2026](https://doi.org/10.5194/gmd-19-10123-2026)).

### 4.1 Governing Equations (Implemented Pixel Operator)

Given pixel area $A_p$ ($\text{km}^2$), approximated here from FIRMS `scan × track` (`pixel_area_km2`), fire radiative power $P$ ($\text{MW}$), sensor product (`modis` or `viirs-snpp`), and overpass illumination (`daynight` $\in \{\text{D}, \text{N}\}$):

1. **Pixel Detection Limit $D_p(A_p)$ (Eq. 5):**
   $$D_p(A_p) = \lambda A_p + \mu \quad [\text{MW}]$$

2. **Sigmoid Transition Rate $r(A_p)$ (Eq. 6, with $A_0 = 0$):**
   $$r(A_p) = r_0 + \frac{r_1}{A_p - A_0} \quad [\text{MW}^{-1}]$$

3. **Pixel Detection Probability $p(P)$ (Eq. 7):**
   $$p(P) = \operatorname{clip}_{[0, 1]}\!\left( \frac{\sigma\!\big((P - D_p)\,r\big) - c_o}{1 - c_o} \right), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}, \qquad p(0) = 0$$
   where exponent $z$ is numerically safeguarded to $[-60, 60]$.

4. **Expected Retrieved FRP Diagnostic (`expected_frp`):**
   $$E_{\text{ret}}(P) = p(P) \cdot P \quad [\text{MW}]$$
   *Critical scientific interpretation:* FIRMS supplies already-retrieved $P$, not unobserved true surface FRP, and undetected fires are absent from FIRMS tables. Therefore $E_{\text{ret}}(P)$ measures a **detection-discounted diagnostic conditional on detection**, **never** true emitted fire power or an extinction correction.

5. **Diagnostic Detectability Regime (`detection_regime` — UI convention):**
   With margin $M = P - D_p$ and convention half-width $\Delta = 2 / r$:
   - `no-fire`: $P = 0$
   - `below-limit`: $M < -\Delta$
   - `transition`: $|M| \le \Delta$
   - `above-limit`: $M > \Delta$

### 4.2 Published Coefficients (`coefficients.py`, Tables 1–2)

| Product | Overpass (`D`/`N`) | $\lambda$ ($\text{MW}/\text{km}^2$) | $\mu$ ($\text{MW}$) | $r_0$ ($\text{MW}^{-1}$) | $r_1$ ($\text{km}^2/\text{MW}$) | $c_o$ |
|---|---|---:|---:|---:|---:|---:|
| `modis` | `D` (Day) | `4.44` | `0.52` | `0.07` | `1.26` | `0.045` |
| `modis` | `N` (Night) | `4.43` | `1.01` | `0.11` | `1.06` | `0.045` |
| `viirs-snpp` | `D` (Day) | `6.17` | `1.44` | `0.61` | `0.21` | `0.050` |
| `viirs-snpp` | `N` (Night) | `1.38` | `0.35` | `2.12` | `1.02` | `0.050` |

*Sensor exclusion rule (`validation.py`):* VIIRS `NOAA-20` and `NOAA-21` do **not** have published Tables 1–2 operator coefficients in Sofiev & Kouznetsov (2026) and are explicitly excluded with traceable exclusion reasons rather than reusing `viirs-snpp` weights.

### 4.3 Exploratory, Indicative & Unimplemented Operator Forms
- **Cross-sensor mapping (`cross_map_expected`, Eq. 13 exploratory):** evaluates the target sensor operator at the source detection's $P$ and $A_p$ (`validation_status = "exploratory-unvalidated"`).
- **Atmospheric attenuation (`aod_attenuation`, Eq. 12 indicative):** $P_{\text{att}} = P \exp(-\tau_{4\,\mu\text{m}})$ (`validation_status = "indicative-unvalidated"`, no production AOD product wired).
- **Gridded Observation Operator (Eqs. 8–11):** **not implemented** in this codebase.

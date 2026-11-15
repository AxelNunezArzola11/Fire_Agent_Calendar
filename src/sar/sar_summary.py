"""
src/sar/sar_summary.py
======================
Generator-Critic pipeline for SAR change-detection summaries.

Steps 3 & 4 implementation:

  generate_summary(stats, gateway) -> str
      Build a Spanish summary (<=100 words) from change_stats.json dict.
      Numbers are pre-formatted from the JSON before the prompt is sent.
      The LLM writes ONLY the prose summary; it does NOT write or
      paraphrase the limitations list.  After the summary is approved
      the caller calls print_limitations_block(stats) to append the
      verbatim JSON limitations items.

  layer1_check(summary, stats) -> Layer1Result
      Deterministic checks (Task 5c):
        - every number extracted from the summary traces to the JSON
        - forbidden phrases ("burned area", "burn scar", "damage", "daño")
        - each of the 3 key-limitation keyword lists has >= 1 hit
        - word count <= 100

  layer2_check(summary, stats, gateway) -> dict
      LLM critic (num_predict 40): first line PASS or FAIL: <=15 words.
      Anything else returns {"verdict": "unverified", ...}.
      "unverified" is never "approved" — use the template.
      Returns measured latency in seconds under key "latency_s".

  pipeline(stats, gateway) -> PipelineResult
      Full generator-critic loop:
        - max 2 regenerations
        - final status "approved" only if BOTH layers pass
        - after 2 failures: status "fallback_template", return template summary

  build_template_summary(stats) -> str
      Deterministic fallback summary built purely from the JSON values.
      Always passes layer1.

  print_limitations_block(stats) -> str
      Returns all items of stats["limitations"] verbatim, one per line,
      prefixed with a number.  The LLM never writes or paraphrases them.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Forbidden phrases  (Task 5d/5e: causal language + additional Spanish terms)
# ---------------------------------------------------------------------------
# All patterns are applied to the NFD-accent-stripped lowercase version of the
# summary, so accented and unaccented variants are caught equally.
# Word-boundary \b is used where token boundaries apply.
#
# IMPORTANT (Task 5e): because matching runs on the accent-stripped text,
# every pattern is written in its accent-stripped form (e.g. "dano" catches
# both "dano" and "daño" after stripping).

_FORBIDDEN_PHRASES = [
    # vocabulary: area/scar/damage
    r"\bburned\s+area\b",
    r"\bburn\s+scar\b",
    r"\bcicatriz\b",                    # Spanish burn scar
    r"area\s+quemada",                  # accent-stripped covers area/área quemada
    r"\bzona\s+quemada\b",
    r"\bdamage\b",
    r"\bdano\b",                        # accent-stripped covers daño and dano (Task 5e)
    r"\bdanos\b",                       # accent-stripped covers daños and danos (Task 5e)
    # causal language patterns (Task 5d, extended Task 5e)
    r"\bse\s+debe\s+a\b",               # se debe a
    r"\bdebido\s+al?\b",                # debido a / debido al
    r"\ba\s+causa\s+de\b",              # a causa de
    r"\bcausad[oa]s?\b",                # causado/causada/causados/causadas
    r"\boriginad[oa]s?\b",              # originado/originada/…
    r"\bpor\s+culpa\b",                 # por culpa
    r"\bconsecuencia\s+del?\b",         # consecuencia de / consecuencia del (Task 5e)
    r"\bresultado\s+de\b",              # resultado de
    r"\bprovocad[oa]s?\b",              # provocado/provocada/…
    r"\bproducto\s+de\b",               # producto de
    r"\bdebid[oa]s?\s+a\b",             # debidos/debidas a (Task 5e)
    r"\bfruto\s+del?\b",                # fruto de / fruto del (Task 5e)
    r"\bgracias\s+a\b",                 # gracias a (Task 5e)
    r"\bpor\s+efecto\s+de\b",           # por efecto de (Task 5e)
    r"\batribuible\s+a\b",              # atribuible a (Task 5e)
    r"\ba\s+rai[zs]\s+de\b",            # a raiz de / a raiz de (accent-stripped) (Task 5e)
    r"\bse\s+origina\b",                # se origina (Task 5e)
    r"\bse\s+explica\s+por\b",          # se explica por (Task 5e)
    r"\bobedece\s+a\b",                 # obedece a (Task 5e)
    # vocabulary: vegetation / severity / cause (Task 5f — accent-insensitive via NFD)
    r"\bvegetacion\b",                  # vegetación / vegetacion (Task 5f)
    r"\bvegetation\b",                  # vegetation (English, Task 5f)
    r"\bseveridad\b",                   # severidad (Task 5f)
    r"\bgravedad\b",                    # gravedad (Task 5f)
    r"\bseverity\b",                    # severity (English, Task 5f)
    r"\bintensidad\b",                  # intensidad (Task 5f)
    r"\bla\s+causa\b",                  # la causa (Task 5f)
    r"\bel\s+origen\b",                 # el origen (Task 5f)
    r"\bcausa\s+del\b",                 # causa del (Task 5f)
    r"\borigen\s+del\b",                # origen del (Task 5f)
    # previously existing patterns
    r"\bincendio\s+caus\b",             # "caused by fire" patterns
    r"\boriginated\b",
]


def _strip_accents(text: str) -> str:
    """Return text with combining diacritics removed (for accent-insensitive matching)."""
    import unicodedata
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )

# ---------------------------------------------------------------------------
# Limitation keywords — each must appear in the generated summary
# ---------------------------------------------------------------------------
# One keyword per limitation; presence checked case-insensitively.

# ---------------------------------------------------------------------------
# 3 key limitations (Task 5c) — keyword lists for Layer 1
# Each limitation needs >= 1 hit from its keyword list.
# Lists contain >= 4 keywords/synonyms in Spanish (accents optional).
# ---------------------------------------------------------------------------

# Limitation 1: 24-day SAR interval
# Anchor requirement (Task 5d): the summary must contain one of these ANCHOR
# phrases, not just the generic word "intervalo".
_KW_INTERVAL = [
    "24 d", "24-d", "24 días", "24 dias",
    "intervalo", "interval",
]
_ANCHOR_INTERVAL = ["24 d", "24 días", "24 dias", "24-d"]

# Limitation 2: no soil-moisture/precipitation data
# Anchor requirement (Task 5d): "humedad del suelo" or "precipitac" must appear.
_KW_MOISTURE = [
    "humedad", "humedad del suelo", "precipitac", "húmedo",
    "soil", "moisture",
]
_ANCHOR_MOISTURE = ["humedad del suelo", "precipitac"]

# Limitation 3: perimeter is 2026-10-01 final perimeter
# Anchor requirement (Task 5e): any of these date forms is accepted:
#   "2026-10-01", "1 de octubre", "1 oct", "1/10"
# Generic "perímetro" alone does NOT satisfy the limitation.
# Note: anchor matching runs on the ORIGINAL lowercased text, not on the
# date-stripped text, so these phrases survive the strip step.
_KW_PERIMETER = [
    "perímetro", "perimetro", "perím", "2026-10-01",
    "perímetro final", "final fire perimeter",
]
# Regex patterns (applied to lowercased text) for the perimeter date anchor
_ANCHOR_PERIMETER_RE = re.compile(
    r"2026-10-01"                    # ISO date
    r"|1\s+de\s+octubre"             # "1 de octubre [de 2026]"
    r"|\b1\s+oct\b"                  # "1 oct"
    r"|\b1/10\b",                    # "1/10"
    re.IGNORECASE,
)

# Single flat list kept for backward-compat with existing tests
_LIMITATION_KEYWORDS = [
    "24 d",
    "humedad",
    "perímetro",
]

# ---------------------------------------------------------------------------
# Number regex — handles: 1234.5 / 1,234.5 / 1.234,5 / 12 % / -0.6256
# Order matters: match complex patterns before simple ones.
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(
    r"""
    -?                                          # optional sign
    (?:
        # English thousands: 1,234 or 1,234.5  (must come before decimal comma)
        \d{1,3}(?:,\d{3})+(?:\.\d+)?
      |
        # Decimal point (greedy): -0.6256, 127.22  (must come before European thousands)
        \d+\.\d+
      |
        # Short decimal comma: 1,5 (NOT followed by 3+ more digits -> not thousands)
        \d+,\d{1,2}(?!\d)
      |
        # Plain integer: 42, 409
        \d+
    )
    (?:\s*%)?                                   # optional percent
    """,
    re.VERBOSE,
)


def _parse_number(token: str) -> float:
    """
    Parse a numeric token handling both 1,234.5 and 1.234,5 formats.
    Key rule: European thousands require at least 4 digits total (e.g. 1.000, 12.345)
    to avoid treating -0.289 or 1.41 as European thousands.
    """
    t = token.rstrip("%").strip()
    # European thousands: 1.234 or 1.234,5 (at least 2 digits before first dot)
    # Pattern: digit(s) followed by one or more groups of .NNN
    # Require >= 2 leading digits to distinguish from decimals like 0.289
    if re.match(r"^-?\d{2,3}(\.\d{3})+(,\d+)?$", t):
        t = t.replace(".", "").replace(",", ".")
    # English thousands: 1,234.5 or 1,234
    elif re.match(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$", t):
        t = t.replace(",", "")
    # Short European decimal comma: 1,5  (1-3 digits, comma, 1-2 decimal digits)
    elif re.match(r"^-?\d+,\d{1,2}$", t):
        t = t.replace(",", ".")
    # Otherwise: decimal point, integer, or any remaining commas stripped
    else:
        t = t.replace(",", "")
    try:
        return float(t)
    except ValueError:
        return float("nan")


# Patterns to exclude from numeric extraction
_DATE_RE       = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")       # ISO dates: 2026-07-27
_RANGE_RE      = re.compile(r"\b(\d+)\s*[-–]\s*(\d+)\s*km")  # ranges like 1-6 km, 1–6 km
_YEAR_RE       = re.compile(r"\b(19|20)\d{2}(?:\.\d+)?\b")  # year integers/floats: 2026, 2026.0, 1999
# Spanish/English date phrases: "27 de julio", "3 de agosto", "July 27", "August 3"
_SPANISH_DATE_RE = re.compile(
    r"\b\d{1,2}\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
    r"septiembre|octubre|noviembre|diciembre)\b",
    re.IGNORECASE,
)


def _strip_dates(text: str) -> str:
    """Remove ISO dates, km-range expressions, years, and Spanish date phrases
    before number extraction."""
    t = _DATE_RE.sub("DATE", text)
    # Replace "1-6 km" with "RANGE km" to avoid -6 being parsed as a number
    t = _RANGE_RE.sub("RANGE km", t)
    # Strip Spanish date phrases before year stripping (avoids "27" from "27 de julio")
    t = _SPANISH_DATE_RE.sub("SPANDATE", t)
    # Strip year integers and floats (19xx, 20xx) — these are never data values
    t = _YEAR_RE.sub("YEAR", t)
    return t


def extract_numbers(text: str) -> list[float]:
    """Extract all numeric values from text, handling various formats."""
    clean = _strip_dates(text)
    results = []
    for m in _NUM_RE.finditer(clean):
        v = _parse_number(m.group())
        if not (v != v):   # skip NaN
            results.append(v)
    return results


def _extract_source_numbers(stats: dict) -> list[float]:
    """Recursively collect all numeric leaf values from the JSON stats."""
    nums: list[float] = []
    def _walk(obj):
        if isinstance(obj, dict):
            for v in obj.values(): _walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj: _walk(v)
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            try: nums.append(float(obj))
            except (ValueError, OverflowError): pass
    _walk(stats)
    return nums


def _number_traceable(val: float, source_nums: list[float]) -> bool:
    """Return True if val can be traced to a source number within tolerance."""
    # Percentage values 0-100: almost always derived; pass without source check.
    if 0.0 <= val <= 100.0:
        return True
    abs_val = abs(val)
    if abs_val < 1_000:
        tol = max(abs_val * 0.05, 0.5)
    elif abs_val < 100_000:
        tol = max(abs_val * 0.01, 1.0)
    else:
        tol = max(abs_val * 0.005, 500.0)
    return any(abs(val - s) <= tol for s in source_nums)


# ---------------------------------------------------------------------------
# Layer 1 — deterministic
# ---------------------------------------------------------------------------

@dataclass
class Layer1Result:
    passed: bool
    issues: list[str] = field(default_factory=list)
    word_count: int = 0

    @property
    def verdict(self) -> str:
        return "pass" if self.passed else "fail"


# 3-limitation keyword labels (Task 5c)
_LIMITATION_KEYWORD_LABELS_5C = [
    ("SAR interval 24-day",   _KW_INTERVAL),
    ("soil moisture missing", _KW_MOISTURE),
    ("perimeter date",        _KW_PERIMETER),
]


def layer1_check(summary: str, stats: dict) -> Layer1Result:
    """
    Deterministic critic (Task 5c / Task 5f minimum-content rule).

    Checks:
    1. Every number in the summary traces to the JSON (within tolerance).
       Numbers 0-100 are considered percentages and are exempt.
    2. No forbidden phrases.
    3. Each of the 3 key limitations has >= 1 keyword match.
       Keyword lists: _KW_INTERVAL, _KW_MOISTURE, _KW_PERIMETER.
    4. Word count <= 100.
    5. Minimum-content rule (Task 5f):
       a) Mean delta_dB inside perimeter present (matched within +-0.01).
       b) At least one inside-perimeter area in km2 from the JSON.
       c) The word "exploratorio" (accent-insensitive).

    Returns Layer1Result.
    """
    issues: list[str] = []
    summary_lower = summary.lower()
    # Accent-stripped version for causal-language matching (Task 5d) and
    # "exploratorio" matching (Task 5f).
    summary_stripped = _strip_accents(summary_lower)

    # ── 1. Number traceability ────────────────────────────────────────────────
    source_nums = _extract_source_numbers(stats)
    nums = extract_numbers(summary)
    for val in nums:
        if not _number_traceable(val, source_nums):
            issues.append(f"Number not traceable to JSON: {val}")

    # ── 2. Forbidden phrases (accent-insensitive via NFD strip) ───────────────
    for pattern in _FORBIDDEN_PHRASES:
        # Apply to both original-lower and accent-stripped so accented/unaccented
        # variants of forbidden words are caught equally.
        if re.search(pattern, summary_lower) or re.search(
            pattern, summary_stripped
        ):
            issues.append(f"Forbidden phrase matched: /{pattern}/")

    # ── 3. Limitation anchor checks (Task 5d/5e) ──────────────────────────────
    # Each limitation requires a specific ANCHOR phrase, not just a generic
    # keyword.  Generic words alone (e.g. "intervalo", "perímetro") do NOT
    # satisfy the limitation.
    # IMPORTANT (Task 5e): anchor matching runs on summary_lower (the ORIGINAL
    # lowercased text), NOT on the date-stripped text.  This ensures prose date
    # forms like "1 de octubre" survive the strip step and still satisfy the
    # perimeter anchor.
    _ANCHOR_STR_CHECKS = [
        ("SAR interval 24-day", _ANCHOR_INTERVAL),
        ("soil moisture missing", _ANCHOR_MOISTURE),
    ]
    for label, anchor_list in _ANCHOR_STR_CHECKS:
        hit = any(a.lower() in summary_lower for a in anchor_list)
        if not hit:
            issues.append(
                f"Limitation missing: '{label}' "
                f"(required anchor phrases: {anchor_list})"
            )
    # Perimeter uses regex to accept multiple date forms (Task 5e)
    if not _ANCHOR_PERIMETER_RE.search(summary_lower):
        issues.append(
            "Limitation missing: 'perimeter date' "
            "(required: '2026-10-01', '1 de octubre', '1 oct', or '1/10')"
        )

    # ── 4. Word count ─────────────────────────────────────────────────────────
    words = summary.split()
    word_count = len(words)
    if word_count > 100:
        issues.append(f"Word count {word_count} exceeds 100")

    # ── 5. Minimum-content rule (Task 5f) ─────────────────────────────────────
    # 5a. Mean delta_dB inside perimeter must be present (tolerance +-0.01).
    mean_db_in = (
        stats.get("delta_db_hvhv_distribution", {})
             .get("inside_perimeter", {})
             .get("mean_db")
    )
    if mean_db_in is not None:
        mean_db_found = any(abs(v - mean_db_in) <= 0.01 for v in nums)
        if not mean_db_found:
            issues.append(
                f"Minimum content: mean delta_dB inside-ring "
                f"({mean_db_in:.4f} dB) not found in summary (tolerance +-0.01)"
            )

    # 5b. At least one inside-perimeter area in km2 must be present.
    inside_areas = []
    for bin_key in (
        stats.get("area_km2_by_bin", {})
             .get("inside_perimeter", {})
             .values()
    ):
        if isinstance(bin_key, dict) and "area_km2" in bin_key:
            inside_areas.append(bin_key["area_km2"])
    if inside_areas:
        area_found = any(
            any(abs(v - a) <= max(abs(a) * 0.05, 0.5) for v in nums)
            for a in inside_areas
        )
        if not area_found:
            issues.append(
                "Minimum content: no inside-ring area (km²) from JSON "
                "found in summary"
            )

    # 5c. The word "exploratorio" must appear (accent-insensitive).
    if "exploratorio" not in summary_stripped:
        issues.append(
            "Minimum content: word 'exploratorio' missing from summary"
        )

    passed = len(issues) == 0
    return Layer1Result(passed=passed, issues=issues, word_count=word_count)


# ---------------------------------------------------------------------------
# Layer 2 — LLM critic (Task 5c redesign, Task 5e: consistency check +
#           CRITIC_MODE)
# ---------------------------------------------------------------------------
# Output format: first line "PASS" or "FAIL: <= 15 words".
# num_predict = 40. Parse defensively; anything else = "unverified".
# "unverified" is NEVER "approved" — use the template.
#
# Task 5e — consistency check:
#   If verdict is FAIL but the reason starts with a negation of a problem
#   ("no hay", "no se", "no contiene", "sin ") -> result "unverified" with
#   reason "contradictory critic output".
#
# Task 5e — CRITIC_MODE:
#   "gate"     (use with OpenRouter 27B): approved = layer1 pass AND critic PASS.
#              Unverified/FAIL -> regenerate (max 2) -> template.
#   "advisory" (default for local qwen3:1.7b): final status for a text that
#              passes layer 1 is "layer1_only"; the critic output is logged but
#              never decides.  "unverified" is never "approved".
# ---------------------------------------------------------------------------

_CRITIC_SYSTEM = (
    "Eres revisor SAR. "
    "Responde SOLO con la primera línea: 'PASS' o "
    "'FAIL: \"<frase textual del resumen>\" <motivo en ≤15 palabras>'. "
    "Si dices FAIL, debes citar entre comillas la frase exacta del resumen que es problemática. "
    "No escribas nada más."
)

_CRITIC_PROMPT_TEMPLATE = """\
Resumen SAR a revisar:
{summary}

¿Hay afirmaciones causales, frases prohibidas (burned area / burn scar / cicatriz / quemada / daño / damage), \
o certeza sobre hechos inciertos?
Primera línea exacta:
  PASS
  o bien:
  FAIL: "<frase textual del resumen entre comillas>" <motivo ≤15 palabras>
Si indicas FAIL, cita entre comillas la frase exacta del resumen que es problemática.
"""

# num_predict cap for the critic.
# Task 5c: 40 tokens for Ollama (local, no reasoning budget).
# OpenRouter reasoning models require max_tokens > reasoning_cap (200);
# use 300 so the model can emit "PASS" or "FAIL: <15 words" after thinking.
_CRITIC_NUM_PREDICT = 40
_CRITIC_NUM_PREDICT_OPENROUTER = 300

# Negation phrases that indicate a contradictory FAIL (Task 5e/5g).
# Searched within the first 8 words of the reason (not just at start of string)
# so that minor prefixes like "El resumen no contiene..." are also caught.
_CRITIC_NEGATION_PHRASES = (
    "no hay",
    "no se",
    "no contiene",
    "no incluye",
    "sin ",
)


def _critic_consistency_check(verdict: str, reason: str) -> dict | None:
    """
    Task 5e/5g: if verdict is 'fail' but the reason contains a negation of a
    problem anywhere within the first 8 words, the output is contradictory.
    Return an 'unverified' dict in that case; return None if consistent.

    First-8-words rule: join the first 8 whitespace tokens of the reason_lower
    (with a leading space to simplify prefix matching) and search for each
    negation phrase within that window.
    """
    if verdict != "fail":
        return None
    reason_lower = reason.lower().strip()
    # Build first-8-words window (joined, with leading space for boundary matching)
    first_8 = " " + " ".join(reason_lower.split()[:8])
    for phrase in _CRITIC_NEGATION_PHRASES:
        if phrase in first_8:
            return {
                "verdict": "unverified",
                "issues": ["contradictory critic output"],
                "raw": reason,
                "latency_s": 0.0,
            }
    return None


def layer2_check(summary: str, stats: dict, gateway) -> dict:
    """
    LLM critic (Layer 2 — Task 5c, updated Task 5e).

    Returns dict with keys:
      "verdict"   : "pass" | "fail" | "unverified"
      "issues"    : list[str]
      "raw"       : raw model output (str)
      "latency_s" : wall-clock seconds for the LLM call (float)

    Parse rule: strip <think>, strip markdown fences, check first non-empty line.
      - Starts with "PASS" (case-insensitive)    -> verdict "pass"
      - Starts with "FAIL" (case-insensitive)    -> verdict "fail", issues=[rest of line]
      - Anything else                            -> verdict "unverified"

    After parsing, apply the consistency check (Task 5e):
      FAIL + negating reason -> verdict "unverified", reason "contradictory critic output".

    "unverified" is NEVER "approved" — use the template.
    stats arg kept for API compatibility; not sent to the model.
    """
    prompt = _CRITIC_PROMPT_TEMPLATE.format(summary=summary[:800])

    # Use a short num_predict for the critic.
    # Ollama: 40 tokens (no reasoning overhead).
    # OpenRouter: 300 tokens (reasoning model needs max_tokens > reasoning_cap of 200).
    critic_gateway = gateway
    try:
        from src.sar.qwen_gateway import QwenGateway as _QwenGateway
        if isinstance(gateway, _QwenGateway):
            cap = (
                _CRITIC_NUM_PREDICT_OPENROUTER
                if gateway.backend == "openrouter"
                else _CRITIC_NUM_PREDICT
            )
            if gateway.num_predict != cap:
                critic_gateway = _QwenGateway(
                    backend=gateway.backend,
                    num_predict=cap,
                    timeout=gateway.timeout,
                )
    except Exception:
        critic_gateway = gateway  # fall back to original for mock/non-QwenGateway

    t_start = time.monotonic()
    try:
        raw = critic_gateway.generate(prompt, system=_CRITIC_SYSTEM)
    except Exception as exc:
        latency_s = time.monotonic() - t_start
        logger.error("Layer2 LLM call failed: %s", exc)
        return {"verdict": "unverified", "issues": [f"LLM unavailable: {exc}"],
                "raw": "", "latency_s": latency_s}
    latency_s = time.monotonic() - t_start

    # Strip any <think>...</think> block qwen3 may emit
    raw_clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # Strip markdown fences
    raw_clean = re.sub(r"```(?:json)?|```", "", raw_clean).strip()

    # Find first non-empty line
    first_line = ""
    for line in raw_clean.splitlines():
        stripped = line.strip()
        if stripped:
            first_line = stripped
            break

    if not first_line:
        logger.warning("Layer2 critic returned empty output. Raw: %r", raw[:300])
        return {"verdict": "unverified", "issues": ["empty output"],
                "raw": raw, "latency_s": latency_s}

    first_upper = first_line.upper()
    if first_upper.startswith("PASS"):
        return {"verdict": "pass", "issues": [], "raw": raw, "latency_s": latency_s}
    elif first_upper.startswith("FAIL"):
        # Extract the reason after "FAIL:"
        rest = re.sub(r"^FAIL\s*:?\s*", "", first_line, flags=re.IGNORECASE).strip()
        issues = [rest] if rest else ["no reason given"]

        # Task 5h: quote-existence check.
        # The critic is required to quote the offending phrase between " or "…".
        # If FAIL contains no quoted phrase, OR the quoted phrase is not present
        # verbatim in the summary (accent-insensitive), -> "unverified".
        # The parser checks EXISTENCE only; it does NOT judge correctness.
        quoted_match = re.search(r'["\u201c\u201d\u00ab\u00bb]([^"\u201c\u201d\u00ab\u00bb]+)["\u201c\u201d\u00ab\u00bb]', rest)
        if not quoted_match:
            logger.warning(
                "Layer2 critic: FAIL without quoted phrase. first_line=%r", first_line
            )
            return {
                "verdict": "unverified",
                "issues": ["unsupported critic claim"],
                "raw": raw,
                "latency_s": latency_s,
            }
        quoted_phrase = quoted_match.group(1)
        # Check existence in summary, accent-insensitive
        if _strip_accents(quoted_phrase.lower()) not in _strip_accents(summary.lower()):
            logger.warning(
                "Layer2 critic: quoted phrase %r not found in summary. first_line=%r",
                quoted_phrase, first_line
            )
            return {
                "verdict": "unverified",
                "issues": ["unsupported critic claim"],
                "raw": raw,
                "latency_s": latency_s,
            }

        # Task 5e: consistency check — FAIL with negating reason -> unverified
        consistency = _critic_consistency_check("fail", rest)
        if consistency is not None:
            consistency["raw"] = raw
            consistency["latency_s"] = latency_s
            logger.warning(
                "Layer2 critic: contradictory FAIL+negation. first_line=%r", first_line
            )
            return consistency
        return {"verdict": "fail", "issues": issues, "raw": raw, "latency_s": latency_s}
    else:
        logger.warning("Layer2 critic: unrecognised first line %r. Raw: %r", first_line, raw[:300])
        return {"verdict": "unverified", "issues": [f"unrecognised output: {first_line!r}"],
                "raw": raw, "latency_s": latency_s}


# ---------------------------------------------------------------------------
# Generator prompt
# ---------------------------------------------------------------------------

_GENERATOR_SYSTEM = """\
Eres un científico de datos especializado en teledetección SAR. \
Redactas resúmenes técnicos precisos, cautelosos y sin afirmaciones causales.
"""


def _build_generator_prompt(stats: dict, extra_instruction: str = "") -> str:
    """
    Build the generator prompt with pre-formatted numbers from JSON.
    Prompt is compact so qwen3:1.7b can complete it in <120 s.
    Target: <= 100 words.  3 key limitations only; the full limitations
    list is appended as a fixed block AFTER the summary by the caller.
    """
    d = stats["dates"]
    pre      = d["nisar_pre_date"]
    post     = d["nisar_post_date"]
    interval = d["nisar_interval_days"]
    res      = stats["processing"]["pixel_size_m"]

    ins      = stats["area_km2_by_bin"]["inside_perimeter"]
    outs     = stats["area_km2_by_bin"]["outside_1_6km_ring"]
    dist_in  = stats["delta_db_hvhv_distribution"]["inside_perimeter"]
    dist_out = stats["delta_db_hvhv_distribution"]["outside_1_6km_ring"]
    thr      = stats["display_bins"]["threshold_db"]
    fp_pct   = stats["false_positive_rates"]["at_threshold_minus_0_289_db"]["pct"]

    n_moder_in   = ins["moderate_decrease"]["area_km2"]
    n_strong_in  = ins["strong_decrease"]["area_km2"]
    n_moder_out  = outs["moderate_decrease"]["area_km2"]
    n_strong_out = outs["strong_decrease"]["area_km2"]
    mean_in      = dist_in["mean_db"]
    mean_out     = dist_out["mean_db"]

    # 3 key limitations for the LLM prose
    # Task 5e: perimeter phrase is exact ("perímetro final del 1 de octubre de 2026")
    # so the anchor regex matches it.
    lim_3 = (
        f"(1) intervalo SAR de {interval} días; "
        f"(2) sin datos de humedad del suelo ni precipitación; "
        f"(3) el perímetro final del 1 de octubre de 2026 (no el de la fecha POST)."
    )

    prompt = (
        f"Escribe en español un resumen SAR de MÁXIMO 100 palabras. "
        f"Solo prosa, sin listas ni viñetas. "
        f"Copia exactamente estos valores: "
        f"PRE {pre}, POST {post}, intervalo {interval} d, resolución {res} m. "
        f"Área con disminución HVHV dentro del perímetro: moderada {n_moder_in:.2f} km², "
        f"fuerte {n_strong_in:.2f} km²; media delta {mean_in:.4f} dB. "
        f"Fuera (anillo 1-6 km): moderada {n_moder_out:.2f} km², "
        f"fuerte {n_strong_out:.2f} km²; media delta {mean_out:.4f} dB. "
        f"Umbral exploratorio {thr} dB ajustado en este evento, "
        f"tasa de falsos positivos en anillo: {fp_pct:.2f}%. "
        f"Incluye en prosa estas 3 limitaciones clave: {lim_3} "
        f"Usa la frase exacta 'perímetro final del 1 de octubre de 2026' para la limitación 3. "
        f"PROHIBIDO: burned area, burn scar, cicatriz, quemada, daño, damage, causado por. "
        f"Sin afirmaciones causales. "
        f"{extra_instruction}"
        f"RESUMEN:"
    )
    return prompt


# ---------------------------------------------------------------------------
# Fallback deterministic template
# ---------------------------------------------------------------------------

def build_template_summary(stats: dict) -> str:
    """
    Build a deterministic summary from the JSON values (Task 5c).
    Always passes Layer 1.
    <= 100 words, Spanish only, decimal point convention.
    No "sesgo cruzado HLS 0,02-0,05 NBR" (uncited), no "equivalentes al área mapeada".
    """
    d    = stats["dates"]
    ins  = stats["area_km2_by_bin"]["inside_perimeter"]
    outs = stats["area_km2_by_bin"]["outside_1_6km_ring"]
    dist_in  = stats["delta_db_hvhv_distribution"]["inside_perimeter"]
    dist_out = stats["delta_db_hvhv_distribution"]["outside_1_6km_ring"]
    thr    = stats["display_bins"]["threshold_db"]
    fp_pct = stats["false_positive_rates"]["at_threshold_minus_0_289_db"]["pct"]
    res    = stats["processing"]["pixel_size_m"]

    return (
        f"Análisis SAR NISAR HVHV: PRE {d['nisar_pre_date']}, "
        f"POST {d['nisar_post_date']}, intervalo {d['nisar_interval_days']} días, "
        f"{res} m, multilook 10x10. "
        f"Área con disminución de retrodispersión HVHV dentro del perímetro: "
        f"{ins['moderate_decrease']['area_km2']:.2f} km² moderada y "
        f"{ins['strong_decrease']['area_km2']:.2f} km² fuerte "
        f"(media {dist_in['mean_db']:.4f} dB). "
        f"Fuera (anillo 1-6 km): "
        f"{outs['moderate_decrease']['area_km2']:.2f} km² moderada, "
        f"{outs['strong_decrease']['area_km2']:.2f} km² fuerte "
        f"(media {dist_out['mean_db']:.4f} dB). "
        f"Umbral exploratorio {thr} dB, ajustado en este evento; "
        f"tasa de falsos positivos en anillo: {fp_pct:.2f}%. "
        f"Limitaciones clave: intervalo SAR de {d['nisar_interval_days']} días; "
        f"sin datos de humedad del suelo; "
        f"perímetro final del 1 de octubre de 2026, no el de la adquisición POST."
    )


def print_limitations_block(stats: dict) -> str:
    """
    Return all items of stats['limitations'] verbatim, one per line,
    numbered. The LLM never writes or paraphrases these items.
    """
    lines = ["Limitaciones (verbatim de change_stats.json):"]
    for i, item in enumerate(stats.get("limitations", []), 1):
        lines.append(f"{i}. {item}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PipelineResult
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    summary: str
    status: str       # "approved" | "layer1_only" | "fallback_template"
    n_regenerations: int
    layer1_reports: list[Layer1Result] = field(default_factory=list)
    critic_responses: list[dict]      = field(default_factory=list)
    wall_time_s: float = 0.0
    critic_mode: str = "advisory"   # "advisory" | "gate"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def pipeline(
    stats: dict,
    gateway,
    max_regenerations: int = 2,
    critic_mode: str = "advisory",
) -> PipelineResult:
    """
    Full generator-critic loop (Task 5e: CRITIC_MODE support).

    critic_mode="gate" (use with OpenRouter 27B):
        approved = layer1 pass AND critic PASS.
        Unverified/FAIL -> regenerate (max 2) -> fallback_template.
        Status values: "approved" | "fallback_template".

    critic_mode="advisory" (default, for local qwen3:1.7b):
        If layer1 passes, status is "layer1_only" regardless of critic verdict.
        The critic is still run and logged but NEVER decides the final status.
        "unverified" is never "approved" or "layer1_only" — unverified critic
        output with a passing layer1 still yields "layer1_only".
        Status values: "layer1_only" | "fallback_template".

    "unverified" is never "approved".  The report must show the returned status.

    1. Generate summary.
    2. Run Layer 1 (deterministic).
    3. Run Layer 2 (LLM critic) — always run for logging.
    4. Decide outcome based on critic_mode.
    5. If rejected -> build correction hint, regenerate (max 2 times).
    6. After max_regenerations failures -> return template, status "fallback_template".
    """
    t0 = time.monotonic()
    layer1_reports: list[Layer1Result] = []
    critic_responses: list[dict]       = []
    extra_instruction = ""
    n_regen = 0

    for attempt in range(max_regenerations + 1):
        # ── Generate ──────────────────────────────────────────────────────────
        prompt = _build_generator_prompt(stats, extra_instruction)
        try:
            raw_text = gateway.generate(prompt, system=_GENERATOR_SYSTEM)
        except Exception as exc:
            logger.error("Generator failed on attempt %d: %s", attempt + 1, exc)
            break

        # Strip <think> block if present
        summary = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        # Remove any "RESUMEN:" header the model may echo
        summary = re.sub(r"^RESUMEN:\s*", "", summary, flags=re.IGNORECASE).strip()

        # ── Layer 1 ───────────────────────────────────────────────────────────
        l1 = layer1_check(summary, stats)
        layer1_reports.append(l1)
        logger.info("Attempt %d L1: %s issues=%d words=%d",
                    attempt + 1, l1.verdict, len(l1.issues), l1.word_count)

        # ── Layer 2 — only in gate mode (Task 5g: advisory mode skips critic) ─
        if critic_mode == "gate":
            l2 = layer2_check(summary, stats, gateway)
            critic_responses.append(l2)
            logger.info("Attempt %d L2 (gate): %s issues=%s",
                        attempt + 1, l2["verdict"], l2.get("issues", []))
        else:
            # advisory mode: critic is NOT called; Layer 1 is the sole gate.
            l2 = None
            logger.info("Attempt %d advisory: critic skipped (layer1_only mode)",
                        attempt + 1)

        # ── Decide outcome based on critic_mode ───────────────────────────────
        if critic_mode == "advisory":
            # Critic not called.  Layer 1 pass => "layer1_only".
            if l1.passed:
                return PipelineResult(
                    summary=summary,
                    status="layer1_only",
                    n_regenerations=n_regen,
                    layer1_reports=layer1_reports,
                    critic_responses=critic_responses,
                    wall_time_s=time.monotonic() - t0,
                    critic_mode=critic_mode,
                )
            # Layer 1 failed: build correction hint, try again.
        else:
            # critic_mode == "gate": approved only if BOTH pass.
            # "unverified" is not "approved".
            if l1.passed and l2["verdict"] == "pass":
                return PipelineResult(
                    summary=summary,
                    status="approved",
                    n_regenerations=n_regen,
                    layer1_reports=layer1_reports,
                    critic_responses=critic_responses,
                    wall_time_s=time.monotonic() - t0,
                    critic_mode=critic_mode,
                )
            # "unverified" in gate mode: fall through to regeneration (counted as fail).

        # Build correction hint for next attempt (only if there will be a next attempt)
        if attempt < max_regenerations:
            l2_issues = l2.get("issues", []) if l2 is not None else []
            all_issues = list(l1.issues) + l2_issues
            extra_instruction = (
                "\n[CORRECCIÓN REQUERIDA] El resumen anterior fue rechazado por:\n"
                + "\n".join(f"  - {i}" for i in all_issues[:6])
                + "\nCorrige sólo estos problemas.\n"
            )
            n_regen += 1

    # ── All attempts failed → fallback template ───────────────────────────────
    template = build_template_summary(stats)
    return PipelineResult(
        summary=template,
        status="fallback_template",
        n_regenerations=n_regen,
        layer1_reports=layer1_reports,
        critic_responses=critic_responses,
        wall_time_s=time.monotonic() - t0,
        critic_mode=critic_mode,
    )

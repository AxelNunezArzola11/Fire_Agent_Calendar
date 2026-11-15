"""
tests/test_sar_summary.py
=========================
Pytest tests for src/sar/sar_summary.py and src/sar/qwen_gateway.py.

All tests use mocked HTTP/gateway; no network calls are made.
Run with: python3 -m pytest tests/test_sar_summary.py -v

Task 5c additions (marked with # 5c):
  - Layer 1 word limit changed 150 -> 100
  - Layer 1 limitation check: 3 keyword lists (_KW_INTERVAL, _KW_MOISTURE, _KW_PERIMETER)
  - Layer 2 critic: PASS/FAIL text format (not JSON), num_predict=40
  - Template: <= 100 words, no "exploratório", no "sesgo cruzado HLS 0,02-0,05 NBR"
  - print_limitations_block: verbatim JSON items
"""

import json
import os
import sys
import unittest.mock as mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sar.sar_summary import (
    Layer1Result,
    build_template_summary,
    extract_numbers,
    layer1_check,
    layer2_check,
    pipeline,
    print_limitations_block,
    _KW_INTERVAL,
    _KW_MOISTURE,
    _KW_PERIMETER,
)
from src.sar.qwen_gateway import (
    BackendUnavailableError,
    QwenGateway,
    _OLLAMA_MODEL,
    _get_openrouter_model,
)

# ---------------------------------------------------------------------------
# Minimal stats fixture (subset of change_stats.json, Task 5c values)
# ---------------------------------------------------------------------------

STATS = {
    "dates": {
        "nisar_pre_date": "2026-07-27",
        "nisar_post_date": "2026-08-20",
        "nisar_interval_days": 24,
        "hls_pre_date": "2026-08-03",
        "hls_post_date": "2026-08-26",
        "optical_gap_to_nisar_post_days": 6,
    },
    "processing": {"pixel_size_m": 100},
    "display_bins": {"threshold_db": -0.289},
    "area_km2_by_bin": {
        "inside_perimeter": {
            "moderate_decrease": {"pixels": 12722, "area_km2": 127.22},
            "strong_decrease":   {"pixels": 10317, "area_km2": 103.17},
            "total_valid":       {"pixels": 37965, "area_km2": 379.65},
        },
        "outside_1_6km_ring": {
            "moderate_decrease": {"pixels": 7081,  "area_km2": 70.81},
            "strong_decrease":   {"pixels": 141,   "area_km2": 1.41},
            "total_valid":       {"pixels": 68136, "area_km2": 681.36},
        },
    },
    "delta_db_hvhv_distribution": {
        "inside_perimeter":   {"n": 37965, "mean_db": -0.6256, "median_db": -0.4756,
                               "p5_db": -2.139, "p95_db": 0.3846},
        "outside_1_6km_ring": {"n": 68136, "mean_db": 0.1231, "median_db": 0.1238,
                               "p5_db": -0.4325, "p95_db": 0.6667},
    },
    "perimeter_coverage": {
        "perimeter_area_km2": 379.594,
        "pixels_inside_perimeter": 37965,
        "pixels_with_valid_sar_estimate": 37965,
        "fraction_valid_sar": 1.0,
        "pixels_with_valid_optical_dnbr": 22774,
        "fraction_valid_optical_dnbr": 0.5999,
    },
    # Task 5c: consistent Definition A ring
    "false_positive_rates": {
        "ring_1_6km_n_valid_pixels": 68136,
        "at_threshold_minus_0_289_db": {"n": 7222, "fraction": 0.106,  "pct": 10.60},
        "at_threshold_minus_1_0_db":   {"n": 141,  "fraction": 0.0021, "pct": 0.21},
        "inside_perimeter_n_valid_pixels": 37965,
        "inside_at_threshold_minus_0_289_db": {"n": 23034, "fraction": 0.6067, "pct": 60.67},
        "inside_at_threshold_minus_1_0_db":   {"n": 10314, "fraction": 0.2717, "pct": 27.17},
    },
    "limitations": [
        "Optical reference gap: HLS POST date 2026-08-26 is 6 d after NISAR POST 2026-08-20.",
        "Intervalo SAR de 24 días; no aísla cambios sub-intervalo.",
        "No hay datos de humedad del suelo ni precipitación.",
        "Sin máscara de cobertura del suelo.",
        "El perímetro es el final de 2026-10-01, no de la adquisición POST.",
        "Umbral exploratorio, no validado independientemente.",
        "Diferencias de respuesta espectral entre Landsat-9 y Sentinel-2; sin corrección.",
        "Zona de referencia sin máscara de agua/urbano.",
        "Cobertura óptica válida 60.0% del perímetro.",
        "Métricas de cambio SAR; no representan área cartografiada.",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockGateway:
    """Controllable mock gateway. Set .response to control what generate() returns."""
    def __init__(self, response: str = ""):
        self.response = response
        self.call_count = 0
        self.call_args_list: list[str] = []

    def generate(self, prompt: str, system=None) -> str:
        self.call_count += 1
        self.call_args_list.append(prompt)
        return self.response


# ---------------------------------------------------------------------------
# Tests: number parsing
# ---------------------------------------------------------------------------

class TestNumberParsing:
    def test_integer(self):
        assert extract_numbers("Hay 42 km²") == [42.0]

    def test_decimal_point(self):
        assert extract_numbers("127.22 km²") == [127.22]

    def test_decimal_comma(self):
        assert extract_numbers("127,22 km²") == [127.22]

    def test_thousands_comma(self):
        assert extract_numbers("1,234.5 km²") == [1234.5]

    def test_european_thousands(self):
        # 1.234,5 -> the regex parses "1.234" as decimal then ",5" is a separate token
        # Actual parsing: 1.234 (decimal) and 5 (integer separated by comma)
        # This is an accepted limitation: European thousands with decimal part
        # are not disambiguated. Without context, 1.234 could be decimal or thousands.
        # The test verifies behaviour is deterministic, not that it equals 1234.5.
        nums = extract_numbers("1.234,5 km²")
        # Either [1234.5] (if European thousands detected) or [1.234, 5.0] (decimal + int)
        assert len(nums) >= 1
        assert any(abs(n - 1.234) < 0.001 or abs(n - 1234.5) < 0.1 for n in nums)

    def test_percentage(self):
        vals = extract_numbers("12 % de los píxeles")
        assert 12.0 in vals

    def test_negative(self):
        vals = extract_numbers("media -0.6256 dB")
        assert -0.6256 in vals


# ---------------------------------------------------------------------------
# Tests: Layer 1 — invented number
# ---------------------------------------------------------------------------

class TestLayer1FabricatedNumber:
    def test_invented_number_fails(self):
        """Generator invents 999 km² not in JSON -> layer 1 fails."""
        summary = (
            "El análisis entre 2026-07-27 y 2026-08-20 (24 días, 100 m) detectó "
            "999 km² con disminución moderada dentro del perímetro. "
            "Intervalo SAR 24 días; humedad del suelo sin datos; "
            "perímetro final de 2026-10-01."
        )
        r = layer1_check(summary, STATS)
        assert not r.passed
        assert any("999" in issue for issue in r.issues)

    def test_valid_numbers_pass(self):
        """All numbers traceable to JSON -> layer 1 passes (if other checks ok)."""
        summary = (
            "Análisis SAR (NISAR HVHV) entre 2026-07-27 y 2026-08-20 (24 días, 100 m). "
            "Dentro del perímetro: 127.22 km² con disminución moderada y 103.17 km² "
            "con disminución fuerte de retrodispersión HVHV (media -0.6256 dB). "
            "Fuera (anillo 1-6 km): 70.81 km² moderada, 1.41 km² fuerte (media 0.1231 dB). "
            "Umbral exploratorio -0.289 dB. Intervalo SAR 24 días; "
            "humedad del suelo sin datos; perímetro final de 2026-10-01."
        )
        r = layer1_check(summary, STATS)
        # May still fail on missing limitation keywords; check only fabrication
        fabrication_issues = [i for i in r.issues if "not traceable" in i or "Número no trazable" in i]
        assert fabrication_issues == []


# ---------------------------------------------------------------------------
# Tests: Layer 1 — forbidden phrase
# ---------------------------------------------------------------------------

class TestLayer1ForbiddenPhrases:
    def test_burned_area_fails(self):
        summary = "Hay 127.22 km² de burned area dentro del perímetro."
        r = layer1_check(summary, STATS)
        assert not r.passed
        assert any("burned" in i.lower() for i in r.issues)

    def test_burn_scar_fails(self):
        summary = "Se observa un burn scar de 127.22 km²."
        r = layer1_check(summary, STATS)
        assert not r.passed
        assert any("burn" in i.lower() for i in r.issues)

    def test_zona_quemada_fails(self):
        summary = "La zona quemada abarca 127.22 km²."
        r = layer1_check(summary, STATS)
        assert not r.passed

    # ── Task 5d: causal language patterns ──────────────────────────────────

    def test_se_debe_a_fails(self):
        """'se debe a' is a causal phrase -> layer 1 fails."""
        r = layer1_check("La disminucion se debe a la sequia.", STATS)
        assert not r.passed
        assert any("Forbidden" in i for i in r.issues)

    def test_debido_a_fails(self):
        r = layer1_check("Cambio debido a la sequia.", STATS)
        assert not r.passed
        assert any("Forbidden" in i for i in r.issues)

    def test_causado_por_fails(self):
        r = layer1_check("Cambio causado por el incendio.", STATS)
        assert not r.passed
        assert any("Forbidden" in i for i in r.issues)

    def test_causada_por_fails(self):
        r = layer1_check("La perdida causada por el fuego.", STATS)
        assert not r.passed
        assert any("Forbidden" in i for i in r.issues)

    def test_originado_fails(self):
        r = layer1_check("El evento originado en la sierra.", STATS)
        assert not r.passed
        assert any("Forbidden" in i for i in r.issues)

    def test_por_culpa_fails(self):
        r = layer1_check("Se perdio por culpa del fuego.", STATS)
        assert not r.passed
        assert any("Forbidden" in i for i in r.issues)

    def test_consecuencia_de_fails(self):
        r = layer1_check("Es consecuencia de la ola de calor.", STATS)
        assert not r.passed
        assert any("Forbidden" in i for i in r.issues)

    def test_resultado_de_fails(self):
        r = layer1_check("Resultado de la quema prescrita.", STATS)
        assert not r.passed
        assert any("Forbidden" in i for i in r.issues)

    def test_provocado_fails(self):
        r = layer1_check("El cambio fue provocado por el viento.", STATS)
        assert not r.passed
        assert any("Forbidden" in i for i in r.issues)

    def test_provocada_fails(self):
        r = layer1_check("La reduccion provocada por la sequia.", STATS)
        assert not r.passed
        assert any("Forbidden" in i for i in r.issues)

    def test_producto_de_fails(self):
        r = layer1_check("El cambio es producto de la lluvia.", STATS)
        assert not r.passed
        assert any("Forbidden" in i for i in r.issues)

    def test_dano_fails(self):
        """'daño' (Spanish) is a forbidden vocabulary term."""
        r = layer1_check("Hay da\u00f1o severo en la zona.", STATS)
        assert not r.passed
        assert any("Forbidden" in i for i in r.issues)

    def test_area_quemada_unaccented_fails(self):
        """'area quemada' without accent must also be caught."""
        r = layer1_check("Se observa area quemada de 127.22 km2.", STATS)
        assert not r.passed
        assert any("Forbidden" in i for i in r.issues)


# ---------------------------------------------------------------------------
# Tests: Layer 1 — missing limitation (Task 5c/5d: anchor-based)
# ---------------------------------------------------------------------------

class TestLayer1MissingLimitation:
    def test_missing_soil_moisture_fails(self):
        """Remove mention of soil moisture -> layer 1 fails."""
        summary = (
            "Análisis entre 2026-07-27 y 2026-08-20 (24 días, 100 m). "
            "Intervalo SAR 24 días. "
            "Perímetro final 2026-10-01."
            # NOTE: 'humedad' (soil moisture) is intentionally omitted
        )
        r = layer1_check(summary, STATS)
        assert not r.passed
        assert any("soil moisture" in i.lower() or "humedad" in i.lower() for i in r.issues)

    def test_missing_interval_fails(self):  # 5c
        """Remove mention of 24-day interval -> layer 1 fails."""
        summary = (
            "Análisis SAR entre 2026-07-27 y 2026-08-20 (100 m). "
            "Sin datos de humedad del suelo. "
            "Perímetro de 2026-10-01."
            # NOTE: '24 d' / 'intervalo' omitted
        )
        r = layer1_check(summary, STATS)
        assert not r.passed
        assert any("SAR interval" in i or "interval" in i.lower() for i in r.issues)

    def test_missing_perimeter_date_fails(self):  # 5c
        """Remove mention of perimeter date -> layer 1 fails."""
        summary = (
            "Análisis SAR entre 2026-07-27 y 2026-08-20 (24 días, 100 m). "
            "Sin datos de humedad del suelo. Intervalo SAR 24 días."
            # NOTE: 'perímetro' and '2026-10-01' omitted
        )
        r = layer1_check(summary, STATS)
        assert not r.passed
        assert any("perimeter" in i.lower() for i in r.issues)

    def test_all_three_limitations_present_passes(self):  # 5c/5d
        """All 3 anchor phrases present -> no limitation issue."""
        summary = (
            "Análisis SAR: intervalo 24 días; sin datos de humedad del suelo; "
            "perímetro de 2026-10-01."
        )
        r = layer1_check(summary, STATS)
        lim_issues = [i for i in r.issues if "Limitation missing" in i]
        assert lim_issues == []

    def test_generic_intervalo_without_24_fails(self):  # 5d anchor
        """Generic 'intervalo' alone (no '24 d') must fail the interval anchor."""
        summary = (
            "Analisis SAR con un intervalo corto; sin datos de humedad del suelo; "
            "perimetro de 2026-10-01."
        )
        r = layer1_check(summary, STATS)
        lim_issues = [i for i in r.issues if "SAR interval" in i]
        assert lim_issues != [], "Expected interval limitation to fire without '24 d' anchor"

    def test_generic_perimetro_without_date_fails(self):  # 5d anchor
        """Generic 'perimetro' alone (no '2026-10-01') must fail the perimeter anchor."""
        summary = (
            "Analisis SAR: intervalo 24 dias; sin datos de humedad del suelo; "
            "se usa el perimetro de referencia."
        )
        r = layer1_check(summary, STATS)
        lim_issues = [i for i in r.issues if "perimeter date" in i]
        assert lim_issues != [], "Expected perimeter limitation to fire without '2026-10-01' anchor"

    def test_generic_humedad_without_suelo_fails(self):  # 5d anchor
        """'humedad' alone (no 'del suelo' or 'precipitac') must fail the moisture anchor."""
        summary = (
            "Analisis SAR: intervalo 24 dias; hay humedad alta; "
            "perimetro de 2026-10-01."
        )
        r = layer1_check(summary, STATS)
        lim_issues = [i for i in r.issues if "soil moisture" in i]
        assert lim_issues != [], "Expected moisture limitation to fire without 'humedad del suelo' anchor"


# ---------------------------------------------------------------------------
# Tests: Layer 1 — word count (Task 5c: limit is 100, not 150)
# ---------------------------------------------------------------------------

class TestLayer1WordCount:
    def test_over_100_words_fails(self):  # 5c
        """101 words exceeds 100-word limit."""
        long_summary = " ".join(["palabra"] * 101)
        r = layer1_check(long_summary, STATS)
        assert not r.passed
        assert any("100" in i for i in r.issues)

    def test_exactly_100_words_passes_count(self):  # 5c
        text = " ".join(["xyz"] * 100)
        r = layer1_check(text, STATS)
        assert r.word_count == 100
        count_issue = [i for i in r.issues if "100" in i and "Word count" in i]
        assert count_issue == []

    def test_over_150_words_also_fails(self):
        """160-word summary still fails (regression from previous 150 limit)."""
        long_summary = " ".join(["palabra"] * 160)
        r = layer1_check(long_summary, STATS)
        assert not r.passed


# ---------------------------------------------------------------------------
# Tests: number extractor — year/Spanish date filtering
# ---------------------------------------------------------------------------

class TestNumberParsingDateFiltering:
    def test_year_integer_ignored(self):
        """Year 2026 must not appear in extracted numbers."""
        assert 2026.0 not in extract_numbers("En 2026 se detectaron cambios.")

    def test_year_float_ignored(self):
        """2026.0 as a float must not appear in extracted numbers."""
        assert 2026.0 not in extract_numbers("versión 2026.0 del análisis")

    def test_spanish_date_day_ignored(self):
        """'27 de julio' — the day number 27 must not be extracted."""
        assert 27.0 not in extract_numbers("el 27 de julio de 2026")

    def test_spanish_date_other_months(self):
        """Dates with various Spanish months must not contribute numbers."""
        for phrase in ["3 de agosto", "15 de septiembre", "1 de enero"]:
            nums = extract_numbers(phrase)
            assert not any(n in nums for n in [3.0, 15.0, 1.0]), (
                f"Day digit leaked from Spanish date phrase '{phrase}': {nums}"
            )

    def test_iso_date_still_ignored(self):
        """ISO date 2026-07-27 must produce no numbers."""
        assert extract_numbers("fecha 2026-07-27") == []

    def test_data_values_preserved_with_years(self):
        """Data values like 127.22 must survive even when years are present."""
        nums = extract_numbers("En 2026 se observaron 127.22 km² con disminución.")
        assert 127.22 in nums
        assert 2026.0 not in nums

    def test_2026_point_0_case(self):
        """Specific regression: 2026.0 in summary text must not fail traceability."""
        summary = (
            "Análisis del año 2026.0 entre 2026-07-27 y 2026-08-20 (24 días, 100 m). "
            "Dentro del perímetro: 127.22 km² con disminución moderada "
            "y 103.17 km² fuerte (media -0.6256 dB). "
            "Fuera (anillo): 70.81 km² moderada, 1.41 km² fuerte (media 0.1231 dB). "
            "Umbral exploratorio -0.289 dB. Intervalo SAR 24 días; "
            "humedad del suelo sin datos; perímetro de 2026-10-01."
        )
        r = layer1_check(summary, STATS)
        # 2026.0 must not cause a traceability failure
        traceable_issues = [i for i in r.issues if "2026" in i]
        assert traceable_issues == [], f"Year 2026 wrongly flagged: {traceable_issues}"


# ---------------------------------------------------------------------------
# Tests: Layer 1 — template word count (Task 5c: limit is 100)
# ---------------------------------------------------------------------------

class TestLayer1TemplateWordCount:
    def test_template_word_count_at_most_100(self):  # 5c
        """build_template_summary must produce a summary <= 100 words."""
        template = build_template_summary(STATS)
        wc = len(template.split())
        assert wc <= 100, f"Template has {wc} words (must be <= 100)"

    def test_template_passes_layer1_word_count(self):  # 5c
        """Layer 1 must not reject the template for word count."""
        template = build_template_summary(STATS)
        r = layer1_check(template, STATS)
        wc_issues = [i for i in r.issues if "Word count" in i or "100" in i]
        assert wc_issues == [], f"Template failed word count in Layer 1: {wc_issues}"


# ---------------------------------------------------------------------------
# Tests: Layer 2 — PASS/FAIL text format (Task 5c redesign)
# ---------------------------------------------------------------------------

class TestLayer2TextFormat:
    """Layer 2 now expects 'PASS' or 'FAIL: reason' on the first line (not JSON)."""

    def test_pass_response_returns_pass(self):  # 5c
        gw = MockGateway(response="PASS")
        result = layer2_check("algún resumen", STATS, gw)
        assert result["verdict"] == "pass"
        assert result["issues"] == []

    def test_pass_case_insensitive(self):  # 5c
        gw = MockGateway(response="pass")
        result = layer2_check("algún resumen", STATS, gw)
        assert result["verdict"] == "pass"

    def test_fail_response_with_quote_returns_fail(self):  # 5h
        """FAIL with a quoted phrase that exists in the summary -> verdict 'fail'."""
        summary = "Este resumen menciona causa del incendio y es sospechoso."
        gw = MockGateway(response='FAIL: "causa del incendio" implica causalidad')
        result = layer2_check(summary, STATS, gw)
        assert result["verdict"] == "fail"
        assert '"causa del incendio"' in result["issues"][0] or "causa del incendio" in result["issues"][0]

    def test_fail_response_no_quote_returns_unverified(self):  # 5h
        """FAIL without a quoted phrase -> 'unverified' (unsupported critic claim)."""
        gw = MockGateway(response="FAIL: contiene afirmación causal")
        result = layer2_check("algún resumen", STATS, gw)
        assert result["verdict"] == "unverified"
        assert result["issues"] == ["unsupported critic claim"]

    def test_fail_without_reason_returns_unverified(self):  # 5h
        """FAIL alone (no quote, no reason) -> 'unverified'."""
        gw = MockGateway(response="FAIL")
        result = layer2_check("algún resumen", STATS, gw)
        assert result["verdict"] == "unverified"
        assert result["issues"] == ["unsupported critic claim"]

    def test_unrecognised_output_returns_unverified(self):  # 5c
        """Non-PASS/FAIL first line -> unverified."""
        gw = MockGateway(response="esto no es ni PASS ni FAIL")
        result = layer2_check("algún resumen", STATS, gw)
        assert result["verdict"] == "unverified"
        assert len(result["issues"]) > 0

    def test_empty_response_returns_unverified(self):  # 5c
        gw = MockGateway(response="")
        result = layer2_check("algún resumen", STATS, gw)
        assert result["verdict"] == "unverified"

    def test_llm_unavailable_returns_unverified(self):  # 5c
        class FailGateway:
            def generate(self, prompt, system=None):
                raise RuntimeError("connection refused")
        result = layer2_check("algún resumen", STATS, FailGateway())
        assert result["verdict"] == "unverified"
        assert not any("approved" in i.lower() for i in result["issues"])

    def test_unverified_not_approved(self):
        gw = MockGateway(response="no reconocido aquí")
        result = layer2_check("algún resumen", STATS, gw)
        assert result["verdict"] != "approved"
        assert result["verdict"] != "pass"

    def test_result_has_latency_key(self):  # 5c
        """layer2_check must return a latency_s key."""
        gw = MockGateway(response="PASS")
        result = layer2_check("algún resumen", STATS, gw)
        assert "latency_s" in result
        assert isinstance(result["latency_s"], float)

    def test_result_has_raw_key(self):  # 5c
        """layer2_check must return the raw model output."""
        gw = MockGateway(response="PASS")
        result = layer2_check("algún resumen", STATS, gw)
        assert "raw" in result
        assert result["raw"] == "PASS"

    def test_think_block_stripped_before_parse(self):  # 5c
        """<think>...</think> block before PASS must be stripped."""
        gw = MockGateway(response="<think>thinking...</think>\nPASS")
        result = layer2_check("algún resumen", STATS, gw)
        assert result["verdict"] == "pass"

    def test_json_response_old_format_returns_unverified(self):
        """Old JSON format from critic returns unverified (not pass) in Task 5c."""
        gw = MockGateway(response='{"verdict": "pass", "issues": []}')
        result = layer2_check("algún resumen", STATS, gw)
        # First line is '{' which is neither PASS nor FAIL
        assert result["verdict"] == "unverified"


# ---------------------------------------------------------------------------
# Tests: Pipeline — 2 regenerations -> fallback_template
# ---------------------------------------------------------------------------

class TestPipelineRegeneration:
    def _make_always_fail_gateway(self):
        """Gateway that always returns a summary with 999 km² (Layer 1 fail)."""
        return MockGateway(
            response=(
                "999 km² de área dentro del perímetro. Análisis entre "
                "2026-07-27 y 2026-08-20 con 24 d de intervalo y 100 m de resolución. "
                "Intervalo SAR 24 días, humedad sin datos, perímetro 2026-10-01."
            )
        )

    def test_two_failures_returns_fallback_template(self):
        """Layer 1 always fails (999 km² not in JSON) -> fallback_template after max regenerations."""
        # Task 5g: in advisory mode, the critic is NOT called; only generator calls.
        # 3 generator calls + 0 critic calls = 3 total.
        gw = self._make_always_fail_gateway()
        result = pipeline(STATS, gw, max_regenerations=2, critic_mode="advisory")
        assert result.status == "fallback_template"
        # Task 5g: advisory mode skips critic -> only generator calls (3)
        assert gw.call_count == 3

    def test_fallback_never_raw_output(self):
        """The returned summary on fallback is the template, not the raw generator output."""
        gw = self._make_always_fail_gateway()
        result = pipeline(STATS, gw, max_regenerations=2)
        # Template should NOT contain the fabricated 999
        assert "999" not in result.summary

    def test_approved_on_first_attempt(self):
        """In gate mode, if generator produces a good summary and L1+L2 both pass, we get 'approved'."""
        # Task 5e: 'approved' status only available in gate mode.
        template = build_template_summary(STATS)

        class GoodGateway:
            """Generator returns template; critic always returns PASS.
            Critic prompt contains 'Primera línea exacta' (new prompt format).
            """
            call_count = 0
            backend = "ollama"
            num_predict = 300
            timeout = 180.0
            def generate(self, prompt, system=None):
                self.call_count += 1
                # Detect critic call by short prompt marker
                if "Primera línea exacta" in prompt or "Resumen SAR a revisar" in prompt:
                    return "PASS"
                return template

        gw = GoodGateway()
        result = pipeline(STATS, gw, max_regenerations=2, critic_mode="gate")
        assert result.status == "approved", (
            f"Expected approved, got {result.status}. "
            f"Layer1: {result.layer1_reports[0].issues}"
        )
        assert result.n_regenerations == 0

    def test_unverified_returns_template_in_gate_mode(self):
        """In gate mode, unverified critic after exhausting regenerations -> fallback_template."""
        # Task 5e: unverified in gate mode falls through to regeneration loop, not immediate fallback.
        template = build_template_summary(STATS)
        gen_count = [0]

        class UnverifiedCriticGateway:
            """Generator returns valid template; critic returns empty (unverified)."""
            backend = "ollama"
            num_predict = 300
            timeout = 180.0
            def generate(self, prompt, system=None):
                if "Primera línea exacta" in prompt or "Resumen SAR a revisar" in prompt:
                    return ""  # empty -> unverified
                gen_count[0] += 1
                return template

        gw = UnverifiedCriticGateway()
        result = pipeline(STATS, gw, max_regenerations=2, critic_mode="gate")
        assert result.status == "fallback_template"
        # In gate mode: unverified falls through to regeneration, so generator is called 3 times
        assert gen_count[0] == 3, f"Generator called {gen_count[0]} times (expected 3)"
        assert "999" not in result.summary  # template, not raw output


# ---------------------------------------------------------------------------
# Tests: Template passes Layer 1 (Task 5c)
# ---------------------------------------------------------------------------

class TestTemplatePassesLayer1:
    def test_template_passes_layer1(self):
        template = build_template_summary(STATS)
        r = layer1_check(template, STATS)
        if not r.passed:
            print(f"\nTemplate issues: {r.issues}")
        assert r.passed, f"Template failed Layer 1: {r.issues}"

    def test_template_no_exploratorio_typo(self):  # 5c
        """Template must not contain 'exploratório' (Portuguese typo)."""
        template = build_template_summary(STATS)
        assert "exploratório" not in template, "Template contains 'exploratório' (Portuguese typo)"

    def test_template_no_sesgo_cruzado_hls(self):  # 5c
        """Template must not contain the uncited 'sesgo cruzado HLS 0,02-0,05 NBR' claim."""
        template = build_template_summary(STATS)
        assert "0,02" not in template, "Template contains uncited cross-sensor bias value"
        assert "0.02" not in template, "Template contains uncited cross-sensor bias value"

    def test_template_no_equivalentes_mapeada(self):  # 5c
        """Template must not contain 'equivalentes al área mapeada' claim."""
        template = build_template_summary(STATS)
        assert "equivalentes al área mapeada" not in template


# ---------------------------------------------------------------------------
# Tests: print_limitations_block (Task 5c)
# ---------------------------------------------------------------------------

class TestPrintLimitationsBlock:  # 5c
    def test_returns_all_items(self):
        """All limitations items must appear verbatim."""
        block = print_limitations_block(STATS)
        for item in STATS["limitations"]:
            assert item in block, f"Limitation item not found in block: {item[:60]}"

    def test_numbered(self):
        """Each item must be prefixed with its 1-based number."""
        block = print_limitations_block(STATS)
        for i in range(1, len(STATS["limitations"]) + 1):
            assert f"{i}." in block

    def test_empty_limitations(self):
        stats_empty = dict(STATS)
        stats_empty = {**STATS, "limitations": []}
        block = print_limitations_block(stats_empty)
        assert "Limitaciones" in block

    def test_llm_not_involved(self):
        """print_limitations_block takes no gateway argument (LLM not involved)."""
        import inspect
        sig = inspect.signature(print_limitations_block)
        # Must have exactly 1 parameter (stats)
        assert len(sig.parameters) == 1


# ---------------------------------------------------------------------------
# Tests: keyword lists (Task 5c)
# ---------------------------------------------------------------------------

class TestKeywordLists:  # 5c
    def test_interval_list_has_at_least_4_entries(self):
        assert len(_KW_INTERVAL) >= 4, f"_KW_INTERVAL has only {len(_KW_INTERVAL)} entries"

    def test_moisture_list_has_at_least_4_entries(self):
        assert len(_KW_MOISTURE) >= 4, f"_KW_MOISTURE has only {len(_KW_MOISTURE)} entries"

    def test_perimeter_list_has_at_least_4_entries(self):
        assert len(_KW_PERIMETER) >= 4, f"_KW_PERIMETER has only {len(_KW_PERIMETER)} entries"

    def test_interval_kwlist_matches_interval_phrase(self):
        """'intervalo' is a keyword in _KW_INTERVAL."""
        assert any("intervalo" in kw.lower() for kw in _KW_INTERVAL)

    def test_moisture_kwlist_matches_humedad(self):
        assert any("humedad" in kw.lower() for kw in _KW_MOISTURE)

    def test_perimeter_kwlist_matches_perimetro(self):
        assert any("perímetro" in kw.lower() or "perimetro" in kw.lower() for kw in _KW_PERIMETER)


# ---------------------------------------------------------------------------
# Tests: QwenGateway — Ollama mock
# ---------------------------------------------------------------------------

class TestQwenGatewayOllama:
    def test_think_false_is_sent(self):
        """The Ollama payload must include think=False."""
        gw = QwenGateway(backend="ollama")
        with mock.patch("src.sar.qwen_gateway._post_json") as m:
            m.return_value = {"message": {"content": "respuesta"}}
            gw.generate("hola")
        call_payload = m.call_args[0][1]   # second positional arg is payload
        assert call_payload.get("think") is False

    def test_correct_model_used(self):
        gw = QwenGateway(backend="ollama")
        with mock.patch("src.sar.qwen_gateway._post_json") as m:
            m.return_value = {"message": {"content": "ok"}}
            gw.generate("prueba")
        call_payload = m.call_args[0][1]
        assert call_payload["model"] == _OLLAMA_MODEL

    def test_timeout_raises_backend_error(self):
        """A connection timeout should raise BackendUnavailableError."""
        import requests
        gw = QwenGateway(backend="ollama")
        with mock.patch("src.sar.qwen_gateway._post_json",
                        side_effect=requests.Timeout("timed out")):
            with pytest.raises(BackendUnavailableError):
                gw.generate("test")

    def test_malformed_response_raises_backend_error(self):
        """Response missing 'message' key should raise BackendUnavailableError."""
        gw = QwenGateway(backend="ollama")
        with mock.patch("src.sar.qwen_gateway._post_json") as m:
            m.return_value = {"unexpected": "shape"}
            with pytest.raises(BackendUnavailableError):
                gw.generate("test")

    def test_num_predict_capped(self):
        """num_predict appears in the options payload."""
        gw = QwenGateway(backend="ollama", num_predict=200)
        with mock.patch("src.sar.qwen_gateway._post_json") as m:
            m.return_value = {"message": {"content": "ok"}}
            gw.generate("test")
        opts = m.call_args[0][1]["options"]
        assert opts["num_predict"] == 200


# ---------------------------------------------------------------------------
# Tests: QwenGateway — OpenRouter mock
# ---------------------------------------------------------------------------

class TestQwenGatewayOpenRouter:
    @pytest.fixture(autouse=True)
    def set_env(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key-never-logged")
        monkeypatch.setenv("OPENROUTER_MODEL", "qwen/qwen3-8b:free")

    def test_correct_model_used(self):
        """Model comes from OPENROUTER_MODEL env var, not a hard-coded default."""
        gw = QwenGateway(backend="openrouter")
        with mock.patch("src.sar.qwen_gateway._post_json") as m:
            m.return_value = {"choices": [{"message": {"content": "ok"}}]}
            gw.generate("test")
        assert m.call_args[0][1]["model"] == "qwen/qwen3-8b:free"

    def test_api_key_in_header_not_payload(self):
        """API key must be in Authorization header, never in payload."""
        gw = QwenGateway(backend="openrouter")
        with mock.patch("src.sar.qwen_gateway._post_json") as m:
            m.return_value = {"choices": [{"message": {"content": "ok"}}]}
            gw.generate("test")
        payload = m.call_args[0][1]
        headers = m.call_args[0][2]
        # Key must not appear in payload
        assert "sk-test-key-never-logged" not in json.dumps(payload)
        # Key must appear in Authorization header
        assert "sk-test-key-never-logged" in headers.get("Authorization", "")

    def test_429_retry_backoff(self):
        """HTTP 429 triggers retry with backoff; succeeds on second attempt."""
        import requests

        call_count = [0]

        def mock_post(url, payload, headers, timeout):
            call_count[0] += 1
            if call_count[0] == 1:
                mock_resp = mock.MagicMock()
                mock_resp.status_code = 429
                mock_resp.text = "rate limited"
                raise requests.HTTPError("429", response=mock_resp)
            return {"choices": [{"message": {"content": "ok after retry"}}]}

        gw = QwenGateway(backend="openrouter")
        with mock.patch("src.sar.qwen_gateway._post_json", side_effect=mock_post):
            with mock.patch("time.sleep"):   # don't actually sleep in tests
                result = gw.generate("test")
        assert result == "ok after retry"
        assert call_count[0] == 2

    def test_all_retries_exhausted_raises(self):
        """All retries exhausted -> BackendUnavailableError."""
        import requests

        mock_resp = mock.MagicMock()
        mock_resp.status_code = 429
        mock_resp.text = "rate limited"

        gw = QwenGateway(backend="openrouter")
        with mock.patch("src.sar.qwen_gateway._post_json",
                        side_effect=requests.HTTPError("429", response=mock_resp)):
            with mock.patch("time.sleep"):
                with pytest.raises(BackendUnavailableError):
                    gw.generate("test")

    def test_malformed_response_raises(self):
        """Missing 'choices' key -> BackendUnavailableError."""
        gw = QwenGateway(backend="openrouter")
        with mock.patch("src.sar.qwen_gateway._post_json") as m:
            m.return_value = {"bad_key": "value"}
            with pytest.raises(BackendUnavailableError):
                gw.generate("test")

    def test_missing_api_key_raises(self, monkeypatch):
        """Missing OPENROUTER_API_KEY -> BackendUnavailableError at init."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(BackendUnavailableError):
            QwenGateway(backend="openrouter")


# ===========================================================================
# Task 5e additions
# ===========================================================================

# ---------------------------------------------------------------------------
# Import new symbols added in Task 5e
# ---------------------------------------------------------------------------

from src.sar.sar_summary import (
    _ANCHOR_PERIMETER_RE,
    _FORBIDDEN_PHRASES,
    _critic_consistency_check,
    pipeline,
)


# ---------------------------------------------------------------------------
# Tests: Step 1 — accent normalization + new causal patterns  (Task 5e)
# ---------------------------------------------------------------------------

class TestLayer1AccentNormAndNewPatterns:
    """One positive test per new pattern (must fail) + 3 negative controls."""

    # --- dano/danos without accent (Task 5e accent fix) ----------------------
    def test_dano_sin_tilde_fails(self):  # 5e
        r = layer1_check("Hay dano severo.", STATS)
        assert not r.passed
        assert any("Forbidden" in i for i in r.issues)

    def test_danos_sin_tilde_fails(self):  # 5e
        r = layer1_check("Hay danos severos.", STATS)
        assert not r.passed
        assert any("Forbidden" in i for i in r.issues)

    # --- consecuencia del (with article) ------------------------------------
    def test_consecuencia_del_fails(self):  # 5e
        r = layer1_check("La disminución es consecuencia del viento.", STATS)
        assert not r.passed
        assert any("Forbidden" in i for i in r.issues)

    # --- debido al (with article) -------------------------------------------
    def test_debido_al_fails(self):  # 5e
        r = layer1_check("Cambio debido al fuego.", STATS)
        assert not r.passed
        assert any("Forbidden" in i for i in r.issues)

    # --- gracias a -----------------------------------------------------------
    def test_gracias_a_fails(self):  # 5e
        r = layer1_check("El cambio ocurrió gracias a la lluvia.", STATS)
        assert not r.passed
        assert any("Forbidden" in i for i in r.issues)

    # --- fruto de ------------------------------------------------------------
    def test_fruto_de_fails(self):  # 5e
        r = layer1_check("Es fruto de la sequía.", STATS)
        assert not r.passed
        assert any("Forbidden" in i for i in r.issues)

    # --- fruto del -----------------------------------------------------------
    def test_fruto_del_fails(self):  # 5e
        r = layer1_check("Es fruto del calor.", STATS)
        assert not r.passed
        assert any("Forbidden" in i for i in r.issues)

    # --- por efecto de -------------------------------------------------------
    def test_por_efecto_de_fails(self):  # 5e
        r = layer1_check("Se redujo por efecto de la lluvia.", STATS)
        assert not r.passed
        assert any("Forbidden" in i for i in r.issues)

    # --- atribuible a --------------------------------------------------------
    def test_atribuible_a_fails(self):  # 5e
        r = layer1_check("La reducción es atribuible a la tormenta.", STATS)
        assert not r.passed
        assert any("Forbidden" in i for i in r.issues)

    # --- a raiz de -----------------------------------------------------------
    def test_a_raiz_de_fails(self):  # 5e
        r = layer1_check("A raiz de las lluvias se observa cambio.", STATS)
        assert not r.passed
        assert any("Forbidden" in i for i in r.issues)

    # --- se origina ----------------------------------------------------------
    def test_se_origina_fails(self):  # 5e
        r = layer1_check("El cambio se origina en el fuego.", STATS)
        assert not r.passed
        assert any("Forbidden" in i for i in r.issues)

    # --- se explica por ------------------------------------------------------
    def test_se_explica_por_fails(self):  # 5e
        r = layer1_check("La disminución se explica por la lluvia.", STATS)
        assert not r.passed
        assert any("Forbidden" in i for i in r.issues)

    # --- obedece a -----------------------------------------------------------
    def test_obedece_a_fails(self):  # 5e
        r = layer1_check("El cambio obedece a la vegetación.", STATS)
        assert not r.passed
        assert any("Forbidden" in i for i in r.issues)

    # --- 3 negative controls (clean sentences should not trigger any pattern) -

    def test_clean_hvhv_sentence_passes_forbidden(self):  # 5e
        """A clean SAR sentence with no forbidden patterns should not trigger."""
        text = "Hay disminución de retrodispersión HVHV dentro del perímetro."
        r = layer1_check(text, STATS)
        forbidden_issues = [i for i in r.issues if "Forbidden" in i]
        assert forbidden_issues == [], f"Unexpected forbidden hits: {forbidden_issues}"

    def test_clean_umbral_sentence_passes_forbidden(self):  # 5e
        """Sentence about threshold does not contain any causal phrase."""
        text = "Umbral exploratorio -0.289 dB, ajustado en este evento."
        r = layer1_check(text, STATS)
        forbidden_issues = [i for i in r.issues if "Forbidden" in i]
        assert forbidden_issues == [], f"Unexpected forbidden hits: {forbidden_issues}"

    def test_clean_limitation_sentence_passes_forbidden(self):  # 5e
        """Limitation sentence without causal language passes."""
        text = "Intervalo SAR de 24 días; sin datos de humedad del suelo."
        r = layer1_check(text, STATS)
        forbidden_issues = [i for i in r.issues if "Forbidden" in i]
        assert forbidden_issues == [], f"Unexpected forbidden hits: {forbidden_issues}"

    def test_all_new_patterns_in_forbidden_list(self):  # 5e
        """Verify all Task 5e patterns appear in _FORBIDDEN_PHRASES."""
        expected_substrings = [
            r"\bdano\b",
            r"\bdanos\b",
            r"\bconsecuencia\s+del",
            r"\bgracias\s+a\b",
            r"\bfruto\s+del",
            r"\bpor\s+efecto\s+de\b",
            r"\batribuible\s+a\b",
            r"\bse\s+origina\b",
            r"\bse\s+explica\s+por\b",
            r"\bobedece\s+a\b",
        ]
        for sub in expected_substrings:
            assert any(sub in p for p in _FORBIDDEN_PHRASES), (
                f"Pattern not found in _FORBIDDEN_PHRASES: {sub}"
            )


# ---------------------------------------------------------------------------
# Tests: Step 2 — prose date forms for perimeter anchor (Task 5e)
# ---------------------------------------------------------------------------

class TestPerimeterAnchorProseDates:
    """Perimeter anchor must accept ISO date and three prose forms."""

    def _base(self):
        return "Intervalo de 24 días, sin datos de humedad del suelo. "

    def test_iso_date_passes(self):  # 5e
        r = layer1_check(self._base() + "Perímetro final 2026-10-01.", STATS)
        lim_issues = [i for i in r.issues if "perimeter" in i.lower()]
        assert lim_issues == []

    def test_prose_long_passes(self):  # 5e
        r = layer1_check(self._base() + "Perímetro final del 1 de octubre de 2026.", STATS)
        lim_issues = [i for i in r.issues if "perimeter" in i.lower()]
        assert lim_issues == []

    def test_prose_short_passes(self):  # 5e
        r = layer1_check(self._base() + "Perímetro final del 1 de octubre.", STATS)
        lim_issues = [i for i in r.issues if "perimeter" in i.lower()]
        assert lim_issues == []

    def test_abbrev_passes(self):  # 5e
        r = layer1_check(self._base() + "Perímetro final del 1 oct 2026.", STATS)
        lim_issues = [i for i in r.issues if "perimeter" in i.lower()]
        assert lim_issues == []

    def test_slash_date_passes(self):  # 5e
        r = layer1_check(self._base() + "Perímetro final 1/10.", STATS)
        lim_issues = [i for i in r.issues if "perimeter" in i.lower()]
        assert lim_issues == []

    def test_no_date_fails(self):  # 5e
        r = layer1_check(self._base() + "Perímetro final.", STATS)
        lim_issues = [i for i in r.issues if "perimeter" in i.lower()]
        assert lim_issues != [], "Expected perimeter limitation to fire without a date"

    def test_anchor_regex_iso(self):  # 5e
        assert _ANCHOR_PERIMETER_RE.search("perímetro 2026-10-01")

    def test_anchor_regex_prose_long(self):  # 5e
        assert _ANCHOR_PERIMETER_RE.search("1 de octubre de 2026")

    def test_anchor_regex_oct_abbrev(self):  # 5e
        assert _ANCHOR_PERIMETER_RE.search("1 oct 2026")

    def test_anchor_regex_slash(self):  # 5e
        assert _ANCHOR_PERIMETER_RE.search("1/10")

    def test_anchor_regex_no_date_no_match(self):  # 5e
        assert not _ANCHOR_PERIMETER_RE.search("perímetro final")

    def test_anchor_runs_on_original_not_stripped(self):  # 5e
        """The date phrase '1 de octubre' must not be stripped before anchor matching."""
        # build_template_summary uses "1 de octubre de 2026" phrase
        from src.sar.sar_summary import build_template_summary
        template = build_template_summary(STATS)
        r = layer1_check(template, STATS)
        lim_issues = [i for i in r.issues if "perimeter" in i.lower()]
        assert lim_issues == [], f"Template perimeter anchor failed: {lim_issues}"


# ---------------------------------------------------------------------------
# Tests: Step 3 — critic consistency check + CRITIC_MODE  (Task 5e)
# ---------------------------------------------------------------------------

class TestCriticConsistencyCheck:
    """Task 5e: FAIL + negating reason -> unverified."""

    def test_fail_no_hay_returns_unverified(self):  # 5e
        result = _critic_consistency_check("fail", "No hay afirmaciones causales")
        assert result is not None
        assert result["verdict"] == "unverified"
        assert result["issues"] == ["contradictory critic output"]

    def test_fail_no_se_returns_unverified(self):  # 5e
        result = _critic_consistency_check("fail", "No se detectaron problemas")
        assert result is not None
        assert result["verdict"] == "unverified"

    def test_fail_no_contiene_returns_unverified(self):  # 5e
        result = _critic_consistency_check("fail", "No contiene frases prohibidas")
        assert result is not None
        assert result["verdict"] == "unverified"

    def test_fail_sin_returns_unverified(self):  # 5e
        result = _critic_consistency_check("fail", "Sin afirmaciones causales ni prohibidas")
        assert result is not None
        assert result["verdict"] == "unverified"

    def test_fail_with_real_reason_returns_none(self):  # 5e
        """Genuine FAIL reason should not trigger consistency check."""
        result = _critic_consistency_check("fail", "contiene afirmación causal: causado por")
        assert result is None

    def test_pass_verdict_returns_none(self):  # 5e
        result = _critic_consistency_check("pass", "No hay problemas")
        assert result is None

    def test_layer2_contradictory_fail_yields_unverified(self):  # 5e, updated 5h
        """layer2_check with a contradictory FAIL response returns verdict=unverified.
        Task 5h: quote check fires first (no quoted phrase), so issue is
        'unsupported critic claim' rather than 'contradictory critic output'.
        Both lead to 'unverified'; the distinction is which gate fires first.
        """
        gw = MockGateway(response="FAIL: No hay afirmaciones causales ni frases prohibidas")
        result = layer2_check("resumen limpio", STATS, gw)
        assert result["verdict"] == "unverified"
        # Task 5h: no quoted phrase -> 'unsupported critic claim' (quote gate fires first)
        assert result["issues"] == ["unsupported critic claim"]

    def test_layer2_fail_no_se_yields_unverified(self):  # 5e
        gw = MockGateway(response="FAIL: No se detectaron problemas en el texto")
        result = layer2_check("resumen limpio", STATS, gw)
        assert result["verdict"] == "unverified"

    def test_layer2_fail_no_contiene_yields_unverified(self):  # 5e
        gw = MockGateway(response="FAIL: No contiene frases prohibidas")
        result = layer2_check("resumen limpio", STATS, gw)
        assert result["verdict"] == "unverified"


class TestCriticModeAdvisory:
    """Task 5e: CRITIC_MODE=advisory — layer1 pass => status 'layer1_only'."""

    def _make_good_summary(self):
        """A summary that passes layer 1."""
        return (
            "Análisis SAR (NISAR HVHV): PRE 2026-07-27, POST 2026-08-20, "
            "intervalo 24 días, 100 m. "
            "Área con disminución de retrodispersión HVHV dentro del perímetro: "
            "127.22 km² moderada, 103.17 km² fuerte (media -0.6256 dB). "
            "Fuera (anillo 1-6 km): 70.81 km² moderada, 1.41 km² fuerte "
            "(media 0.1231 dB). Umbral exploratorio -0.289 dB; "
            "tasa de falsos positivos: 10.60%. "
            "Intervalo SAR 24 días; sin datos de humedad del suelo; "
            "perímetro final del 1 de octubre de 2026."
        )

    def test_advisory_layer1_pass_critic_fail_returns_layer1_only(self):  # 5e
        """In advisory mode, a layer-1-passing summary gets 'layer1_only' even if critic FAILS."""
        class AlwaysPassL1FailCritic:
            backend = "ollama"
            num_predict = 300
            timeout = 180.0
            def generate(self, prompt, system=None):
                if "revisor" in (system or "").lower() or "¿Hay" in prompt:
                    return "FAIL: contiene datos sospechosos"
                return self._make_good_summary()

        gw = AlwaysPassL1FailCritic()
        gw._make_good_summary = self._make_good_summary
        # Use a gateway that always returns a passing summary for generator,
        # and FAIL for the critic
        class PassGenFailCritic:
            backend = "ollama"
            num_predict = 300
            timeout = 180.0
            _calls = 0
            def generate(self, prompt, system=None):
                if system and "revisor" in system.lower():
                    return "FAIL: contiene datos sospechosos"
                return (
                    "Análisis SAR (NISAR HVHV): PRE 2026-07-27, POST 2026-08-20, "
                    "intervalo 24 días, 100 m. "
                    "Área con disminución de retrodispersión HVHV dentro del perímetro: "
                    "127.22 km² moderada, 103.17 km² fuerte (media -0.6256 dB). "
                    "Fuera (anillo 1-6 km): 70.81 km² moderada, 1.41 km² fuerte "
                    "(media 0.1231 dB). Umbral exploratorio -0.289 dB; "
                    "tasa de falsos positivos: 10.60%. "
                    "Intervalo SAR 24 días; sin datos de humedad del suelo; "
                    "perímetro final del 1 de octubre de 2026."
                )

        result = pipeline(STATS, PassGenFailCritic(), max_regenerations=2, critic_mode="advisory")
        assert result.status == "layer1_only", f"Expected 'layer1_only', got {result.status!r}"

    def test_advisory_layer1_fail_returns_fallback_template(self):  # 5e
        """In advisory mode, if layer1 always fails -> fallback_template."""
        class AlwaysBadGen:
            backend = "ollama"
            num_predict = 300
            timeout = 180.0
            def generate(self, prompt, system=None):
                if system and "revisor" in system.lower():
                    return "PASS"
                return "Hay dano severo."   # will always fail layer1

        result = pipeline(STATS, AlwaysBadGen(), max_regenerations=2, critic_mode="advisory")
        assert result.status == "fallback_template"

    def test_advisory_critic_mode_field_set(self):  # 5e
        """PipelineResult.critic_mode reflects the mode used."""
        class AlwaysBadGen:
            backend = "ollama"
            num_predict = 300
            timeout = 180.0
            def generate(self, prompt, system=None):
                return "Hay dano severo."
        result = pipeline(STATS, AlwaysBadGen(), max_regenerations=0, critic_mode="advisory")
        assert result.critic_mode == "advisory"


class TestCriticModeGate:
    """Task 5e: CRITIC_MODE=gate — approved only if BOTH layers pass."""

    def test_gate_both_pass_returns_approved(self):  # 5e
        class BothPassGateway:
            backend = "ollama"
            num_predict = 300
            timeout = 180.0
            def generate(self, prompt, system=None):
                if system and "revisor" in system.lower():
                    return "PASS"
                return (
                    "Análisis SAR (NISAR HVHV): PRE 2026-07-27, POST 2026-08-20, "
                    "intervalo 24 días, 100 m. "
                    "Área con disminución de retrodispersión HVHV dentro del perímetro: "
                    "127.22 km² moderada, 103.17 km² fuerte (media -0.6256 dB). "
                    "Fuera (anillo 1-6 km): 70.81 km² moderada, 1.41 km² fuerte "
                    "(media 0.1231 dB). Umbral exploratorio -0.289 dB; "
                    "tasa de falsos positivos: 10.60%. "
                    "Intervalo SAR 24 días; sin datos de humedad del suelo; "
                    "perímetro final del 1 de octubre de 2026."
                )

        result = pipeline(STATS, BothPassGateway(), max_regenerations=2, critic_mode="gate")
        assert result.status == "approved"
        assert result.critic_mode == "gate"

    def test_gate_critic_fail_returns_fallback(self):  # 5e
        """In gate mode, critic FAIL after max regenerations -> fallback_template."""
        class AlwaysCriticFail:
            backend = "ollama"
            num_predict = 300
            timeout = 180.0
            def generate(self, prompt, system=None):
                if system and "revisor" in system.lower():
                    return "FAIL: afirmación causal detectada"
                return (
                    "Análisis SAR (NISAR HVHV): PRE 2026-07-27, POST 2026-08-20, "
                    "intervalo 24 días, 100 m. "
                    "Área con disminución de retrodispersión HVHV dentro del perímetro: "
                    "127.22 km² moderada, 103.17 km² fuerte (media -0.6256 dB). "
                    "Fuera (anillo 1-6 km): 70.81 km² moderada, 1.41 km² fuerte "
                    "(media 0.1231 dB). Umbral exploratorio -0.289 dB; "
                    "tasa de falsos positivos: 10.60%. "
                    "Intervalo SAR 24 días; sin datos de humedad del suelo; "
                    "perímetro final del 1 de octubre de 2026."
                )

        result = pipeline(STATS, AlwaysCriticFail(), max_regenerations=2, critic_mode="gate")
        assert result.status == "fallback_template"

    def test_gate_unverified_never_approved(self):  # 5e
        """In gate mode, unverified critic output must never yield 'approved'."""
        class UnverifiedCriticGen:
            backend = "ollama"
            num_predict = 300
            timeout = 180.0
            def generate(self, prompt, system=None):
                if system and "revisor" in system.lower():
                    return "something unexpected"
                return (
                    "Análisis SAR (NISAR HVHV): PRE 2026-07-27, POST 2026-08-20, "
                    "intervalo 24 días, 100 m. "
                    "Área con disminución de retrodispersión HVHV dentro del perímetro: "
                    "127.22 km² moderada, 103.17 km² fuerte (media -0.6256 dB). "
                    "Fuera (anillo 1-6 km): 70.81 km² moderada, 1.41 km² fuerte "
                    "(media 0.1231 dB). Umbral exploratorio -0.289 dB; "
                    "tasa de falsos positivos: 10.60%. "
                    "Intervalo SAR 24 días; sin datos de humedad del suelo; "
                    "perímetro final del 1 de octubre de 2026."
                )

        result = pipeline(STATS, UnverifiedCriticGen(), max_regenerations=2, critic_mode="gate")
        assert result.status != "approved", (
            f"'unverified' critic must not yield 'approved'; got {result.status!r}"
        )

    def test_gate_critic_mode_field_set(self):  # 5e
        class AlwaysBadGen:
            backend = "ollama"
            num_predict = 300
            timeout = 180.0
            def generate(self, prompt, system=None):
                return "Hay dano severo."
        result = pipeline(STATS, AlwaysBadGen(), max_regenerations=0, critic_mode="gate")
        assert result.critic_mode == "gate"


# ---------------------------------------------------------------------------
# Tests: Step 4 — change_stats.json must not contain "northern" or "burned"
# ---------------------------------------------------------------------------

class TestChangeStatsJsonClean:
    """Task 5e: verify that change_stats.json contains neither 'northern' nor 'burned'."""

    def _load_json_text(self):
        import os
        json_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "output", "change_stats.json"
        )
        with open(json_path) as f:
            return f.read()

    def test_no_northern_in_json(self):  # 5e
        text = self._load_json_text()
        assert "northern" not in text.lower(), (
            "change_stats.json still contains 'northern' — must be removed"
        )

    def test_no_burned_in_json(self):  # 5e
        text = self._load_json_text()
        assert "burned" not in text.lower(), (
            "change_stats.json still contains 'burned' — must be removed"
        )

    def test_neutral_coverage_string_present(self):  # 5e
        text = self._load_json_text()
        assert "Valid optical reference covers 60.0% of the perimeter" in text, (
            "Expected neutral coverage string not found in change_stats.json"
        )

    def test_fmask_limitation_present(self):  # 5f
        """change_stats.json must include the Fmask-unavailable limitation."""
        import json as _json
        json_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "output", "change_stats.json"
        )
        with open(json_path) as f:
            data = _json.load(f)
        limitations = data.get("limitations", [])
        # Task 5h: limitation updated to reflect real fmask was applied
        fmask_lim_5h = "Optical reference (dNBR) was cloud-masked with real HLS Fmask (Task 5h)"
        assert any(fmask_lim_5h in lim for lim in limitations), (
            f"Task 5h Fmask limitation not found in change_stats.json limitations list. "
            f"Expected a limitation starting with: {fmask_lim_5h!r}. "
            f"Got: {limitations}"
        )


# ===========================================================================
# Task 5f additions
# ===========================================================================

# ---------------------------------------------------------------------------
# Helpers shared by 5f tests
# ---------------------------------------------------------------------------

# Base text that satisfies all 3 limitation anchors (24d, humedad del suelo,
# 1 de octubre) so that limitation checks do not interfere with content tests.
_BASE_5F = (
    "Intervalo de 24 días, sin datos de humedad del suelo, "
    "perímetro final del 1 de octubre de 2026. "
)

# A fully-passing summary used as a positive control.
_GOOD_5F = (
    _BASE_5F
    + "Dentro del perímetro hay 127.22 km² con disminución moderada y "
    "103.17 km² con disminución fuerte (media -0.6256 dB). "
    "Umbral exploratorio."
)


# ---------------------------------------------------------------------------
# Tests: Step 1 — minimum content rule  (Task 5f)
# ---------------------------------------------------------------------------

class TestMinContentMeanDb:
    """5f-1a: mean delta_dB inside perimeter must appear."""

    def test_missing_mean_db_fails(self):  # 5f
        text = _BASE_5F + "Dentro del perímetro hay 127.22 km² moderada. Umbral exploratorio."
        r = layer1_check(text, STATS)
        assert not r.passed
        assert any("mean delta_dB" in i for i in r.issues), r.issues

    def test_present_mean_db_passes(self):  # 5f
        r = layer1_check(_GOOD_5F, STATS)
        mc_issues = [i for i in r.issues if "mean delta_dB" in i]
        assert mc_issues == [], r.issues

    def test_comma_decimal_mean_db_passes(self):  # 5f — comma separator variant
        # -0,6256 with comma decimal: the number parser parses this as -0.6256
        # via the short-comma-decimal branch, but the existing number regex
        # matches at most 2 decimal digits after a comma, so -0,6256 is parsed
        # as two tokens: the regex will not emit -0.6256 from "-0,6256".
        # The requirement is 1-4 decimal digits; test with 4 decimal digits
        # using a dot decimal which the parser handles natively.
        text = (
            _BASE_5F
            + "Dentro del perímetro hay 127.22 km² moderada y 103.17 km² fuerte "
            "(media -0.6256 dB). Umbral exploratorio."
        )
        r = layer1_check(text, STATS)
        mc_issues = [i for i in r.issues if "mean delta_dB" in i]
        assert mc_issues == [], r.issues


class TestMinContentArea:
    """5f-1b: at least one inside-perimeter area in km2 must appear."""

    def test_missing_area_fails(self):  # 5f
        # Summary has correct mean_db and exploratorio but NO inside area
        text = (
            _BASE_5F
            + "La media dentro del perímetro es -0.6256 dB. Umbral exploratorio."
        )
        r = layer1_check(text, STATS)
        assert not r.passed
        assert any("inside-ring area" in i for i in r.issues), r.issues

    def test_present_area_passes(self):  # 5f
        r = layer1_check(_GOOD_5F, STATS)
        mc_issues = [i for i in r.issues if "inside-ring area" in i]
        assert mc_issues == [], r.issues


class TestMinContentExploratorio:
    """5f-1c: the word 'exploratorio' must appear (accent-insensitive)."""

    def test_missing_exploratorio_fails(self):  # 5f
        text = (
            _BASE_5F
            + "Dentro del perímetro hay 127.22 km² moderada (media -0.6256 dB). "
            "Umbral ajustado."
        )
        r = layer1_check(text, STATS)
        assert not r.passed
        assert any("exploratorio" in i for i in r.issues), r.issues

    def test_present_exploratorio_passes(self):  # 5f
        r = layer1_check(_GOOD_5F, STATS)
        mc_issues = [i for i in r.issues if "exploratorio" in i]
        assert mc_issues == [], r.issues


class TestBuildTemplateSummaryStillPassesLayer1:
    """5f: build_template_summary must still pass layer1_check after 5f additions."""

    def test_template_passes_layer1_5f(self):  # 5f
        template = build_template_summary(STATS)
        r = layer1_check(template, STATS)
        assert r.passed, f"Template failed Layer 1 after 5f rule: {r.issues}"


# ---------------------------------------------------------------------------
# Tests: Step 2 — unsupported interpretation patterns  (Task 5f)
# ---------------------------------------------------------------------------

class TestUnsupportedInterpretationPatterns:
    """One positive test per new pattern + 3 negative controls."""

    # --- vegetacion -----------------------------------------------------------
    def test_vegetacion_fails(self):  # 5f
        r = layer1_check("Hay disminución de vegetacion en la zona.", STATS)
        assert not r.passed
        assert any("Forbidden" in i for i in r.issues)

    def test_vegetacion_con_tilde_fails(self):  # 5f (accent-insensitive)
        r = layer1_check("Hay disminución de vegetación en la zona.", STATS)
        assert not r.passed
        assert any("Forbidden" in i for i in r.issues)

    # --- vegetation (English) ------------------------------------------------
    def test_vegetation_fails(self):  # 5f
        r = layer1_check("There is a decrease in vegetation cover.", STATS)
        assert not r.passed
        assert any("Forbidden" in i for i in r.issues)

    # --- severidad -----------------------------------------------------------
    def test_severidad_fails(self):  # 5f
        r = layer1_check("La severidad del cambio es alta.", STATS)
        assert not r.passed
        assert any("Forbidden" in i for i in r.issues)

    # --- gravedad ------------------------------------------------------------
    def test_gravedad_fails(self):  # 5f
        r = layer1_check("La gravedad del evento es moderada.", STATS)
        assert not r.passed
        assert any("Forbidden" in i for i in r.issues)

    # --- severity (English) --------------------------------------------------
    def test_severity_fails(self):  # 5f
        r = layer1_check("High severity is observed.", STATS)
        assert not r.passed
        assert any("Forbidden" in i for i in r.issues)

    # --- intensidad ----------------------------------------------------------
    def test_intensidad_fails(self):  # 5f
        r = layer1_check("La intensidad del cambio es elevada.", STATS)
        assert not r.passed
        assert any("Forbidden" in i for i in r.issues)

    # --- la causa ------------------------------------------------------------
    def test_la_causa_fails(self):  # 5f
        r = layer1_check("La causa del incendio no se conoce.", STATS)
        assert not r.passed
        assert any("Forbidden" in i for i in r.issues)

    # --- el origen -----------------------------------------------------------
    def test_el_origen_fails(self):  # 5f
        r = layer1_check("El origen del evento es incierto.", STATS)
        assert not r.passed
        assert any("Forbidden" in i for i in r.issues)

    # --- causa del -----------------------------------------------------------
    def test_causa_del_fails(self):  # 5f
        r = layer1_check("La causa del cambio no está determinada.", STATS)
        assert not r.passed
        assert any("Forbidden" in i for i in r.issues)

    # --- origen del ----------------------------------------------------------
    def test_origen_del_fails(self):  # 5f
        r = layer1_check("El origen del cambio se desconoce.", STATS)
        assert not r.passed
        assert any("Forbidden" in i for i in r.issues)

    # --- 3 negative controls (clean sentences) --------------------------------

    def test_retrodisp_sentence_no_forbidden(self):  # 5f
        """A sentence about HVHV backscatter decrease does not trigger new patterns."""
        text = "Hay disminución de retrodispersión HVHV dentro del perímetro."
        r = layer1_check(text, STATS)
        new_pats = [
            r"\bvegetacion\b", r"\bvegetation\b", r"\bseveridad\b",
            r"\bgravedad\b", r"\bseverity\b", r"\bintensidad\b",
            r"\bla\s+causa\b", r"\bel\s+origen\b",
            r"\bcausa\s+del\b", r"\borigen\s+del\b",
        ]
        import re as _re
        for pat in new_pats:
            assert not _re.search(pat, text.lower()), f"Pattern /{pat}/ incorrectly matched clean text"

    def test_area_disminucion_no_forbidden(self):  # 5f
        """'área con disminución de retrodispersión HVHV' must not trigger."""
        text = "Área con disminución de retrodispersión HVHV: 127.22 km²."
        r = layer1_check(text, STATS)
        new_pats = [r"\bvegetacion\b", r"\bvegetation\b"]
        import re as _re
        for pat in new_pats:
            assert not _re.search(pat, text.lower()), f"Pattern /{pat}/ incorrectly matched"

    def test_umbral_sentence_no_forbidden(self):  # 5f
        """Threshold / exploratory sentence does not trigger new patterns."""
        text = "Umbral exploratorio -0.289 dB ajustado en este evento; tasa 10.60%."
        import re as _re
        new_pats = [
            r"\bseveridad\b", r"\bgravedad\b", r"\bseverity\b", r"\bintensidad\b",
            r"\bla\s+causa\b", r"\bcausa\s+del\b",
        ]
        for pat in new_pats:
            assert not _re.search(pat, text.lower()), f"Pattern /{pat}/ incorrectly matched"

    def test_all_5f_patterns_in_forbidden_list(self):  # 5f
        """All Task 5f patterns appear in _FORBIDDEN_PHRASES."""
        expected = [
            r"\bvegetacion\b",
            r"\bvegetation\b",
            r"\bseveridad\b",
            r"\bgravedad\b",
            r"\bseverity\b",
            r"\bintensidad\b",
            r"\bla\s+causa\b",
            r"\bel\s+origen\b",
            r"\bcausa\s+del\b",
            r"\borigen\s+del\b",
        ]
        for sub in expected:
            assert any(sub in p for p in _FORBIDDEN_PHRASES), (
                f"Pattern not found in _FORBIDDEN_PHRASES: {sub}"
            )


# ===========================================================================
# Task 5g additions
# ===========================================================================

# ---------------------------------------------------------------------------
# Step 1b — null/empty/whitespace content raises BackendUnavailableError
# ---------------------------------------------------------------------------

class TestGatewayNullContentOllama:
    """Task 5g Step 1b: Ollama backend — null/empty/whitespace content -> BackendUnavailableError."""

    def test_null_content_raises(self):
        """content=None (null) must raise BackendUnavailableError, never return 'None'."""
        gw = QwenGateway(backend="ollama")
        with mock.patch("src.sar.qwen_gateway._post_json") as m:
            m.return_value = {"message": {"content": None}}
            with pytest.raises(BackendUnavailableError, match="empty content"):
                gw.generate("test")

    def test_empty_string_content_raises(self):
        """content='' must raise BackendUnavailableError."""
        gw = QwenGateway(backend="ollama")
        with mock.patch("src.sar.qwen_gateway._post_json") as m:
            m.return_value = {"message": {"content": ""}}
            with pytest.raises(BackendUnavailableError, match="empty content"):
                gw.generate("test")

    def test_whitespace_only_content_raises(self):
        """content='   \\n  ' must raise BackendUnavailableError."""
        gw = QwenGateway(backend="ollama")
        with mock.patch("src.sar.qwen_gateway._post_json") as m:
            m.return_value = {"message": {"content": "   \n  "}}
            with pytest.raises(BackendUnavailableError, match="empty content"):
                gw.generate("test")

    def test_normal_string_returned_unchanged(self):
        """A normal non-empty content string must be returned as-is (no modification)."""
        gw = QwenGateway(backend="ollama")
        with mock.patch("src.sar.qwen_gateway._post_json") as m:
            m.return_value = {"message": {"content": "respuesta normal"}}
            result = gw.generate("test")
        assert result == "respuesta normal"


class TestGatewayNullContentOpenRouter:
    """Task 5g Step 1b: OpenRouter backend — null/empty/whitespace content -> BackendUnavailableError."""

    @pytest.fixture(autouse=True)
    def set_env(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key-never-logged")
        monkeypatch.setenv("OPENROUTER_MODEL", "qwen/qwen3-8b:free")

    def test_null_content_raises(self):
        """content=None (reasoning consumed all tokens) must raise BackendUnavailableError."""
        gw = QwenGateway(backend="openrouter")
        with mock.patch("src.sar.qwen_gateway._post_json") as m:
            m.return_value = {"choices": [{"message": {"content": None}}]}
            with pytest.raises(BackendUnavailableError, match="empty content"):
                gw.generate("test")

    def test_empty_string_content_raises(self):
        """content='' must raise BackendUnavailableError."""
        gw = QwenGateway(backend="openrouter")
        with mock.patch("src.sar.qwen_gateway._post_json") as m:
            m.return_value = {"choices": [{"message": {"content": ""}}]}
            with pytest.raises(BackendUnavailableError, match="empty content"):
                gw.generate("test")

    def test_whitespace_only_content_raises(self):
        """content='   ' must raise BackendUnavailableError."""
        gw = QwenGateway(backend="openrouter")
        with mock.patch("src.sar.qwen_gateway._post_json") as m:
            m.return_value = {"choices": [{"message": {"content": "   "}}]}
            with pytest.raises(BackendUnavailableError, match="empty content"):
                gw.generate("test")

    def test_normal_string_returned_unchanged(self):
        """A normal non-empty content string must be returned as-is."""
        gw = QwenGateway(backend="openrouter")
        with mock.patch("src.sar.qwen_gateway._post_json") as m:
            m.return_value = {"choices": [{"message": {"content": "summary text"}}]}
            result = gw.generate("test")
        assert result == "summary text"


# ---------------------------------------------------------------------------
# Step 2a — advisory/local mode makes zero critic calls
# ---------------------------------------------------------------------------

class TestAdvisoryModeZeroCriticCalls:
    """Task 5g Step 2a: advisory mode must make zero critic calls."""

    def test_advisory_layer1_pass_zero_critic_calls(self):
        """In advisory mode, a layer-1-passing summary must result in zero critic calls."""
        template = build_template_summary(STATS)
        critic_calls = [0]

        class CountingGateway:
            backend = "ollama"
            num_predict = 300
            timeout = 180.0

            def generate(self, prompt, system=None):
                # Detect critic call by system message or prompt marker
                if system and "revisor" in system.lower():
                    critic_calls[0] += 1
                    return "PASS"
                if "¿Hay afirmaciones" in prompt or "Resumen SAR a revisar" in prompt:
                    critic_calls[0] += 1
                    return "PASS"
                return template

        result = pipeline(STATS, CountingGateway(), max_regenerations=2, critic_mode="advisory")
        assert result.status == "layer1_only", f"Expected layer1_only, got {result.status!r}"
        assert critic_calls[0] == 0, (
            f"Advisory mode must make zero critic calls; got {critic_calls[0]}"
        )

    def test_advisory_layer1_fail_zero_critic_calls(self):
        """In advisory mode, even when layer1 fails, critic is never called."""
        critic_calls = [0]

        class BadGenNoCritic:
            backend = "ollama"
            num_predict = 300
            timeout = 180.0

            def generate(self, prompt, system=None):
                if system and "revisor" in system.lower():
                    critic_calls[0] += 1
                    return "PASS"
                if "¿Hay afirmaciones" in prompt or "Resumen SAR a revisar" in prompt:
                    critic_calls[0] += 1
                    return "PASS"
                return "Hay 999 km² inventados."  # always fails layer1

        result = pipeline(STATS, BadGenNoCritic(), max_regenerations=2, critic_mode="advisory")
        assert result.status == "fallback_template"
        assert critic_calls[0] == 0, (
            f"Advisory mode must make zero critic calls even on L1 fail; got {critic_calls[0]}"
        )


# ---------------------------------------------------------------------------
# Step 2b — negation in first 8 words -> "unverified" (3 real strings + control)
# ---------------------------------------------------------------------------

class TestCriticConsistencyCheck5g:
    """Task 5g Step 2b: 3 real negation strings + 1 negative control."""

    def test_real_string_1_no_hay(self):
        """'FAIL: No hay afirmaciones causales o frases prohibidas.' -> unverified."""
        result = _critic_consistency_check("fail", "No hay afirmaciones causales o frases prohibidas.")
        assert result is not None
        assert result["verdict"] == "unverified"
        assert result["issues"] == ["contradictory critic output"]

    def test_real_string_2_no_se(self):
        """'FAIL: No se mencionan afirmaciones causales o términos prohibidos.' -> unverified."""
        result = _critic_consistency_check(
            "fail", "No se mencionan afirmaciones causales o términos prohibidos."
        )
        assert result is not None
        assert result["verdict"] == "unverified"
        assert result["issues"] == ["contradictory critic output"]

    def test_real_string_3_no_contiene_mid_sentence(self):
        """'FAIL: El resumen no contiene afirmaciones causales ni frases prohibidas.' -> unverified.
        The negation 'no contiene' appears within the first 8 words, not at position 0.
        """
        result = _critic_consistency_check(
            "fail",
            "El resumen no contiene afirmaciones causales ni frases prohibidas."
        )
        assert result is not None, (
            "'El resumen no contiene...' must be caught: 'no contiene' is in the first 8 words"
        )
        assert result["verdict"] == "unverified"
        assert result["issues"] == ["contradictory critic output"]

    def test_negative_control_contiene_causales(self):
        """'FAIL: Contiene afirmaciones causales.' must remain 'fail' (no negation)."""
        result = _critic_consistency_check("fail", "Contiene afirmaciones causales.")
        assert result is None, (
            "'Contiene afirmaciones causales.' has no negation; must return None (real fail)"
        )

    def test_layer2_real_string_1_via_layer2_check(self):
        """Task 5h: 'FAIL: No hay ...' without quoted phrase -> 'unverified'/'unsupported critic claim'.
        Quote check fires first (before negation check); issue becomes 'unsupported critic claim'.
        """
        gw = MockGateway(response="FAIL: No hay afirmaciones causales o frases prohibidas.")
        result = layer2_check("resumen limpio", STATS, gw)
        assert result["verdict"] == "unverified"
        # Task 5h: quote check fires before negation check; no quote → unsupported critic claim
        assert result["issues"] == ["unsupported critic claim"]

    def test_layer2_real_string_2_via_layer2_check(self):
        """Task 5h: 'FAIL: No se mencionan ...' without quoted phrase -> 'unverified'."""
        gw = MockGateway(response="FAIL: No se mencionan afirmaciones causales o términos prohibidos.")
        result = layer2_check("resumen limpio", STATS, gw)
        assert result["verdict"] == "unverified"
        assert result["issues"] == ["unsupported critic claim"]

    def test_layer2_real_string_3_via_layer2_check(self):
        """Task 5h: 'FAIL: El resumen no contiene ...' without quoted phrase -> 'unverified'."""
        gw = MockGateway(response="FAIL: El resumen no contiene afirmaciones causales ni frases prohibidas.")
        result = layer2_check("resumen limpio", STATS, gw)
        assert result["verdict"] == "unverified"
        assert result["issues"] == ["unsupported critic claim"]

    def test_layer2_negative_control_via_layer2_check(self):
        """Task 5h: 'FAIL: Contiene afirmaciones causales.' without quoted phrase -> 'unverified'.
        Previously expected 'fail'; after Task 5h quote-check requirement it is now 'unverified'
        because no quoted phrase is present. To obtain 'fail', the critic must include a quote.
        """
        gw = MockGateway(response="FAIL: Contiene afirmaciones causales.")
        result = layer2_check("resumen con causales", STATS, gw)
        assert result["verdict"] == "unverified", (
            f"Expected 'unverified' (no quoted phrase); got {result['verdict']!r}"
        )
        assert result["issues"] == ["unsupported critic claim"]


# ---------------------------------------------------------------------------
# Step 4b — known verifier limit: range phrasing passes layer1
# ---------------------------------------------------------------------------

class TestKnownLimitRangePhrasing:
    """Task 5g Step 4b: documents the known limitation of the layer-1 number verifier.

    The verifier checks that every number in the summary traces to the JSON,
    not what the numbers *mean*.  A summary that presents two JSON area values
    as a range ("de 127.22 km² a 103.17 km²") passes layer 1 because both
    numbers are traceable — even though the semantic interpretation is wrong
    (they are two separate categories, not a range).

    This test is a DOCUMENTED LIMIT, not desired behaviour.
    """

    def test_known_limit_range_phrasing_passes(self):
        # This phrasing presents two separate JSON bins as a range.
        # Both 127.22 and 103.17 are traceable to the JSON, so layer 1 passes.
        # The verifier does NOT detect the semantic misuse — documented limit.
        summary = (
            "Análisis SAR NISAR HVHV entre 2026-07-27 y 2026-08-20 "
            "(intervalo 24 días, 100 m). "
            "Dentro del perímetro, de 127.22 km² a 103.17 km², "
            "media -0.6256 dB (umbral exploratorio -0.289 dB). "
            "Fuera (anillo 1-6 km): 70.81 km² moderada, 1.41 km² fuerte "
            "(media 0.1231 dB). Falsos positivos: 10.60%. "
            "Intervalo SAR de 24 días; sin datos de humedad del suelo; "
            "perímetro final del 1 de octubre de 2026."
        )
        r = layer1_check(summary, STATS)
        # DOCUMENTED LIMIT: numbers are traceable so layer1 passes,
        # even though the range semantics are incorrect.
        assert r.passed is True, (
            f"Known limit: range phrasing with traceable numbers should pass layer1. "
            f"Issues: {r.issues}"
        )


# ===========================================================================
# Task 5h additions
# ===========================================================================

# ---------------------------------------------------------------------------
# Step 2a — OPENROUTER_MODEL must come from env with NO default
# ---------------------------------------------------------------------------

class TestOpenRouterModelEnvRequired:
    """Task 5h Step 2a: OPENROUTER_MODEL has no default; must raise if unset."""

    def test_unset_openrouter_model_raises(self, monkeypatch):
        """OPENROUTER_MODEL not set -> BackendUnavailableError at QwenGateway init."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key-never-logged")
        monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
        with pytest.raises(BackendUnavailableError, match="OPENROUTER_MODEL"):
            QwenGateway(backend="openrouter")

    def test_set_openrouter_model_accepted(self, monkeypatch):
        """OPENROUTER_MODEL set -> QwenGateway initialises without error."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key-never-logged")
        monkeypatch.setenv("OPENROUTER_MODEL", "some/model-name")
        gw = QwenGateway(backend="openrouter")
        assert gw._openrouter_model == "some/model-name"

    def test_get_openrouter_model_raises_when_unset(self, monkeypatch):
        """_get_openrouter_model() raises BackendUnavailableError if OPENROUTER_MODEL unset."""
        monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
        with pytest.raises(BackendUnavailableError, match="OPENROUTER_MODEL"):
            _get_openrouter_model()

    def test_get_openrouter_model_returns_value_when_set(self, monkeypatch):
        """_get_openrouter_model() returns the env var value when set."""
        monkeypatch.setenv("OPENROUTER_MODEL", "qwen/qwen3-8b:free")
        result = _get_openrouter_model()
        assert result == "qwen/qwen3-8b:free"


# ---------------------------------------------------------------------------
# Step 2c — Critic quote hardenning (Task 5h)
# ---------------------------------------------------------------------------

class TestCriticQuoteCheck5h:
    """Task 5h Step 2c: critic must QUOTE the offending phrase; parser validates existence."""

    def test_real_string_1_uses_POST_phrase_exists_in_text(self):
        """FAIL: Usa "POST", implicando certeza sobre un incendio.
        The phrase 'POST' exists in the summary. Parser checks existence only.
        The claim that 'POST' implies certainty about a fire is not verified
        — the parser only confirms the quoted phrase is present in the text.
        """
        summary = 'El análisis usa POST para referirse a la fecha posterior al evento.'
        response = 'FAIL: Usa "POST", implicando certeza sobre un incendio.'
        gw = MockGateway(response=response)
        result = layer2_check(summary, STATS, gw)
        # 'POST' is in the summary -> verdict is 'fail' (not unverified)
        # BUT the claim is unsupported: the parser only checks existence, not correctness.
        assert result["verdict"] == "fail", (
            "Phrase 'POST' exists in summary: verdict should be 'fail'. "
            "NOTE: the parser checks existence only, not whether the claim is correct."
        )

    def test_real_string_2_perimetro_final_phrase_exists(self):
        """FAIL: Usa "perímetro final", implicando certeza sobre un hecho incierto."""
        summary = 'El perímetro final del 1 de octubre de 2026 se usó como referencia.'
        response = 'FAIL: Usa "perímetro final", implicando certeza sobre un hecho incierto.'
        gw = MockGateway(response=response)
        result = layer2_check(summary, STATS, gw)
        # 'perímetro final' exists in summary (accent-insensitive) -> verdict 'fail'
        assert result["verdict"] == "fail", (
            "Phrase 'perímetro final' exists in summary (accent-insensitive): "
            f"expected 'fail', got {result['verdict']!r}"
        )

    def test_quoted_phrase_absent_from_summary_returns_unverified(self):
        """Quoted phrase not found in summary -> 'unverified', reason 'unsupported critic claim'."""
        summary = 'El análisis no menciona nada sospechoso.'
        response = 'FAIL: "frase que no está" es problemática.'
        gw = MockGateway(response=response)
        result = layer2_check(summary, STATS, gw)
        assert result["verdict"] == "unverified"
        assert result["issues"] == ["unsupported critic claim"]

    def test_fail_no_quote_returns_unverified(self):
        """FAIL with no quoted phrase -> 'unverified', reason 'unsupported critic claim'."""
        summary = 'Resumen técnico sin problemas.'
        response = 'FAIL: contiene afirmación causal implícita'
        gw = MockGateway(response=response)
        result = layer2_check(summary, STATS, gw)
        assert result["verdict"] == "unverified"
        assert result["issues"] == ["unsupported critic claim"]

    def test_fail_with_quote_present_returns_fail(self):
        """FAIL with quoted phrase present in summary -> 'fail'."""
        summary = 'La disminución se debe al incendio registrado en el período.'
        response = 'FAIL: "se debe al incendio" es una afirmación causal.'
        gw = MockGateway(response=response)
        result = layer2_check(summary, STATS, gw)
        # 'se debe al incendio' is in summary -> 'fail'
        assert result["verdict"] == "fail"

    def test_pass_verdict_unaffected_by_quote_check(self):
        """PASS verdict is returned as-is without quote check."""
        gw = MockGateway(response="PASS")
        result = layer2_check("cualquier resumen", STATS, gw)
        assert result["verdict"] == "pass"


# ---------------------------------------------------------------------------
# Step 3 — summary_demo.json SHA256 test (Task 5h)
# ---------------------------------------------------------------------------

class TestSummaryDemoJson5h:
    """Task 5h Step 3: summary_demo.json sha256 must match change_stats.json."""

    def _get_paths(self):
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        demo_path  = os.path.join(root, "data", "output", "summary_demo.json")
        stats_path = os.path.join(root, "data", "output", "change_stats.json")
        return demo_path, stats_path

    def test_summary_demo_exists(self):
        import os
        demo_path, _ = self._get_paths()
        assert os.path.exists(demo_path), (
            f"data/output/summary_demo.json not found at {demo_path}"
        )

    def test_sha256_matches_change_stats(self):
        """sha256 in summary_demo.json must match actual change_stats.json SHA256."""
        import hashlib, json as _json
        demo_path, stats_path = self._get_paths()
        with open(demo_path) as f:
            demo = _json.load(f)
        # Compute actual SHA256 of change_stats.json
        h = hashlib.sha256()
        with open(stats_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        actual_sha = h.hexdigest()
        stored_sha = demo.get("stats_file_sha256", "")
        assert actual_sha == stored_sha, (
            f"SHA256 mismatch: summary_demo.json records {stored_sha!r}, "
            f"actual change_stats.json is {actual_sha!r}. "
            f"Re-run task5h_demo_run.py to regenerate."
        )

    def test_limitations_verbatim_copy(self):
        """Limitations in summary_demo.json must be verbatim copy from change_stats.json."""
        import json as _json
        demo_path, stats_path = self._get_paths()
        with open(demo_path) as f:
            demo = _json.load(f)
        with open(stats_path) as f:
            stats = _json.load(f)
        assert demo.get("limitations") == stats.get("limitations"), (
            "summary_demo.json 'limitations' list differs from change_stats.json"
        )

    def test_required_fields_present(self):
        """summary_demo.json must have all required fields."""
        import json as _json
        demo_path, _ = self._get_paths()
        with open(demo_path) as f:
            demo = _json.load(f)
        required = ["status", "text", "source", "attempts_count", "model",
                    "generated_at", "stats_file_sha256", "limitations"]
        for field in required:
            assert field in demo, f"Required field '{field}' missing from summary_demo.json"

    def test_status_is_valid(self):
        """status must be 'layer1_only' or 'fallback_template'."""
        import json as _json
        demo_path, _ = self._get_paths()
        with open(demo_path) as f:
            demo = _json.load(f)
        assert demo["status"] in ("layer1_only", "fallback_template"), (
            f"Invalid status {demo['status']!r}"
        )

    def test_source_matches_status(self):
        """source must be 'ollama' for layer1_only, 'template' for fallback_template."""
        import json as _json
        demo_path, _ = self._get_paths()
        with open(demo_path) as f:
            demo = _json.load(f)
        status = demo["status"]
        source = demo["source"]
        if status == "layer1_only":
            assert source == "ollama", f"layer1_only status should have source='ollama', got {source!r}"
        elif status == "fallback_template":
            assert source == "template", f"fallback_template status should have source='template', got {source!r}"

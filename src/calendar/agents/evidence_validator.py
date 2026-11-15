"""Agent 4 — Validation and Evidence Agent.

Verifies structured LLM outputs against their evidence payloads
(schema, banned claims, evidence grounding, number grounding) and
validates deterministic findings structurally. Never generates
explanatory text; returns verdicts only. Every entry point fails
closed (valid=False or empty tuple) on unexpected types.

Enforced valid-case schema (mirrors qwen.validate_response):
parsed must be a dict with keys summary (non-empty str),
evidence_used/uncertainty/limitations (lists of str); summary must
contain none of the banned phrases; every evidence_used id must be a
payload id; every number must ground to a payload value. Word limits
in agent SYSTEM prompts are guidance only and are NOT enforced.
"""

from __future__ import annotations

import json

from ..events import qwen as qwen_mod
from .contracts import Finding

MAX_RULE_CHARS = 200
MAX_FINDING_CHARS = 2000


def _strip_fences(text: str) -> str:
    """Remove one leading/trailing markdown fence (edges only)."""
    clean = text.strip()
    for fence in ("```json", "```"):
        if clean.startswith(fence):
            clean = clean[len(fence):].strip()
            break
    if clean.endswith("```"):
        clean = clean[:-3].strip()
    return clean


def _extract_json(text: str):
    """Parse the first {...} span (prose-tolerant repair, or None)."""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(text[start:end + 1])
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def check_llm_json(raw_text, payload: dict) -> dict:
    """Validate one raw provider response.

    Returns {valid, reason, parsed}. Unparseable text, non-dict JSON,
    schema violations, banned claims, ungrounded ids, and untraceable
    numbers all fail closed (valid=False, parsed=None).
    """
    try:
        if not isinstance(raw_text, str):
            return {"valid": False,
                    "reason": "response is not text", "parsed": None}
        try:
            parsed = json.loads(_strip_fences(raw_text))
        except (ValueError, AttributeError):
            parsed = _extract_json(raw_text)
            if parsed is None:
                return {"valid": False, "reason": "unparseable response",
                        "parsed": None}
        if not isinstance(parsed, dict):
            return {"valid": False,
                    "reason": "response is not a JSON object",
                    "parsed": None}
        verdict = qwen_mod.validate_response(parsed, payload)
        if not verdict["valid"]:
            return {"valid": False, "reason": verdict["reason"],
                    "parsed": None}
        required = {"summary": str, "evidence_used": list,
                    "uncertainty": list, "limitations": list}
        for key, want in required.items():
            if not isinstance(parsed.get(key), want):
                return {"valid": False,
                        "reason": f"valid response lacks {key!r} "
                                  f"({want.__name__})", "parsed": None}
        for key in ("evidence_used", "uncertainty", "limitations"):
            if not all(isinstance(v, str) for v in parsed[key]):
                return {"valid": False,
                        "reason": f"{key} must be a list of strings",
                        "parsed": None}
        return {"valid": True, "reason": "all checks passed",
                "parsed": parsed}
    except (ValueError, AttributeError, TypeError, KeyError) as exc:
        return {"valid": False,
                "reason": f"validator fault: {type(exc).__name__}",
                "parsed": None}


def check_findings(findings) -> dict:
    """Structural check on deterministic findings (rules + text)."""
    try:
        items = list(findings or [])
    except Exception:  # noqa: BLE001 - fail closed on bad input
        return {"valid": False, "reason": "findings are not a list"}
    for pos, item in enumerate(items):
        if not isinstance(item, Finding):
            return {"valid": False,
                    "reason": f"finding at index {pos} is not a Finding"}
        try:
            ok = bool(item.rule.strip()) and bool(item.finding.strip())
        except AttributeError:
            ok = False
        if not ok:
            return {"valid": False,
                    "reason": f"malformed finding at index {pos}"}
        if item.severity not in ("info", "caution"):
            return {"valid": False,
                    "reason": f"bad severity at index {pos}: "
                              f"{item.severity!r}"}
        if len(item.rule) > MAX_RULE_CHARS \
                or len(item.finding) > MAX_FINDING_CHARS:
            return {"valid": False,
                    "reason": f"finding at index {pos} exceeds length caps"}
    return {"valid": True, "reason": f"{len(items)} findings well-formed"}


def evidence_ids(payload: dict) -> tuple:
    """Traceable identifier set for one payload (sorted, stable).

    Fails closed to () on unexpected payload shapes.
    """
    try:
        if not isinstance(payload, dict):
            return ()
        return tuple(sorted(str(v) for v in qwen_mod.payload_ids(payload)))
    except (ValueError, AttributeError, TypeError):
        return ()

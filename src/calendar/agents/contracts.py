"""Agent input/output contracts (typed, no I/O, no LLM calls)."""

from __future__ import annotations

from dataclasses import dataclass

KINDS = ("detections", "event", "science")
STATUSES = ("ok", "fallback", "error")


REQUIRED_PAYLOAD_KEYS = {
    # kind -> required top-level payload keys (documents the contract;
    # workers read optional keys defensively beyond these).
    "detections": ("cluster_id", "n_detections"),
    "event": ("kind",),
    "science": ("observed_sum", "expected_sum"),
}


@dataclass(frozen=True)
class AgentRequest:
    """One agent invocation (payload carries verified evidence only)."""

    kind: str  # one of KINDS
    target_id: str
    payload: dict

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown agent kind {self.kind!r}; want {KINDS}")
        if not self.target_id or not isinstance(self.target_id, str):
            raise ValueError("target_id must be a non-empty string")
        if not isinstance(self.payload, dict):
            raise ValueError("payload must be a dict")
        missing = [k for k in REQUIRED_PAYLOAD_KEYS[self.kind]
                   if k not in self.payload]
        if missing:
            raise ValueError(f"payload for kind {self.kind!r} lacks "
                             f"required keys: {missing}")


@dataclass(frozen=True)
class Finding:
    """One deterministic check outcome (rule-bound, template-free)."""

    rule: str
    finding: str
    severity: str = "info"  # info | caution

    def __post_init__(self) -> None:
        if self.severity not in ("info", "caution"):
            raise ValueError(f"bad severity {self.severity!r}")
        if not self.rule.strip() or not self.finding.strip():
            raise ValueError("rule and finding must be non-empty")


@dataclass(frozen=True)
class AgentResponse:
    """Structured agent output (LLM text or deterministic fallback).

    provider/model name the successful provider on "ok", else the
    last-attempted provider (possibly empty when nothing ran).
    """

    agent: str
    status: str  # one of STATUSES
    summary: str
    evidence_used: tuple = ()
    uncertainty: tuple = ()
    limitations: tuple = ()
    findings: tuple = ()  # Finding items (deterministic analysis)
    provider: str = ""
    model: str = ""
    latency_ms: float = 0.0
    fallback_reason: str = ""
    validation: str = ""
    retries: int = 0
    response_chars: int = 0
    input_tokens: int | None = None
    generated_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r}")
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise ValueError("summary must be a non-empty string")
        for name in ("evidence_used", "uncertainty", "limitations"):
            items = getattr(self, name)
            if not isinstance(items, tuple) or not all(
                    isinstance(v, str) for v in items):
                raise ValueError(f"{name} must be a tuple of strings")
        if not isinstance(self.findings, tuple) or not all(
                isinstance(v, Finding) for v in self.findings):
            raise ValueError("findings must be a tuple of Finding")
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be >= 0")
        if not self.validation.strip():
            raise ValueError("validation must be non-empty")
        if self.status == "ok" and not self.provider:
            raise ValueError("ok responses must name their provider")
        if self.status in ("fallback", "error") \
                and not self.fallback_reason.strip():
            raise ValueError(
                f"{self.status} responses must state fallback_reason")


@dataclass(frozen=True)
class TraceStep:
    """One coordinator audit step (no secrets, ever)."""

    agent: str
    status: str
    provider: str
    model: str
    latency_ms: float
    validation: str
    fallback_reason: str = ""
    retries: int = 0
    response_chars: int = 0
    input_tokens: int | None = None
    generated_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r}")
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be >= 0")


@dataclass(frozen=True)
class CoordinatorResult:
    """Final aggregated explanation plus execution trace."""

    kind: str
    target_id: str
    text: str
    evidence_ids: tuple = ()
    steps: tuple = ()  # TraceStep items
    limitations: tuple = ()
    audit_refs: tuple = ()

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind {self.kind!r}")
        if not self.target_id or not isinstance(self.target_id, str):
            raise ValueError("target_id must be a non-empty string")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must be a non-empty string")

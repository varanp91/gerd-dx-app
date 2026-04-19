"""
Reasoning-trace data structures.

Every classification step emits structured `Evidence` rather than free text so
the CLI, tests, and future UI can render the trace consistently, and so tests
can assert on specific rule firings instead of string-matching prose.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .enums import Confidence, EvidenceStrength


class Evidence(BaseModel):
    """A single rule firing against the input."""
    rule_id: str                    # e.g. "LYON_CONCLUSIVE_LA_CD"
    mechanism_id: str               # which bucket this evidence contributes to
    strength: EvidenceStrength
    trigger: str                    # human-readable, includes the actual input value
    source: str                     # e.g. "Lyon Consensus 2.0"


class MechanismResult(BaseModel):
    """Per-mechanism classification outcome with its supporting evidence."""
    mechanism_id: str
    label: str                      # human-readable name
    confidence: Confidence
    evidence: list[Evidence] = Field(default_factory=list)


class ProceduralOption(BaseModel):
    """
    One entry in ManagementPlan.procedural.

    Either `indication` (positive match) OR `contraindicated=True` with a
    `reason` must be present. The classifier does not filter options by
    patient anatomy (except via mechanism routing) — clinician matches the
    indication to the patient.
    """
    id: str
    label: str
    indication: str | None = None
    contraindicated: bool = False
    reason: str | None = None


class ManagementPlan(BaseModel):
    mechanism_id: str
    first_line: list[str] = Field(default_factory=list)
    escalation: list[str] = Field(default_factory=list)
    procedural: list[ProceduralOption] = Field(default_factory=list)
    lifestyle: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    modifiers_from_contributing_factors: list[str] = Field(default_factory=list)


class CaseOutput(BaseModel):
    """
    Full classifier output. Consumed by CLI (pretty-prints reasoning trace),
    JSON output mode, and tests.
    """
    disclaimer: str
    ranked_mechanisms: list[MechanismResult]
    combinations_flagged: list[str] = Field(default_factory=list)
    contributing_factors: list[str] = Field(
        default_factory=list,
        description="Factors that modify management but not classification "
                    "(e.g. GLP-1, CCBs, opioids).",
    )
    insufficient_inputs: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    refractory_flag: bool = Field(
        default=False,
        description="True when on-PPI AET > 4% (user-configured threshold).",
    )
    management: dict[str, ManagementPlan] = Field(default_factory=dict)
    red_flags: list[str] = Field(default_factory=list)

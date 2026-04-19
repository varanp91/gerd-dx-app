"""
Pydantic input models for a GERD diagnostic case.

All sub-sections are optional except ClinicalContext, because real workups are
often incomplete. Missing sections must be surfaced as `insufficient_inputs`
downstream rather than silently defaulted.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from .enums import (
    BiopsyFinding,
    ContributingMed,
    DominantSymptom,
    EGJMorphology,
    LAGrade,
    PPIResponse,
    Peristalsis,
    PriorSurgery,
    PhStudyState,
)


class EndoscopyFindings(BaseModel):
    la_grade: LAGrade = LAGrade.NONE
    hiatal_hernia_cm: float | None = Field(
        None, ge=0, description="Axial length in cm. Leave None if not measured."
    )
    hiatal_hernia_present: bool = Field(
        False,
        description="True when a hernia is documented but the size was not measured. "
                    "If hiatal_hernia_cm is set, this is auto-inferred True.",
    )
    intrathoracic_stomach: bool = False
    barretts: bool = False
    barretts_length_cm: float | None = Field(None, ge=0)
    biopsy_finding: BiopsyFinding = BiopsyFinding.NORMAL
    eosinophils_per_hpf: int | None = Field(None, ge=0)
    dilated_intercellular_spaces: bool = False
    peptic_stricture: bool = False

    @model_validator(mode="after")
    def _barretts_coherent(self) -> "EndoscopyFindings":
        if self.barretts and self.barretts_length_cm is None:
            # Length unknown is allowed, but flag via absence downstream.
            pass
        if not self.barretts and self.barretts_length_cm:
            raise ValueError("barretts_length_cm set but barretts is False")
        return self


class HRMFindings(BaseModel):
    """High-resolution manometry — Chicago Classification v4.0."""
    les_resting_pressure_mmhg: float | None = None
    egj_morphology: EGJMorphology | None = None
    peristalsis: Peristalsis | None = None
    dci_median: float | None = Field(None, ge=0)
    ineffective_swallow_pct: float | None = Field(None, ge=0, le=100)
    failed_swallow_pct: float | None = Field(None, ge=0, le=100)
    egj_ci: float | None = Field(None, ge=0)
    irp_supine_mmhg: float | None = None
    premature_swallow_pct: float | None = Field(None, ge=0, le=100)
    jackhammer_swallow_pct: float | None = Field(None, ge=0, le=100)


class PhImpedance(BaseModel):
    """Ambulatory pH-impedance monitoring. Test state matters for interpretation."""
    test_state: PhStudyState
    aet_pct: float = Field(..., ge=0, le=100)
    total_reflux_episodes: int | None = Field(None, ge=0)
    acidic_pct: float | None = Field(None, ge=0, le=100)
    weakly_acidic_pct: float | None = Field(None, ge=0, le=100)
    non_acid_pct: float | None = Field(None, ge=0, le=100)
    symptom_index_pct: float | None = Field(None, ge=0, le=100)
    # SAP is intentionally collected but NOT used in classification rules
    # (per clinical direction). Retained for the reasoning trace only.
    sap_pct: float | None = Field(None, ge=0, le=100)
    mnbi_ohms: float | None = Field(None, ge=0)
    pspw_pct: float | None = Field(None, ge=0, le=100)


class EndoFLIPFindings(BaseModel):
    """EndoFLIP measurements. Cutoffs defined at 60mL only."""
    egj_di_60ml: float | None = Field(None, ge=0, description="mm^2/mmHg at 60mL")
    min_diameter_60ml_mm: float | None = Field(None, ge=0)
    repetitive_antegrade_contractions: bool = False
    repetitive_retrograde_contractions: bool = False


class GastricEmptyingStudy(BaseModel):
    retention_2h_pct: float | None = Field(None, ge=0, le=100)
    retention_4h_pct: float | None = Field(None, ge=0, le=100)


class ClinicalContext(BaseModel):
    ppi_response: PPIResponse
    dominant_symptoms: list[DominantSymptom] = Field(default_factory=list)
    bmi: float | None = Field(None, ge=0)
    prior_anti_reflux_surgery: PriorSurgery = PriorSurgery.NONE
    prior_bariatric_surgery: PriorSurgery = PriorSurgery.NONE
    contributing_medications: list[ContributingMed] = Field(default_factory=list)
    pregnant: bool = False


class CaseInput(BaseModel):
    """Full diagnostic workup input. Only ClinicalContext is required."""
    endoscopy: EndoscopyFindings | None = None
    hrm: HRMFindings | None = None
    ph_impedance: PhImpedance | None = None
    endoflip: EndoFLIPFindings | None = None
    gastric_emptying: GastricEmptyingStudy | None = None
    clinical: ClinicalContext

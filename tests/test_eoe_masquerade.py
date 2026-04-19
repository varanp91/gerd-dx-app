"""
Eosinophilic esophagitis masquerading as GERD.

Patient: dysphagia + heartburn, failed PPI, endoscopy shows rings/furrows
(no erosions, no Barrett's), biopsy with 40 eos/HPF, normal AET off-PPI.

Expected:
  - eoe_masquerade ranks top (CONCLUSIVE on histology).
  - functional_heartburn / reflux_hypersensitivity are excluded by the EoE
    biopsy finding — they should not outrank EoE.
"""

from gerd_dx.engine import classify
from gerd_dx.enums import (
    BiopsyFinding,
    Confidence,
    DominantSymptom,
    LAGrade,
    PPIResponse,
    Peristalsis,
    PhStudyState,
)
from gerd_dx.models import (
    CaseInput,
    ClinicalContext,
    EndoscopyFindings,
    HRMFindings,
    PhImpedance,
)


def _eoe_case() -> CaseInput:
    return CaseInput(
        endoscopy=EndoscopyFindings(
            la_grade=LAGrade.NONE,
            biopsy_finding=BiopsyFinding.EOSINOPHILS,
            eosinophils_per_hpf=40,
        ),
        hrm=HRMFindings(peristalsis=Peristalsis.NORMAL),
        ph_impedance=PhImpedance(
            test_state=PhStudyState.OFF_PPI,
            aet_pct=1.8,
            symptom_index_pct=35,
        ),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.NONE,
            dominant_symptoms=[DominantSymptom.DYSPHAGIA, DominantSymptom.HEARTBURN],
        ),
    )


def test_eoe_ranks_top_with_high_confidence():
    result = classify(_eoe_case())
    top = result.ranked_mechanisms[0]
    assert top.mechanism_id == "eoe_masquerade"
    assert top.confidence == Confidence.HIGH


def test_eoe_trace_cites_histologic_threshold_and_supportive_context():
    result = classify(_eoe_case())
    top = result.ranked_mechanisms[0]
    rule_ids = {ev.rule_id for ev in top.evidence}
    assert "EOE_HISTOLOGIC_THRESHOLD" in rule_ids
    assert "EOE_PPI_NONRESPONSE" in rule_ids
    assert "EOE_DYSPHAGIA_DOMINANT" in rule_ids


def test_eoe_excludes_functional_heartburn():
    """SI is negative and AET is physiologic, but EoE biopsy rules out functional dx."""
    result = classify(_eoe_case())
    fh = next(
        (r for r in result.ranked_mechanisms if r.mechanism_id == "functional_heartburn"),
        None,
    )
    assert fh is not None, "Expected functional_heartburn to be considered and ruled out"
    assert fh.confidence == Confidence.RULED_OUT
    assert any(ev.rule_id == "EOE_EXCLUDES_FUNCTIONAL" for ev in fh.evidence)


def test_eoe_below_threshold_does_not_fire():
    """14 eos/HPF is below the 15 threshold — the EoE bucket should not appear."""
    case = _eoe_case()
    case.endoscopy.eosinophils_per_hpf = 14
    result = classify(case)
    assert not any(r.mechanism_id == "eoe_masquerade" for r in result.ranked_mechanisms)


def test_eoe_coexisting_with_true_gerd_flags_combination():
    """LA grade C + EoE biopsy → both fire, combinations_flagged surfaces coexistence."""
    case = _eoe_case()
    case.endoscopy.la_grade = LAGrade.C
    case.ph_impedance.aet_pct = 8.0  # pathologic off-PPI
    result = classify(case)
    mech_ids = {r.mechanism_id for r in result.ranked_mechanisms}
    assert "eoe_masquerade" in mech_ids
    assert "true_gerd_competent_peristalsis" in mech_ids
    assert "True GERD + coexisting EoE" in result.combinations_flagged

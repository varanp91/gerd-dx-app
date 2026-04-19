"""
Reflux hypersensitivity.

Patient: heartburn with partial PPI response, endoscopy unremarkable (LA none),
AET 2.1% off-PPI (physiologic), Symptom Index 72% (positive).

Expected: reflux_hypersensitivity ranks top with moderate+ confidence.
No conclusive endoscopy means no exclusion; SI positive drives classification.
"""

from gerd_dx.engine import classify
from gerd_dx.enums import (
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


def _hypersensitivity_case() -> CaseInput:
    return CaseInput(
        endoscopy=EndoscopyFindings(la_grade=LAGrade.NONE),
        hrm=HRMFindings(peristalsis=Peristalsis.NORMAL),
        ph_impedance=PhImpedance(
            test_state=PhStudyState.OFF_PPI,
            aet_pct=2.1,
            total_reflux_episodes=35,
            symptom_index_pct=72,
        ),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.HEARTBURN],
        ),
    )


def test_hypersensitivity_ranks_top():
    result = classify(_hypersensitivity_case())
    top = result.ranked_mechanisms[0]
    assert top.mechanism_id == "reflux_hypersensitivity"
    assert top.confidence == Confidence.HIGH  # STRONG SI + SUPPORTIVE AET


def test_hypersensitivity_trace_cites_si_and_physiologic_aet():
    result = classify(_hypersensitivity_case())
    top = result.ranked_mechanisms[0]
    rule_ids = {ev.rule_id for ev in top.evidence}
    assert "SI_POSITIVE" in rule_ids
    assert "AET_PHYSIOLOGIC_OFF_PPI" in rule_ids


def test_hypersensitivity_case_rules_out_true_gerd():
    """Physiologic AET off-PPI should rule out the true-GERD bucket."""
    result = classify(_hypersensitivity_case())
    true_gerd = next(
        (r for r in result.ranked_mechanisms if r.mechanism_id == "true_gerd_competent_peristalsis"),
        None,
    )
    assert true_gerd is not None
    assert true_gerd.confidence == Confidence.RULED_OUT


def test_hypersensitivity_does_not_fire_when_si_negative():
    """SI < 50% → the functional heartburn bucket owns the case, not hypersensitivity."""
    case = _hypersensitivity_case()
    case.ph_impedance.symptom_index_pct = 20
    result = classify(case)
    assert not any(
        r.mechanism_id == "reflux_hypersensitivity" for r in result.ranked_mechanisms
    )


def test_hypersensitivity_excluded_by_conclusive_endoscopy():
    """LA grade C → structural GERD, not hypersensitivity, even with positive SI."""
    case = _hypersensitivity_case()
    case.endoscopy.la_grade = LAGrade.C
    result = classify(case)
    hyper = next(
        (r for r in result.ranked_mechanisms if r.mechanism_id == "reflux_hypersensitivity"),
        None,
    )
    # Either the bucket is ruled out (still appears with exclusionary evidence)
    # or it doesn't appear at all — both are acceptable exclusions.
    if hyper is not None:
        assert hyper.confidence == Confidence.RULED_OUT

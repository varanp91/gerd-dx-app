"""
True acid GERD with ineffective esophageal motility (IEM).

Patient: LA grade B, AET 7.4% off-PPI, HRM shows IEM (ineffective swallows
80%, median DCI 300).

Expected:
  - true_gerd_with_iem ranks top with HIGH confidence.
  - true_gerd_competent_peristalsis is RULED_OUT (HRM not competent).
  - Both buckets appear in the output so the clinician sees the trade-off.
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


def _iem_plus_gerd_case() -> CaseInput:
    return CaseInput(
        endoscopy=EndoscopyFindings(la_grade=LAGrade.B),
        hrm=HRMFindings(
            peristalsis=Peristalsis.IEM,
            dci_median=300,
            ineffective_swallow_pct=80,
        ),
        ph_impedance=PhImpedance(
            test_state=PhStudyState.OFF_PPI,
            aet_pct=7.4,
            total_reflux_episodes=85,
        ),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.REGURGITATION, DominantSymptom.HEARTBURN],
        ),
    )


def test_iem_combo_ranks_top_with_high_confidence():
    result = classify(_iem_plus_gerd_case())
    top = result.ranked_mechanisms[0]
    assert top.mechanism_id == "true_gerd_with_iem"
    assert top.confidence == Confidence.HIGH


def test_iem_combo_trace_includes_both_acid_evidence_and_iem_finding():
    result = classify(_iem_plus_gerd_case())
    iem_bucket = next(
        r for r in result.ranked_mechanisms if r.mechanism_id == "true_gerd_with_iem"
    )
    rule_ids = {ev.rule_id for ev in iem_bucket.evidence}
    assert "LYON_AET_PATHOLOGIC" in rule_ids
    assert "HRM_IEM_CONFIRMED" in rule_ids


def test_iem_case_rules_out_competent_peristalsis_bucket():
    """The competent-peristalsis bucket should appear with RULED_OUT confidence."""
    result = classify(_iem_plus_gerd_case())
    competent = next(
        (r for r in result.ranked_mechanisms if r.mechanism_id == "true_gerd_competent_peristalsis"),
        None,
    )
    assert competent is not None, (
        "Expected the competent bucket to appear with RULED_OUT status so the "
        "clinician sees why it was considered but rejected"
    )
    assert competent.confidence == Confidence.RULED_OUT
    assert any(ev.rule_id == "HRM_NOT_COMPETENT" for ev in competent.evidence)


def test_iem_without_acid_evidence_does_not_fire_combo():
    """IEM alone (no acid evidence) should NOT fire the combo bucket."""
    case = CaseInput(
        hrm=HRMFindings(peristalsis=Peristalsis.IEM, dci_median=300),
        ph_impedance=PhImpedance(
            test_state=PhStudyState.OFF_PPI,
            aet_pct=2.0,
            symptom_index_pct=10,
        ),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.NONE,
            dominant_symptoms=[DominantSymptom.HEARTBURN],
        ),
    )
    result = classify(case)
    assert not any(
        r.mechanism_id == "true_gerd_with_iem" for r in result.ranked_mechanisms
    )

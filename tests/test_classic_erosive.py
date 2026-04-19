"""
Classic erosive GERD with competent peristalsis.

Patient: LA grade C esophagitis on EGD, AET 9.2% off-PPI, 95 reflux episodes,
normal peristalsis on HRM, partial PPI response.

Expected: true_gerd_competent_peristalsis ranked top with HIGH confidence.
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


def _classic_erosive_case() -> CaseInput:
    return CaseInput(
        endoscopy=EndoscopyFindings(la_grade=LAGrade.C),
        hrm=HRMFindings(peristalsis=Peristalsis.NORMAL, dci_median=1500),
        ph_impedance=PhImpedance(
            test_state=PhStudyState.OFF_PPI,
            aet_pct=9.2,
            total_reflux_episodes=95,
        ),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.HEARTBURN],
        ),
    )


def test_classic_erosive_ranks_true_gerd_high_confidence():
    result = classify(_classic_erosive_case())
    assert result.ranked_mechanisms, "Expected at least one mechanism to fire"
    top = result.ranked_mechanisms[0]
    assert top.mechanism_id == "true_gerd_competent_peristalsis"
    assert top.confidence == Confidence.HIGH


def test_classic_erosive_reasoning_trace_cites_all_three_pillars():
    """Reasoning trace must show conclusive endoscopy, pathologic AET, and intact HRM."""
    result = classify(_classic_erosive_case())
    top = result.ranked_mechanisms[0]
    rule_ids = {ev.rule_id for ev in top.evidence}
    assert "LYON_CONCLUSIVE_LA_CD" in rule_ids
    assert "LYON_AET_PATHOLOGIC" in rule_ids
    assert "HRM_NORMAL_PERISTALSIS" in rule_ids


def test_classic_erosive_no_refractory_flag_when_off_ppi():
    """Refractory flag is strictly an on-PPI phenomenon."""
    result = classify(_classic_erosive_case())
    assert result.refractory_flag is False


def test_classic_erosive_no_contributing_factors_when_none_given():
    result = classify(_classic_erosive_case())
    assert result.contributing_factors == []


def test_output_carries_disclaimer():
    result = classify(_classic_erosive_case())
    assert "education" in result.disclaimer.lower()
    assert "diagnostic device" in result.disclaimer.lower()

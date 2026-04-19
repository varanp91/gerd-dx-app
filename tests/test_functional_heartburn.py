"""
Functional heartburn.

Patient: heartburn with NO PPI response, endoscopy normal, AET 1.4% off-PPI
(physiologic), Symptom Index 15% (negative). Rome IV functional heartburn.

Expected: functional_heartburn ranks top. true-GERD bucket is RULED_OUT.
Reflux hypersensitivity should not appear (SI is negative).
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


def _functional_heartburn_case() -> CaseInput:
    return CaseInput(
        endoscopy=EndoscopyFindings(la_grade=LAGrade.NONE),
        hrm=HRMFindings(peristalsis=Peristalsis.NORMAL),
        ph_impedance=PhImpedance(
            test_state=PhStudyState.OFF_PPI,
            aet_pct=1.4,
            total_reflux_episodes=20,
            symptom_index_pct=15,
        ),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.NONE,
            dominant_symptoms=[DominantSymptom.HEARTBURN],
        ),
    )


def test_functional_heartburn_ranks_top():
    result = classify(_functional_heartburn_case())
    top = result.ranked_mechanisms[0]
    assert top.mechanism_id == "functional_heartburn"
    assert top.confidence == Confidence.HIGH


def test_functional_heartburn_trace_cites_negative_si():
    result = classify(_functional_heartburn_case())
    top = result.ranked_mechanisms[0]
    rule_ids = {ev.rule_id for ev in top.evidence}
    assert "SI_NEGATIVE" in rule_ids
    assert "AET_PHYSIOLOGIC_OFF_PPI" in rule_ids


def test_functional_heartburn_does_not_coexist_with_hypersensitivity():
    """These buckets partition the physiologic-AET space by SI — they're mutually exclusive."""
    result = classify(_functional_heartburn_case())
    mech_ids = {r.mechanism_id for r in result.ranked_mechanisms}
    assert "functional_heartburn" in mech_ids
    assert "reflux_hypersensitivity" not in mech_ids


def test_functional_heartburn_case_rules_out_true_gerd():
    result = classify(_functional_heartburn_case())
    true_gerd = next(
        (r for r in result.ranked_mechanisms if r.mechanism_id == "true_gerd_competent_peristalsis"),
        None,
    )
    assert true_gerd is not None
    assert true_gerd.confidence == Confidence.RULED_OUT

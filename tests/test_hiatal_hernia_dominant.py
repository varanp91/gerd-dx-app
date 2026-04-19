"""Hiatal hernia-dominant GERD."""

from gerd_dx.engine import classify
from gerd_dx.enums import (
    Confidence,
    DominantSymptom,
    EGJMorphology,
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


def test_large_measured_hernia_with_intrathoracic_stomach_is_conclusive():
    case = CaseInput(
        endoscopy=EndoscopyFindings(
            hiatal_hernia_cm=6.0,
            intrathoracic_stomach=True,
        ),
        hrm=HRMFindings(peristalsis=Peristalsis.NORMAL, egj_morphology=EGJMorphology.TYPE_III),
        ph_impedance=PhImpedance(test_state=PhStudyState.OFF_PPI, aet_pct=8.0),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.REGURGITATION],
        ),
    )
    result = classify(case)
    hh = next(
        (r for r in result.ranked_mechanisms if r.mechanism_id == "hiatal_hernia_dominant"),
        None,
    )
    assert hh is not None
    assert hh.confidence == Confidence.HIGH
    rule_ids = {e.rule_id for e in hh.evidence}
    assert "HH_LARGE_WITH_INTRATHORACIC" in rule_ids
    assert "HRM_EGJ_TYPE_III" in rule_ids
    assert "HH_REGURG_DOMINANT" in rule_ids


def test_hernia_without_intrathoracic_stomach_fires_at_strong_not_conclusive():
    """A large measured hernia alone (no ITS) is STRONG, not CONCLUSIVE."""
    case = CaseInput(
        endoscopy=EndoscopyFindings(hiatal_hernia_cm=5.0, intrathoracic_stomach=False),
        ph_impedance=PhImpedance(test_state=PhStudyState.OFF_PPI, aet_pct=2.0),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.NONE,
            dominant_symptoms=[DominantSymptom.HEARTBURN],
        ),
    )
    result = classify(case)
    hh = next(
        (r for r in result.ranked_mechanisms if r.mechanism_id == "hiatal_hernia_dominant"),
        None,
    )
    assert hh is not None
    assert any(e.rule_id == "HH_LARGE" for e in hh.evidence)
    assert not any(e.rule_id == "HH_LARGE_WITH_INTRATHORACIC" for e in hh.evidence)


def test_unmeasured_hernia_requires_intrathoracic_stomach():
    """Hernia_present=True without cm AND without ITS → bucket does NOT fire."""
    case = CaseInput(
        endoscopy=EndoscopyFindings(hiatal_hernia_present=True, intrathoracic_stomach=False),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.REGURGITATION],
        ),
    )
    result = classify(case)
    assert not any(r.mechanism_id == "hiatal_hernia_dominant" for r in result.ranked_mechanisms)


def test_unmeasured_hernia_with_intrathoracic_stomach_fires():
    case = CaseInput(
        endoscopy=EndoscopyFindings(hiatal_hernia_present=True, intrathoracic_stomach=True),
        ph_impedance=PhImpedance(test_state=PhStudyState.OFF_PPI, aet_pct=7.0),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.REGURGITATION],
        ),
    )
    result = classify(case)
    hh = next(
        (r for r in result.ranked_mechanisms if r.mechanism_id == "hiatal_hernia_dominant"),
        None,
    )
    assert hh is not None
    assert any(e.rule_id == "HH_DOCUMENTED_WITH_INTRATHORACIC" for e in hh.evidence)


def test_hernia_coexists_with_true_gerd_and_flags_combination():
    case = CaseInput(
        endoscopy=EndoscopyFindings(
            la_grade=LAGrade.C,
            hiatal_hernia_cm=5.5,
            intrathoracic_stomach=True,
        ),
        hrm=HRMFindings(peristalsis=Peristalsis.NORMAL),
        ph_impedance=PhImpedance(test_state=PhStudyState.OFF_PPI, aet_pct=9.0),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.REGURGITATION],
        ),
    )
    result = classify(case)
    mech_ids = {r.mechanism_id for r in result.ranked_mechanisms}
    assert "true_gerd_competent_peristalsis" in mech_ids
    assert "hiatal_hernia_dominant" in mech_ids
    assert "True GERD + hiatal hernia-dominant overlay" in result.combinations_flagged

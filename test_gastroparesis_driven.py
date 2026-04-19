"""Gastroparesis-driven reflux + contributing factor interaction."""

from gerd_dx.engine import classify
from gerd_dx.enums import (
    Confidence,
    ContributingMed,
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
    GastricEmptyingStudy,
    HRMFindings,
    PhImpedance,
)


def test_gastroparesis_fires_on_4h_retention():
    case = CaseInput(
        gastric_emptying=GastricEmptyingStudy(retention_2h_pct=45, retention_4h_pct=22),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.REGURGITATION],
        ),
    )
    result = classify(case)
    gp = next(
        (r for r in result.ranked_mechanisms if r.mechanism_id == "gastroparesis_driven_reflux"),
        None,
    )
    assert gp is not None
    assert any(e.rule_id == "GE_RETENTION_4H" for e in gp.evidence)


def test_gastroparesis_does_not_fire_below_threshold():
    case = CaseInput(
        gastric_emptying=GastricEmptyingStudy(retention_2h_pct=50, retention_4h_pct=8),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.HEARTBURN],
        ),
    )
    result = classify(case)
    assert not any(
        r.mechanism_id == "gastroparesis_driven_reflux" for r in result.ranked_mechanisms
    )


def test_glp1_surfaces_as_contributing_factor_not_mechanism():
    """GLP-1 is captured as a contributing factor, NOT as evidence for gastroparesis bucket."""
    case = CaseInput(
        gastric_emptying=GastricEmptyingStudy(retention_4h_pct=25),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.REGURGITATION],
            contributing_medications=[ContributingMed.GLP1],
        ),
    )
    result = classify(case)
    assert "glp1" in result.contributing_factors
    # And the bucket fires on the retention data regardless of the GLP-1 context.
    gp = next(
        r for r in result.ranked_mechanisms if r.mechanism_id == "gastroparesis_driven_reflux"
    )
    # No rule_id in the mechanism's trace should reference GLP-1 — contributing
    # factors modify management downstream, not classification.
    for ev in gp.evidence:
        assert "glp" not in ev.rule_id.lower()
        assert "glp" not in ev.trigger.lower()


def test_gastroparesis_plus_true_gerd_flags_combination():
    case = CaseInput(
        endoscopy=EndoscopyFindings(la_grade=LAGrade.C),
        hrm=HRMFindings(peristalsis=Peristalsis.NORMAL),
        ph_impedance=PhImpedance(test_state=PhStudyState.OFF_PPI, aet_pct=8.0),
        gastric_emptying=GastricEmptyingStudy(retention_4h_pct=28),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.HEARTBURN],
        ),
    )
    result = classify(case)
    assert "Gastroparesis contributing to reflux mechanism" in result.combinations_flagged

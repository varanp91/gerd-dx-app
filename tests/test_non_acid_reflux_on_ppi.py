"""Non-acid / weakly acidic reflux on PPI."""

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


def test_non_acid_reflux_with_composition_data_ranks_high():
    """On-PPI + SI positive + weakly-acidic/non-acid > 50% → HIGH confidence."""
    case = CaseInput(
        endoscopy=EndoscopyFindings(la_grade=LAGrade.NONE),
        hrm=HRMFindings(peristalsis=Peristalsis.NORMAL),
        ph_impedance=PhImpedance(
            test_state=PhStudyState.ON_PPI,
            aet_pct=2.0,
            symptom_index_pct=65,
            acidic_pct=30,
            weakly_acidic_pct=55,
            non_acid_pct=15,
        ),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.REGURGITATION],
        ),
    )
    result = classify(case)
    top = result.ranked_mechanisms[0]
    assert top.mechanism_id == "non_acid_reflux_on_ppi"
    assert top.confidence == Confidence.HIGH
    rule_ids = {e.rule_id for e in top.evidence}
    assert "NAB_SI_POSITIVE_ON_PPI" in rule_ids
    assert "NAB_NONACID_PREDOMINANT" in rule_ids


def test_non_acid_reflux_falls_back_to_aet_inference_without_composition():
    """On-PPI + SI positive + AET controlled (no composition data) → STRONG + SUPPORTIVE."""
    case = CaseInput(
        ph_impedance=PhImpedance(
            test_state=PhStudyState.ON_PPI,
            aet_pct=2.0,
            symptom_index_pct=70,
        ),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.HEARTBURN],
        ),
    )
    result = classify(case)
    nab = next(
        (r for r in result.ranked_mechanisms if r.mechanism_id == "non_acid_reflux_on_ppi"),
        None,
    )
    assert nab is not None
    rule_ids = {e.rule_id for e in nab.evidence}
    assert "NAB_SI_POSITIVE_ON_PPI" in rule_ids
    assert "NAB_AET_CONTROLLED_ON_PPI" in rule_ids


def test_non_acid_bucket_requires_on_ppi_study():
    """Off-PPI studies go through hypersensitivity / functional / true-GERD logic."""
    case = CaseInput(
        ph_impedance=PhImpedance(
            test_state=PhStudyState.OFF_PPI,
            aet_pct=2.0,
            symptom_index_pct=70,
        ),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.HEARTBURN],
        ),
    )
    result = classify(case)
    assert not any(
        r.mechanism_id == "non_acid_reflux_on_ppi" for r in result.ranked_mechanisms
    )


def test_on_ppi_high_aet_triggers_refractory_flag_not_this_bucket():
    """On-PPI AET > 4% is refractory acid GERD (flag), not non-acid reflux."""
    case = CaseInput(
        endoscopy=EndoscopyFindings(la_grade=LAGrade.B),
        ph_impedance=PhImpedance(
            test_state=PhStudyState.ON_PPI,
            aet_pct=5.5,
            symptom_index_pct=70,
        ),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.HEARTBURN],
        ),
    )
    result = classify(case)
    assert result.refractory_flag is True

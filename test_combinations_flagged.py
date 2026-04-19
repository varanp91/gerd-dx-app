"""
Combination-flag surfacing in CaseOutput.combinations_flagged.

Verifies that multi-bucket clinical combinations produce human-readable
strings the UI layer can render above the per-bucket reasoning trace.
"""

from gerd_dx.engine import classify
from gerd_dx.enums import (
    BiopsyFinding,
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


def test_true_gerd_with_iem_flags_the_iem_combination():
    """The IEM combo bucket is inherently a combination — surface it explicitly."""
    case = CaseInput(
        endoscopy=EndoscopyFindings(la_grade=LAGrade.B),
        hrm=HRMFindings(peristalsis=Peristalsis.IEM, dci_median=300),
        ph_impedance=PhImpedance(test_state=PhStudyState.OFF_PPI, aet_pct=8.0),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.HEARTBURN],
        ),
    )
    result = classify(case)
    assert "True GERD + IEM" in result.combinations_flagged


def test_coexisting_gerd_and_eoe_flags_combination():
    case = CaseInput(
        endoscopy=EndoscopyFindings(
            la_grade=LAGrade.C,
            biopsy_finding=BiopsyFinding.EOSINOPHILS,
            eosinophils_per_hpf=35,
        ),
        hrm=HRMFindings(peristalsis=Peristalsis.NORMAL),
        ph_impedance=PhImpedance(test_state=PhStudyState.OFF_PPI, aet_pct=8.5),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.HEARTBURN, DominantSymptom.DYSPHAGIA],
        ),
    )
    result = classify(case)
    assert "True GERD + coexisting EoE" in result.combinations_flagged


def test_on_ppi_refractory_with_true_gerd_flags_refractory_combination():
    """On-PPI AET > 4% with true-GERD evidence → refractory combination."""
    case = CaseInput(
        endoscopy=EndoscopyFindings(la_grade=LAGrade.C),
        hrm=HRMFindings(peristalsis=Peristalsis.NORMAL),
        ph_impedance=PhImpedance(test_state=PhStudyState.ON_PPI, aet_pct=5.2),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.HEARTBURN],
        ),
    )
    result = classify(case)
    assert result.refractory_flag is True
    assert any(
        "Refractory" in s for s in result.combinations_flagged
    )


def test_simple_functional_case_has_no_combinations():
    case = CaseInput(
        endoscopy=EndoscopyFindings(la_grade=LAGrade.NONE),
        hrm=HRMFindings(peristalsis=Peristalsis.NORMAL),
        ph_impedance=PhImpedance(
            test_state=PhStudyState.OFF_PPI,
            aet_pct=1.5,
            symptom_index_pct=10,
        ),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.NONE,
            dominant_symptoms=[DominantSymptom.HEARTBURN],
        ),
    )
    result = classify(case)
    assert result.combinations_flagged == []

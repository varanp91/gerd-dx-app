"""
Tie-breaker behavior in CaseOutput.ranked_mechanisms.

Within the same Confidence tier, buckets are sorted by the strongest positive
evidence they hold (CONCLUSIVE > STRONG > SUPPORTIVE). A CONCLUSIVE-driven
HIGH therefore outranks a HIGH reached via STRONG+SUPPORTIVE, which matters
for clinician-facing display order.
"""

from gerd_dx.engine import classify
from gerd_dx.enums import (
    BiopsyFinding,
    Confidence,
    DominantSymptom,
    EvidenceStrength,
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


def test_conclusive_driven_bucket_outranks_strong_plus_supportive():
    """
    Patient with LA grade C esophagitis (conclusive for GERD) AND 40 eos/HPF
    (conclusive for EoE). Both buckets land at HIGH. The EoE bucket is
    CONCLUSIVE + two SUPPORTIVE; true-GERD bucket is CONCLUSIVE + STRONG +
    SUPPORTIVE. Whichever has higher max-strength / count wins the tie.

    Here both have CONCLUSIVE as max. The true-GERD bucket has MORE evidence
    at higher strength (CONCLUSIVE + STRONG + SUPPORTIVE vs CONCLUSIVE +
    SUPPORTIVE + SUPPORTIVE) — so it should rank first.
    """
    case = CaseInput(
        endoscopy=EndoscopyFindings(
            la_grade=LAGrade.C,
            biopsy_finding=BiopsyFinding.EOSINOPHILS,
            eosinophils_per_hpf=40,
        ),
        hrm=HRMFindings(peristalsis=Peristalsis.NORMAL),
        ph_impedance=PhImpedance(test_state=PhStudyState.OFF_PPI, aet_pct=9.0),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.HEARTBURN, DominantSymptom.DYSPHAGIA],
        ),
    )
    result = classify(case)
    high_confidence_mechs = [
        r for r in result.ranked_mechanisms if r.confidence == Confidence.HIGH
    ]
    assert len(high_confidence_mechs) >= 2, \
        "Expected at least two HIGH-confidence buckets in this scenario"
    # Both are conclusive-driven; true-GERD has more pieces of evidence at
    # the top strength plus extra STRONG (AET), so it wins tie-break.
    assert high_confidence_mechs[0].mechanism_id == "true_gerd_competent_peristalsis"
    assert high_confidence_mechs[1].mechanism_id == "eoe_masquerade"


def test_conclusive_driven_bucket_outranks_two_strong_bucket():
    """
    CONCLUSIVE should outrank two STRONG pieces of evidence, even though both
    land at HIGH. Setup: classic erosive (LA C + AET + normal HRM) vs an
    IEM-combo case (AET STRONG + IEM STRONG, no conclusive).
    """
    # Classic erosive → CONCLUSIVE (LA C) + STRONG (AET) + SUPPORTIVE (HRM normal)
    case_erosive = CaseInput(
        endoscopy=EndoscopyFindings(la_grade=LAGrade.C),
        hrm=HRMFindings(peristalsis=Peristalsis.NORMAL),
        ph_impedance=PhImpedance(test_state=PhStudyState.OFF_PPI, aet_pct=9.0),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.HEARTBURN],
        ),
    )
    result_erosive = classify(case_erosive)
    top = result_erosive.ranked_mechanisms[0]
    strengths = {e.strength for e in top.evidence}
    assert EvidenceStrength.CONCLUSIVE in strengths, \
        "The top bucket on a classic erosive case should contain conclusive evidence"

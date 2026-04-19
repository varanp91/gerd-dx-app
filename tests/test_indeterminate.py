"""
Indeterminate fallback bucket.

Fires when no mechanism bucket reaches LOW / MODERATE / HIGH confidence.
Provides a "refer to or discuss with a gastroenterologist" management path
so the output is never empty — the clinician always gets a clear next step.
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


def _nothing_fires_case() -> CaseInput:
    """Heartburn + partial PPI response, no objective workup data. Classic indeterminate."""
    return CaseInput(
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.HEARTBURN],
        ),
    )


def test_indeterminate_fires_when_no_mechanism_bucket_is_active():
    result = classify(_nothing_fires_case())
    top = result.ranked_mechanisms[0]
    assert top.mechanism_id == "indeterminate"
    assert top.confidence == Confidence.LOW
    assert any(
        e.rule_id == "INDETERMINATE_NO_BUCKET_FIRED" for e in top.evidence
    )


def test_indeterminate_does_not_fire_when_any_bucket_is_active():
    """A classic erosive case has a HIGH bucket — indeterminate should NOT be injected."""
    case = CaseInput(
        endoscopy=EndoscopyFindings(la_grade=LAGrade.C),
        hrm=HRMFindings(peristalsis=Peristalsis.NORMAL),
        ph_impedance=PhImpedance(test_state=PhStudyState.OFF_PPI, aet_pct=9.0),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.HEARTBURN],
        ),
    )
    result = classify(case)
    assert not any(
        r.mechanism_id == "indeterminate" for r in result.ranked_mechanisms
    )


def test_indeterminate_does_not_fire_when_only_functional_bucket_active():
    """Functional heartburn at HIGH is a real classification — no fallback needed."""
    case = CaseInput(
        endoscopy=EndoscopyFindings(la_grade=LAGrade.NONE),
        hrm=HRMFindings(peristalsis=Peristalsis.NORMAL),
        ph_impedance=PhImpedance(
            test_state=PhStudyState.OFF_PPI,
            aet_pct=1.4,
            symptom_index_pct=10,
        ),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.NONE,
            dominant_symptoms=[DominantSymptom.HEARTBURN],
        ),
    )
    result = classify(case)
    assert not any(
        r.mechanism_id == "indeterminate" for r in result.ranked_mechanisms
    )


def test_indeterminate_fires_when_only_ruled_out_buckets_present():
    """
    Off-PPI AET of 2% rules out true_gerd_competent_peristalsis. But without
    SI reported, the functional / hypersensitivity buckets cannot classify.
    Indeterminate should fire AND rank ABOVE the ruled-out bucket.
    """
    case = CaseInput(
        endoscopy=EndoscopyFindings(la_grade=LAGrade.NONE),
        hrm=HRMFindings(peristalsis=Peristalsis.NORMAL),
        ph_impedance=PhImpedance(
            test_state=PhStudyState.OFF_PPI,
            aet_pct=2.0,
            # symptom_index_pct intentionally absent
        ),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.HEARTBURN],
        ),
    )
    result = classify(case)
    ids_in_order = [r.mechanism_id for r in result.ranked_mechanisms]
    assert "indeterminate" in ids_in_order
    assert "true_gerd_competent_peristalsis" in ids_in_order
    # LOW sorts before RULED_OUT in the confidence order.
    assert ids_in_order.index("indeterminate") < ids_in_order.index(
        "true_gerd_competent_peristalsis"
    )


def test_indeterminate_management_tells_clinician_to_see_gi():
    result = classify(_nothing_fires_case())
    plan = result.management["indeterminate"]
    joined = " ".join(plan.first_line).lower()
    assert "gastroenterologist" in joined
    # Management must discourage premature anti-reflux surgery.
    proc_labels = [p.label.lower() for p in plan.procedural]
    assert any("deferred" in label for label in proc_labels)


def test_indeterminate_gets_a_management_plan_despite_low_confidence():
    """
    LOW confidence normally gets management (only RULED_OUT / INSUFFICIENT
    are skipped). Verify indeterminate actually appears in result.management.
    """
    result = classify(_nothing_fires_case())
    assert "indeterminate" in result.management

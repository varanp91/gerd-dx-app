"""Post-surgical sub-buckets: sleeve, myotomy, fundoplication."""

from gerd_dx.engine import classify
from gerd_dx.enums import (
    Confidence,
    DominantSymptom,
    LAGrade,
    PPIResponse,
    Peristalsis,
    PhStudyState,
    PriorSurgery,
)
from gerd_dx.models import (
    CaseInput,
    ClinicalContext,
    EndoscopyFindings,
    HRMFindings,
    PhImpedance,
)


def _acid_evidence_pieces():
    return dict(
        endoscopy=EndoscopyFindings(la_grade=LAGrade.B),
        hrm=HRMFindings(peristalsis=Peristalsis.NORMAL),
        ph_impedance=PhImpedance(test_state=PhStudyState.OFF_PPI, aet_pct=8.5),
    )


def test_post_sleeve_fires_and_routes_general_gerd_bucket_to_it():
    case = CaseInput(
        **_acid_evidence_pieces(),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.REGURGITATION],
            prior_bariatric_surgery=PriorSurgery.SLEEVE_GASTRECTOMY,
        ),
    )
    result = classify(case)
    top = result.ranked_mechanisms[0]
    assert top.mechanism_id == "post_sleeve_gerd"
    assert top.confidence == Confidence.HIGH

    # General true-GERD bucket is routed / ruled out.
    general = next(
        r for r in result.ranked_mechanisms
        if r.mechanism_id == "true_gerd_competent_peristalsis"
    )
    assert general.confidence == Confidence.RULED_OUT
    assert any(e.rule_id == "ROUTED_TO_POST_SLEEVE" for e in general.evidence)


def test_post_myotomy_fires_on_heller_or_poem():
    for surgery in (PriorSurgery.HELLER_MYOTOMY, PriorSurgery.POEM):
        case = CaseInput(
            **_acid_evidence_pieces(),
            clinical=ClinicalContext(
                ppi_response=PPIResponse.PARTIAL,
                dominant_symptoms=[DominantSymptom.HEARTBURN],
                prior_anti_reflux_surgery=surgery,
            ),
        )
        result = classify(case)
        top = result.ranked_mechanisms[0]
        assert top.mechanism_id == "post_myotomy_gerd", f"failed for {surgery.value}"
        general = next(
            r for r in result.ranked_mechanisms
            if r.mechanism_id == "true_gerd_competent_peristalsis"
        )
        assert general.confidence == Confidence.RULED_OUT


def test_post_fundoplication_failure_fires_on_nissen():
    case = CaseInput(
        **_acid_evidence_pieces(),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.HEARTBURN],
            prior_anti_reflux_surgery=PriorSurgery.NISSEN,
        ),
    )
    result = classify(case)
    top = result.ranked_mechanisms[0]
    assert top.mechanism_id == "post_fundoplication_failure"
    assert any(
        e.rule_id == "POST_FUNDOPLICATION_FAILURE_CONTEXT" for e in top.evidence
    )


def test_post_surgical_buckets_require_acid_evidence():
    """Prior sleeve without acid evidence → sleeve bucket does NOT fire."""
    case = CaseInput(
        hrm=HRMFindings(peristalsis=Peristalsis.NORMAL),
        ph_impedance=PhImpedance(
            test_state=PhStudyState.OFF_PPI,
            aet_pct=2.0,
            symptom_index_pct=10,
        ),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.HEARTBURN],
            prior_bariatric_surgery=PriorSurgery.SLEEVE_GASTRECTOMY,
        ),
    )
    result = classify(case)
    assert not any(r.mechanism_id == "post_sleeve_gerd" for r in result.ranked_mechanisms)


def test_iem_combo_also_routes_to_post_surgical():
    """A patient with IEM + prior sleeve + acid evidence → post_sleeve wins, IEM combo ruled out."""
    case = CaseInput(
        endoscopy=EndoscopyFindings(la_grade=LAGrade.B),
        hrm=HRMFindings(peristalsis=Peristalsis.IEM, dci_median=300),
        ph_impedance=PhImpedance(test_state=PhStudyState.OFF_PPI, aet_pct=8.5),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.HEARTBURN],
            prior_bariatric_surgery=PriorSurgery.SLEEVE_GASTRECTOMY,
        ),
    )
    result = classify(case)
    top = result.ranked_mechanisms[0]
    assert top.mechanism_id == "post_sleeve_gerd"
    iem = next(
        (r for r in result.ranked_mechanisms if r.mechanism_id == "true_gerd_with_iem"),
        None,
    )
    assert iem is not None
    assert iem.confidence == Confidence.RULED_OUT

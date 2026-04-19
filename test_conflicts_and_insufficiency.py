"""
Conflict + insufficient-input detection.

Each detector in conflicts.py gets a positive test (condition present →
string emitted) and where relevant a negative test (condition absent →
not emitted). The exact wording is not asserted; substring checks on the
clinically load-bearing keywords are used so future tightening of the
prose doesn't break the suite.
"""

from gerd_dx.engine import classify
from gerd_dx.enums import (
    BiopsyFinding,
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
    EndoFLIPFindings,
    GastricEmptyingStudy,
    HRMFindings,
    PhImpedance,
)


# ---------------------------------------------------------------------------
# Conflicts
# ---------------------------------------------------------------------------

def test_conflict_conclusive_endoscopy_plus_physiologic_aet():
    case = CaseInput(
        endoscopy=EndoscopyFindings(la_grade=LAGrade.D),
        ph_impedance=PhImpedance(test_state=PhStudyState.OFF_PPI, aet_pct=1.5),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.HEARTBURN],
        ),
    )
    result = classify(case)
    assert any("LA grade D" in c and "physiologic" in c for c in result.conflicts)


def test_conflict_achalasia_plus_pathologic_aet():
    case = CaseInput(
        hrm=HRMFindings(peristalsis=Peristalsis.ACHALASIA_II, irp_supine_mmhg=24),
        ph_impedance=PhImpedance(test_state=PhStudyState.OFF_PPI, aet_pct=8.2),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.NONE,
            dominant_symptoms=[DominantSymptom.REGURGITATION],
        ),
    )
    result = classify(case)
    assert any(
        "achalasia" in c.lower() and "stasis" in c.lower() for c in result.conflicts
    )


def test_conflict_achalasia_plus_full_ppi_response():
    case = CaseInput(
        hrm=HRMFindings(peristalsis=Peristalsis.ACHALASIA_I),
        endoflip=EndoFLIPFindings(egj_di_60ml=1.5),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.FULL,
            dominant_symptoms=[DominantSymptom.REGURGITATION],
        ),
    )
    result = classify(case)
    assert any("implausible" in c.lower() for c in result.conflicts)


def test_conflict_eos_count_vs_biopsy_label_mismatch():
    """Eos >= 15 but biopsy_finding still labeled NORMAL → input inconsistency."""
    case = CaseInput(
        endoscopy=EndoscopyFindings(
            biopsy_finding=BiopsyFinding.NORMAL,  # mis-labeled
            eosinophils_per_hpf=30,
        ),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.DYSPHAGIA],
        ),
    )
    result = classify(case)
    assert any("biopsy_finding" in c.lower() for c in result.conflicts)


def test_conflict_full_ppi_response_vs_functional_heartburn_classification():
    case = CaseInput(
        endoscopy=EndoscopyFindings(la_grade=LAGrade.NONE),
        hrm=HRMFindings(peristalsis=Peristalsis.NORMAL),
        ph_impedance=PhImpedance(
            test_state=PhStudyState.OFF_PPI,
            aet_pct=1.2,
            symptom_index_pct=10,
        ),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.FULL,
            dominant_symptoms=[DominantSymptom.HEARTBURN],
        ),
    )
    result = classify(case)
    assert any(
        "functional heartburn" in c.lower() and "placebo" in c.lower()
        for c in result.conflicts
    )


def test_no_conflicts_for_clean_erosive_case():
    """Classic erosive GERD with consistent data should emit zero conflicts."""
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
    assert result.conflicts == []


# ---------------------------------------------------------------------------
# Insufficient inputs
# ---------------------------------------------------------------------------

def test_insufficient_no_ph_impedance_in_ppi_refractory_workup():
    """PPI nonresponse + no conclusive endoscopy + no pH study → flag."""
    case = CaseInput(
        endoscopy=EndoscopyFindings(la_grade=LAGrade.A),
        hrm=HRMFindings(peristalsis=Peristalsis.NORMAL),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.NONE,
            dominant_symptoms=[DominantSymptom.HEARTBURN],
        ),
    )
    result = classify(case)
    assert any("pH-impedance" in g for g in result.insufficient_inputs)


def test_no_ph_gap_when_conclusive_endoscopy_present():
    """LA grade C already conclusive → no need to flag missing pH study."""
    case = CaseInput(
        endoscopy=EndoscopyFindings(la_grade=LAGrade.C),
        hrm=HRMFindings(peristalsis=Peristalsis.NORMAL),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.HEARTBURN],
        ),
    )
    result = classify(case)
    assert not any("pH-impedance" in g for g in result.insufficient_inputs)


def test_insufficient_no_hrm_with_high_confidence_true_gerd():
    """HIGH true-GERD bucket without HRM → flag HRM as required pre-surgery."""
    case = CaseInput(
        endoscopy=EndoscopyFindings(la_grade=LAGrade.C),
        ph_impedance=PhImpedance(test_state=PhStudyState.OFF_PPI, aet_pct=9.0),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.HEARTBURN],
        ),
    )
    result = classify(case)
    assert any("HRM not documented" in g for g in result.insufficient_inputs)


def test_insufficient_borderline_aet_with_missing_adjuncts():
    case = CaseInput(
        ph_impedance=PhImpedance(test_state=PhStudyState.OFF_PPI, aet_pct=5.2),
        hrm=HRMFindings(peristalsis=Peristalsis.NORMAL),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.HEARTBURN],
        ),
    )
    result = classify(case)
    assert any(
        "Borderline AET" in g and "adjuncts" in g for g in result.insufficient_inputs
    )


def test_no_borderline_adjunct_gap_when_adjuncts_present():
    """All four adjuncts supplied → no gap message, even at borderline AET."""
    case = CaseInput(
        endoscopy=EndoscopyFindings(dilated_intercellular_spaces=True),
        ph_impedance=PhImpedance(
            test_state=PhStudyState.OFF_PPI,
            aet_pct=5.2,
            total_reflux_episodes=75,
            mnbi_ohms=1200,
            pspw_pct=35,
        ),
        hrm=HRMFindings(peristalsis=Peristalsis.NORMAL),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.HEARTBURN],
        ),
    )
    result = classify(case)
    assert not any("Borderline AET" in g for g in result.insufficient_inputs)


def test_insufficient_no_biopsy_with_dysphagia():
    case = CaseInput(
        endoscopy=EndoscopyFindings(la_grade=LAGrade.NONE, eosinophils_per_hpf=None),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.NONE,
            dominant_symptoms=[DominantSymptom.DYSPHAGIA],
        ),
    )
    result = classify(case)
    assert any(
        "biopsies" in g.lower() and "eoe" in g.lower()
        for g in result.insufficient_inputs
    )


def test_insufficient_no_gastric_emptying_on_glp1_with_regurgitation():
    case = CaseInput(
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.REGURGITATION],
            contributing_medications=[ContributingMed.GLP1],
        ),
    )
    result = classify(case)
    assert any(
        "gastric emptying" in g.lower() and "glp1" in g.lower()
        for g in result.insufficient_inputs
    )


def test_no_gastric_emptying_gap_when_study_is_documented():
    case = CaseInput(
        gastric_emptying=GastricEmptyingStudy(retention_4h_pct=12),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.REGURGITATION],
            contributing_medications=[ContributingMed.GLP1],
        ),
    )
    result = classify(case)
    assert not any("gastric emptying" in g.lower() for g in result.insufficient_inputs)


def test_insufficient_only_on_ppi_study_without_conclusive_endoscopy():
    case = CaseInput(
        endoscopy=EndoscopyFindings(la_grade=LAGrade.A),
        ph_impedance=PhImpedance(
            test_state=PhStudyState.ON_PPI,
            aet_pct=2.5,
            symptom_index_pct=65,
        ),
        hrm=HRMFindings(peristalsis=Peristalsis.NORMAL),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.HEARTBURN],
        ),
    )
    result = classify(case)
    assert any(
        "on-PPI" in g and "off-PPI" in g for g in result.insufficient_inputs
    )


def test_no_conflicts_or_gaps_for_full_clean_workup():
    """A fully worked-up case should return empty conflicts and no hard gaps."""
    case = CaseInput(
        endoscopy=EndoscopyFindings(la_grade=LAGrade.C, eosinophils_per_hpf=3),
        hrm=HRMFindings(peristalsis=Peristalsis.NORMAL),
        ph_impedance=PhImpedance(
            test_state=PhStudyState.OFF_PPI,
            aet_pct=9.0,
            total_reflux_episodes=92,
            mnbi_ohms=1100,
            pspw_pct=40,
            symptom_index_pct=60,
        ),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.HEARTBURN],
        ),
    )
    result = classify(case)
    assert result.conflicts == []
    assert result.insufficient_inputs == []

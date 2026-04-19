"""
Management-layer tests.

Verifies:
  - Every implemented mechanism has a management plan.
  - Clinical conventions encoded in user directives:
      * H2RA wording is "as needed when breakthrough occurs"
      * MSA / LINX is NOT listed anywhere in procedural options
      * c-TIF appears with the 2–4 cm hernia window
      * IEM bucket marks Nissen contraindicated with a reason
      * Achalasia plan tells the clinician to NOT pursue anti-reflux therapy
  - Contributing-factor modifiers render into the top-ranked plan.
  - Management is not emitted for RULED_OUT / INSUFFICIENT buckets.
"""

from gerd_dx.engine import classify
from gerd_dx.enums import (
    ContributingMed,
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
    EndoFLIPFindings,
    HRMFindings,
    PhImpedance,
)
from gerd_dx.rules.management import load_management_catalog
from gerd_dx.rules.mechanisms import ALL_MECHANISMS


def test_every_mechanism_has_a_management_plan():
    catalog = load_management_catalog()
    missing = [
        rule.mechanism_id
        for rule in ALL_MECHANISMS
        if catalog.plan_for(rule.mechanism_id) is None
    ]
    assert missing == [], f"Missing management plans for: {missing}"


def test_msa_is_never_listed_in_any_procedural_option():
    """User removed MSA from the procedural menu entirely."""
    catalog = load_management_catalog()
    for mech_id in catalog.mechanism_ids:
        plan = catalog.plan_for(mech_id)
        for proc in plan.procedural:
            id_text = proc.id.lower()
            label_text = proc.label.lower()
            assert "magnetic_sphincter" not in id_text, f"MSA listed in {mech_id}"
            assert "linx" not in id_text, f"LINX listed in {mech_id}"
            assert "magnetic sphincter" not in label_text, f"MSA label in {mech_id}"
            assert "linx" not in label_text, f"LINX label in {mech_id}"


def test_h2ra_wording_is_breakthrough_not_bedtime():
    """Every H2RA mention should say 'as needed when breakthrough occurs'."""
    catalog = load_management_catalog()
    for mech_id in catalog.mechanism_ids:
        plan = catalog.plan_for(mech_id)
        all_strings = [*plan.first_line, *plan.escalation]
        for s in all_strings:
            if "h2ra" in s.lower():
                assert "breakthrough" in s.lower(), (
                    f"H2RA line in {mech_id} should mention 'breakthrough': {s!r}"
                )
                assert "bedtime" not in s.lower(), (
                    f"H2RA in {mech_id} should not be limited to bedtime: {s!r}"
                )


def test_c_tif_appears_with_2_4cm_window():
    """c-TIF must be listed with an indication naming the 2–4 cm hernia window."""
    catalog = load_management_catalog()
    found_c_tif_with_window = False
    for mech_id in catalog.mechanism_ids:
        plan = catalog.plan_for(mech_id)
        for proc in plan.procedural:
            if "c_tif" in proc.id.lower() or "c-tif" in proc.label.lower():
                if proc.indication and "2" in proc.indication and "4" in proc.indication:
                    found_c_tif_with_window = True
    assert found_c_tif_with_window, \
        "c-TIF must appear with its 2–4 cm hiatal hernia window in at least one plan"


def test_iem_bucket_contraindicates_nissen_with_reason():
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
    plan = result.management["true_gerd_with_iem"]
    nissen_entries = [p for p in plan.procedural if p.id == "nissen"]
    assert nissen_entries, "Nissen should still be listed in the IEM plan, just marked contraindicated"
    nissen = nissen_entries[0]
    assert nissen.contraindicated is True
    assert nissen.reason is not None
    assert "dysphagia" in nissen.reason.lower()


def test_iem_bucket_still_offers_tif_and_c_tif():
    """User specified TIF / c-TIF can still be considered in IEM."""
    catalog = load_management_catalog()
    plan = catalog.plan_for("true_gerd_with_iem")
    procedural_ids = {p.id for p in plan.procedural}
    assert "tif" in procedural_ids
    assert "c_tif" in procedural_ids


def test_achalasia_plan_tells_clinician_not_to_pursue_anti_reflux():
    case = CaseInput(
        hrm=HRMFindings(peristalsis=Peristalsis.ACHALASIA_II, irp_supine_mmhg=24),
        endoflip=EndoFLIPFindings(egj_di_60ml=1.4),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.NONE,
            dominant_symptoms=[DominantSymptom.REGURGITATION],
        ),
    )
    result = classify(case)
    plan = result.management["achalasia_masquerade"]
    combined = " ".join(plan.first_line).lower()
    assert "do not" in combined and "anti-reflux" in combined


def test_post_sleeve_plan_contraindicates_fundoplication():
    case = CaseInput(
        endoscopy=EndoscopyFindings(la_grade=LAGrade.B),
        hrm=HRMFindings(peristalsis=Peristalsis.NORMAL),
        ph_impedance=PhImpedance(test_state=PhStudyState.OFF_PPI, aet_pct=8.0),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.REGURGITATION],
            prior_bariatric_surgery=PriorSurgery.SLEEVE_GASTRECTOMY,
        ),
    )
    result = classify(case)
    plan = result.management["post_sleeve_gerd"]
    # RYGB conversion should be the definitive surgical option
    assert any("roux-en-y" in p.label.lower() for p in plan.procedural)
    # Nissen after sleeve must be flagged contraindicated
    nissen_after = [p for p in plan.procedural if "nissen" in p.id.lower()]
    assert nissen_after and nissen_after[0].contraindicated is True


def test_contributing_factors_render_into_top_plans():
    case = CaseInput(
        endoscopy=EndoscopyFindings(la_grade=LAGrade.C),
        hrm=HRMFindings(peristalsis=Peristalsis.NORMAL),
        ph_impedance=PhImpedance(test_state=PhStudyState.OFF_PPI, aet_pct=8.5),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.PARTIAL,
            dominant_symptoms=[DominantSymptom.HEARTBURN],
            contributing_medications=[ContributingMed.GLP1, ContributingMed.CCB],
        ),
    )
    result = classify(case)
    plan = result.management["true_gerd_competent_peristalsis"]
    modifiers_joined = " ".join(plan.modifiers_from_contributing_factors).lower()
    assert "glp-1" in modifiers_joined
    assert "calcium channel" in modifiers_joined


def test_pregnancy_modifier_prefers_lifestyle_alginate_h2ra():
    case = CaseInput(
        endoscopy=EndoscopyFindings(la_grade=LAGrade.A),
        ph_impedance=PhImpedance(
            test_state=PhStudyState.OFF_PPI, aet_pct=2.5, symptom_index_pct=70
        ),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.NOT_TRIED,
            dominant_symptoms=[DominantSymptom.HEARTBURN],
            pregnant=True,
        ),
    )
    result = classify(case)
    assert "pregnancy" in result.contributing_factors
    # At least one management plan should carry the pregnancy modifier string.
    any_plan_with_pregnancy = any(
        any("pregnancy" in m.lower() for m in plan.modifiers_from_contributing_factors)
        for plan in result.management.values()
    )
    assert any_plan_with_pregnancy


def test_ruled_out_buckets_do_not_receive_management_plans():
    """A case where a bucket is RULED_OUT should NOT appear in result.management."""
    case = CaseInput(
        endoscopy=EndoscopyFindings(la_grade=LAGrade.NONE),
        hrm=HRMFindings(peristalsis=Peristalsis.NORMAL),
        ph_impedance=PhImpedance(
            test_state=PhStudyState.OFF_PPI, aet_pct=1.8, symptom_index_pct=10
        ),
        clinical=ClinicalContext(
            ppi_response=PPIResponse.NONE,
            dominant_symptoms=[DominantSymptom.HEARTBURN],
        ),
    )
    result = classify(case)
    # true_gerd_competent_peristalsis is RULED_OUT in this functional case.
    assert "true_gerd_competent_peristalsis" not in result.management
    # functional_heartburn is the active bucket.
    assert "functional_heartburn" in result.management

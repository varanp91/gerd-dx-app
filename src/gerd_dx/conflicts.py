"""
Conflict and insufficiency detection.

These checks run AFTER classification and populate:
  - CaseOutput.conflicts:            contradictions in the input data that
                                     the clinician should reconcile before
                                     acting on the classification.
  - CaseOutput.insufficient_inputs:  gaps in the workup that limit what the
                                     classifier can safely conclude.

Neither list changes the ranked mechanism output. The classifier still emits
its best call; these fields tell the clinician what to question or gather.
"""

from __future__ import annotations

from .enums import (
    BiopsyFinding,
    Confidence,
    ContributingMed,
    DominantSymptom,
    LAGrade,
    PPIResponse,
    Peristalsis,
    PhStudyState,
)
from .models import CaseInput
from .reasoning import MechanismResult
from .rules import Thresholds


_ACHALASIA_PATTERNS = frozenset({
    Peristalsis.ACHALASIA_I,
    Peristalsis.ACHALASIA_II,
    Peristalsis.ACHALASIA_III,
})

_TRUE_GERD_BUCKETS_NEEDING_HRM = frozenset({
    "true_gerd_competent_peristalsis",
    "true_gerd_with_iem",
    "post_sleeve_gerd",
    "post_myotomy_gerd",
    "post_fundoplication_failure",
    "hiatal_hernia_dominant",
})


def _has_conclusive_endoscopy(case: CaseInput, t: Thresholds) -> bool:
    if case.endoscopy is None:
        return False
    endo = case.endoscopy
    if endo.la_grade.value in t.lyon.conclusive_endoscopic.la_grades:
        return True
    if (
        endo.barretts
        and endo.barretts_length_cm is not None
        and endo.barretts_length_cm >= t.lyon.conclusive_endoscopic.barrett_min_length_cm
    ):
        return True
    if endo.peptic_stricture:
        return True
    return False


def _describe_conclusive_endoscopy(case: CaseInput, t: Thresholds) -> str | None:
    if case.endoscopy is None:
        return None
    endo = case.endoscopy
    if endo.la_grade.value in t.lyon.conclusive_endoscopic.la_grades:
        return f"LA grade {endo.la_grade.value}"
    if (
        endo.barretts
        and endo.barretts_length_cm is not None
        and endo.barretts_length_cm >= t.lyon.conclusive_endoscopic.barrett_min_length_cm
    ):
        return f"Barrett's esophagus {endo.barretts_length_cm} cm"
    if endo.peptic_stricture:
        return "peptic stricture"
    return None


# ---------------------------------------------------------------------------
# Conflict detection — contradictions in the input data.
# ---------------------------------------------------------------------------

def detect_conflicts(
    case: CaseInput, results: list[MechanismResult], t: Thresholds
) -> list[str]:
    conflicts: list[str] = []

    # 1. Conclusive erosive endoscopy + physiologic off-PPI AET
    finding = _describe_conclusive_endoscopy(case, t)
    if (
        finding is not None
        and case.ph_impedance is not None
        and case.ph_impedance.test_state == PhStudyState.OFF_PPI
        and case.ph_impedance.aet_pct < t.lyon.aet.physiologic_upper
    ):
        conflicts.append(
            f"Conclusive erosive endoscopy ({finding}) coexists with physiologic "
            f"off-PPI AET ({case.ph_impedance.aet_pct}%). Reconcile: consider pill "
            "esophagitis, EoE, recent PPI exposure distorting the 'off-PPI' window, "
            "or test-state mis-labeling."
        )

    # 2. Achalasia HRM pattern + pathologic AET off-PPI
    if (
        case.hrm is not None
        and case.hrm.peristalsis in _ACHALASIA_PATTERNS
        and case.ph_impedance is not None
        and case.ph_impedance.test_state == PhStudyState.OFF_PPI
        and case.ph_impedance.aet_pct > t.lyon.aet.pathologic_lower
    ):
        conflicts.append(
            f"Achalasia HRM pattern ({case.hrm.peristalsis.value}) with elevated "
            f"off-PPI AET ({case.ph_impedance.aet_pct}%). 'Acid' likely reflects "
            "food stasis and fermentation in the dilated esophagus, not pathologic "
            "reflux. Achalasia management takes priority over anti-reflux therapy."
        )

    # 3. Achalasia HRM pattern + reported full PPI response
    if (
        case.hrm is not None
        and case.hrm.peristalsis in _ACHALASIA_PATTERNS
        and case.clinical.ppi_response == PPIResponse.FULL
    ):
        conflicts.append(
            "Achalasia HRM pattern with reported full PPI response — implausible. "
            "Achalasia symptoms (regurgitation, dysphagia, chest pain) should not "
            "resolve on PPI. Reconcile history vs HRM before treating."
        )

    # 4. Eosinophils >= threshold but biopsy_finding not labeled eosinophils
    if (
        case.endoscopy is not None
        and case.endoscopy.eosinophils_per_hpf is not None
        and case.endoscopy.eosinophils_per_hpf >= t.eoe.eos_per_hpf_min
        and case.endoscopy.biopsy_finding != BiopsyFinding.EOSINOPHILS
    ):
        conflicts.append(
            f"Peak eosinophils {case.endoscopy.eosinophils_per_hpf}/HPF "
            f"(>= {t.eoe.eos_per_hpf_min}) recorded but primary biopsy finding is "
            f"labeled '{case.endoscopy.biopsy_finding.value}'. Reconcile input — "
            "biopsy_finding should likely be EOSINOPHILS."
        )

    # 5. Full PPI response reported while top classification is functional heartburn
    functional_fh_active = any(
        r.mechanism_id == "functional_heartburn"
        and r.confidence in (Confidence.HIGH, Confidence.MODERATE)
        for r in results
    )
    if functional_fh_active and case.clinical.ppi_response == PPIResponse.FULL:
        conflicts.append(
            "Classification is functional heartburn (physiologic AET, negative SI) "
            "but patient reports full PPI response. Consider placebo effect, a "
            "prior acid component since treated, or imprecise symptom reporting "
            "before committing to a functional management pathway."
        )

    # 6. Hiatal hernia present flag set without plausible size
    if (
        case.endoscopy is not None
        and case.endoscopy.hiatal_hernia_present
        and case.endoscopy.hiatal_hernia_cm is not None
        and case.endoscopy.hiatal_hernia_cm == 0
    ):
        conflicts.append(
            "Hiatal hernia flagged as present but size recorded as 0 cm — reconcile "
            "measurement."
        )

    return conflicts


# ---------------------------------------------------------------------------
# Insufficiency detection — workup gaps that limit interpretation.
# ---------------------------------------------------------------------------

def detect_insufficient_inputs(
    case: CaseInput, results: list[MechanismResult], t: Thresholds
) -> list[str]:
    gaps: list[str] = []
    has_conclusive = _has_conclusive_endoscopy(case, t)

    # 1. No pH-impedance in PPI-refractory workup without conclusive endoscopy
    if (
        case.ph_impedance is None
        and case.clinical.ppi_response in (PPIResponse.PARTIAL, PPIResponse.NONE)
        and not has_conclusive
    ):
        gaps.append(
            "No pH-impedance study documented. For PPI-refractory symptoms without "
            "conclusive erosive endoscopy, off-PPI pH-impedance is required to "
            "distinguish true GERD, reflux hypersensitivity, and functional heartburn."
        )

    # 2. No HRM but a true-GERD-family bucket at HIGH (surgical planning implied)
    if case.hrm is None:
        surgical_candidate = any(
            r.mechanism_id in _TRUE_GERD_BUCKETS_NEEDING_HRM
            and r.confidence == Confidence.HIGH
            for r in results
        )
        if surgical_candidate:
            gaps.append(
                "HRM not documented. Before anti-reflux surgery, HRM is required to "
                "confirm intact peristalsis (or identify IEM / absent contractility "
                "that contraindicates complete fundoplication) and to exclude major "
                "motility disorders."
            )

    # 3. Borderline AET with adjuncts missing → cannot adjudicate the 4–6% band
    if (
        case.ph_impedance is not None
        and case.ph_impedance.test_state == PhStudyState.OFF_PPI
        and t.lyon.aet.physiologic_upper <= case.ph_impedance.aet_pct <= t.lyon.aet.pathologic_lower
    ):
        missing: list[str] = []
        if case.ph_impedance.total_reflux_episodes is None:
            missing.append("total reflux episodes")
        if case.ph_impedance.mnbi_ohms is None:
            missing.append("MNBI")
        if case.ph_impedance.pspw_pct is None:
            missing.append("PSPW")
        if case.endoscopy is None or not case.endoscopy.dilated_intercellular_spaces:
            # DIS is either not assessed or absent. We don't distinguish here.
            missing.append("DIS on biopsy (assessed + documented)")
        # At least 3 missing means we cannot reach the 3-adjunct bar in thresholds.
        if len(missing) >= 3:
            gaps.append(
                f"Borderline AET ({case.ph_impedance.aet_pct}%) in the Lyon 4–6% "
                f"band with insufficient adjuncts: missing {', '.join(missing)}. "
                "Cannot firmly adjudicate pathologic vs inconclusive — request these "
                "metrics from the pH-impedance report and biopsy."
            )

    # 4. Endoscopy done but no eosinophil count despite dysphagia or PPI nonresponse
    if (
        case.endoscopy is not None
        and case.endoscopy.eosinophils_per_hpf is None
    ):
        has_dysphagia = DominantSymptom.DYSPHAGIA in case.clinical.dominant_symptoms
        nonresponse = case.clinical.ppi_response in (PPIResponse.PARTIAL, PPIResponse.NONE)
        if has_dysphagia or nonresponse:
            gaps.append(
                "Esophageal biopsies (eosinophil count) not documented despite "
                "dysphagia or PPI nonresponse. EoE cannot be excluded — obtain "
                ">=6 biopsies from proximal + distal esophagus per ACG/AGREE."
            )

    # 5. No gastric emptying study when gastroparesis-aggravating meds + regurgitation
    if case.gastric_emptying is None:
        aggravating = {ContributingMed.GLP1, ContributingMed.OPIOID} & set(
            case.clinical.contributing_medications
        )
        if aggravating and DominantSymptom.REGURGITATION in case.clinical.dominant_symptoms:
            med_list = ", ".join(sorted(m.value for m in aggravating))
            gaps.append(
                f"No gastric emptying study documented; patient on {med_list} with "
                "regurgitation. Consider 4-hour gastric emptying scintigraphy to "
                "evaluate gastroparesis contribution."
            )

    # 6. Only on-PPI pH study + no conclusive endoscopy → primary dx still open
    if (
        case.ph_impedance is not None
        and case.ph_impedance.test_state == PhStudyState.ON_PPI
        and not has_conclusive
    ):
        gaps.append(
            "Only on-PPI pH-impedance available without conclusive erosive "
            "endoscopy. For primary diagnostic workup (as opposed to refractory "
            "workup), off-PPI pH-impedance is preferred to distinguish true GERD "
            "from functional overlap."
        )

    return gaps

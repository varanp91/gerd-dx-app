"""
Mechanism rule functions.

Each rule evaluates ONE mechanism bucket and returns a list of Evidence.
Confidence aggregation happens centrally in engine.py — rules MUST NOT decide
confidence themselves. This keeps scoring semantics consistent across every
bucket and makes rule files read as "which clinical findings matter here."

All thresholds come from the injected Thresholds object (loaded from
rules/thresholds.yaml). Never hard-code clinical numbers in this file.

Rule-writing conventions:
  - Rule IDs are UPPER_SNAKE_CASE, stable once shipped (tests key off them).
  - `trigger` strings include the actual input value so the trace is auditable.
  - `source` cites the guideline the rule derives from.
  - Combination buckets (e.g. "true GERD + IEM") return [] when either half
    of the combination is missing, so they don't fire spuriously.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

from ..enums import (
    ContributingMed,
    DominantSymptom,
    EGJMorphology,
    EvidenceStrength,
    PPIResponse,
    Peristalsis,
    PhStudyState,
    PriorSurgery,
)
from ..models import CaseInput
from ..reasoning import Evidence
from . import Thresholds


class MechanismRule(NamedTuple):
    mechanism_id: str
    label: str
    evaluate: Callable[[CaseInput, Thresholds], list[Evidence]]


SOURCE_LYON = "Lyon Consensus 2.0"
SOURCE_CC4 = "Chicago Classification v4.0"
SOURCE_ROME_IV = "Rome IV functional GI disorders"
SOURCE_AGREE = "AGREE 2018 / ACG EoE Guideline 2013"


# Major motility disorders per Chicago v4.0 that should disqualify a case
# from being classified as a *functional* esophageal disorder.
# IEM is intentionally NOT here — it's a minor disorder compatible with
# reflux hypersensitivity / functional heartburn workup.
_MAJOR_MOTILITY_DISORDERS = frozenset({
    Peristalsis.ACHALASIA_I,
    Peristalsis.ACHALASIA_II,
    Peristalsis.ACHALASIA_III,
    Peristalsis.EGJ_OUTFLOW_OBSTRUCTION,
    Peristalsis.DES,
    Peristalsis.JACKHAMMER,
    Peristalsis.ABSENT,
})


# ---------------------------------------------------------------------------
# Shared helpers.
#
# These build Evidence objects from raw case data and are called by multiple
# mechanism rules. Centralizing them means a change to (e.g.) how borderline
# AET is adjudicated propagates to every bucket that cares about it.
# ---------------------------------------------------------------------------

def _lyon_conclusive_endoscopy(
    case: CaseInput, t: Thresholds, mechanism_id: str
) -> list[Evidence]:
    """Conclusive endoscopic findings per Lyon 2.0: LA C/D, Barrett's >=1cm, peptic stricture."""
    ev: list[Evidence] = []
    if case.endoscopy is None:
        return ev
    endo = case.endoscopy

    if endo.la_grade.value in t.lyon.conclusive_endoscopic.la_grades:
        ev.append(Evidence(
            rule_id="LYON_CONCLUSIVE_LA_CD",
            mechanism_id=mechanism_id,
            strength=EvidenceStrength.CONCLUSIVE,
            trigger=f"LA grade {endo.la_grade.value} esophagitis",
            source=SOURCE_LYON,
        ))
    if (
        endo.barretts
        and endo.barretts_length_cm is not None
        and endo.barretts_length_cm >= t.lyon.conclusive_endoscopic.barrett_min_length_cm
    ):
        ev.append(Evidence(
            rule_id="LYON_CONCLUSIVE_BARRETT",
            mechanism_id=mechanism_id,
            strength=EvidenceStrength.CONCLUSIVE,
            trigger=f"Barrett's esophagus, length {endo.barretts_length_cm} cm "
                    f"(>= {t.lyon.conclusive_endoscopic.barrett_min_length_cm} cm)",
            source=SOURCE_LYON,
        ))
    if endo.peptic_stricture and t.lyon.conclusive_endoscopic.peptic_stricture:
        ev.append(Evidence(
            rule_id="LYON_CONCLUSIVE_STRICTURE",
            mechanism_id=mechanism_id,
            strength=EvidenceStrength.CONCLUSIVE,
            trigger="Peptic stricture on endoscopy",
            source=SOURCE_LYON,
        ))
    return ev


def _lyon_aet_offppi(
    case: CaseInput, t: Thresholds, mechanism_id: str
) -> list[Evidence]:
    """
    AET-based evidence from an OFF-PPI pH-impedance study.
    Emits STRONG for pathologic AET, EXCLUSIONARY for physiologic AET, and
    adjudicates the 4–6% borderline band using adjuncts per Lyon 2.0.
    """
    ev: list[Evidence] = []
    if case.ph_impedance is None or case.ph_impedance.test_state != PhStudyState.OFF_PPI:
        return ev
    ph = case.ph_impedance
    aet = ph.aet_pct

    if aet > t.lyon.aet.pathologic_lower:
        ev.append(Evidence(
            rule_id="LYON_AET_PATHOLOGIC",
            mechanism_id=mechanism_id,
            strength=EvidenceStrength.STRONG,
            trigger=f"AET {aet}% (> {t.lyon.aet.pathologic_lower}% off-PPI)",
            source=SOURCE_LYON,
        ))
    elif aet < t.lyon.aet.physiologic_upper:
        ev.append(Evidence(
            rule_id="LYON_AET_PHYSIOLOGIC",
            mechanism_id=mechanism_id,
            strength=EvidenceStrength.EXCLUSIONARY,
            trigger=f"AET {aet}% (< {t.lyon.aet.physiologic_upper}% off-PPI) — "
                    "rules out pathologic acid reflux",
            source=SOURCE_LYON,
        ))
    else:
        # Borderline 4–6% — adjudicate with adjuncts.
        positive_adjuncts: list[str] = []
        if (
            ph.total_reflux_episodes is not None
            and ph.total_reflux_episodes >= t.lyon.reflux_episodes.pathologic_lower
        ):
            positive_adjuncts.append(
                f"reflux episodes {ph.total_reflux_episodes} "
                f">= {t.lyon.reflux_episodes.pathologic_lower}"
            )
        if ph.mnbi_ohms is not None and ph.mnbi_ohms < t.lyon.adjunctive.mnbi_low_ohms:
            positive_adjuncts.append(
                f"MNBI {ph.mnbi_ohms} ohms < {t.lyon.adjunctive.mnbi_low_ohms}"
            )
        if ph.pspw_pct is not None and ph.pspw_pct < t.lyon.adjunctive.pspw_low_pct:
            positive_adjuncts.append(
                f"PSPW {ph.pspw_pct}% < {t.lyon.adjunctive.pspw_low_pct}%"
            )
        if (
            t.lyon.adjunctive.dis_supportive
            and case.endoscopy is not None
            and case.endoscopy.dilated_intercellular_spaces
        ):
            positive_adjuncts.append("Dilated intercellular spaces on biopsy")

        n = len(positive_adjuncts)
        need = t.lyon.borderline_adjudication.min_adjunctive_for_pathologic
        if n >= need:
            ev.append(Evidence(
                rule_id="LYON_AET_BORDERLINE_ADJUDICATED_POS",
                mechanism_id=mechanism_id,
                strength=EvidenceStrength.STRONG,
                trigger=f"AET {aet}% (borderline 4–6%) with {n} adjuncts positive "
                        f"(>= {need} required): {'; '.join(positive_adjuncts)}",
                source=SOURCE_LYON,
            ))
        else:
            ev.append(Evidence(
                rule_id="LYON_AET_BORDERLINE_INCONCLUSIVE",
                mechanism_id=mechanism_id,
                strength=EvidenceStrength.BORDERLINE,
                trigger=f"AET {aet}% (borderline); only {n} of {need} required "
                        f"adjuncts positive"
                        + (f" ({'; '.join(positive_adjuncts)})" if positive_adjuncts else ""),
                source=SOURCE_LYON,
            ))
    return ev


def _has_positive_acid_evidence(ev: list[Evidence]) -> bool:
    """True if any evidence in the list contributes positively to a GERD diagnosis."""
    positive_ids = {
        "LYON_CONCLUSIVE_LA_CD",
        "LYON_CONCLUSIVE_BARRETT",
        "LYON_CONCLUSIVE_STRICTURE",
        "LYON_AET_PATHOLOGIC",
        "LYON_AET_BORDERLINE_ADJUDICATED_POS",
    }
    return any(e.rule_id in positive_ids for e in ev)


# Groupings of PriorSurgery values used by post-surgical buckets.
_POST_SLEEVE_VALUES = frozenset({PriorSurgery.SLEEVE_GASTRECTOMY})
_POST_MYOTOMY_VALUES = frozenset({PriorSurgery.HELLER_MYOTOMY, PriorSurgery.POEM})
_POST_FUNDOPLICATION_VALUES = frozenset({
    PriorSurgery.NISSEN,
    PriorSurgery.TOUPET,
    PriorSurgery.TIF,
    PriorSurgery.C_TIF,
})


def _post_surgical_routing(
    case: CaseInput, mechanism_id: str
) -> list[Evidence]:
    """
    For the GENERAL true-GERD buckets: when a patient has a relevant prior
    surgery, route them to the post-surgical sub-bucket instead. Emits
    EXCLUSIONARY on the caller so the general bucket ends up RULED_OUT
    with a pointer to where the classification landed.
    """
    ev: list[Evidence] = []
    bariatric = case.clinical.prior_bariatric_surgery
    anti_reflux = case.clinical.prior_anti_reflux_surgery

    if bariatric in _POST_SLEEVE_VALUES:
        ev.append(Evidence(
            rule_id="ROUTED_TO_POST_SLEEVE",
            mechanism_id=mechanism_id,
            strength=EvidenceStrength.EXCLUSIONARY,
            trigger=f"Prior bariatric surgery: {bariatric.value} — routed to "
                    "post-sleeve GERD bucket",
            source="Clinical history",
        ))
    if anti_reflux in _POST_MYOTOMY_VALUES:
        ev.append(Evidence(
            rule_id="ROUTED_TO_POST_MYOTOMY",
            mechanism_id=mechanism_id,
            strength=EvidenceStrength.EXCLUSIONARY,
            trigger=f"Prior anti-reflux/motility surgery: {anti_reflux.value} — "
                    "routed to post-myotomy GERD bucket",
            source="Clinical history",
        ))
    if anti_reflux in _POST_FUNDOPLICATION_VALUES:
        ev.append(Evidence(
            rule_id="ROUTED_TO_POST_FUNDOPLICATION",
            mechanism_id=mechanism_id,
            strength=EvidenceStrength.EXCLUSIONARY,
            trigger=f"Prior fundoplication: {anti_reflux.value} — routed to "
                    "post-fundoplication failure bucket",
            source="Clinical history",
        ))
    return ev


def _eoe_or_major_motility_excludes(
    case: CaseInput, t: Thresholds, mechanism_id: str
) -> list[Evidence]:
    """
    EoE on biopsy and any major motility disorder on HRM disqualify the
    functional esophageal buckets (hypersensitivity, functional heartburn).
    EoE is a histologic diagnosis that takes precedence; major motility
    disorders require their own workup and should not be re-labeled as
    functional symptoms.
    """
    ev: list[Evidence] = []
    if (
        case.endoscopy is not None
        and case.endoscopy.eosinophils_per_hpf is not None
        and case.endoscopy.eosinophils_per_hpf >= t.eoe.eos_per_hpf_min
    ):
        ev.append(Evidence(
            rule_id="EOE_EXCLUDES_FUNCTIONAL",
            mechanism_id=mechanism_id,
            strength=EvidenceStrength.EXCLUSIONARY,
            trigger=f"Eosinophils {case.endoscopy.eosinophils_per_hpf}/HPF "
                    f"(>= {t.eoe.eos_per_hpf_min}) — EoE takes precedence",
            source=SOURCE_AGREE,
        ))
    if case.hrm is not None and case.hrm.peristalsis in _MAJOR_MOTILITY_DISORDERS:
        ev.append(Evidence(
            rule_id="MAJOR_MOTILITY_DISORDER_EXCLUDES_FUNCTIONAL",
            mechanism_id=mechanism_id,
            strength=EvidenceStrength.EXCLUSIONARY,
            trigger=f"HRM shows {case.hrm.peristalsis.value} — major motility "
                    "disorder, not a functional esophageal disorder",
            source=SOURCE_CC4,
        ))
    return ev


def _conclusive_endoscopy_excludes(
    case: CaseInput, t: Thresholds, mechanism_id: str
) -> list[Evidence]:
    """
    For buckets that are defined by the *absence* of structural GERD
    (reflux hypersensitivity, functional heartburn): emit EXCLUSIONARY when
    any Lyon-conclusive endoscopic finding is present.
    """
    ev: list[Evidence] = []
    if case.endoscopy is None:
        return ev
    endo = case.endoscopy

    if endo.la_grade.value in t.lyon.conclusive_endoscopic.la_grades:
        ev.append(Evidence(
            rule_id="CONCLUSIVE_ENDOSCOPY_LA_CD_EXCLUDES",
            mechanism_id=mechanism_id,
            strength=EvidenceStrength.EXCLUSIONARY,
            trigger=f"LA grade {endo.la_grade.value} is Lyon-conclusive for GERD",
            source=SOURCE_LYON,
        ))
    if (
        endo.barretts
        and endo.barretts_length_cm is not None
        and endo.barretts_length_cm >= t.lyon.conclusive_endoscopic.barrett_min_length_cm
    ):
        ev.append(Evidence(
            rule_id="CONCLUSIVE_ENDOSCOPY_BARRETT_EXCLUDES",
            mechanism_id=mechanism_id,
            strength=EvidenceStrength.EXCLUSIONARY,
            trigger=f"Barrett's {endo.barretts_length_cm} cm is Lyon-conclusive for GERD",
            source=SOURCE_LYON,
        ))
    if endo.peptic_stricture:
        ev.append(Evidence(
            rule_id="CONCLUSIVE_ENDOSCOPY_STRICTURE_EXCLUDES",
            mechanism_id=mechanism_id,
            strength=EvidenceStrength.EXCLUSIONARY,
            trigger="Peptic stricture is Lyon-conclusive for GERD",
            source=SOURCE_LYON,
        ))
    return ev


# ---------------------------------------------------------------------------
# Bucket 1: True acid GERD with competent peristalsis
# ---------------------------------------------------------------------------
def evaluate_true_gerd_competent_peristalsis(
    case: CaseInput, t: Thresholds
) -> list[Evidence]:
    """
    Fires when there is pathologic acid reflux evidence AND peristalsis is
    competent (or HRM not available). IEM / absent contractility / major
    motility disorders route to EXCLUSIONARY and land in other buckets.
    """
    M = "true_gerd_competent_peristalsis"
    ev = _lyon_conclusive_endoscopy(case, t, M)
    ev.extend(_lyon_aet_offppi(case, t, M))
    ev.extend(_post_surgical_routing(case, M))

    if case.hrm is not None and case.hrm.peristalsis is not None:
        per = case.hrm.peristalsis
        if per == Peristalsis.NORMAL:
            ev.append(Evidence(
                rule_id="HRM_NORMAL_PERISTALSIS",
                mechanism_id=M,
                strength=EvidenceStrength.SUPPORTIVE,
                trigger="HRM shows normal peristalsis",
                source=SOURCE_CC4,
            ))
        elif per in (Peristalsis.IEM, Peristalsis.ABSENT):
            ev.append(Evidence(
                rule_id="HRM_NOT_COMPETENT",
                mechanism_id=M,
                strength=EvidenceStrength.EXCLUSIONARY,
                trigger=f"HRM shows {per.value} — routed to the IEM / absent-"
                        "contractility bucket instead",
                source=SOURCE_CC4,
            ))
        elif per in (
            Peristalsis.ACHALASIA_I,
            Peristalsis.ACHALASIA_II,
            Peristalsis.ACHALASIA_III,
        ):
            ev.append(Evidence(
                rule_id="HRM_ACHALASIA_MASQUERADE",
                mechanism_id=M,
                strength=EvidenceStrength.EXCLUSIONARY,
                trigger=f"HRM shows {per.value} — major motility disorder, "
                        "not true GERD",
                source=SOURCE_CC4,
            ))
    return ev


# ---------------------------------------------------------------------------
# Bucket 2: True acid GERD with ineffective esophageal motility (IEM)
#
# COMBO bucket: fires only when BOTH conditions present. When either is
# missing, returns []. This ensures the single-condition buckets (true-GERD-
# competent or IEM-alone) don't lose their ranking.
# ---------------------------------------------------------------------------
def evaluate_true_gerd_with_iem(
    case: CaseInput, t: Thresholds
) -> list[Evidence]:
    M = "true_gerd_with_iem"

    # Gate 1: HRM must show IEM.
    if case.hrm is None or case.hrm.peristalsis != Peristalsis.IEM:
        return []

    # Gate 2: there must be positive acid reflux evidence.
    acid_ev = _lyon_conclusive_endoscopy(case, t, M)
    acid_ev.extend(_lyon_aet_offppi(case, t, M))
    if not _has_positive_acid_evidence(acid_ev):
        return []

    # Both gates passed — add the IEM finding.
    acid_ev.append(Evidence(
        rule_id="HRM_IEM_CONFIRMED",
        mechanism_id=M,
        strength=EvidenceStrength.STRONG,
        trigger="HRM shows ineffective esophageal motility (IEM) — combination with "
                "confirmed acid reflux",
        source=SOURCE_CC4,
    ))
    # Post-surgical history routes this case to the relevant sub-bucket.
    acid_ev.extend(_post_surgical_routing(case, M))
    return acid_ev


# ---------------------------------------------------------------------------
# Bucket 3: Reflux hypersensitivity
#
# Normal AET off-PPI + positive Symptom Index (SI >= 50%).
# Any conclusive endoscopic finding excludes this bucket (those patients have
# structural GERD, not hypersensitivity).
# ---------------------------------------------------------------------------
def evaluate_reflux_hypersensitivity(
    case: CaseInput, t: Thresholds
) -> list[Evidence]:
    M = "reflux_hypersensitivity"

    # Requires an off-PPI study with SI reported.
    if case.ph_impedance is None or case.ph_impedance.test_state != PhStudyState.OFF_PPI:
        return []
    ph = case.ph_impedance
    if ph.symptom_index_pct is None:
        return []
    if ph.aet_pct >= t.lyon.aet.physiologic_upper:
        # AET not physiologic — hypersensitivity requires normal AET.
        return []

    ev: list[Evidence] = []
    if ph.symptom_index_pct >= t.symptom_association.si_positive_pct:
        ev.append(Evidence(
            rule_id="SI_POSITIVE",
            mechanism_id=M,
            strength=EvidenceStrength.STRONG,
            trigger=f"Symptom Index {ph.symptom_index_pct}% "
                    f">= {t.symptom_association.si_positive_pct}%",
            source=SOURCE_LYON,
        ))
        ev.append(Evidence(
            rule_id="AET_PHYSIOLOGIC_OFF_PPI",
            mechanism_id=M,
            strength=EvidenceStrength.SUPPORTIVE,
            trigger=f"AET {ph.aet_pct}% (< {t.lyon.aet.physiologic_upper}% off-PPI)",
            source=SOURCE_LYON,
        ))
    else:
        # SI negative → this is functional heartburn territory, not
        # hypersensitivity. Emit nothing so we don't compete with that bucket.
        return []

    # Conclusive endoscopy, EoE on biopsy, or major motility disorder all
    # exclude a functional esophageal diagnosis.
    ev.extend(_conclusive_endoscopy_excludes(case, t, M))
    ev.extend(_eoe_or_major_motility_excludes(case, t, M))
    return ev


# ---------------------------------------------------------------------------
# Bucket 4: Functional heartburn
#
# Normal AET off-PPI + negative Symptom Index (SI < 50%). Rome IV criteria.
# Any conclusive endoscopic finding excludes.
# ---------------------------------------------------------------------------
def evaluate_functional_heartburn(
    case: CaseInput, t: Thresholds
) -> list[Evidence]:
    M = "functional_heartburn"

    if case.ph_impedance is None or case.ph_impedance.test_state != PhStudyState.OFF_PPI:
        return []
    ph = case.ph_impedance
    if ph.symptom_index_pct is None:
        return []
    if ph.aet_pct >= t.lyon.aet.physiologic_upper:
        return []

    ev: list[Evidence] = []
    if ph.symptom_index_pct < t.symptom_association.si_positive_pct:
        ev.append(Evidence(
            rule_id="SI_NEGATIVE",
            mechanism_id=M,
            strength=EvidenceStrength.STRONG,
            trigger=f"Symptom Index {ph.symptom_index_pct}% "
                    f"< {t.symptom_association.si_positive_pct}%",
            source=SOURCE_ROME_IV,
        ))
        ev.append(Evidence(
            rule_id="AET_PHYSIOLOGIC_OFF_PPI",
            mechanism_id=M,
            strength=EvidenceStrength.SUPPORTIVE,
            trigger=f"AET {ph.aet_pct}% (< {t.lyon.aet.physiologic_upper}% off-PPI)",
            source=SOURCE_LYON,
        ))
    else:
        return []  # SI positive → hypersensitivity bucket owns this case.

    # Conclusive endoscopy, EoE on biopsy, or major motility disorder all
    # exclude a functional esophageal diagnosis.
    ev.extend(_conclusive_endoscopy_excludes(case, t, M))
    ev.extend(_eoe_or_major_motility_excludes(case, t, M))
    return ev


# ---------------------------------------------------------------------------
# Bucket 5: Eosinophilic esophagitis masquerading as GERD
#
# EoE is a histologic diagnosis: >= 15 eos/HPF on biopsy. This bucket does NOT
# exclude true GERD (the two can coexist) — combination-flag detection in
# engine.py surfaces "True GERD + coexisting EoE" when both fire. It DOES
# conflict with the functional esophageal buckets (handled there).
# ---------------------------------------------------------------------------
def evaluate_eoe_masquerade(case: CaseInput, t: Thresholds) -> list[Evidence]:
    M = "eoe_masquerade"
    ev: list[Evidence] = []
    if case.endoscopy is None or case.endoscopy.eosinophils_per_hpf is None:
        return ev

    eos = case.endoscopy.eosinophils_per_hpf
    if eos >= t.eoe.eos_per_hpf_min:
        ev.append(Evidence(
            rule_id="EOE_HISTOLOGIC_THRESHOLD",
            mechanism_id=M,
            strength=EvidenceStrength.CONCLUSIVE,
            trigger=f"Peak eosinophils {eos}/HPF (>= {t.eoe.eos_per_hpf_min}/HPF) "
                    "on esophageal biopsy",
            source=SOURCE_AGREE,
        ))

    # Supportive clinical context (only if the histologic threshold already met).
    if ev:
        if case.clinical.ppi_response in (PPIResponse.NONE, PPIResponse.PARTIAL):
            ev.append(Evidence(
                rule_id="EOE_PPI_NONRESPONSE",
                mechanism_id=M,
                strength=EvidenceStrength.SUPPORTIVE,
                trigger=f"PPI response: {case.clinical.ppi_response.value} — "
                        "consistent with EoE presenting as PPI-refractory 'GERD'",
                source=SOURCE_AGREE,
            ))
        if DominantSymptom.DYSPHAGIA in case.clinical.dominant_symptoms:
            ev.append(Evidence(
                rule_id="EOE_DYSPHAGIA_DOMINANT",
                mechanism_id=M,
                strength=EvidenceStrength.SUPPORTIVE,
                trigger="Dysphagia is a dominant symptom — classic EoE presentation",
                source=SOURCE_AGREE,
            ))
    return ev


# ---------------------------------------------------------------------------
# Bucket 6: Achalasia (or other major motility disorder) masquerading as GERD
#
# Classic pitfall: regurgitation of undigested food + "heartburn" in a patient
# with achalasia. HRM is diagnostic (Chicago v4.0). EndoFLIP severely reduced
# DI at 60mL is strongly supportive.
# ---------------------------------------------------------------------------
def evaluate_achalasia_masquerade(case: CaseInput, t: Thresholds) -> list[Evidence]:
    M = "achalasia_masquerade"
    ev: list[Evidence] = []

    # HRM-defined achalasia — CC v4.0 diagnostic.
    if case.hrm is not None and case.hrm.peristalsis in (
        Peristalsis.ACHALASIA_I,
        Peristalsis.ACHALASIA_II,
        Peristalsis.ACHALASIA_III,
    ):
        ev.append(Evidence(
            rule_id="HRM_ACHALASIA_DIAGNOSTIC",
            mechanism_id=M,
            strength=EvidenceStrength.CONCLUSIVE,
            trigger=f"HRM shows {case.hrm.peristalsis.value} — "
                    "Chicago v4.0 diagnostic for achalasia",
            source=SOURCE_CC4,
        ))
    elif case.hrm is not None and case.hrm.peristalsis == Peristalsis.EGJ_OUTFLOW_OBSTRUCTION:
        ev.append(Evidence(
            rule_id="HRM_EGJOO",
            mechanism_id=M,
            strength=EvidenceStrength.STRONG,
            trigger="HRM shows EGJ outflow obstruction — requires exclusion of "
                    "mechanical obstruction; can be achalasia variant",
            source=SOURCE_CC4,
        ))

    # EndoFLIP — severely reduced DI at 60mL is strong corroboration.
    if case.endoflip is not None and case.endoflip.egj_di_60ml is not None:
        di = case.endoflip.egj_di_60ml
        if di <= t.endoflip.egj_di_severely_reduced:
            ev.append(Evidence(
                rule_id="ENDOFLIP_DI_SEVERELY_REDUCED",
                mechanism_id=M,
                strength=EvidenceStrength.STRONG,
                trigger=f"EndoFLIP EGJ-DI {di} mm^2/mmHg at 60mL "
                        f"(<= {t.endoflip.egj_di_severely_reduced}) — "
                        "severely reduced distensibility",
                source=SOURCE_CC4,
            ))
        elif di <= t.endoflip.egj_di_reduced:
            ev.append(Evidence(
                rule_id="ENDOFLIP_DI_REDUCED",
                mechanism_id=M,
                strength=EvidenceStrength.SUPPORTIVE,
                trigger=f"EndoFLIP EGJ-DI {di} mm^2/mmHg at 60mL "
                        f"(<= {t.endoflip.egj_di_reduced}) — reduced distensibility",
                source=SOURCE_CC4,
            ))

    # Only surface supportive clinical context when at least one positive finding
    # is already present; we don't want "dysphagia" alone to place a patient in
    # this bucket at LOW confidence.
    if ev:
        if DominantSymptom.DYSPHAGIA in case.clinical.dominant_symptoms:
            ev.append(Evidence(
                rule_id="ACHALASIA_DYSPHAGIA",
                mechanism_id=M,
                strength=EvidenceStrength.SUPPORTIVE,
                trigger="Dysphagia is a dominant symptom",
                source=SOURCE_CC4,
            ))
        if DominantSymptom.REGURGITATION in case.clinical.dominant_symptoms:
            ev.append(Evidence(
                rule_id="ACHALASIA_REGURG",
                mechanism_id=M,
                strength=EvidenceStrength.SUPPORTIVE,
                trigger="Regurgitation dominant — may represent retained esophageal "
                        "contents rather than reflux",
                source=SOURCE_CC4,
            ))
        if case.clinical.ppi_response == PPIResponse.NONE:
            ev.append(Evidence(
                rule_id="ACHALASIA_PPI_NONRESPONSE",
                mechanism_id=M,
                strength=EvidenceStrength.SUPPORTIVE,
                trigger="No response to PPI — consistent with non-acid mechanism",
                source=SOURCE_CC4,
            ))
    return ev


# ---------------------------------------------------------------------------
# Bucket 7: Hiatal hernia-dominant GERD
#
# Trigger: hiatal hernia ≥3cm OR (documented hiatal hernia + intrathoracic
# stomach). Intrathoracic stomach is the clinical anchor that distinguishes
# hernia-dominant cases from incidental hiatal laxity.
#
# Does NOT exclude true GERD — hernia-dominant commonly coexists with
# true acid GERD; combinations_flagged surfaces that overlay in engine.py.
# ---------------------------------------------------------------------------
def evaluate_hiatal_hernia_dominant(
    case: CaseInput, t: Thresholds
) -> list[Evidence]:
    M = "hiatal_hernia_dominant"
    if case.endoscopy is None:
        return []
    endo = case.endoscopy

    # Two entry paths for hernia presence. Intrathoracic stomach is required
    # in the second path because it's the clinical anchor for "dominant"
    # when the exact size isn't measured.
    large_measured = (
        endo.hiatal_hernia_cm is not None
        and endo.hiatal_hernia_cm >= t.hiatal_hernia.dominant_min_cm
    )
    qualifying_unmeasured = (
        endo.hiatal_hernia_cm is None
        and endo.hiatal_hernia_present
        and endo.intrathoracic_stomach
    )
    if not (large_measured or qualifying_unmeasured):
        return []

    ev: list[Evidence] = []
    if large_measured and endo.intrathoracic_stomach:
        ev.append(Evidence(
            rule_id="HH_LARGE_WITH_INTRATHORACIC",
            mechanism_id=M,
            strength=EvidenceStrength.CONCLUSIVE,
            trigger=f"Hiatal hernia {endo.hiatal_hernia_cm} cm "
                    f"(>= {t.hiatal_hernia.dominant_min_cm} cm) with intrathoracic stomach",
            source=SOURCE_LYON,
        ))
    elif large_measured:
        ev.append(Evidence(
            rule_id="HH_LARGE",
            mechanism_id=M,
            strength=EvidenceStrength.STRONG,
            trigger=f"Hiatal hernia {endo.hiatal_hernia_cm} cm "
                    f"(>= {t.hiatal_hernia.dominant_min_cm} cm)",
            source=SOURCE_LYON,
        ))
    else:  # qualifying_unmeasured
        ev.append(Evidence(
            rule_id="HH_DOCUMENTED_WITH_INTRATHORACIC",
            mechanism_id=M,
            strength=EvidenceStrength.STRONG,
            trigger="Hiatal hernia documented (size not measured) with intrathoracic stomach",
            source=SOURCE_LYON,
        ))

    # HRM type III EGJ morphology corroborates a large hernia.
    if case.hrm is not None and case.hrm.egj_morphology == EGJMorphology.TYPE_III:
        ev.append(Evidence(
            rule_id="HRM_EGJ_TYPE_III",
            mechanism_id=M,
            strength=EvidenceStrength.SUPPORTIVE,
            trigger="HRM shows type III EGJ morphology (LES-crural separation >= 3cm)",
            source=SOURCE_CC4,
        ))

    # Regurgitation-dominant presentation aligns with mechanical hernia physiology.
    if DominantSymptom.REGURGITATION in case.clinical.dominant_symptoms:
        ev.append(Evidence(
            rule_id="HH_REGURG_DOMINANT",
            mechanism_id=M,
            strength=EvidenceStrength.SUPPORTIVE,
            trigger="Regurgitation is a dominant symptom — consistent with "
                    "mechanical hernia physiology",
            source=SOURCE_LYON,
        ))
    return ev


# ---------------------------------------------------------------------------
# Bucket 8: Non-acid / weakly acidic reflux on PPI
#
# Patient on PPI with positive symptom-reflux association but the refluxate
# is weakly acidic or non-acid. Distinct from "refractory acid GERD"
# (on-PPI AET >4%, handled as a flag, not a bucket).
# ---------------------------------------------------------------------------
def evaluate_non_acid_reflux_on_ppi(
    case: CaseInput, t: Thresholds
) -> list[Evidence]:
    M = "non_acid_reflux_on_ppi"
    if case.ph_impedance is None or case.ph_impedance.test_state != PhStudyState.ON_PPI:
        return []
    ph = case.ph_impedance
    if ph.symptom_index_pct is None:
        return []
    if ph.symptom_index_pct < t.symptom_association.si_positive_pct:
        return []  # No symptom-reflux association → not this bucket.

    ev: list[Evidence] = [
        Evidence(
            rule_id="NAB_SI_POSITIVE_ON_PPI",
            mechanism_id=M,
            strength=EvidenceStrength.STRONG,
            trigger=f"Symptom Index {ph.symptom_index_pct}% "
                    f">= {t.symptom_association.si_positive_pct}% on PPI",
            source=SOURCE_LYON,
        )
    ]

    # Refluxate composition: weakly-acidic + non-acid dominance confirms NAB.
    if ph.weakly_acidic_pct is not None and ph.non_acid_pct is not None:
        combined = ph.weakly_acidic_pct + ph.non_acid_pct
        if combined > 50:
            ev.append(Evidence(
                rule_id="NAB_NONACID_PREDOMINANT",
                mechanism_id=M,
                strength=EvidenceStrength.STRONG,
                trigger=f"Weakly acidic + non-acid episodes {combined:.0f}% "
                        f"(> 50%) of total reflux on PPI",
                source=SOURCE_LYON,
            ))
    elif ph.aet_pct < t.lyon.aet.physiologic_upper:
        # No composition data, but AET is controlled — symptoms aren't from acid.
        ev.append(Evidence(
            rule_id="NAB_AET_CONTROLLED_ON_PPI",
            mechanism_id=M,
            strength=EvidenceStrength.SUPPORTIVE,
            trigger=f"AET {ph.aet_pct}% on PPI (< {t.lyon.aet.physiologic_upper}%) — "
                    "acid is controlled; ongoing symptoms imply non-acid mechanism",
            source=SOURCE_LYON,
        ))
    return ev


# ---------------------------------------------------------------------------
# Bucket 9: Gastroparesis-driven reflux
#
# Trigger: retention >=10% at 4h or >=60% at 2h on gastric emptying study.
# Contributing meds (GLP-1, opioid) are NOT required but are surfaced in
# the separate contributing-factors list by the engine.
# ---------------------------------------------------------------------------
def evaluate_gastroparesis_driven_reflux(
    case: CaseInput, t: Thresholds
) -> list[Evidence]:
    M = "gastroparesis_driven_reflux"
    if case.gastric_emptying is None:
        return []
    ge = case.gastric_emptying

    ev: list[Evidence] = []
    has_positive_retention = False
    if (
        ge.retention_4h_pct is not None
        and ge.retention_4h_pct >= t.gastroparesis.retention_4h_min_pct
    ):
        ev.append(Evidence(
            rule_id="GE_RETENTION_4H",
            mechanism_id=M,
            strength=EvidenceStrength.STRONG,
            trigger=f"Gastric retention {ge.retention_4h_pct}% at 4h "
                    f"(>= {t.gastroparesis.retention_4h_min_pct}%)",
            source="ANMS/SNM Gastric Emptying Scintigraphy Guideline",
        ))
        has_positive_retention = True
    if (
        ge.retention_2h_pct is not None
        and ge.retention_2h_pct >= t.gastroparesis.retention_2h_min_pct
    ):
        ev.append(Evidence(
            rule_id="GE_RETENTION_2H",
            mechanism_id=M,
            strength=EvidenceStrength.STRONG,
            trigger=f"Gastric retention {ge.retention_2h_pct}% at 2h "
                    f"(>= {t.gastroparesis.retention_2h_min_pct}%)",
            source="ANMS/SNM Gastric Emptying Scintigraphy Guideline",
        ))
        has_positive_retention = True

    if not has_positive_retention:
        return []

    # Regurgitation or extra-esophageal symptoms are consistent with delayed
    # emptying promoting reflux.
    if DominantSymptom.REGURGITATION in case.clinical.dominant_symptoms:
        ev.append(Evidence(
            rule_id="GE_REGURG_DOMINANT",
            mechanism_id=M,
            strength=EvidenceStrength.SUPPORTIVE,
            trigger="Regurgitation dominant — consistent with retained gastric contents",
            source="ANMS/SNM Gastric Emptying Scintigraphy Guideline",
        ))
    return ev


# ---------------------------------------------------------------------------
# Post-surgical sub-buckets
#
# Each fires when BOTH (a) the relevant prior surgery is documented AND
# (b) there is positive acid reflux evidence. The general true-GERD buckets
# route to these via _post_surgical_routing so the classifier does not emit
# "true GERD" plus "post-sleeve GERD" separately.
# ---------------------------------------------------------------------------
def _post_surgical_bucket(
    case: CaseInput,
    t: Thresholds,
    mechanism_id: str,
    *,
    rule_id: str,
    surgery_label: str,
    explanation: str,
) -> list[Evidence]:
    acid_ev = _lyon_conclusive_endoscopy(case, t, mechanism_id)
    acid_ev.extend(_lyon_aet_offppi(case, t, mechanism_id))
    if not _has_positive_acid_evidence(acid_ev):
        return []
    acid_ev.append(Evidence(
        rule_id=rule_id,
        mechanism_id=mechanism_id,
        strength=EvidenceStrength.STRONG,
        trigger=f"Prior {surgery_label}. {explanation}",
        source="Clinical history + guideline context",
    ))
    return acid_ev


def evaluate_post_sleeve_gerd(case: CaseInput, t: Thresholds) -> list[Evidence]:
    if case.clinical.prior_bariatric_surgery not in _POST_SLEEVE_VALUES:
        return []
    return _post_surgical_bucket(
        case, t, "post_sleeve_gerd",
        rule_id="POST_SLEEVE_CONTEXT",
        surgery_label="sleeve gastrectomy",
        explanation="Sleeve anatomy (intrathoracic migration, absent fundus) "
                    "promotes de novo GERD; anti-reflux surgery is generally "
                    "not advised — conversion to RYGB is the definitive option.",
    )


def evaluate_post_myotomy_gerd(case: CaseInput, t: Thresholds) -> list[Evidence]:
    if case.clinical.prior_anti_reflux_surgery not in _POST_MYOTOMY_VALUES:
        return []
    return _post_surgical_bucket(
        case, t, "post_myotomy_gerd",
        rule_id="POST_MYOTOMY_CONTEXT",
        surgery_label=case.clinical.prior_anti_reflux_surgery.value.replace("_", " "),
        explanation="Post-myotomy (Heller / POEM) patients have intentionally "
                    "disrupted LES function; PPI is first-line and surgical "
                    "anti-reflux is generally avoided.",
    )


def evaluate_post_fundoplication_failure(
    case: CaseInput, t: Thresholds
) -> list[Evidence]:
    if case.clinical.prior_anti_reflux_surgery not in _POST_FUNDOPLICATION_VALUES:
        return []
    return _post_surgical_bucket(
        case, t, "post_fundoplication_failure",
        rule_id="POST_FUNDOPLICATION_FAILURE_CONTEXT",
        surgery_label=f"fundoplication ({case.clinical.prior_anti_reflux_surgery.value})",
        explanation="Recurrent objective GERD after fundoplication indicates "
                    "wrap failure; imaging (barium, endoscopy) to evaluate "
                    "wrap integrity before re-do surgical decisions.",
    )


# ---------------------------------------------------------------------------
# Mechanism registry — engine.py iterates this list.
# ---------------------------------------------------------------------------
ALL_MECHANISMS: list[MechanismRule] = [
    MechanismRule(
        mechanism_id="true_gerd_competent_peristalsis",
        label="True acid GERD with competent peristalsis",
        evaluate=evaluate_true_gerd_competent_peristalsis,
    ),
    MechanismRule(
        mechanism_id="true_gerd_with_iem",
        label="True acid GERD with ineffective esophageal motility",
        evaluate=evaluate_true_gerd_with_iem,
    ),
    MechanismRule(
        mechanism_id="reflux_hypersensitivity",
        label="Reflux hypersensitivity",
        evaluate=evaluate_reflux_hypersensitivity,
    ),
    MechanismRule(
        mechanism_id="functional_heartburn",
        label="Functional heartburn",
        evaluate=evaluate_functional_heartburn,
    ),
    MechanismRule(
        mechanism_id="eoe_masquerade",
        label="Eosinophilic esophagitis masquerading as GERD",
        evaluate=evaluate_eoe_masquerade,
    ),
    MechanismRule(
        mechanism_id="achalasia_masquerade",
        label="Achalasia (or major motility disorder) masquerading as GERD",
        evaluate=evaluate_achalasia_masquerade,
    ),
    MechanismRule(
        mechanism_id="hiatal_hernia_dominant",
        label="Hiatal hernia-dominant GERD",
        evaluate=evaluate_hiatal_hernia_dominant,
    ),
    MechanismRule(
        mechanism_id="non_acid_reflux_on_ppi",
        label="Non-acid / weakly acidic reflux on PPI",
        evaluate=evaluate_non_acid_reflux_on_ppi,
    ),
    MechanismRule(
        mechanism_id="gastroparesis_driven_reflux",
        label="Gastroparesis-driven reflux",
        evaluate=evaluate_gastroparesis_driven_reflux,
    ),
    MechanismRule(
        mechanism_id="post_sleeve_gerd",
        label="Post-sleeve gastrectomy GERD",
        evaluate=evaluate_post_sleeve_gerd,
    ),
    MechanismRule(
        mechanism_id="post_myotomy_gerd",
        label="Post-myotomy GERD (Heller / POEM)",
        evaluate=evaluate_post_myotomy_gerd,
    ),
    MechanismRule(
        mechanism_id="post_fundoplication_failure",
        label="Post-fundoplication failure",
        evaluate=evaluate_post_fundoplication_failure,
    ),
]

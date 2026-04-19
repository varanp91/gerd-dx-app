"""
Classifier engine.

Orchestrates mechanism rule evaluation, aggregates evidence into confidence
levels, detects the on-PPI refractory flag, and emits contributing factors.

Design invariants:
  - Mechanism rule functions are pure: (CaseInput, Thresholds) -> list[Evidence].
    They do not compute confidence; the engine does. This keeps confidence
    semantics in one place so every mechanism is scored identically.
  - Every positive recommendation must trace to specific Evidence objects.
  - When required inputs are missing, mechanisms emit an INSUFFICIENT marker
    rather than silently scoring zero.
"""

from __future__ import annotations

from collections import defaultdict

from . import DISCLAIMER
from .conflicts import detect_conflicts, detect_insufficient_inputs
from .enums import Confidence, EvidenceStrength, PhStudyState
from .models import CaseInput
from .reasoning import CaseOutput, Evidence, ManagementPlan, MechanismResult
from .rules import Thresholds, load_thresholds
from .rules.management import ManagementCatalog, load_management_catalog
from .rules.mechanisms import ALL_MECHANISMS


# --- Confidence aggregation ------------------------------------------------

_CONFIDENCE_SORT_ORDER: dict[Confidence, int] = {
    Confidence.HIGH: 0,
    Confidence.MODERATE: 1,
    Confidence.LOW: 2,
    Confidence.INSUFFICIENT: 3,
    Confidence.RULED_OUT: 4,
}

# Within the same Confidence tier, buckets are tie-broken by the strongest
# positive evidence they hold (CONCLUSIVE > STRONG > SUPPORTIVE > BORDERLINE)
# and then by the count at that strength. A CONCLUSIVE-driven HIGH therefore
# outranks a HIGH reached via two STRONG findings or STRONG+SUPPORTIVE.
_POSITIVE_STRENGTH_RANK: dict[EvidenceStrength, int] = {
    EvidenceStrength.CONCLUSIVE: 5,
    EvidenceStrength.STRONG: 4,
    EvidenceStrength.SUPPORTIVE: 3,
    EvidenceStrength.BORDERLINE: 2,
    EvidenceStrength.NEGATIVE: 1,
    EvidenceStrength.EXCLUSIONARY: 0,  # ignored in positive tie-break
}


def _ranking_key(r: MechanismResult) -> tuple[int, int, int]:
    """
    Sort key for MechanismResult.

    Lower tuples sort first (Python default). We want:
      1. higher Confidence first  → confidence_order ascending
      2. higher max-strength first → negate
      3. more evidence at that strength first → negate
    """
    positives = [
        _POSITIVE_STRENGTH_RANK[e.strength]
        for e in r.evidence
        if e.strength != EvidenceStrength.EXCLUSIONARY
    ]
    if positives:
        max_rank = max(positives)
        count_at_max = sum(1 for rank in positives if rank == max_rank)
    else:
        max_rank = 0
        count_at_max = 0
    return (_CONFIDENCE_SORT_ORDER[r.confidence], -max_rank, -count_at_max)


def _compute_confidence(evidence: list[Evidence]) -> Confidence | None:
    """
    Reduce a bag of Evidence to a single Confidence level.

    Returns None when the mechanism has no evidence at all — the caller then
    drops it from the ranked list to avoid 11 "no evidence" rows per case.

    Rules (see EvidenceStrength enum for definitions):
      - Any EXCLUSIONARY     -> RULED_OUT   (trade-off preserved in trace)
      - Any CONCLUSIVE       -> HIGH
      - >=2 STRONG           -> HIGH        (independent strong findings converge)
      - STRONG + SUPPORTIVE  -> HIGH
      - 1 STRONG alone       -> MODERATE
      - >=2 SUPPORTIVE       -> MODERATE
      - 1 SUPPORTIVE         -> LOW
      - Only BORDERLINE      -> LOW
      - Only NEGATIVE        -> LOW   (considered and argued against, not excluded)
      - NEGATIVE downgrades the above by one tier.
    """
    if not evidence:
        return None

    by = defaultdict(int)
    for ev in evidence:
        by[ev.strength] += 1

    if by[EvidenceStrength.EXCLUSIONARY] > 0:
        return Confidence.RULED_OUT

    conclusive = by[EvidenceStrength.CONCLUSIVE]
    strong = by[EvidenceStrength.STRONG]
    supportive = by[EvidenceStrength.SUPPORTIVE]
    borderline = by[EvidenceStrength.BORDERLINE]
    negative = by[EvidenceStrength.NEGATIVE]

    if conclusive >= 1:
        base = Confidence.HIGH
    elif strong >= 2:
        base = Confidence.HIGH
    elif strong >= 1 and supportive >= 1:
        base = Confidence.HIGH
    elif strong >= 1:
        base = Confidence.MODERATE
    elif supportive >= 2:
        base = Confidence.MODERATE
    elif supportive >= 1:
        base = Confidence.LOW
    elif borderline >= 1:
        base = Confidence.LOW
    elif negative >= 1:
        # Only negative evidence — mechanism looked at and argued against, but
        # not conclusively ruled out (that would be EXCLUSIONARY).
        return Confidence.LOW
    else:
        return None

    if negative >= 1:
        downgrade = {
            Confidence.HIGH: Confidence.MODERATE,
            Confidence.MODERATE: Confidence.LOW,
            Confidence.LOW: Confidence.LOW,
        }
        base = downgrade[base]
    return base


# --- Refractory and contributing factors -----------------------------------

def _check_refractory(case: CaseInput, t: Thresholds) -> bool:
    """On-PPI AET above threshold triggers the refractory flag, orthogonal to classification."""
    if case.ph_impedance is None:
        return False
    if case.ph_impedance.test_state != PhStudyState.ON_PPI:
        return False
    return case.ph_impedance.aet_pct > t.on_ppi_refractory.aet_pct_min


def _detect_combinations(
    results: list[MechanismResult], refractory: bool
) -> list[str]:
    """
    Surface clinically meaningful combinations across mechanism buckets.

    Only considers buckets at LOW confidence or above (RULED_OUT and
    INSUFFICIENT do not count). The list returned is human-readable and is
    intended to be rendered above the per-bucket reasoning trace.
    """
    active = {
        r.mechanism_id
        for r in results
        if r.confidence in (Confidence.HIGH, Confidence.MODERATE, Confidence.LOW)
    }
    combos: list[str] = []

    true_gerd_buckets = {"true_gerd_competent_peristalsis", "true_gerd_with_iem"}
    has_true_gerd = bool(active & true_gerd_buckets)

    if "true_gerd_with_iem" in active:
        combos.append("True GERD + IEM")

    if has_true_gerd and "eoe_masquerade" in active:
        combos.append("True GERD + coexisting EoE")

    if has_true_gerd and "hiatal_hernia_dominant" in active:
        combos.append("True GERD + hiatal hernia-dominant overlay")

    if "gastroparesis_driven_reflux" in active and (
        has_true_gerd or "hiatal_hernia_dominant" in active
    ):
        combos.append("Gastroparesis contributing to reflux mechanism")

    if refractory and has_true_gerd:
        combos.append("Refractory GERD (on-PPI AET above threshold)")

    return combos


def _contributing_factors(case: CaseInput, t: Thresholds) -> list[str]:
    """
    Surface medications (and pregnancy) that modify management but do NOT
    change mechanism classification. Kept deliberately narrow per clinical
    direction — classifier is conservative about what moves buckets.
    """
    factors: list[str] = []
    allowed = set(t.contributing_factors.medications)
    for med in case.clinical.contributing_medications:
        if med.value in allowed:
            factors.append(med.value)
    if case.clinical.pregnant:
        factors.append("pregnancy")
    return factors


# --- Public entrypoint -----------------------------------------------------

_INDETERMINATE_MECHANISM_ID = "indeterminate"
_INDETERMINATE_LABEL = "Indeterminate — insufficient or non-specific workup"


def _synthesize_indeterminate(results: list[MechanismResult]) -> MechanismResult | None:
    """
    When no mechanism bucket reached LOW / MODERATE / HIGH confidence, emit
    an explicit 'indeterminate' result so the output is never empty — the
    clinician gets a clear "refer to / discuss with GI" pathway via the
    management layer rather than a silent null classification.

    RULED_OUT and INSUFFICIENT results do not count as active classification:
    the former tell the clinician what was considered and excluded; neither
    constitutes a positive diagnosis.
    """
    any_positive = any(
        r.confidence in (Confidence.HIGH, Confidence.MODERATE, Confidence.LOW)
        for r in results
    )
    if any_positive:
        return None

    evidence = Evidence(
        rule_id="INDETERMINATE_NO_BUCKET_FIRED",
        mechanism_id=_INDETERMINATE_MECHANISM_ID,
        strength=EvidenceStrength.SUPPORTIVE,
        trigger=(
            "No mechanism bucket met even low-confidence criteria from the "
            "submitted workup. Classification deferred pending additional "
            "data; see 'insufficient_inputs' for specific gaps."
        ),
        source="Classifier fallback",
    )
    return MechanismResult(
        mechanism_id=_INDETERMINATE_MECHANISM_ID,
        label=_INDETERMINATE_LABEL,
        confidence=Confidence.LOW,
        evidence=[evidence],
    )


# Module-level cache of the management catalog. The YAML is read once per
# process; tests that exercise alternative catalogs can inject their own via
# classify(..., management=my_catalog).
_default_management: ManagementCatalog | None = None


def _get_default_management() -> ManagementCatalog:
    global _default_management
    if _default_management is None:
        _default_management = load_management_catalog()
    return _default_management


def _attach_management(
    results: list[MechanismResult],
    catalog: ManagementCatalog,
    contributing_factor_keys: list[str],
) -> dict[str, ManagementPlan]:
    """
    Produce a mechanism_id -> ManagementPlan dict for every mechanism in
    `results` that is actively firing (confidence LOW / MODERATE / HIGH).
    RULED_OUT and INSUFFICIENT mechanisms do not receive a management plan —
    there is nothing to manage.

    Contributing-factor modifier strings are appended to each plan so a UI
    rendering only the top-ranked plan still surfaces the medication context.
    """
    modifier_strings = catalog.modifier_strings(contributing_factor_keys)
    out: dict[str, ManagementPlan] = {}
    for r in results:
        if r.confidence in (Confidence.RULED_OUT, Confidence.INSUFFICIENT):
            continue
        plan = catalog.plan_for(r.mechanism_id)
        if plan is None:
            continue
        plan.modifiers_from_contributing_factors = list(modifier_strings)
        out[r.mechanism_id] = plan
    return out


def classify(
    case: CaseInput,
    thresholds: Thresholds | None = None,
    management: ManagementCatalog | None = None,
) -> CaseOutput:
    """
    Run every mechanism rule against a case, score each, and return a ranked
    CaseOutput. This is the single entrypoint consumed by the CLI, tests, and
    any future UI — keep it free of I/O.
    """
    t = thresholds or load_thresholds()
    catalog = management or _get_default_management()

    results: list[MechanismResult] = []
    for rule in ALL_MECHANISMS:
        evidence = rule.evaluate(case, t)
        confidence = _compute_confidence(evidence)
        if confidence is None:
            continue
        results.append(
            MechanismResult(
                mechanism_id=rule.mechanism_id,
                label=rule.label,
                confidence=confidence,
                evidence=evidence,
            )
        )

    # Indeterminate fallback: injected before sorting so the existing sort
    # places LOW-confidence indeterminate correctly (after any real LOW+
    # bucket, if somehow both coexisted — by construction, they do not).
    indeterminate = _synthesize_indeterminate(results)
    if indeterminate is not None:
        results.append(indeterminate)

    results.sort(key=_ranking_key)

    refractory = _check_refractory(case, t)
    contributing = _contributing_factors(case, t)
    return CaseOutput(
        disclaimer=DISCLAIMER,
        ranked_mechanisms=results,
        refractory_flag=refractory,
        contributing_factors=contributing,
        combinations_flagged=_detect_combinations(results, refractory),
        conflicts=detect_conflicts(case, results, t),
        insufficient_inputs=detect_insufficient_inputs(case, results, t),
        management=_attach_management(results, catalog, contributing),
        # red_flags is populated in a later slice.
    )

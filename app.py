"""
Local Streamlit UI for the GERD mechanism-based diagnostic reasoner.

Run:
    streamlit run app.py

Opens a form in your browser (localhost only). No data leaves the machine.
This file is pure UI — it imports `classify()` from the logic layer and
renders the returned `CaseOutput`. No clinical logic lives here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from gerd_dx import DISCLAIMER
from gerd_dx.engine import classify
from gerd_dx.enums import (
    BiopsyFinding,
    Confidence,
    ContributingMed,
    DominantSymptom,
    EGJMorphology,
    EvidenceStrength,
    LAGrade,
    PPIResponse,
    Peristalsis,
    PhStudyState,
    PriorSurgery,
)
from gerd_dx.models import (
    CaseInput,
    ClinicalContext,
    EndoFLIPFindings,
    EndoscopyFindings,
    GastricEmptyingStudy,
    HRMFindings,
    PhImpedance,
)
from gerd_dx.reasoning import CaseOutput, ManagementPlan

RULESET_VERSION = "Lyon 2.0 / Chicago v4.0 — ruleset v0.1"
FIXTURE_DIR = Path(__file__).parent / "tests" / "fixtures"

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="GERD Diagnostic Reasoner",
    page_icon=None,
    layout="wide",
)

# Sidebar — disclaimer + version + load-example.
with st.sidebar:
    st.markdown("### GERD Diagnostic Reasoner")
    st.caption(RULESET_VERSION)
    st.markdown("---")
    st.markdown("**Disclaimer**")
    st.caption(DISCLAIMER)
    st.markdown("---")
    st.markdown("**Load example case**")
    example_files = sorted(FIXTURE_DIR.glob("*.json")) if FIXTURE_DIR.exists() else []
    example_choice = st.selectbox(
        "Example",
        options=["(none)"] + [p.name for p in example_files],
        label_visibility="collapsed",
    )
    if st.button("Load into form", use_container_width=True) and example_choice != "(none)":
        fixture_path = FIXTURE_DIR / example_choice
        data = json.loads(fixture_path.read_text())
        st.session_state["loaded_case"] = data
        st.rerun()


# ---------------------------------------------------------------------------
# Helpers — pull optional values from a pre-loaded case if present.
# ---------------------------------------------------------------------------

_loaded: dict[str, Any] = st.session_state.get("loaded_case", {})


def _section(name: str) -> dict[str, Any]:
    """Return the section dict from the loaded example, or {} if absent."""
    return _loaded.get(name) or {}


def _enum_index(enum_cls, value, default_index: int = 0) -> int:
    if value is None:
        return default_index
    values = [e.value for e in enum_cls]
    return values.index(value) if value in values else default_index


# ---------------------------------------------------------------------------
# Form
# ---------------------------------------------------------------------------

st.title("GERD Mechanism-Based Diagnostic Reasoner")
st.caption(
    "Enter the workup data for one case. Sections are optional — leave them "
    "collapsed if you don't have that data. Only 'Clinical context' is required."
)

# ----- Clinical context (required) ----------------------------------------
st.markdown("### Clinical context (required)")
clinical_section = _section("clinical")
col1, col2 = st.columns(2)
with col1:
    ppi_response = st.radio(
        "PPI response",
        options=[e.value for e in PPIResponse],
        index=_enum_index(PPIResponse, clinical_section.get("ppi_response"), 3),
        horizontal=True,
    )
    bmi = st.number_input(
        "BMI (kg/m²)",
        min_value=0.0,
        max_value=90.0,
        value=clinical_section.get("bmi") if clinical_section.get("bmi") is not None else None,
        step=0.1,
        placeholder="optional",
    )
    pregnant = st.checkbox("Pregnant", value=clinical_section.get("pregnant", False))
with col2:
    dominant_symptoms = st.multiselect(
        "Dominant symptoms",
        options=[e.value for e in DominantSymptom],
        default=clinical_section.get("dominant_symptoms", []),
    )
    prior_anti_reflux = st.selectbox(
        "Prior anti-reflux / motility surgery",
        options=[e.value for e in PriorSurgery],
        index=_enum_index(PriorSurgery, clinical_section.get("prior_anti_reflux_surgery"), 0),
    )
    prior_bariatric = st.selectbox(
        "Prior bariatric surgery",
        options=[e.value for e in PriorSurgery],
        index=_enum_index(PriorSurgery, clinical_section.get("prior_bariatric_surgery"), 0),
    )
contributing_meds = st.multiselect(
    "Contributing medications",
    options=[e.value for e in ContributingMed],
    default=clinical_section.get("contributing_medications", []),
    help="These modify management but do NOT change mechanism classification.",
)

# ----- Endoscopy ----------------------------------------------------------
with st.expander("Endoscopy", expanded=bool(_section("endoscopy"))):
    endo_section = _section("endoscopy")
    include_endoscopy = st.checkbox(
        "Include endoscopy findings",
        value=bool(endo_section),
        key="include_endoscopy",
    )
    if include_endoscopy:
        c1, c2 = st.columns(2)
        with c1:
            la_grade = st.radio(
                "LA grade",
                options=[e.value for e in LAGrade],
                index=_enum_index(LAGrade, endo_section.get("la_grade"), 0),
                horizontal=True,
            )
            hh_cm = st.number_input(
                "Hiatal hernia size (cm)",
                min_value=0.0,
                max_value=25.0,
                value=endo_section.get("hiatal_hernia_cm")
                if endo_section.get("hiatal_hernia_cm") is not None
                else None,
                step=0.5,
                placeholder="optional",
            )
            hh_present = st.checkbox(
                "Hernia documented but size NOT measured",
                value=endo_section.get("hiatal_hernia_present", False),
            )
            its = st.checkbox(
                "Intrathoracic stomach",
                value=endo_section.get("intrathoracic_stomach", False),
            )
            stricture = st.checkbox(
                "Peptic stricture",
                value=endo_section.get("peptic_stricture", False),
            )
        with c2:
            barretts = st.checkbox(
                "Barrett's esophagus",
                value=endo_section.get("barretts", False),
            )
            barretts_length = st.number_input(
                "Barrett's length (cm)",
                min_value=0.0,
                max_value=20.0,
                value=endo_section.get("barretts_length_cm")
                if endo_section.get("barretts_length_cm") is not None
                else None,
                step=0.5,
                placeholder="optional",
                disabled=not barretts,
            )
            biopsy_finding = st.selectbox(
                "Primary biopsy finding",
                options=[e.value for e in BiopsyFinding],
                index=_enum_index(BiopsyFinding, endo_section.get("biopsy_finding"), 0),
            )
            eos_hpf = st.number_input(
                "Peak eosinophils / HPF",
                min_value=0,
                max_value=500,
                value=endo_section.get("eosinophils_per_hpf")
                if endo_section.get("eosinophils_per_hpf") is not None
                else None,
                step=1,
                placeholder="optional",
            )
            dis = st.checkbox(
                "Dilated intercellular spaces (DIS) on biopsy",
                value=endo_section.get("dilated_intercellular_spaces", False),
            )

# ----- HRM ---------------------------------------------------------------
with st.expander("High-resolution manometry (Chicago v4.0)", expanded=bool(_section("hrm"))):
    hrm_section = _section("hrm")
    include_hrm = st.checkbox(
        "Include HRM findings",
        value=bool(hrm_section),
        key="include_hrm",
    )
    if include_hrm:
        c1, c2 = st.columns(2)
        with c1:
            peristalsis = st.selectbox(
                "Peristalsis pattern",
                options=[e.value for e in Peristalsis],
                index=_enum_index(Peristalsis, hrm_section.get("peristalsis"), 0),
            )
            egj_morph_set = st.checkbox(
                "Record EGJ morphology",
                value=hrm_section.get("egj_morphology") is not None,
            )
            egj_morph = None
            if egj_morph_set:
                egj_morph = st.radio(
                    "EGJ morphology",
                    options=[e.value for e in EGJMorphology],
                    index=_enum_index(EGJMorphology, hrm_section.get("egj_morphology"), 0),
                    horizontal=True,
                )
            les_resting = st.number_input(
                "LES resting pressure (mmHg)",
                value=hrm_section.get("les_resting_pressure_mmhg"),
                placeholder="optional",
            )
            irp = st.number_input(
                "IRP supine (mmHg)",
                value=hrm_section.get("irp_supine_mmhg"),
                placeholder="optional",
            )
        with c2:
            dci_median = st.number_input(
                "Median DCI",
                min_value=0.0,
                value=hrm_section.get("dci_median"),
                placeholder="optional",
            )
            ineffective_pct = st.number_input(
                "Ineffective swallow %",
                min_value=0.0,
                max_value=100.0,
                value=hrm_section.get("ineffective_swallow_pct"),
                placeholder="optional",
            )
            failed_pct = st.number_input(
                "Failed swallow %",
                min_value=0.0,
                max_value=100.0,
                value=hrm_section.get("failed_swallow_pct"),
                placeholder="optional",
            )
            premature_pct = st.number_input(
                "Premature swallow %",
                min_value=0.0,
                max_value=100.0,
                value=hrm_section.get("premature_swallow_pct"),
                placeholder="optional",
            )
            jackhammer_pct = st.number_input(
                "Jackhammer swallow %",
                min_value=0.0,
                max_value=100.0,
                value=hrm_section.get("jackhammer_swallow_pct"),
                placeholder="optional",
            )
            egj_ci = st.number_input(
                "EGJ contractile integral",
                min_value=0.0,
                value=hrm_section.get("egj_ci"),
                placeholder="optional",
            )

# ----- pH-impedance ------------------------------------------------------
with st.expander("pH-impedance monitoring", expanded=bool(_section("ph_impedance"))):
    ph_section = _section("ph_impedance")
    include_ph = st.checkbox(
        "Include pH-impedance findings",
        value=bool(ph_section),
        key="include_ph",
    )
    if include_ph:
        c1, c2 = st.columns(2)
        with c1:
            test_state = st.radio(
                "Test state",
                options=[e.value for e in PhStudyState],
                index=_enum_index(PhStudyState, ph_section.get("test_state"), 1),
                horizontal=True,
            )
            aet_pct = st.number_input(
                "Acid Exposure Time (%)  — required",
                min_value=0.0,
                max_value=100.0,
                value=ph_section.get("aet_pct") if ph_section.get("aet_pct") is not None else 4.0,
                step=0.1,
            )
            total_reflux = st.number_input(
                "Total reflux episodes",
                min_value=0,
                value=ph_section.get("total_reflux_episodes"),
                step=1,
                placeholder="optional",
            )
            si_pct = st.number_input(
                "Symptom Index (SI) %",
                min_value=0.0,
                max_value=100.0,
                value=ph_section.get("symptom_index_pct"),
                placeholder="optional",
            )
            sap_pct = st.number_input(
                "Symptom Association Probability (SAP) %",
                min_value=0.0,
                max_value=100.0,
                value=ph_section.get("sap_pct"),
                placeholder="optional  — not used in classification, shown for reference",
            )
        with c2:
            acidic_pct = st.number_input(
                "Acidic episodes %",
                min_value=0.0,
                max_value=100.0,
                value=ph_section.get("acidic_pct"),
                placeholder="optional",
            )
            weakly_acidic_pct = st.number_input(
                "Weakly acidic episodes %",
                min_value=0.0,
                max_value=100.0,
                value=ph_section.get("weakly_acidic_pct"),
                placeholder="optional",
            )
            non_acid_pct = st.number_input(
                "Non-acid episodes %",
                min_value=0.0,
                max_value=100.0,
                value=ph_section.get("non_acid_pct"),
                placeholder="optional",
            )
            mnbi_ohms = st.number_input(
                "MNBI (ohms)",
                min_value=0.0,
                value=ph_section.get("mnbi_ohms"),
                placeholder="optional",
            )
            pspw_pct = st.number_input(
                "PSPW %",
                min_value=0.0,
                max_value=100.0,
                value=ph_section.get("pspw_pct"),
                placeholder="optional",
            )

# ----- EndoFLIP ----------------------------------------------------------
with st.expander("EndoFLIP (60 mL only)", expanded=bool(_section("endoflip"))):
    flip_section = _section("endoflip")
    include_flip = st.checkbox(
        "Include EndoFLIP findings",
        value=bool(flip_section),
        key="include_flip",
    )
    if include_flip:
        c1, c2 = st.columns(2)
        with c1:
            egj_di_60 = st.number_input(
                "EGJ distensibility index at 60 mL (mm²/mmHg)",
                min_value=0.0,
                max_value=20.0,
                value=flip_section.get("egj_di_60ml"),
                step=0.1,
                placeholder="optional",
            )
            min_dia_60 = st.number_input(
                "Minimum diameter at 60 mL (mm)",
                min_value=0.0,
                max_value=40.0,
                value=flip_section.get("min_diameter_60ml_mm"),
                step=0.5,
                placeholder="optional",
            )
        with c2:
            rac = st.checkbox(
                "Repetitive antegrade contractions",
                value=flip_section.get("repetitive_antegrade_contractions", False),
            )
            rrc = st.checkbox(
                "Repetitive retrograde contractions",
                value=flip_section.get("repetitive_retrograde_contractions", False),
            )

# ----- Gastric emptying --------------------------------------------------
with st.expander("Gastric emptying study", expanded=bool(_section("gastric_emptying"))):
    ge_section = _section("gastric_emptying")
    include_ge = st.checkbox(
        "Include gastric emptying findings",
        value=bool(ge_section),
        key="include_ge",
    )
    if include_ge:
        c1, c2 = st.columns(2)
        with c1:
            ret_2h = st.number_input(
                "Retention at 2 hours (%)",
                min_value=0.0,
                max_value=100.0,
                value=ge_section.get("retention_2h_pct"),
                placeholder="optional",
            )
        with c2:
            ret_4h = st.number_input(
                "Retention at 4 hours (%)",
                min_value=0.0,
                max_value=100.0,
                value=ge_section.get("retention_4h_pct"),
                placeholder="optional",
            )

st.markdown("---")

# ---------------------------------------------------------------------------
# Run classifier
# ---------------------------------------------------------------------------

run_col, reset_col, _spacer = st.columns([1, 1, 4])
do_classify = run_col.button("Classify", type="primary", use_container_width=True)
if reset_col.button("Clear loaded example", use_container_width=True):
    st.session_state.pop("loaded_case", None)
    st.rerun()


def _assemble_case() -> CaseInput:
    endoscopy_obj = None
    if include_endoscopy:
        endoscopy_obj = EndoscopyFindings(
            la_grade=LAGrade(la_grade),
            hiatal_hernia_cm=hh_cm,
            hiatal_hernia_present=hh_present,
            intrathoracic_stomach=its,
            barretts=barretts,
            barretts_length_cm=barretts_length if barretts else None,
            biopsy_finding=BiopsyFinding(biopsy_finding),
            eosinophils_per_hpf=int(eos_hpf) if eos_hpf is not None else None,
            dilated_intercellular_spaces=dis,
            peptic_stricture=stricture,
        )
    hrm_obj = None
    if include_hrm:
        hrm_obj = HRMFindings(
            les_resting_pressure_mmhg=les_resting,
            egj_morphology=EGJMorphology(egj_morph) if egj_morph else None,
            peristalsis=Peristalsis(peristalsis),
            dci_median=dci_median,
            ineffective_swallow_pct=ineffective_pct,
            failed_swallow_pct=failed_pct,
            egj_ci=egj_ci,
            irp_supine_mmhg=irp,
            premature_swallow_pct=premature_pct,
            jackhammer_swallow_pct=jackhammer_pct,
        )
    ph_obj = None
    if include_ph:
        ph_obj = PhImpedance(
            test_state=PhStudyState(test_state),
            aet_pct=aet_pct,
            total_reflux_episodes=int(total_reflux) if total_reflux is not None else None,
            acidic_pct=acidic_pct,
            weakly_acidic_pct=weakly_acidic_pct,
            non_acid_pct=non_acid_pct,
            symptom_index_pct=si_pct,
            sap_pct=sap_pct,
            mnbi_ohms=mnbi_ohms,
            pspw_pct=pspw_pct,
        )
    flip_obj = None
    if include_flip:
        flip_obj = EndoFLIPFindings(
            egj_di_60ml=egj_di_60,
            min_diameter_60ml_mm=min_dia_60,
            repetitive_antegrade_contractions=rac,
            repetitive_retrograde_contractions=rrc,
        )
    ge_obj = None
    if include_ge:
        ge_obj = GastricEmptyingStudy(
            retention_2h_pct=ret_2h,
            retention_4h_pct=ret_4h,
        )
    clinical_obj = ClinicalContext(
        ppi_response=PPIResponse(ppi_response),
        dominant_symptoms=[DominantSymptom(s) for s in dominant_symptoms],
        bmi=bmi,
        prior_anti_reflux_surgery=PriorSurgery(prior_anti_reflux),
        prior_bariatric_surgery=PriorSurgery(prior_bariatric),
        contributing_medications=[ContributingMed(m) for m in contributing_meds],
        pregnant=pregnant,
    )
    return CaseInput(
        endoscopy=endoscopy_obj,
        hrm=hrm_obj,
        ph_impedance=ph_obj,
        endoflip=flip_obj,
        gastric_emptying=ge_obj,
        clinical=clinical_obj,
    )


# ---------------------------------------------------------------------------
# Render results
# ---------------------------------------------------------------------------

_CONF_COLOR: dict[Confidence, str] = {
    Confidence.HIGH: "green",
    Confidence.MODERATE: "blue",
    Confidence.LOW: "orange",
    Confidence.INSUFFICIENT: "violet",
    Confidence.RULED_OUT: "red",
}

_STRENGTH_COLOR: dict[EvidenceStrength, str] = {
    EvidenceStrength.CONCLUSIVE: "green",
    EvidenceStrength.STRONG: "green",
    EvidenceStrength.SUPPORTIVE: "blue",
    EvidenceStrength.BORDERLINE: "orange",
    EvidenceStrength.NEGATIVE: "gray",
    EvidenceStrength.EXCLUSIONARY: "red",
}


def _render_plan(plan: ManagementPlan) -> None:
    if plan.first_line:
        st.markdown("**First-line**")
        for item in plan.first_line:
            st.markdown(f"- {item}")
    if plan.escalation:
        st.markdown("**Escalation**")
        for item in plan.escalation:
            st.markdown(f"- {item}")
    if plan.procedural:
        st.markdown("**Procedural options**")
        for p in plan.procedural:
            if p.contraindicated:
                st.markdown(f"- :red[**CONTRAINDICATED:** {p.label}]")
                if p.reason:
                    st.caption(f"    reason: {p.reason}")
            else:
                st.markdown(f"- {p.label}")
                if p.indication:
                    st.caption(f"    indication: {p.indication}")
    if plan.lifestyle:
        st.markdown("**Lifestyle**")
        for item in plan.lifestyle:
            st.markdown(f"- {item}")
    if plan.red_flags:
        st.markdown("**Red flags**")
        for item in plan.red_flags:
            st.markdown(f"- :orange[{item}]")
    if plan.modifiers_from_contributing_factors:
        st.markdown("**Modifiers from contributing factors**")
        for item in plan.modifiers_from_contributing_factors:
            st.markdown(f"- {item}")


def _render_result(result: CaseOutput) -> None:
    st.markdown("## Results")

    # Top-priority banners
    if result.refractory_flag:
        st.error(
            "Refractory GERD flag: on-PPI AET is above threshold. Symptoms remain "
            "despite acid suppression — see management for next steps."
        )
    if result.combinations_flagged:
        st.info(
            "**Combinations flagged:** " + " · ".join(result.combinations_flagged)
        )
    if result.conflicts:
        with st.container(border=True):
            st.markdown(":red[**Conflicts — reconcile before acting**]")
            for c in result.conflicts:
                st.markdown(f"- {c}")
    if result.insufficient_inputs:
        with st.container(border=True):
            st.markdown(":orange[**Insufficient inputs**]")
            for g in result.insufficient_inputs:
                st.markdown(f"- {g}")
    if result.contributing_factors:
        st.caption(
            "Contributing factors (modify management, not classification): "
            + ", ".join(result.contributing_factors)
        )

    # Ranked mechanisms
    st.markdown("### Ranked mechanisms")
    if not result.ranked_mechanisms:
        st.warning(
            "No mechanism triggered — insufficient data or all inputs physiologic."
        )
    for i, r in enumerate(result.ranked_mechanisms, 1):
        color = _CONF_COLOR[r.confidence]
        header = f"**{i}. {r.label}**  :{color}[\\[{r.confidence.value.upper()}\\]]"
        default_open = i <= 2  # top 2 mechanisms expanded by default
        with st.expander(header, expanded=default_open):
            for ev in r.evidence:
                strength_color = _STRENGTH_COLOR[ev.strength]
                st.markdown(
                    f"- :{strength_color}[**\\[{ev.strength.value.upper()}\\]**]  {ev.trigger}"
                )
                st.caption(f"    source: {ev.source}")

    # Management for top active bucket + compact summary for others
    active = [
        r for r in result.ranked_mechanisms
        if r.confidence in (Confidence.HIGH, Confidence.MODERATE, Confidence.LOW)
    ]
    if active:
        top = active[0]
        top_plan = result.management.get(top.mechanism_id)
        if top_plan is not None:
            st.markdown(f"### Management — {top.label}")
            _render_plan(top_plan)

        if len(active) > 1:
            st.markdown("### Other active mechanisms")
            for r in active[1:]:
                p = result.management.get(r.mechanism_id)
                color = _CONF_COLOR[r.confidence]
                with st.expander(
                    f"{r.label}  :{color}[\\[{r.confidence.value.upper()}\\]]",
                    expanded=False,
                ):
                    if p is not None:
                        _render_plan(p)

    # Raw JSON, for copy-paste into notes or another tool
    with st.expander("Raw CaseOutput JSON (for copy-paste)"):
        st.code(result.model_dump_json(indent=2), language="json")


if do_classify:
    try:
        case = _assemble_case()
    except Exception as exc:  # pydantic validation errors surface here
        st.error(f"Input validation failed: {exc}")
    else:
        result = classify(case)
        _render_result(result)

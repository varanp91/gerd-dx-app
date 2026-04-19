"""
Command-line interface for the GERD mechanism-based diagnostic reasoner.

Usage:
  gerd-dx                              # interactive prompts (skip by default)
  gerd-dx --json path/to/case.json     # classify a JSON-encoded CaseInput
  gerd-dx --json ... --format json     # emit CaseOutput as JSON instead of pretty

Kept deliberately free of clinical logic — this is a thin I/O + rendering
shell over engine.classify(). Any future Streamlit / web UI imports
gerd_dx.engine.classify directly; the pretty renderer here is one way of
several to display a CaseOutput.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Optional, TypeVar

import typer

from .enums import (
    BiopsyFinding,
    Confidence,
    ContributingMed,
    DominantSymptom,
    EGJMorphology,
    LAGrade,
    PPIResponse,
    Peristalsis,
    PhStudyState,
    PriorSurgery,
)
from .engine import classify
from .models import (
    CaseInput,
    ClinicalContext,
    EndoFLIPFindings,
    EndoscopyFindings,
    GastricEmptyingStudy,
    HRMFindings,
    PhImpedance,
)
from .reasoning import CaseOutput, ManagementPlan

app = typer.Typer(
    help="Mechanism-based GERD diagnostic reasoner (educational / decision-support).",
    no_args_is_help=False,
    add_completion=False,
)

E = TypeVar("E", bound=Enum)


# ---------------------------------------------------------------------------
# Interactive prompting helpers.
# ---------------------------------------------------------------------------

def _prompt_yn(text: str, default: bool = False) -> bool:
    default_token = "y" if default else "n"
    while True:
        resp = typer.prompt(f"{text} [y/n]", default=default_token).strip().lower()
        if resp in ("y", "yes"):
            return True
        if resp in ("n", "no"):
            return False
        typer.echo("Please enter y or n.")


def _prompt_enum(text: str, enum_cls: type[E], default: E) -> E:
    options = ", ".join(e.value for e in enum_cls)
    while True:
        resp = typer.prompt(f"{text} [{options}]", default=default.value).strip()
        try:
            return enum_cls(resp)
        except ValueError:
            typer.secho(f"Invalid. Options: {options}", fg="red")


def _prompt_float_opt(text: str) -> Optional[float]:
    resp = typer.prompt(f"{text} (blank=skip)", default="").strip()
    if not resp:
        return None
    try:
        return float(resp)
    except ValueError:
        typer.secho(f"Invalid number: {resp!r}", fg="red")
        return _prompt_float_opt(text)


def _prompt_float_required(text: str, min_val: float = 0, max_val: float = 100) -> float:
    while True:
        resp = typer.prompt(text).strip()
        try:
            val = float(resp)
        except ValueError:
            typer.secho(f"Invalid number: {resp!r}", fg="red")
            continue
        if min_val <= val <= max_val:
            return val
        typer.secho(f"Value must be in [{min_val}, {max_val}].", fg="red")


def _prompt_int_opt(text: str) -> Optional[int]:
    resp = typer.prompt(f"{text} (blank=skip)", default="").strip()
    if not resp:
        return None
    try:
        return int(resp)
    except ValueError:
        typer.secho(f"Invalid int: {resp!r}", fg="red")
        return _prompt_int_opt(text)


def _prompt_enum_list(text: str, enum_cls: type[E]) -> list[E]:
    options = ", ".join(e.value for e in enum_cls)
    resp = typer.prompt(
        f"{text} (comma-separated, blank=none) [{options}]", default=""
    ).strip()
    if not resp:
        return []
    out: list[E] = []
    for item in resp.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            out.append(enum_cls(item))
        except ValueError:
            typer.secho(f"  Unknown value {item!r} ignored.", fg="yellow")
    return out


# ---------------------------------------------------------------------------
# Interactive case builder. Each section defaults to "skip" — real workups
# are often partial, and this matches the classifier's design invariant that
# missing sections become `insufficient_inputs` entries rather than defaults.
# ---------------------------------------------------------------------------

def _prompt_case() -> CaseInput:
    typer.secho(
        "GERD Diagnostic Workup — Interactive Entry", fg="bright_white", bold=True
    )
    typer.echo(
        "Press Enter to accept defaults. Optional sections default to SKIP.\n"
    )

    endoscopy: Optional[EndoscopyFindings] = None
    if _prompt_yn("Include endoscopy findings?", default=False):
        typer.secho("[Endoscopy]", fg="cyan")
        endoscopy = EndoscopyFindings(
            la_grade=_prompt_enum("  LA grade", LAGrade, LAGrade.NONE),
            hiatal_hernia_cm=_prompt_float_opt("  Hiatal hernia size (cm)"),
            hiatal_hernia_present=_prompt_yn(
                "  Hernia documented but size NOT measured?", default=False
            ),
            intrathoracic_stomach=_prompt_yn("  Intrathoracic stomach?", default=False),
            barretts=_prompt_yn("  Barrett's esophagus present?", default=False),
            barretts_length_cm=_prompt_float_opt("  Barrett's length (cm)"),
            biopsy_finding=_prompt_enum(
                "  Primary biopsy finding", BiopsyFinding, BiopsyFinding.NORMAL
            ),
            eosinophils_per_hpf=_prompt_int_opt("  Peak eosinophils/HPF"),
            dilated_intercellular_spaces=_prompt_yn(
                "  Dilated intercellular spaces (DIS) on biopsy?", default=False
            ),
            peptic_stricture=_prompt_yn("  Peptic stricture?", default=False),
        )

    hrm: Optional[HRMFindings] = None
    if _prompt_yn("Include HRM (Chicago v4.0)?", default=False):
        typer.secho("[HRM]", fg="cyan")
        peristalsis = _prompt_enum(
            "  Peristalsis pattern", Peristalsis, Peristalsis.NORMAL
        )
        egj_morph = None
        if _prompt_yn("  Record EGJ morphology?", default=False):
            egj_morph = _prompt_enum(
                "  EGJ morphology", EGJMorphology, EGJMorphology.TYPE_I
            )
        hrm = HRMFindings(
            les_resting_pressure_mmhg=_prompt_float_opt("  LES resting pressure (mmHg)"),
            egj_morphology=egj_morph,
            peristalsis=peristalsis,
            dci_median=_prompt_float_opt("  Median DCI"),
            ineffective_swallow_pct=_prompt_float_opt("  Ineffective swallow %"),
            failed_swallow_pct=_prompt_float_opt("  Failed swallow %"),
            egj_ci=_prompt_float_opt("  EGJ contractile integral"),
            irp_supine_mmhg=_prompt_float_opt("  IRP supine (mmHg)"),
            premature_swallow_pct=_prompt_float_opt("  Premature swallow %"),
            jackhammer_swallow_pct=_prompt_float_opt("  Jackhammer swallow %"),
        )

    ph_impedance: Optional[PhImpedance] = None
    if _prompt_yn("Include pH-impedance study?", default=False):
        typer.secho("[pH-Impedance]", fg="cyan")
        ph_impedance = PhImpedance(
            test_state=_prompt_enum(
                "  Test state", PhStudyState, PhStudyState.OFF_PPI
            ),
            aet_pct=_prompt_float_required("  AET (%)", 0, 100),
            total_reflux_episodes=_prompt_int_opt("  Total reflux episodes"),
            acidic_pct=_prompt_float_opt("  Acidic episodes %"),
            weakly_acidic_pct=_prompt_float_opt("  Weakly acidic episodes %"),
            non_acid_pct=_prompt_float_opt("  Non-acid episodes %"),
            symptom_index_pct=_prompt_float_opt("  Symptom Index (SI) %"),
            sap_pct=_prompt_float_opt("  Symptom Association Probability (SAP) %"),
            mnbi_ohms=_prompt_float_opt("  MNBI (ohms)"),
            pspw_pct=_prompt_float_opt("  PSPW %"),
        )

    endoflip: Optional[EndoFLIPFindings] = None
    if _prompt_yn("Include EndoFLIP (60 mL only)?", default=False):
        typer.secho("[EndoFLIP — 60 mL]", fg="cyan")
        endoflip = EndoFLIPFindings(
            egj_di_60ml=_prompt_float_opt("  EGJ-DI at 60 mL (mm^2/mmHg)"),
            min_diameter_60ml_mm=_prompt_float_opt("  Min diameter at 60 mL (mm)"),
            repetitive_antegrade_contractions=_prompt_yn(
                "  Repetitive antegrade contractions?", default=False
            ),
            repetitive_retrograde_contractions=_prompt_yn(
                "  Repetitive retrograde contractions?", default=False
            ),
        )

    gastric_emptying: Optional[GastricEmptyingStudy] = None
    if _prompt_yn("Include gastric emptying study?", default=False):
        typer.secho("[Gastric emptying]", fg="cyan")
        gastric_emptying = GastricEmptyingStudy(
            retention_2h_pct=_prompt_float_opt("  % retention at 2 hours"),
            retention_4h_pct=_prompt_float_opt("  % retention at 4 hours"),
        )

    typer.secho("[Clinical context] (required)", fg="cyan", bold=True)
    clinical = ClinicalContext(
        ppi_response=_prompt_enum(
            "  PPI response", PPIResponse, PPIResponse.NOT_TRIED
        ),
        dominant_symptoms=_prompt_enum_list("  Dominant symptoms", DominantSymptom),
        bmi=_prompt_float_opt("  BMI"),
        prior_anti_reflux_surgery=_prompt_enum(
            "  Prior anti-reflux / motility surgery",
            PriorSurgery,
            PriorSurgery.NONE,
        ),
        prior_bariatric_surgery=_prompt_enum(
            "  Prior bariatric surgery", PriorSurgery, PriorSurgery.NONE
        ),
        contributing_medications=_prompt_enum_list(
            "  Contributing medications", ContributingMed
        ),
        pregnant=_prompt_yn("  Pregnant?", default=False),
    )

    return CaseInput(
        endoscopy=endoscopy,
        hrm=hrm,
        ph_impedance=ph_impedance,
        endoflip=endoflip,
        gastric_emptying=gastric_emptying,
        clinical=clinical,
    )


# ---------------------------------------------------------------------------
# Pretty output.
# ---------------------------------------------------------------------------

_CONFIDENCE_COLORS: dict[str, str] = {
    "high": "green",
    "moderate": "cyan",
    "low": "yellow",
    "insufficient": "magenta",
    "ruled_out": "red",
}

_STRENGTH_MARK: dict[str, str] = {
    "conclusive": "*",
    "strong": "+",
    "supportive": ".",
    "borderline": "?",
    "negative": "-",
    "exclusionary": "x",
}


def _section_header(title: str) -> None:
    typer.echo("")
    typer.secho("-" * 68, fg="bright_black")
    typer.secho(title, fg="bright_white", bold=True)
    typer.secho("-" * 68, fg="bright_black")


def _render_plan(plan: ManagementPlan) -> None:
    if plan.first_line:
        typer.secho("First-line:", bold=True)
        for item in plan.first_line:
            typer.echo(f"  - {item}")
    if plan.escalation:
        typer.secho("\nEscalation:", bold=True)
        for item in plan.escalation:
            typer.echo(f"  - {item}")
    if plan.procedural:
        typer.secho("\nProcedural options:", bold=True)
        for p in plan.procedural:
            if p.contraindicated:
                typer.secho(f"  [CONTRAINDICATED] {p.label}", fg="red")
                if p.reason:
                    typer.echo(f"      reason: {p.reason}")
            else:
                typer.echo(f"  - {p.label}")
                if p.indication:
                    typer.echo(f"      indication: {p.indication}")
    if plan.lifestyle:
        typer.secho("\nLifestyle:", bold=True)
        for item in plan.lifestyle:
            typer.echo(f"  - {item}")
    if plan.red_flags:
        typer.secho("\nRed flags:", bold=True)
        for item in plan.red_flags:
            typer.secho(f"  ! {item}", fg="yellow")
    if plan.modifiers_from_contributing_factors:
        typer.secho("\nContributing-factor modifiers:", bold=True)
        for item in plan.modifiers_from_contributing_factors:
            typer.echo(f"  - {item}")


def _render_pretty(result: CaseOutput) -> None:
    # Disclaimer at the top of every non-help invocation (per design).
    typer.secho("=" * 68, fg="bright_black")
    typer.secho("GERD MECHANISM-BASED DIAGNOSTIC REASONER", fg="bright_white", bold=True)
    typer.secho("=" * 68, fg="bright_black")
    typer.secho(result.disclaimer, fg="bright_black")

    _section_header("RANKED MECHANISMS")
    if not result.ranked_mechanisms:
        typer.secho(
            "No mechanism triggered — insufficient data, or all inputs physiologic.",
            fg="yellow",
        )
    for i, r in enumerate(result.ranked_mechanisms, 1):
        color = _CONFIDENCE_COLORS.get(r.confidence.value, "white")
        typer.echo("")
        typer.secho(f"{i}. {r.label}  ", bold=True, nl=False)
        typer.secho(f"[{r.confidence.value.upper()}]", fg=color, bold=True)
        for ev in r.evidence:
            mark = _STRENGTH_MARK.get(ev.strength.value, "-")
            typer.echo(
                f"   {mark} [{ev.strength.value.upper()}] {ev.trigger}"
            )
            typer.echo(f"       source: {ev.source}")

    if result.combinations_flagged:
        _section_header("COMBINATIONS FLAGGED")
        for c in result.combinations_flagged:
            typer.secho(f"  > {c}", fg="cyan")

    if result.refractory_flag:
        _section_header("REFRACTORY FLAG")
        typer.secho(
            "  On-PPI AET above threshold — refractory GERD flag set.",
            fg="red",
            bold=True,
        )

    if result.contributing_factors:
        _section_header("CONTRIBUTING FACTORS (modify management, not classification)")
        for f in result.contributing_factors:
            typer.echo(f"  - {f}")

    if result.conflicts:
        _section_header("CONFLICTS — RECONCILE BEFORE ACTING")
        for c in result.conflicts:
            typer.secho(f"  ! {c}", fg="red")

    if result.insufficient_inputs:
        _section_header("INSUFFICIENT INPUTS")
        for g in result.insufficient_inputs:
            typer.secho(f"  ! {g}", fg="yellow")

    # Management: full plan for top active bucket; compact summary for others.
    active = [
        r for r in result.ranked_mechanisms
        if r.confidence in (Confidence.HIGH, Confidence.MODERATE, Confidence.LOW)
    ]
    if active:
        top = active[0]
        top_plan = result.management.get(top.mechanism_id)
        if top_plan is not None:
            _section_header(f"MANAGEMENT — {top.label}")
            _render_plan(top_plan)

        others = active[1:]
        if others:
            _section_header("OTHER ACTIVE MECHANISMS (summary)")
            for r in others:
                p = result.management.get(r.mechanism_id)
                color = _CONFIDENCE_COLORS.get(r.confidence.value, "white")
                typer.secho(
                    f"\n  {r.label}  [{r.confidence.value.upper()}]",
                    fg=color,
                    bold=True,
                )
                if p and p.first_line:
                    typer.echo(f"    First-line: {p.first_line[0]}")
                if p and p.procedural:
                    procs = [x.label for x in p.procedural if not x.contraindicated]
                    if procs:
                        typer.echo(f"    Procedural: {', '.join(procs[:2])}")

    typer.echo("")


# ---------------------------------------------------------------------------
# Entrypoint.
# ---------------------------------------------------------------------------

@app.command()
def main(
    json_input: Annotated[
        Optional[Path],
        typer.Option(
            "--json",
            help="Path to a CaseInput JSON file. Skips interactive prompts.",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            help="Output format: 'pretty' (default) or 'json' (CaseOutput as JSON).",
        ),
    ] = "pretty",
) -> None:
    if json_input is not None:
        case = CaseInput.model_validate_json(json_input.read_text())
    else:
        case = _prompt_case()

    result = classify(case)

    if output_format == "json":
        typer.echo(result.model_dump_json(indent=2))
    elif output_format == "pretty":
        _render_pretty(result)
    else:
        typer.secho(
            f"Unknown --format {output_format!r}. Use 'pretty' or 'json'.",
            fg="red",
        )
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()

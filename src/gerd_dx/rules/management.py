"""
Management-plan loader.

Reads `management.yaml` and produces typed `ManagementPlan` objects keyed by
mechanism_id, plus a contributing-factor modifier table the engine uses to
append management notes driven by medication/pregnancy context.

This module is pure data loading — it does NOT decide which factor strings
apply to a given case. The engine does that using the
`ContributingFactors` list from Thresholds.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ..reasoning import ManagementPlan, ProceduralOption

_MANAGEMENT_PATH = Path(__file__).parent / "management.yaml"

# Keys at the top level of management.yaml that are NOT mechanism plans.
_RESERVED_KEYS = frozenset({"contributing_factor_modifiers"})


def _build_plan(mechanism_id: str, raw: dict) -> ManagementPlan:
    procedural = [
        ProceduralOption(**entry) for entry in raw.get("procedural", [])
    ]
    return ManagementPlan(
        mechanism_id=mechanism_id,
        first_line=list(raw.get("first_line", [])),
        escalation=list(raw.get("escalation", [])),
        procedural=procedural,
        lifestyle=list(raw.get("lifestyle", [])),
        red_flags=list(raw.get("red_flags", [])),
        # modifiers_from_contributing_factors is populated by the engine,
        # not from the YAML — it's case-specific, not mechanism-specific.
    )


class ManagementCatalog:
    """
    Typed wrapper around the loaded YAML:
      - .plan_for(mechanism_id)        -> ManagementPlan or None
      - .modifier_strings(factor_keys) -> list[str]
    """

    def __init__(
        self,
        plans: dict[str, ManagementPlan],
        modifiers: dict[str, list[str]],
    ) -> None:
        self._plans = plans
        self._modifiers = modifiers

    def plan_for(self, mechanism_id: str) -> ManagementPlan | None:
        plan = self._plans.get(mechanism_id)
        if plan is None:
            return None
        # Return a fresh copy so callers can set modifiers without mutating
        # the shared catalog entry.
        return plan.model_copy(deep=True)

    def modifier_strings(self, factor_keys: list[str]) -> list[str]:
        """Render contributing-factor advice strings for the keys that have entries."""
        out: list[str] = []
        for key in factor_keys:
            out.extend(self._modifiers.get(key, []))
        return out

    @property
    def mechanism_ids(self) -> list[str]:
        return list(self._plans.keys())


def load_management_catalog(path: Path | None = None) -> ManagementCatalog:
    target = path or _MANAGEMENT_PATH
    with target.open("r") as f:
        raw = yaml.safe_load(f)

    plans: dict[str, ManagementPlan] = {}
    for key, value in raw.items():
        if key in _RESERVED_KEYS:
            continue
        plans[key] = _build_plan(key, value)

    modifiers = raw.get("contributing_factor_modifiers", {})
    return ManagementCatalog(plans=plans, modifiers=modifiers)

"""
Thresholds loader.

Loads `thresholds.yaml` into a typed, immutable `Thresholds` object so rule
code can reference values as `t.lyon.aet.pathologic_lower` instead of
string-indexing dicts. Keeping the loader here (rather than scattered across
rule modules) ensures the YAML is the single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_THRESHOLDS_PATH = Path(__file__).parent / "thresholds.yaml"


@dataclass(frozen=True)
class AETThresholds:
    physiologic_upper: float
    pathologic_lower: float


@dataclass(frozen=True)
class RefluxEpisodeThresholds:
    physiologic_upper: int
    pathologic_lower: int


@dataclass(frozen=True)
class AdjunctiveThresholds:
    mnbi_low_ohms: float
    pspw_low_pct: float
    dis_supportive: bool


@dataclass(frozen=True)
class ConclusiveEndoscopicThresholds:
    la_grades: tuple[str, ...]
    barrett_min_length_cm: float
    peptic_stricture: bool


@dataclass(frozen=True)
class BorderlineAdjudication:
    """Lyon 4–6% band reclassified as pathologic only with >= this many adjuncts positive."""
    min_adjunctive_for_pathologic: int


@dataclass(frozen=True)
class LyonThresholds:
    aet: AETThresholds
    reflux_episodes: RefluxEpisodeThresholds
    adjunctive: AdjunctiveThresholds
    conclusive_endoscopic: ConclusiveEndoscopicThresholds
    borderline_adjudication: BorderlineAdjudication


@dataclass(frozen=True)
class OnPPIRefractory:
    aet_pct_min: float


@dataclass(frozen=True)
class HiatalHerniaThresholds:
    dominant_min_cm: float
    dominant_requires_intrathoracic_stomach: bool
    tif_eligible_max_cm: float
    c_tif_min_cm: float
    c_tif_max_cm: float


@dataclass(frozen=True)
class IEMThresholds:
    ineffective_swallow_dci_max: float
    failed_swallow_dci_max: float
    ineffective_pct_min: float
    mixed_ineffective_pct_min: float
    mixed_failed_pct_min: float


@dataclass(frozen=True)
class JackhammerThresholds:
    dci_min: float
    pct_min: float


@dataclass(frozen=True)
class ChicagoV4Thresholds:
    iem: IEMThresholds
    absent_contractility_failed_pct: float
    jackhammer: JackhammerThresholds
    des_premature_pct_min: float
    achalasia_irp_supine_min: float


@dataclass(frozen=True)
class EndoFLIPThresholds:
    volume_ml: int
    egj_di_reduced: float
    egj_di_severely_reduced: float


@dataclass(frozen=True)
class EoEThresholds:
    eos_per_hpf_min: int


@dataclass(frozen=True)
class GastroparesisThresholds:
    retention_2h_min_pct: float
    retention_4h_min_pct: float


@dataclass(frozen=True)
class SymptomAssociationThresholds:
    si_positive_pct: float
    # SAP intentionally absent — not used in classification per clinical direction.


@dataclass(frozen=True)
class ContributingFactors:
    medications: tuple[str, ...]


@dataclass(frozen=True)
class Thresholds:
    lyon: LyonThresholds
    on_ppi_refractory: OnPPIRefractory
    hiatal_hernia: HiatalHerniaThresholds
    chicago_v4: ChicagoV4Thresholds
    endoflip: EndoFLIPThresholds
    eoe: EoEThresholds
    gastroparesis: GastroparesisThresholds
    symptom_association: SymptomAssociationThresholds
    contributing_factors: ContributingFactors


def _build_thresholds(raw: dict) -> Thresholds:
    """Pure function — easy to unit-test with synthetic YAML payloads."""
    lyon = raw["lyon_2_0"]
    chicago = raw["chicago_v4"]
    return Thresholds(
        lyon=LyonThresholds(
            aet=AETThresholds(**lyon["aet"]),
            reflux_episodes=RefluxEpisodeThresholds(**lyon["reflux_episodes"]),
            adjunctive=AdjunctiveThresholds(**lyon["adjunctive"]),
            conclusive_endoscopic=ConclusiveEndoscopicThresholds(
                la_grades=tuple(lyon["conclusive_endoscopic"]["la_grades"]),
                barrett_min_length_cm=lyon["conclusive_endoscopic"]["barrett_min_length_cm"],
                peptic_stricture=lyon["conclusive_endoscopic"]["peptic_stricture"],
            ),
            borderline_adjudication=BorderlineAdjudication(
                **lyon["borderline_adjudication"]
            ),
        ),
        on_ppi_refractory=OnPPIRefractory(**raw["on_ppi_refractory"]),
        hiatal_hernia=HiatalHerniaThresholds(**raw["hiatal_hernia"]),
        chicago_v4=ChicagoV4Thresholds(
            iem=IEMThresholds(**chicago["iem"]),
            absent_contractility_failed_pct=chicago["absent_contractility_failed_pct"],
            jackhammer=JackhammerThresholds(**chicago["jackhammer"]),
            des_premature_pct_min=chicago["des_premature_pct_min"],
            achalasia_irp_supine_min=chicago["achalasia_irp_supine_min"],
        ),
        endoflip=EndoFLIPThresholds(**raw["endoflip"]),
        eoe=EoEThresholds(**raw["eoe"]),
        gastroparesis=GastroparesisThresholds(**raw["gastroparesis"]),
        symptom_association=SymptomAssociationThresholds(**raw["symptom_association"]),
        contributing_factors=ContributingFactors(
            medications=tuple(raw["contributing_factors"]["medications"]),
        ),
    )


def load_thresholds(path: Path | None = None) -> Thresholds:
    """Load the default `thresholds.yaml` (or a custom path for tests)."""
    target = path or _THRESHOLDS_PATH
    with target.open("r") as f:
        raw = yaml.safe_load(f)
    return _build_thresholds(raw)

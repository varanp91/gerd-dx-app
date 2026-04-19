"""Enumerations for GERD diagnostic inputs."""

from enum import Enum


class LAGrade(str, Enum):
    NONE = "none"
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class EGJMorphology(str, Enum):
    TYPE_I = "I"    # LES and crural diaphragm superimposed
    TYPE_II = "II"  # LES and crural diaphragm separated <3cm
    TYPE_III = "III"  # Separation >=3cm (true hiatal hernia on HRM)


class Peristalsis(str, Enum):
    NORMAL = "normal"
    IEM = "ineffective_esophageal_motility"
    ABSENT = "absent_contractility"
    DES = "distal_esophageal_spasm"
    JACKHAMMER = "jackhammer"
    ACHALASIA_I = "achalasia_type_i"
    ACHALASIA_II = "achalasia_type_ii"
    ACHALASIA_III = "achalasia_type_iii"
    EGJ_OUTFLOW_OBSTRUCTION = "egj_outflow_obstruction"


class PPIResponse(str, Enum):
    NONE = "none"
    PARTIAL = "partial"
    FULL = "full"
    NOT_TRIED = "not_tried"


class DominantSymptom(str, Enum):
    HEARTBURN = "heartburn"
    REGURGITATION = "regurgitation"
    DYSPHAGIA = "dysphagia"
    EXTRA_ESOPHAGEAL = "extra_esophageal"
    CHEST_PAIN = "chest_pain"


class PhStudyState(str, Enum):
    """Whether the pH-impedance study was performed on or off PPI therapy."""
    ON_PPI = "on_ppi"
    OFF_PPI = "off_ppi"


class PriorSurgery(str, Enum):
    NONE = "none"
    SLEEVE_GASTRECTOMY = "sleeve_gastrectomy"
    ROUX_EN_Y = "roux_en_y"
    HELLER_MYOTOMY = "heller_myotomy"
    POEM = "poem"
    NISSEN = "nissen"
    TOUPET = "toupet"
    TIF = "tif"
    C_TIF = "c_tif"
    HIATAL_HERNIA_REPAIR = "hiatal_hernia_repair"
    OTHER = "other"


class BiopsyFinding(str, Enum):
    """
    Primary biopsy finding. DIS is intentionally NOT here — it often co-occurs
    with other findings and is captured as a separate bool on EndoscopyFindings.
    """
    NORMAL = "normal"
    EOSINOPHILS = "eosinophils"
    BASAL_CELL_HYPERPLASIA = "basal_cell_hyperplasia"
    OTHER = "other"


class ContributingMed(str, Enum):
    """Medications that modify GERD management but do not change classification."""
    GLP1 = "glp1"
    CCB = "ccb"
    NITRATE = "nitrate"
    OPIOID = "opioid"
    ANTICHOLINERGIC = "anticholinergic"
    BISPHOSPHONATE = "bisphosphonate"
    TRICYCLIC = "tricyclic"
    BENZODIAZEPINE = "benzodiazepine"


class Confidence(str, Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    RULED_OUT = "ruled_out"
    INSUFFICIENT = "insufficient"


class EvidenceStrength(str, Enum):
    CONCLUSIVE = "conclusive"      # Lyon 2.0 conclusive tier (LA C/D, Barrett's, peptic stricture, AET > pathologic)
    STRONG = "strong"              # Strong positive contribution
    SUPPORTIVE = "supportive"      # Adjunctive / borderline-band support
    BORDERLINE = "borderline"      # Inconclusive band (e.g. AET 4-6%)
    NEGATIVE = "negative"          # Argues against this mechanism (does not rule out alone)
    EXCLUSIONARY = "exclusionary"  # Rules out this mechanism

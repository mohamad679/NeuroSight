from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Any

COGNITIVE_FEATURES: tuple[str, ...] = (
    "MMSE",
    "MOCA",
    "CDRSB",
    "ADAS11",
    "RAVLT_immediate",
    "RAVLT_learning",
    "FAQ",
    "AGE",
)

OBSOLETE_COGNITIVE_FIELDS: tuple[str, ...] = (
    "cdr",
    "trail_a",
    "trail_b",
    "verbal_fluency",
    "sex",
)


class CognitiveSchema(BaseModel):
    """Canonical 8-feature cognitive assessment schema aligned with ADNI."""
    model_config = ConfigDict(extra="forbid")

    MMSE: float = Field(..., ge=0.0, le=30.0, description="Mini-Mental State Examination, range [0, 30]")
    MOCA: float = Field(..., ge=0.0, le=30.0, description="Montreal Cognitive Assessment, range [0, 30]")
    CDRSB: float = Field(..., ge=0.0, le=18.0, description="Clinical Dementia Rating Sum of Boxes, range [0, 18]")
    ADAS11: float = Field(..., ge=0.0, le=70.0, description="Alzheimer's Disease Assessment Scale 11, range [0, 70]")
    RAVLT_immediate: float = Field(..., ge=0.0, le=75.0, description="Rey Auditory Verbal Learning Test Immediate, range [0, 75]")
    RAVLT_learning: float = Field(..., ge=-15.0, le=15.0, description="Rey Auditory Verbal Learning Test Learning, range [-15, 15]")
    FAQ: float = Field(..., ge=0.0, le=30.0, description="Functional Activities Questionnaire, range [0, 30]")
    AGE: float = Field(..., ge=0.0, le=120.0, description="Age, range [0, 120]")

    @model_validator(mode="before")
    @classmethod
    def normalize_canonical_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            obsolete = sorted(
                key for key in data
                if str(key).lower() in OBSOLETE_COGNITIVE_FIELDS
            )
            if obsolete:
                canonical = ", ".join(COGNITIVE_FEATURES)
                raise ValueError(
                    f"Obsolete cognitive fields are not accepted: {', '.join(obsolete)}. "
                    f"Use canonical fields: {canonical}."
                )
            mapping = {
                "mmse": "MMSE",
                "moca": "MOCA",
                "cdrsb": "CDRSB",
                "adas11": "ADAS11",
                "ravlt_immediate": "RAVLT_immediate",
                "ravlt_learning": "RAVLT_learning",
                "faq": "FAQ",
                "age": "AGE",
            }
            new_data = {}
            for k, v in data.items():
                k_lower = k.lower()
                if k_lower in mapping:
                    new_data[mapping[k_lower]] = v
                else:
                    new_data[k] = v
            return new_data
        return data

    def to_features_dict(self) -> dict[str, float]:
        """Convert fields to a dict matching the canonical order."""
        return {field: float(getattr(self, field)) for field in COGNITIVE_FEATURES}

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Literal

from pydantic import BaseModel, Field, PositiveInt, validator


class StrictBaseModel(BaseModel):
    class Config:
        extra = "forbid"
        validate_assignment = True


class CriterionSchema(StrictBaseModel):
    name: str = Field(..., min_length=1)
    weight: PositiveInt = Field(..., ge=1, le=10, description="Importance weight from 1 (lowest) to 10 (highest).")
    description: Optional[str] = None
    kind: Literal["benefit", "cost"] = Field(
        "benefit",
        description="How this criterion behaves: 'benefit' (higher is better) or 'cost' (lower is better).",
    )


class OptionSchema(StrictBaseModel):
    name: str = Field(..., min_length=1)
    description: Optional[str] = None


class TriangularFuzzyNumber(StrictBaseModel):
    """
    Triangular fuzzy number representing a range of possible scores
    on a 1–10 scale: (l, m, u) with 1.0 <= l <= m <= u <= 10.0.
    """

    l: float = Field(..., description="Lower (worst-case) bound of the score.")
    m: float = Field(..., description="Most-likely score.")
    u: float = Field(..., description="Upper (best-case) bound of the score.")

    @validator("l", "m", "u")
    def within_scale(cls, v: float) -> float:
        if not (1.0 <= v <= 10.0):
            raise ValueError("Fuzzy score components must be between 1.0 and 10.0.")
        return v

    @validator("m")
    def m_not_less_than_l(cls, v: float, values) -> float:
        l = values.get("l")
        if l is not None and v < l:
            raise ValueError("m must be greater than or equal to l.")
        return v

    @validator("u")
    def u_not_less_than_m(cls, v: float, values) -> float:
        m = values.get("m")
        if m is not None and v < m:
            raise ValueError("u must be greater than or equal to m.")
        return v


class NormalizedTriangularFuzzyNumber(StrictBaseModel):
    """
    Triangular fuzzy number used internally for normalized scores in [0, 1].
    This is separate from TriangularFuzzyNumber, which represents user-facing
    scores on the original 1–10 scale.
    """

    l: float = Field(..., description="Lower bound of the normalized score (0.0–1.0).")
    m: float = Field(..., description="Most-likely normalized score (0.0–1.0).")
    u: float = Field(..., description="Upper bound of the normalized score (0.0–1.0).")

    @validator("l", "m", "u")
    def within_normalized_scale(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Normalized fuzzy score components must be between 0.0 and 1.0.")
        return v

    @validator("m")
    def m_not_less_than_l(cls, v: float, values) -> float:
        l = values.get("l")
        if l is not None and v < l:
            raise ValueError("For normalized TFNs, m must be greater than or equal to l.")
        return v

    @validator("u")
    def u_not_less_than_m(cls, v: float, values) -> float:
        m = values.get("m")
        if m is not None and v < m:
            raise ValueError("For normalized TFNs, u must be greater than or equal to m.")
        return v


class WeightedTriangularFuzzyNumber(StrictBaseModel):
    """
    Triangular fuzzy number used internally for weighted scores.
    Values may exceed 1.0 after applying criterion weights; we only
    enforce the usual ordering l <= m <= u.
    """

    l: float = Field(..., description="Lower bound of the weighted score.")
    m: float = Field(..., description="Most-likely weighted score.")
    u: float = Field(..., description="Upper bound of the weighted score.")

    @validator("m")
    def weighted_m_not_less_than_l(cls, v: float, values) -> float:
        l = values.get("l")
        if l is not None and v < l:
            raise ValueError("For weighted TFNs, m must be greater than or equal to l.")
        return v

    @validator("u")
    def weighted_u_not_less_than_m(cls, v: float, values) -> float:
        m = values.get("m")
        if m is not None and v < m:
            raise ValueError("For weighted TFNs, u must be greater than or equal to m.")
        return v


class OptionCriterionScore(StrictBaseModel):
    option_name: str
    criterion_name: str
    score_tfn: TriangularFuzzyNumber = Field(
        ...,
        description="Triangular fuzzy score (l, m, u) on a 1–10 scale.",
    )
    justification: str = Field(..., min_length=1)

    @validator("justification")
    def justification_not_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Justification must not be empty.")
        return v


class DecisionInputState(StrictBaseModel):
    problem_description: str = Field(..., min_length=1)
    options: List[OptionSchema] = Field(..., min_items=2)
    criteria: List[CriterionSchema] = Field(..., min_items=1)


class AIResearchResultState(StrictBaseModel):
    scores: Dict[Tuple[str, str], OptionCriterionScore] = Field(
        default_factory=dict,
        description="Mapping of (option_name, criterion_name) to researched score and justification.",
    )


class FinalScoresState(StrictBaseModel):
    scores: Dict[Tuple[str, str], OptionCriterionScore] = Field(
        default_factory=dict,
        description="Final human-approved fuzzy scores and justifications per (option, criterion).",
    )


class FuzzyOptionResult(StrictBaseModel):
    option_name: str
    closeness_coefficient: float
    distance_to_fpis: float
    distance_to_fnis: float


class FuzzyTopsisResult(StrictBaseModel):
    options: List[FuzzyOptionResult]
    winner: str
    loser: Optional[str] = None


class SynthesisState(StrictBaseModel):
    explanation: str


class GraphState(StrictBaseModel):
    """
    Aggregate state container used by LangGraph.
    Fields are optional so that different nodes can progressively enrich the state.
    """

    inputs: Optional[DecisionInputState] = None
    ai_scores: Optional[AIResearchResultState] = None
    final_scores: Optional[FinalScoresState] = None
    topsis_result: Optional[FuzzyTopsisResult] = None
    explanation: Optional[str] = None
    rag_context: Optional[str] = None


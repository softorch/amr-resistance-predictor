"""
schemas.py
──────────
Pydantic request and response models for the AMR Resistance Predictor API.
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator


# ── REQUEST MODELS ────────────────────────────────────────────────────────────

class SampleInput(BaseModel):
    """Mode B — single organism + antibiotic + optional MIC."""
    organism: str = Field(
        ...,
        example="Klebsiella",
        description="Organism name. One of: E_coli, Enterobacter, Klebsiella, "
                    "Pseudomonas, Staphylococcus, Acinetobacter"
    )
    antibiotic: str = Field(
        ...,
        example="meropenem",
        description="Antibiotic name in lowercase (e.g. meropenem, ciprofloxacin)"
    )
    mic_value: Optional[float] = Field(
        None,
        example=8.0,
        description="MIC value in mg/L. If omitted, historical prior is used."
    )
    mic_sign: Optional[str] = Field(
        None,
        example=">",
        description="Inequality sign: '<', '<=', '=', '>=', '>'. Defaults to '='"
    )
    lab_method: Optional[str] = Field(
        "Unknown",
        example="MIC",
        description="Laboratory typing method (e.g. MIC, DISK, ETEST)"
    )
    evidence: Optional[str] = Field(
        "Unknown",
        example="Laboratory Method",
        description="Evidence type: 'Laboratory Method' or 'Computational Method'"
    )
    model_score: Optional[float] = Field(
        0.0,
        example=0.95,
        description="Computational model confidence score (0–1). Use 0 if unknown."
    )

    @field_validator("organism")
    @classmethod
    def validate_organism(cls, v: str) -> str:
        allowed = {"E_coli", "Enterobacter", "Klebsiella",
                   "Pseudomonas", "Staphylococcus", "Acinetobacter"}
        if v.strip() not in allowed:
            raise ValueError(
                f"'{v}' is not a recognised organism. "
                f"Allowed values: {sorted(allowed)}"
            )
        return v.strip()

    @field_validator("mic_sign")
    @classmethod
    def validate_sign(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        allowed = {"<", "<=", "=", ">=", ">"}
        if v not in allowed:
            raise ValueError(f"mic_sign must be one of {allowed}")
        return v


class OrganismInput(BaseModel):
    """Mode C — organism name only."""
    organism: str = Field(
        ...,
        example="Acinetobacter",
        description="Organism name. Returns predictions across all known antibiotics."
    )

    @field_validator("organism")
    @classmethod
    def validate_organism(cls, v: str) -> str:
        allowed = {"E_coli", "Enterobacter", "Klebsiella",
                   "Pseudomonas", "Staphylococcus", "Acinetobacter"}
        if v.strip() not in allowed:
            raise ValueError(
                f"'{v}' is not a recognised organism. "
                f"Allowed values: {sorted(allowed)}"
            )
        return v.strip()


# ── RESPONSE MODELS ───────────────────────────────────────────────────────────

class PredictionRow(BaseModel):
    """A single resistance prediction result."""
    organism: str
    antibiotic: str
    antibiotic_class: str
    predicted_label: str = Field(
        description="Resistant | Susceptible | Intermediate"
    )
    confidence_pct: float = Field(
        description="Model confidence in the prediction (0–100)"
    )
    probabilities: Dict[str, float] = Field(
        description="Per-class probability percentages"
    )
    advisory: str = Field(
        default=(
            "This prediction is a decision-support tool only. "
            "Clinical laboratory confirmation is required before "
            "any treatment decision is made."
        )
    )


class PredictionResponse(BaseModel):
    """Response for Mode B — single sample prediction."""
    status: str = "success"
    mode: str = "single_sample"
    prediction: PredictionRow


class OrganismProfileResponse(BaseModel):
    """Response for Mode C — full organism resistance profile."""
    status: str = "success"
    mode: str = "organism_profile"
    organism: str
    total_antibiotics_tested: int
    resistant_count: int
    susceptible_count: int
    intermediate_count: int
    resistance_rate_pct: float
    predictions: List[PredictionRow]
    advisory: str = (
        "This profile is based on population-level priors from training data. "
        "It reflects typical resistance patterns for this organism — not a "
        "specific clinical isolate. Laboratory confirmation is required."
    )


class BatchPredictionResponse(BaseModel):
    """Response for Mode A — batch CSV predictions."""
    status: str = "success"
    mode: str = "batch_csv"
    total_rows_processed: int
    label_distribution: Dict[str, int]
    predictions: List[PredictionRow]


class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    target_classes: List[str]
    known_organisms: List[str]
    total_organism_antibiotic_pairs: int
    test_macro_f1: Optional[float]
    test_balanced_accuracy: Optional[float]


class AntibioticListResponse(BaseModel):
    antibiotics: List[str]
    total: int


class OrganismListResponse(BaseModel):
    organisms: List[str]
    total: int
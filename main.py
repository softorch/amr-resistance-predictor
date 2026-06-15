"""
main.py
───────
FastAPI application for the AMR Resistance Predictor.

Endpoints
─────────
GET  /                       → API info
GET  /health                 → health check + model metadata
GET  /organisms              → list all known organisms
GET  /antibiotics            → list all known antibiotics (optional ?organism=)
POST /predict/sample         → Mode B: single organism + antibiotic + MIC
POST /predict/organism       → Mode C: organism name only (full resistance profile)
POST /predict/batch          → Mode A: CSV file upload (up to 500K rows)
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import predictor
from schemas import (
    SampleInput,
    OrganismInput,
    PredictionResponse,
    OrganismProfileResponse,
    BatchPredictionResponse,
    HealthResponse,
    AntibioticListResponse,
    OrganismListResponse,
    PredictionRow,
)

# ── Model directory ───────────────────────────────────────────────────────────
# On Render: set MODEL_DIR as an environment variable in the dashboard
# Locally   : defaults to ./model (matches your PyCharm directory structure)
MODEL_DIR = os.environ.get("MODEL_DIR", "./model")


# ── Startup: load all artefacts once into memory ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup, release at shutdown."""
    print(f"[startup] Loading model artefacts from: {MODEL_DIR}")

    required = [
        "lgbm_model.pkl",
        "xgb_model.pkl",
        "label_encoders.pkl",
        "target_encoder.pkl",
        "scaler.pkl",
        "org_ab_stats.csv",
        "model_config.json",
    ]
    missing = [f for f in required
               if not os.path.exists(os.path.join(MODEL_DIR, f))]
    if missing:
        raise RuntimeError(
            f"[startup] Missing model files: {missing}. "
            f"Check MODEL_DIR='{MODEL_DIR}'"
        )

    models = {}

    # Load with graceful fallback if one model is absent
    for pkl in ("lgbm_model.pkl", "xgb_model.pkl"):
        path = os.path.join(MODEL_DIR, pkl)
        key  = pkl.replace(".pkl", "").replace("lgbm", "lgbm").replace("xgb", "xgb")
        try:
            models[key] = joblib.load(path)
            print(f"  ✅ {pkl} loaded")
        except Exception as e:
            models[key] = None
            print(f"  ⚠️  {pkl} could not be loaded: {e}")

    models["label_encoders"] = joblib.load(
        os.path.join(MODEL_DIR, "label_encoders.pkl"))
    models["target_le"]      = joblib.load(
        os.path.join(MODEL_DIR, "target_encoder.pkl"))
    models["scaler"]         = joblib.load(
        os.path.join(MODEL_DIR, "scaler.pkl"))
    models["org_ab_stats"]   = pd.read_csv(
        os.path.join(MODEL_DIR, "org_ab_stats.csv"))

    with open(os.path.join(MODEL_DIR, "model_config.json")) as f:
        models["config"] = json.load(f)

    predictor.initialise(models)

    loaded = [k for k, v in models.items()
              if v is not None and k not in ("config", "org_ab_stats")]
    print(f"[startup] Ready — {len(loaded)} artefacts loaded")
    print(f"[startup] Target classes: {list(models['target_le'].classes_)}")

    yield   # ← server runs here

    print("[shutdown] Releasing model artefacts")
    models.clear()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "AMR Resistance Predictor",
    description = (
        "Predicts antibiotic resistance (Resistant / Susceptible / Intermediate) "
        "for six clinically important organisms using LightGBM + XGBoost ensemble.\n\n"
        "**Organisms:** E_coli · Enterobacter · Klebsiella · "
        "Pseudomonas · Staphylococcus · Acinetobacter\n\n"
        "⚠️ **Advisory:** This API is a decision-support tool only. "
        "All predictions must be confirmed by clinical laboratory testing "
        "before treatment decisions are made."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)

# CORS — allow all origins for now; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Info"])
async def root():
    return {
        "api"         : "AMR Resistance Predictor",
        "version"     : "1.0.0",
        "docs"        : "/docs",
        "health"      : "/health",
        "endpoints"   : {
            "single_sample"    : "POST /predict/sample",
            "organism_profile" : "POST /predict/organism",
            "batch_csv"        : "POST /predict/batch",
            "list_organisms"   : "GET  /organisms",
            "list_antibiotics" : "GET  /antibiotics",
        },
        "advisory": (
            "This API is a clinical decision-support tool. "
            "Predictions must be confirmed by laboratory testing."
        ),
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health():
    cfg = predictor.config or {}
    loaded_models = []
    if predictor.lgbm_model is not None:
        loaded_models.append("LightGBM")
    if predictor.xgb_model is not None:
        loaded_models.append("XGBoost")

    return HealthResponse(
        status                        = "healthy" if loaded_models else "degraded",
        models_loaded                 = loaded_models,
        target_classes                = list(predictor.target_le.classes_),
        known_organisms               = predictor.get_known_organisms(),
        total_organism_antibiotic_pairs = len(predictor.ORG_AB_PRIORS),
        test_macro_f1                 = cfg.get("test_macro_f1"),
        test_balanced_accuracy        = cfg.get("test_balanced_accuracy"),
    )


@app.get("/organisms", response_model=OrganismListResponse, tags=["Reference"])
async def list_organisms():
    """Return all organisms the model was trained on."""
    orgs = predictor.get_known_organisms()
    return OrganismListResponse(organisms=orgs, total=len(orgs))


@app.get("/antibiotics", response_model=AntibioticListResponse, tags=["Reference"])
async def list_antibiotics(
    organism: Optional[str] = Query(
        None,
        description="Filter antibiotics by organism name"
    )
):
    """
    Return all known antibiotics.
    Pass ?organism=Klebsiella to filter to antibiotics tested against that organism.
    """
    abs_ = predictor.get_known_antibiotics(organism)
    if organism and not abs_:
        raise HTTPException(
            status_code=404,
            detail=f"No antibiotics found for organism '{organism}'. "
                   f"Known organisms: {predictor.get_known_organisms()}"
        )
    return AntibioticListResponse(antibiotics=abs_, total=len(abs_))


# ── MODE B ────────────────────────────────────────────────────────────────────
@app.post("/predict/sample",
          response_model=PredictionResponse,
          tags=["Predict"])
async def predict_sample(body: SampleInput):
    """
    **Mode B — Single sample prediction.**

    Provide an organism name, antibiotic, and optionally a measured MIC value.
    Returns a single Resistant / Susceptible / Intermediate prediction with
    confidence score and per-class probabilities.

    If `mic_value` is omitted, the model uses the learned historical prior
    for that organism-antibiotic combination.
    """
    try:
        result = predictor.predict_single(
            organism    = body.organism,
            antibiotic  = body.antibiotic,
            mic_value   = body.mic_value,
            mic_sign    = body.mic_sign,
            lab_method  = body.lab_method or "Unknown",
            evidence    = body.evidence   or "Unknown",
            model_score = body.model_score or 0.0,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictionResponse(
        prediction=PredictionRow(**result)
    )


# ── MODE C ────────────────────────────────────────────────────────────────────
@app.post("/predict/organism",
          response_model=OrganismProfileResponse,
          tags=["Predict"])
async def predict_organism(body: OrganismInput):
    """
    **Mode C — Full organism resistance profile.**

    Provide only an organism name. The model predicts resistance against
    every antibiotic it was trained on for that organism, using historical
    MIC priors. Returns a complete resistance map sorted by antibiotic class.

    Useful for empirical treatment guidance before lab results are available.
    """
    try:
        rows = predictor.predict_organism_profile(body.organism)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No prior data found for organism '{body.organism}'"
        )

    clean_rows = [r for r in rows if "low confidence" not in r["predicted_label"]]
    resistant  = sum(1 for r in clean_rows if r["predicted_label"] == "Resistant")
    susceptible= sum(1 for r in clean_rows if r["predicted_label"] == "Susceptible")
    intermed   = sum(1 for r in clean_rows if r["predicted_label"] == "Intermediate")
    total      = len(clean_rows)

    return OrganismProfileResponse(
        organism                = body.organism,
        total_antibiotics_tested= total,
        resistant_count         = resistant,
        susceptible_count       = susceptible,
        intermediate_count      = intermed,
        resistance_rate_pct     = round(resistant / total * 100, 2) if total else 0.0,
        predictions             = [PredictionRow(**r) for r in rows],
    )


# ── MODE A ────────────────────────────────────────────────────────────────────
@app.post("/predict/batch",
          response_model=BatchPredictionResponse,
          tags=["Predict"])
async def predict_batch(
    file: UploadFile = File(
        ...,
        description=(
            "CSV file with columns: Organism, Antibiotic, "
            "and optionally: Measurement Value, Measurement Sign, "
            "Laboratory Typing Method, Evidence"
        )
    ),
    max_rows: int = Query(
        10_000,
        ge=1,
        le=500_000,
        description="Maximum rows to process (default 10,000; max 500,000)"
    ),
):
    """
    **Mode A — Batch CSV prediction.**

    Upload a CSV file. Required columns: `Organism`, `Antibiotic`.
    Optional columns: `Measurement Value`, `Measurement Sign`,
    `Laboratory Typing Method`, `Evidence`.

    Processed in 25,000-row chunks. Response capped at `max_rows`
    (default 10,000) to prevent timeout on large files.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are accepted."
        )

    try:
        file_bytes = await file.read()
        rows = predictor.predict_batch_csv(file_bytes, max_rows=max_rows)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not rows:
        raise HTTPException(
            status_code=422,
            detail="No valid rows could be processed. "
                   "Ensure the CSV has 'Organism' and 'Antibiotic' columns."
        )

    label_dist = {}
    for r in rows:
        base_label = r["predicted_label"].replace(" (low confidence)", "")
        label_dist[base_label] = label_dist.get(base_label, 0) + 1

    return BatchPredictionResponse(
        total_rows_processed = len(rows),
        label_distribution   = label_dist,
        predictions          = [PredictionRow(**r) for r in rows],
    )
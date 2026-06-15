"""
predictor.py
────────────
Inference engine for the AMR Resistance Predictor.
Handles all three input modes:
  Mode B — single feature dict  (organism + antibiotic + MIC)
  Mode C — organism name only   (sweeps all known antibiotics using priors)
  Mode A — raw CSV file         (batch, chunked, memory-safe)
"""

from __future__ import annotations

import gc
import io
import re
from typing import Optional

import numpy  as np
import pandas as pd

# ── These are injected at startup by main.py ──────────────────────────────────
lgbm_model     = None
xgb_model      = None
label_encoders = None
target_le      = None
scaler         = None
org_ab_stats   = None
config         = None

ALL_FEATURES     : list = []
CAT_FEATURES     : list = []
NUM_FEATURES     : list = []
ANTIBIOTIC_CLASS : dict = {}
GRAM_MAP         : dict = {}
SIGN_MAP         : dict = {}
ORG_AB_PRIORS    : dict = {}

CONFIDENCE_THRESHOLD = 60.0   # predictions below this % are flagged as low-confidence


def initialise(models: dict):
    """Called once at FastAPI startup to inject all loaded artefacts."""
    global lgbm_model, xgb_model, label_encoders, target_le, scaler
    global org_ab_stats, config, ALL_FEATURES, CAT_FEATURES, NUM_FEATURES
    global ANTIBIOTIC_CLASS, GRAM_MAP, SIGN_MAP, ORG_AB_PRIORS

    lgbm_model     = models["lgbm_model"]
    xgb_model      = models["xgb_model"]
    label_encoders = models["label_encoders"]
    target_le      = models["target_le"]
    scaler         = models["scaler"]
    org_ab_stats   = models["org_ab_stats"]
    config         = models["config"]

    ALL_FEATURES     = config["ALL_FEATURES"]
    CAT_FEATURES     = config["CAT_FEATURES"]
    NUM_FEATURES     = config["NUM_FEATURES"]
    ANTIBIOTIC_CLASS = config["ANTIBIOTIC_CLASS"]
    GRAM_MAP         = config["GRAM_MAP"]
    SIGN_MAP         = {k: int(v) for k, v in config["SIGN_MAP"].items()}

    ORG_AB_PRIORS = (
        org_ab_stats
        .set_index(["Organism", "Antibiotic"])
        .to_dict("index")
    )


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _parse_mic(v) -> float:
    """Parse raw MIC string ('<=0.25', '16/4', '8.0') → float."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return -1.0
    s = str(v).strip().split("/")[0]
    s = re.sub(r"[^0-9.]", "", s)
    try:
        return float(s)
    except ValueError:
        return -1.0


def _build_feature_matrix(
    organisms   : list[str],
    antibiotics : list[str],
    mic_values  : list[float],
    mic_signs   : list[int],
    lab_methods : list[str],
    evidences   : list[str],
    model_scores: list[float],
) -> pd.DataFrame:
    """
    Vectorised feature builder.
    Returns a DataFrame with ALL_FEATURES columns, ready for encode + scale.
    """
    n           = len(organisms)
    antibiotics = [str(a).lower().strip() for a in antibiotics]
    mic_arr     = np.array(mic_values, dtype=float)

    # Fill missing MIC from organism-antibiotic priors
    for i in range(n):
        if mic_arr[i] <= 0:
            key = (organisms[i], antibiotics[i])
            mic_arr[i] = float(
                ORG_AB_PRIORS.get(key, {}).get("prior_MIC", -1)
            )

    mic_available = (mic_arr > 0).astype(int)
    log2_mic = np.where(
        mic_arr > 0,
        np.log2(np.clip(mic_arr, 0.001, None)),
        -99.0
    )

    # Per-(organism, antibiotic) aggregate stats
    org_ab_mean = np.empty(n)
    org_ab_std  = np.zeros(n)
    org_ab_rate = np.full(n, 0.5)
    for i in range(n):
        key   = (organisms[i], antibiotics[i])
        prior = ORG_AB_PRIORS.get(key, {})
        org_ab_mean[i] = float(prior.get("org_ab_mean_MIC", log2_mic[i]))
        org_ab_std[i]  = float(prior.get("org_ab_std_MIC",  0))
        org_ab_rate[i] = float(prior.get("org_ab_res_rate", 0.5))

    return pd.DataFrame({
        "Organism":                 organisms,
        "Antibiotic":               antibiotics,
        "Antibiotic_Class":         [ANTIBIOTIC_CLASS.get(ab, "Other") for ab in antibiotics],
        "Gram_Stain":               [GRAM_MAP.get(org, "Unknown")       for org in organisms],
        "Laboratory Typing Method": [str(m).upper() for m in lab_methods],
        "Evidence":                 evidences,
        "MIC_numeric":              mic_arr,
        "log2_MIC":                 log2_mic,
        "MIC_available":            mic_available,
        "Measurement Sign":         mic_signs,
        "org_ab_mean_MIC":          org_ab_mean,
        "org_ab_std_MIC":           org_ab_std,
        "org_ab_res_rate":          org_ab_rate,
        "Model_Score":              model_scores,
    })


def _encode_and_predict(feat_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Encode → scale → ensemble predict.
    Returns (pred_labels, confidence_pct, proba_matrix).
    """
    df = feat_df[ALL_FEATURES].copy()

    for col in CAT_FEATURES:
        le    = label_encoders[col]
        known = set(le.classes_)
        df[col] = (df[col].astype(str)
                   .apply(lambda x: x if x in known else le.classes_[0]))
        df[col] = le.transform(df[col])

    df[NUM_FEATURES] = scaler.transform(df[NUM_FEATURES])

    X = df[ALL_FEATURES]

    if lgbm_model is not None and xgb_model is not None:
        proba = (lgbm_model.predict_proba(X) + xgb_model.predict_proba(X)) / 2
    elif lgbm_model is not None:
        proba = lgbm_model.predict_proba(X)
    else:
        proba = xgb_model.predict_proba(X)

    pred_idx   = np.argmax(proba, axis=1)
    pred_label = target_le.inverse_transform(pred_idx)
    confidence = (proba.max(axis=1) * 100).round(2)

    del df, X
    gc.collect()

    return pred_label, confidence, proba


def _build_result_rows(
    organisms   : list,
    antibiotics : list,
    pred_labels : np.ndarray,
    confidence  : np.ndarray,
    proba       : np.ndarray,
) -> list[dict]:
    """Assemble prediction dicts for the response schema."""
    classes = list(target_le.classes_)
    rows    = []
    for i in range(len(organisms)):
        conf = float(confidence[i])
        label = str(pred_labels[i])

        # Flag low-confidence predictions
        if conf < CONFIDENCE_THRESHOLD:
            label = f"{label} (low confidence)"

        rows.append({
            "organism":         str(organisms[i]),
            "antibiotic":       str(antibiotics[i]),
            "antibiotic_class": ANTIBIOTIC_CLASS.get(
                                    str(antibiotics[i]).lower(), "Other"),
            "predicted_label":  label,
            "confidence_pct":   conf,
            "probabilities":    {
                cls: round(float(proba[i, j]) * 100, 2)
                for j, cls in enumerate(classes)
            },
            "advisory": (
                "This prediction is a decision-support tool only. "
                "Clinical laboratory confirmation is required before "
                "any treatment decision is made."
            ),
        })
    return rows


# ════════════════════════════════════════════════════════════════════════════
#  MODE B — Single sample prediction
# ════════════════════════════════════════════════════════════════════════════

def predict_single(
    organism    : str,
    antibiotic  : str,
    mic_value   : Optional[float] = None,
    mic_sign    : Optional[str]   = None,
    lab_method  : str = "Unknown",
    evidence    : str = "Unknown",
    model_score : float = 0.0,
) -> dict:
    sign_num = SIGN_MAP.get(str(mic_sign).strip() if mic_sign else "", 0)
    mic      = mic_value if mic_value is not None else -1.0

    feat_df = _build_feature_matrix(
        organisms    = [organism],
        antibiotics  = [antibiotic],
        mic_values   = [mic],
        mic_signs    = [sign_num],
        lab_methods  = [lab_method],
        evidences    = [evidence],
        model_scores = [model_score],
    )

    pred_label, confidence, proba = _encode_and_predict(feat_df)
    rows = _build_result_rows(
        [organism], [antibiotic], pred_label, confidence, proba)

    del feat_df, proba
    gc.collect()

    return rows[0]


# ════════════════════════════════════════════════════════════════════════════
#  MODE C — Organism profile (all known antibiotics)
# ════════════════════════════════════════════════════════════════════════════

def predict_organism_profile(organism: str) -> list[dict]:
    known_abs = sorted({
        ab for (org, ab) in ORG_AB_PRIORS.keys() if org == organism
    })
    if not known_abs:
        return []

    feat_df = _build_feature_matrix(
        organisms    = [organism] * len(known_abs),
        antibiotics  = known_abs,
        mic_values   = [-1.0] * len(known_abs),
        mic_signs    = [0]    * len(known_abs),
        lab_methods  = ["Unknown"] * len(known_abs),
        evidences    = ["Unknown"] * len(known_abs),
        model_scores = [0.0]  * len(known_abs),
    )

    pred_label, confidence, proba = _encode_and_predict(feat_df)
    rows = _build_result_rows(
        [organism] * len(known_abs), known_abs,
        pred_label, confidence, proba
    )

    del feat_df, proba
    gc.collect()

    return rows


# ════════════════════════════════════════════════════════════════════════════
#  MODE A — Batch CSV (chunked, memory-safe)
# ════════════════════════════════════════════════════════════════════════════

CHUNK_SIZE = 25_000

def predict_batch_csv(file_bytes: bytes, max_rows: int = 500_000) -> list[dict]:
    """
    Accepts raw CSV bytes (from FastAPI UploadFile.read()).
    Processes in chunks of 25K rows.
    Returns a list of prediction dicts (capped at max_rows for API safety).
    """
    all_results = []
    rows_done   = 0

    reader = pd.read_csv(
        io.BytesIO(file_bytes),
        chunksize  = CHUNK_SIZE,
        low_memory = False,
    )

    for chunk in reader:
        if rows_done >= max_rows:
            break

        # Trim to cap
        remaining = max_rows - rows_done
        if len(chunk) > remaining:
            chunk = chunk.iloc[:remaining]

        # Minimal cleaning
        chunk = chunk[chunk["Antibiotic"].notna() &
                      chunk["Organism"].notna()].copy()
        if len(chunk) == 0:
            continue

        chunk["MIC_numeric"] = chunk.get("Measurement Value",
                                          pd.Series([-1.0] * len(chunk)))
        chunk["MIC_numeric"] = chunk["MIC_numeric"].apply(_parse_mic)

        sign_col = chunk.get("Measurement Sign",
                              pd.Series([""] * len(chunk)))
        chunk["Sign_num"] = (sign_col.fillna("")
                             .map(SIGN_MAP).fillna(0).astype(int))

        lab_col  = chunk.get("Laboratory Typing Method",
                              pd.Series(["Unknown"] * len(chunk)))
        evid_col = chunk.get("Evidence",
                              pd.Series(["Unknown"] * len(chunk)))

        feat_df = _build_feature_matrix(
            organisms    = chunk["Organism"].tolist(),
            antibiotics  = chunk["Antibiotic"].tolist(),
            mic_values   = chunk["MIC_numeric"].tolist(),
            mic_signs    = chunk["Sign_num"].tolist(),
            lab_methods  = lab_col.fillna("Unknown").tolist(),
            evidences    = evid_col.fillna("Unknown").tolist(),
            model_scores = [0.0] * len(chunk),
        )

        pred_label, confidence, proba = _encode_and_predict(feat_df)
        rows = _build_result_rows(
            chunk["Organism"].tolist(),
            chunk["Antibiotic"].tolist(),
            pred_label, confidence, proba
        )
        all_results.extend(rows)
        rows_done += len(chunk)

        del feat_df, proba, chunk
        gc.collect()

    return all_results


# ── Utility helpers for /antibiotics and /organisms endpoints ─────────────────

def get_known_organisms() -> list[str]:
    return sorted({org for (org, _) in ORG_AB_PRIORS.keys()})


def get_known_antibiotics(organism: Optional[str] = None) -> list[str]:
    if organism:
        return sorted({
            ab for (org, ab) in ORG_AB_PRIORS.keys() if org == organism
        })
    return sorted({ab for (_, ab) in ORG_AB_PRIORS.keys()})
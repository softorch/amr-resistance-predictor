# AMR Resistance Predictor API

Predicts antibiotic resistance (**Resistant / Susceptible / Intermediate**) for six clinically important organisms using a LightGBM + XGBoost soft-vote ensemble trained on 1.5 million AMR records.

> ⚠️ **Advisory:** This API is a clinical decision-support tool only. All predictions must be confirmed by laboratory testing before any treatment decision is made.

---

## Organisms Supported

`E_coli` · `Enterobacter` · `Klebsiella` · `Pseudomonas` · `Staphylococcus` · `Acinetobacter`

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | API info and endpoint map |
| GET | `/health` | Health check + model metadata |
| GET | `/organisms` | List all supported organisms |
| GET | `/antibiotics` | List all known antibiotics (filter by `?organism=`) |
| POST | `/predict/sample` | **Mode B** — single organism + antibiotic + MIC |
| POST | `/predict/organism` | **Mode C** — organism name only, full resistance profile |
| POST | `/predict/batch` | **Mode A** — CSV file upload, batch predictions |

Interactive docs available at `/docs` when the server is running.

---

## Input Modes

### Mode B — Single Sample (`POST /predict/sample`)

Use when you have a specific organism, antibiotic, and MIC measurement.

```json
{
  "organism": "Klebsiella",
  "antibiotic": "meropenem",
  "mic_value": 8.0,
  "mic_sign": ">",
  "lab_method": "MIC",
  "evidence": "Laboratory Method",
  "model_score": 0.0
}
```

`mic_value` is optional — if omitted, the model uses the learned historical prior.

### Mode C — Organism Profile (`POST /predict/organism`)

Use when you have identified the organism but have no MIC data yet. Returns predictions across all antibiotics the model knows about for that organism.

```json
{
  "organism": "Acinetobacter"
}
```

### Mode A — Batch CSV (`POST /predict/batch`)

Upload a CSV file with at minimum `Organism` and `Antibiotic` columns. Optional columns: `Measurement Value`, `Measurement Sign`, `Laboratory Typing Method`, `Evidence`.

```
POST /predict/batch?max_rows=10000
Content-Type: multipart/form-data
file: your_samples.csv
```

---

## Project Structure

```
BioInformatics/
├── main.py              ← FastAPI app, all routes
├── predictor.py         ← Inference engine, all 3 modes
├── schemas.py           ← Pydantic request/response models
├── requirements.txt     ← Pinned dependencies
├── render.yaml          ← Render deployment config
├── .gitignore
├── README.md
└── model/               ← Model artefacts (from amr_deployment_package.zip)
    ├── lgbm_model.pkl
    ├── xgb_model.pkl
    ├── label_encoders.pkl
    ├── target_encoder.pkl
    ├── scaler.pkl
    ├── org_ab_stats.csv
    └── model_config.json
```

---

## Local Development

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the server
uvicorn main:app --reload --port 8000

# 4. Open docs
# http://localhost:8000/docs
```

---

## Deploy to Render

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial AMR predictor API"
git remote add origin https://github.com/Nathansparks19/amr-resistance-predictor.git
git push -u origin main
```

### Step 2 — Create a Render Web Service

1. Go to [render.com](https://render.com) → **New → Web Service**
2. Connect your GitHub repository
3. Render auto-detects `render.yaml` — settings are pre-filled
4. Set the environment variable:
   - Key: `MODEL_DIR`
   - Value: `./model`
5. Click **Deploy**

### Step 3 — Verify

```
https://amr-resistance-predictor.onrender.com/health
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_DIR` | `./model` | Path to the folder containing all 7 model artefacts |
| `PORT` | Set by Render | HTTP port — injected automatically by Render |

---

## Important Notes

**Model file size:** The 7 model files total ~11 MB and are committed directly to the repository. This is acceptable at this scale. If the models grow beyond ~100 MB in future retraining, use Git LFS or store on an S3-compatible bucket and download at startup.

**Render free tier cold starts:** The free tier spins down after 15 minutes of inactivity. The first request after spin-down takes 20–40 seconds while the server restarts and models reload. Upgrade to a paid instance ($7/month) to eliminate cold starts for production clinical use.

**Batch endpoint timeout:** Render's free tier has a 30-second request timeout. The batch endpoint is capped at `max_rows=10,000` by default for this reason. For larger batches, increase `max_rows` on a paid instance with a longer timeout, or process offline and use the results CSV directly.
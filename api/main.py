"""
APS Failure Prediction FastAPI Service
=======================================
Serves LightGBM model for Air Pressure System (APS) failure prediction
on Scania trucks. Applies the same preprocessing pipeline used in training.

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /            → health check + model metadata
    POST /predict     → single truck prediction
    POST /predict/batch → batch prediction (up to 1000 records)
    GET  /docs        → interactive Swagger UI
    GET  /redoc       → ReDoc documentation
"""

import json
import sys
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add parent dir to path so we can import schemas
sys.path.insert(0, str(Path(__file__).parent))
from schemas import (
    SensorInput, PredictionResponse,
    BatchSensorInput, BatchPredictionResponse, HealthResponse
)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
MODEL_DIR = Path(__file__).parent / "model_artifacts"


# ─────────────────────────────────────────────
# Load artifacts at startup
# ─────────────────────────────────────────────
def load_artifacts():
    try:
        model       = joblib.load(MODEL_DIR / "model.pkl")
        preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")
        with open(MODEL_DIR / "model_meta.json") as f:
            meta = json.load(f)
        return model, preprocessor, meta
    except FileNotFoundError as e:
        print(f"[ERROR] Artifact not found: {e}")
        print("  Run 'python run_analysis.py' first to generate model artifacts.")
        return None, None, None


model, preprocessor, meta = load_artifacts()

# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────
app = FastAPI(
    title="APS Failure Prediction API",
    description="""
## Scania Trucks — Air Pressure System (APS) Failure Predictor

This API uses a trained **LightGBM** classifier to predict whether a Scania truck
is likely to experience an **Air Pressure System (APS) failure** based on sensor readings.

### Label Mapping
| Prediction | Meaning |
|-----------|---------|
| `0` | Non-APS failure (negative class) |
| `1` | APS failure (positive class — alert!) |

### Risk Buckets
| Bucket | Probability | Action |
|--------|------------|--------|
| 🔴 **High** | ≥ 0.70 | Immediate inspection |
| 🟡 **Medium** | 0.40 – 0.69 | Schedule within 48h |
| 🟢 **Low** | < 0.40 | Routine check |

### Notes
- All sensor fields are optional — missing values are handled via median imputation
- The model was trained on the official Scania APS failure dataset (UCI/Kaggle)
- Primary metric: **PR-AUC** (prioritized for rare-event detection)
    """,
    version="1.0.0",
    contact={"name": "Praneeth Ramisetti", "email": "praneeth@example.com"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────
def add_miss_flags(df: pd.DataFrame, high_miss_cols: List[str]) -> pd.DataFrame:
    """Add binary missingness indicator columns (same as training)."""
    for c in high_miss_cols:
        if c in df.columns:
            df[f"{c}_missing"] = df[c].isnull().astype(int)
        else:
            df[f"{c}_missing"] = 1  # column not provided → treat as missing
    return df


def add_row_features(df: pd.DataFrame, orig_cols: List[str]) -> pd.DataFrame:
    """Add row-level aggregate features (same as training)."""
    present_cols = [c for c in orig_cols if c in df.columns]
    df['row_null_count'] = df[present_cols].isnull().sum(axis=1)
    df['row_null_frac']  = df['row_null_count'] / max(len(orig_cols), 1)
    df['row_mean']       = df[present_cols].mean(axis=1)
    df['row_std']        = df[present_cols].std(axis=1)
    df['row_max']        = df[present_cols].max(axis=1)
    df['row_min']        = df[present_cols].min(axis=1)
    return df


def preprocess_input(records: List[dict]) -> np.ndarray:
    """Convert a list of sensor dicts into preprocessed feature matrix."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run run_analysis.py first.")

    feature_cols   = meta['feature_cols']
    high_miss_cols = meta['high_miss_cols']
    all_feature_cols = meta['all_feature_cols']

    # Build dataframe with only the model's kept feature columns
    df = pd.DataFrame(records)
    # Ensure all expected columns present (fill with NaN if absent)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df[feature_cols].copy()
    df = df.astype(float, errors='ignore')

    # Feature engineering (match training)
    df = add_miss_flags(df, high_miss_cols)
    df = add_row_features(df, feature_cols)

    # Ensure column order matches training
    for col in all_feature_cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[all_feature_cols]

    X = preprocessor.transform(df)
    return X


def classify(prob: float, threshold: float) -> tuple:
    """Return (predicted_class, label, risk_bucket, recommendation, confidence)."""
    pred_class = int(prob >= threshold)
    label = "APS Failure" if pred_class == 1 else "Non-APS Failure"

    if prob >= 0.70:
        risk, rec = "High", "Immediate inspection required"
    elif prob >= 0.40:
        risk, rec = "Medium", "Schedule inspection within 48 hours"
    else:
        risk, rec = "Low", "Routine scheduled check"

    if prob >= 0.85 or prob <= 0.10:
        confidence = "High"
    elif prob >= 0.60 or prob <= 0.30:
        confidence = "Medium"
    else:
        confidence = "Low"

    return pred_class, label, risk, rec, confidence


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@app.get("/", response_model=HealthResponse, summary="Health Check & Model Info")
async def root():
    """Returns service health status and model metadata."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifacts not found. Run run_analysis.py.")
    return HealthResponse(
        status="healthy",
        model="LightGBM Classifier",
        version="1.0.0",
        test_pr_auc=meta.get("test_prauc", 0.0),
        test_roc_auc=meta.get("test_rocauc", 0.0),
        threshold=meta.get("best_threshold", 0.5),
        top_features=meta.get("top_features", []),
        endpoints=["/", "/predict", "/predict/batch", "/docs", "/redoc"]
    )


@app.post("/predict", response_model=PredictionResponse,
          summary="Predict APS failure for a single truck")
async def predict(sensor_data: SensorInput):
    """
    Accepts sensor readings for a single truck and returns:
    - Predicted class (0 or 1)
    - APS failure probability
    - Risk bucket (High / Medium / Low)
    - Operational recommendation

    **All sensor fields are optional** — missing values are handled automatically.
    """
    record = sensor_data.model_dump(exclude_none=False)
    X = preprocess_input([record])
    threshold = meta['best_threshold']
    prob = float(model.predict_proba(X)[0, 1])
    pred_class, label, risk, rec, confidence = classify(prob, threshold)

    return PredictionResponse(
        predicted_class=pred_class,
        predicted_label=label,
        probability=round(prob, 6),
        risk_bucket=risk,
        recommendation=rec,
        confidence=confidence,
        threshold_used=threshold
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse,
          summary="Batch predict APS failure for multiple trucks")
async def predict_batch(batch: BatchSensorInput):
    """
    Accepts up to **1,000** truck sensor records and returns predictions for each.
    Also returns a summary with risk distribution counts.
    """
    if len(batch.records) > 1000:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 1000 records.")
    if len(batch.records) == 0:
        raise HTTPException(status_code=400, detail="Batch must contain at least 1 record.")

    records = [r.model_dump(exclude_none=False) for r in batch.records]
    X = preprocess_input(records)
    threshold = meta['best_threshold']
    probs = model.predict_proba(X)[:, 1]

    predictions = []
    risk_counts = {"High": 0, "Medium": 0, "Low": 0}
    aps_count = 0

    for prob in probs:
        pred_class, label, risk, rec, confidence = classify(float(prob), threshold)
        risk_counts[risk] += 1
        if pred_class == 1:
            aps_count += 1
        predictions.append(PredictionResponse(
            predicted_class=pred_class,
            predicted_label=label,
            probability=round(float(prob), 6),
            risk_bucket=risk,
            recommendation=rec,
            confidence=confidence,
            threshold_used=threshold
        ))

    summary = {
        "total_trucks": len(records),
        "aps_failures_predicted": aps_count,
        "aps_failure_rate": round(aps_count / len(records), 4),
        "risk_distribution": risk_counts,
        "threshold_used": threshold
    }

    return BatchPredictionResponse(
        total=len(records),
        predictions=predictions,
        summary=summary
    )


@app.get("/health", summary="Simple health ping")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


# ─────────────────────────────────────────────
# Run directly
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

"""
FastAPI backend for the Intelligent Diabetes Risk Predictor.

Loads the trained pipeline + metadata once at startup, reproduces the training-time
preprocessing (impossible-zero -> median imputation), applies the optimized decision
threshold, and serves predictions.

Run from the project root:
    uvicorn app.main:app --reload
Then open http://127.0.0.1:8000  (web UI)  or  /docs  (Swagger).
"""
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.schemas import PatientData, PredictionResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger("diabetes-api")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "model.joblib"
META_PATH = BASE_DIR / "model" / "metadata.json"
STATIC_DIR = BASE_DIR / "app" / "static"

# Populated at startup and reused for every request.
state: dict = {"model": None, "meta": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise RuntimeError(
            "Model artifacts not found. Run `python train.py` before starting the server."
        )
    state["model"] = joblib.load(MODEL_PATH)
    state["meta"] = json.loads(META_PATH.read_text())
    logger.info("Loaded model '%s' (threshold=%.3f)",
                state["meta"]["model_name"], state["meta"].get("decision_threshold", 0.5))
    yield
    state.clear()


app = FastAPI(
    title="Intelligent Diabetes Risk Predictor",
    description="Predicts the probability of diabetes from clinical measurements.",
    version="2.0.0",
    lifespan=lifespan,
)

# Allow the bundled UI (and any front-end) to call the API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _preprocess(patient: PatientData) -> pd.DataFrame:
    """Apply the same impossible-zero -> median imputation used during training."""
    meta = state["meta"]
    row = {f: getattr(patient, f) for f in meta["features"]}
    df = pd.DataFrame([row], columns=meta["features"]).astype(float)
    for col in meta["zero_as_missing"]:
        df[col] = df[col].replace(0, np.nan).fillna(meta["medians"][col])
    return df


def _risk_level(prob: float) -> str:
    bands = state["meta"]["risk_bands"]
    if prob < bands["low"]:
        return "Low"
    if prob < bands["high"]:
        return "Moderate"
    return "High"


@app.get("/health", tags=["ops"])
def health() -> dict:
    meta = state["meta"]
    return {"status": "ok", "model": meta["model_name"], "version": app.version}


@app.get("/metadata", tags=["ops"])
def metadata() -> dict:
    """Model card: algorithm, tuned params, threshold and evaluation metrics."""
    meta = state["meta"]
    return {
        "model_name": meta["model_name"],
        "best_params": meta.get("best_params", {}),
        "decision_threshold": meta.get("decision_threshold", 0.5),
        "risk_bands": meta["risk_bands"],
        "metrics": meta["metrics"],
        "features": meta["features"],
    }


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(patient: PatientData) -> PredictionResponse:
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    meta = state["meta"]
    try:
        X = _preprocess(patient)
        prob = float(state["model"].predict_proba(X)[0, 1])
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    threshold = meta.get("decision_threshold", 0.5)
    return PredictionResponse(
        probability=round(prob, 4),
        risk_percent=round(prob * 100, 2),
        risk_level=_risk_level(prob),
        prediction=int(prob >= threshold),
        threshold=round(threshold, 4),
        model_name=meta["model_name"],
    )


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

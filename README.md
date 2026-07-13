# 🩺 Intelligent Diabetes Risk Predictor

An end-to-end, production-ready machine-learning application that predicts a patient's
probability of diabetes from clinical measurements, served through a **FastAPI**
backend with a simple web UI.

- **ML:** scikit-learn + XGBoost. Seven algorithms are hyperparameter-tuned and the best
  is auto-selected by cross-validated ROC-AUC.
- **Backend:** FastAPI with `/predict`, `/metadata`, `/health` endpoints + Swagger docs.
- **Frontend:** single-page form that calls the API and shows a color-coded risk band.
- **Ops:** Dockerfile, pytest suite, health checks.

## Project structure

```
diabetes_predictor/
├── data/diabetes.csv         # Pima Indians Diabetes dataset (768 records)
├── model/                    # created by train.py
│   ├── model.joblib          # trained scikit-learn/XGBoost pipeline
│   ├── metadata.json         # features, medians, threshold, metrics (used by the API)
│   └── metrics.json          # full report: leaderboard, confusion matrix, importances
├── app/
│   ├── main.py               # FastAPI application (lifespan, CORS, logging)
│   ├── schemas.py            # Pydantic request/response models
│   └── static/index.html     # web UI
├── tests/test_api.py         # pytest smoke + behaviour tests
├── docs/                     # architecture design PDF + generator
├── train.py                  # trains, tunes, evaluates, saves the model
├── Dockerfile                # containerized, ships ready-to-serve
├── requirements.txt
└── README.md
```

## Quick start

```bash
cd diabetes_predictor
python -m pip install -r requirements.txt
python train.py                 # trains + selects the best model (~2 min)
uvicorn app.main:app --reload   # serve
```

- Web UI:  http://127.0.0.1:8000
- Swagger docs:  http://127.0.0.1:8000/docs
- Model card:  http://127.0.0.1:8000/metadata
- Health:  http://127.0.0.1:8000/health

## The ML pipeline (`train.py`)

1. **Missing-data handling** — biologically impossible zeros (Glucose, BloodPressure,
   SkinThickness, Insulin, BMI) are treated as missing and imputed with the
   **training-set median** (medians are stored and re-applied at inference — no leakage,
   no train/serve skew).
2. **Model search** — Logistic Regression, Random Forest, Extra Trees, Gradient Boosting,
   SVM (RBF), KNN and XGBoost are each tuned with `GridSearchCV` (5-fold, ROC-AUC).
3. **Class imbalance** — handled via `class_weight="balanced"` / `scale_pos_weight`.
4. **Threshold optimization** — the decision threshold is tuned to maximize F1 on
   out-of-fold predictions instead of a naive 0.5. This matters for a screener: it lifts
   **recall on the diabetic class from ~0.50 to ~0.78** (fewer missed cases).
5. **Evaluation** — accuracy, ROC-AUC, precision, recall, F1, confusion matrix and
   permutation feature importance, all saved to `model/metrics.json`.

### Current best build

| Metric | Value |
|--------|-------|
| Selected model | **Extra Trees** (tuned, balanced) |
| CV ROC-AUC | 0.851 |
| Test ROC-AUC | 0.821 |
| Test accuracy | 0.721 |
| Test recall (diabetic) | **0.778** |
| Test F1 (diabetic) | 0.661 |
| Decision threshold | 0.468 |

> Top predictive features (permutation importance): **Glucose ≫ BMI ≈ Age > Pregnancies**.
> Accuracy is dataset-bounded — the Pima dataset (768 rows) caps most models near 77–78%.

## API

### `POST /predict`

```json
{
  "Pregnancies": 8, "Glucose": 197, "BloodPressure": 74, "SkinThickness": 45,
  "Insulin": 200, "BMI": 45.0, "DiabetesPedigreeFunction": 1.2, "Age": 58
}
```

Response:

```json
{
  "probability": 0.8347, "risk_percent": 83.47, "risk_level": "High",
  "prediction": 1, "threshold": 0.4679, "model_name": "ExtraTrees"
}
```

Risk bands: **Low** < 30% ≤ **Moderate** < 60% ≤ **High**.

### `GET /metadata`
Returns the model card: algorithm, tuned hyperparameters, threshold and all metrics.

## Tests

```bash
pytest -q
```

Covers the health check, metadata, prediction shape, monotonicity (high-risk profile
scores above low-risk), and input validation.

## Run with Docker (local)

```bash
docker build -t diabetes-predictor .
docker run -p 8000:8000 diabetes-predictor
```

The image trains the model during build and starts the API. It binds to `$PORT` if the
platform provides one, else 8000. Then open http://127.0.0.1:8000.

## Deploy to production

The project ships config for both platforms. **Render is recommended** for a free,
always-available demo (Railway needs a paid plan after its trial credit).

### Render (recommended)

1. Push this project to a GitHub repo.
2. Render dashboard -> **New + -> Blueprint** -> select the repo.
   Render reads [`render.yaml`](render.yaml), builds the Dockerfile (which trains the
   model), and deploys.
3. Your app goes live at `https://<name>.onrender.com`. Health checks hit `/health`.

Notes for the free plan: the service **sleeps after ~15 min idle** and cold-starts in
~30–60s on the next request; runtime RAM is 512 MB (this app fits comfortably).

### Railway (alternative)

1. Push to GitHub. Railway -> **New Project -> Deploy from GitHub repo**.
2. Railway auto-detects the Dockerfile; [`railway.json`](railway.json) sets the health
   check. Add a public domain under the service's **Settings -> Networking**.

Both platforms inject a `PORT` env var; the container binds to it automatically — no
change needed.

## Retraining

Re-run `python train.py` any time (e.g. with new data in `data/diabetes.csv`). It
regenerates `model.joblib`, `metadata.json` and `metrics.json`; the API picks up the new
model on its next start.

## Notes / disclaimer

- Educational/demo project — **not** a certified medical device and not medical advice.
- The dataset covers adult female patients of Pima Indian heritage; predictions outside
  that population are not clinically validated.
```

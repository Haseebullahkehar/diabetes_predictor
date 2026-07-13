"""
Generates the architecture / design PDF for the Intelligent Diabetes Risk Predictor.
Uses pyfpdf (fpdf 1.7.2) -> latin-1 only, so text stays ASCII.

Run:  python docs/generate_design_pdf.py
Output: docs/Architecture_Design.pdf
"""
from pathlib import Path
from fpdf import FPDF

OUT = Path(__file__).resolve().parent / "Architecture_Design.pdf"

# ---- palette -------------------------------------------------------------
NAVY   = (15, 23, 42)
SLATE  = (30, 41, 59)
SKY    = (56, 189, 248)
SKYDK  = (2, 132, 199)
LIGHT  = (241, 245, 249)
GREY   = (100, 116, 139)
WHITE  = (255, 255, 255)
GREEN  = (22, 163, 74)
AMBER  = (202, 138, 4)
RED    = (220, 38, 38)
INK    = (30, 41, 59)

PW, PH = 210, 297
LM, RM = 18, 18
CW = PW - LM - RM  # content width


class PDF(FPDF):
    def footer(self):
        self.set_y(-14)
        self.set_font("Arial", "", 8)
        self.set_text_color(*GREY)
        self.cell(0, 8, "Intelligent Diabetes Risk Predictor  -  Architecture & Design", 0, 0, "L")
        self.cell(0, 8, "Page %s" % self.page_no(), 0, 0, "R")


def sky_to_ink(pdf):
    pdf.set_text_color(*INK)


def section_bar(pdf, number, title):
    if pdf.get_y() > PH - 45:
        pdf.add_page()
    pdf.ln(3)
    y = pdf.get_y()
    pdf.set_fill_color(*NAVY)
    pdf.rect(LM, y, CW, 10, "F")
    pdf.set_fill_color(*SKY)
    pdf.rect(LM, y, 2.5, 10, "F")
    pdf.set_xy(LM + 6, y)
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(*WHITE)
    pdf.cell(0, 10, "%s   %s" % (number, title), 0, 1, "L")
    pdf.ln(3)
    sky_to_ink(pdf)


def sub(pdf, text):
    pdf.ln(1)
    pdf.set_font("Arial", "B", 10.5)
    pdf.set_text_color(*SKYDK)
    pdf.cell(0, 6, text, 0, 1, "L")
    pdf.set_text_color(*INK)


def body(pdf, text):
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(*INK)
    pdf.multi_cell(CW, 5.4, text)
    pdf.ln(1)


def bullet(pdf, text, indent=0):
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(*INK)
    x = LM + 2 + indent
    y = pdf.get_y() + 2.1
    pdf.set_fill_color(*SKY)
    pdf.rect(x, y, 1.8, 1.8, "F")
    pdf.set_xy(x + 5, pdf.get_y())
    pdf.multi_cell(CW - 7 - indent, 5.4, text)
    pdf.ln(0.6)


def kv_bullet(pdf, key, val):
    pdf.set_font("Arial", "B", 10)
    pdf.set_text_color(*INK)
    x = LM + 2
    y = pdf.get_y() + 2.1
    pdf.set_fill_color(*SKY)
    pdf.rect(x, y, 1.8, 1.8, "F")
    pdf.set_xy(x + 5, pdf.get_y())
    pdf.cell(42, 5.4, key, 0, 0, "L")
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(CW - 7 - 42, 5.4, val)
    pdf.ln(0.6)


def box(pdf, x, y, w, h, lines, fill, txtcolor=WHITE, border=None, fontsize=8.5, bold=True):
    pdf.set_fill_color(*fill)
    if border:
        pdf.set_draw_color(*border)
        pdf.set_line_width(0.5)
        pdf.rect(x, y, w, h, "DF")
        pdf.set_line_width(0.2)
    else:
        pdf.rect(x, y, w, h, "F")
    pdf.set_text_color(*txtcolor)
    if isinstance(lines, str):
        lines = [lines]
    total = len(lines)
    lh = 4.3
    start = y + (h - total * lh) / 2
    for i, ln in enumerate(lines):
        pdf.set_font("Arial", "B" if (bold and i == 0) else "", fontsize if i == 0 else fontsize - 0.7)
        pdf.set_xy(x, start + i * lh)
        pdf.cell(w, lh, ln, 0, 0, "C")


def arrow_down(pdf, x, y1, y2, label=None, color=SKYDK):
    pdf.set_draw_color(*color)
    pdf.set_line_width(0.5)
    pdf.line(x, y1, x, y2)
    pdf.line(x, y2, x - 1.8, y2 - 2.6)
    pdf.line(x, y2, x + 1.8, y2 - 2.6)
    pdf.set_line_width(0.2)
    if label:
        pdf.set_font("Arial", "", 7.5)
        pdf.set_text_color(*GREY)
        pdf.set_xy(x + 3, (y1 + y2) / 2 - 2.6)
        pdf.cell(60, 5, label, 0, 0, "L")


def arrow_right(pdf, x1, x2, y, color=SKYDK):
    pdf.set_draw_color(*color)
    pdf.set_line_width(0.5)
    pdf.line(x1, y, x2, y)
    pdf.line(x2, y, x2 - 2.6, y - 1.8)
    pdf.line(x2, y, x2 - 2.6, y + 1.8)
    pdf.set_line_width(0.2)


# =========================================================================
pdf = PDF(orientation="P", unit="mm", format="A4")
pdf.set_auto_page_break(True, margin=18)
pdf.set_margins(LM, 16, RM)

# ---- COVER ---------------------------------------------------------------
pdf.add_page()
pdf.set_fill_color(*NAVY)
pdf.rect(0, 0, PW, PH, "F")
pdf.set_fill_color(*SKY)
pdf.rect(0, 96, PW, 1.2, "F")
pdf.set_xy(0, 40)
pdf.set_font("Arial", "", 12)
pdf.set_text_color(*SKY)
pdf.cell(0, 8, "MACHINE LEARNING  |  FASTAPI  PROJECT", 0, 1, "C")
pdf.ln(6)
pdf.set_font("Arial", "B", 30)
pdf.set_text_color(*WHITE)
pdf.set_x(0)
pdf.cell(0, 16, "Intelligent Diabetes", 0, 1, "C")
pdf.set_x(0)
pdf.cell(0, 16, "Risk Predictor", 0, 1, "C")
pdf.ln(4)
pdf.set_font("Arial", "", 13)
pdf.set_text_color(*LIGHT)
pdf.set_x(0)
pdf.cell(0, 8, "Architecture, Approach & Technology Stack", 0, 1, "C")
pdf.ln(30)
pdf.set_font("Arial", "", 10.5)
pdf.set_text_color(*GREY)
pdf.set_x(0)
pdf.cell(0, 7, "End-to-end ML system: data -> model -> REST API -> web UI", 0, 1, "C")
pdf.set_x(0)
pdf.cell(0, 7, "Dataset: Pima Indians Diabetes (768 records)", 0, 1, "C")
pdf.ln(4)
pdf.set_x(0)
pdf.set_text_color(*SKY)
pdf.cell(0, 7, "Design Document v1.0", 0, 1, "C")

# ---- 1. OVERVIEW ---------------------------------------------------------
pdf.add_page()
section_bar(pdf, "1.", "Project Overview")
body(pdf, "The Intelligent Diabetes Risk Predictor is a machine-learning application that "
          "estimates a patient's probability of having diabetes from eight routine clinical "
          "measurements (glucose, BMI, age, blood pressure, insulin, and others). A trained "
          "scikit-learn model is served through a FastAPI backend and consumed by a simple "
          "web interface, giving an instant, colour-coded risk read-out.")
sub(pdf, "Problem statement")
body(pdf, "Early identification of individuals at risk of diabetes enables timely lifestyle "
          "intervention. Manual screening is slow and inconsistent. A data-driven predictor "
          "provides a fast, repeatable first-pass risk estimate that can support (not replace) "
          "clinical judgement.")
sub(pdf, "Objectives")
bullet(pdf, "Train a reliable binary classifier on the Pima Indians Diabetes dataset.")
bullet(pdf, "Expose the model through a clean REST API with input validation and auto-docs.")
bullet(pdf, "Provide a user-friendly web form that returns a risk percentage and risk band.")
bullet(pdf, "Apply sound ML engineering: no data leakage, missing-value handling, model selection.")

sub(pdf, "Scope & disclaimer")
body(pdf, "This is an educational / demonstration project, not a certified medical device. The "
          "dataset represents adult female patients of Pima Indian heritage, so predictions "
          "outside that population are not clinically validated.")

# ---- 2. ARCHITECTURE (diagram) ------------------------------------------
pdf.add_page()
section_bar(pdf, "2.", "System Architecture")
body(pdf, "The system has two decoupled halves: an OFFLINE training pipeline that produces model "
          "artifacts, and an ONLINE serving path that loads those artifacts and answers requests.")

# ----- Serving flow (vertical) -----
sub(pdf, "A. Serving path (runtime)")
top = pdf.get_y() + 2
bx = LM + 30
bw = CW - 60
bh = 13
gap = 9

box(pdf, bx, top, bw, bh, ["User / Browser  -  Web UI (index.html)",
                           "HTML form + fetch()"], SLATE, WHITE)
y2 = top + bh + gap
arrow_down(pdf, PW / 2, top + bh, y2, "HTTP POST /predict  (JSON body)")
box(pdf, bx, y2, bw, bh, ["FastAPI application  (Uvicorn ASGI server)",
                          "routes: /predict   /health   /   /docs"], SKYDK, WHITE)
y3 = y2 + bh + gap
arrow_down(pdf, PW / 2, y2 + bh, y3, "validated features (Pydantic) + median imputation")
box(pdf, bx, y3, bw, bh, ["scikit-learn Pipeline",
                          "StandardScaler  ->  Classifier  ->  predict_proba"], NAVY, WHITE)
# return arrow up (right side)
pdf.set_draw_color(*GREEN)
rx = bx + bw + 6
pdf.set_line_width(0.5)
pdf.line(rx, y3 + bh / 2, rx, top + bh / 2)
pdf.line(rx, top + bh / 2, rx - 2.6, top + bh / 2 + 1.8)
pdf.line(rx, top + bh / 2, rx - 2.6, top + bh / 2 - 1.8)
pdf.line(bx + bw, y3 + bh / 2, rx, y3 + bh / 2)
pdf.line(rx, top + bh / 2, bx + bw, top + bh / 2)
pdf.set_line_width(0.2)
pdf.set_font("Arial", "", 7.5)
pdf.set_text_color(*GREEN)
pdf.set_xy(rx - 1, (top + y3) / 2)
pdf.rotate(0)
pdf.text(rx + 1, (top + y3) / 2 + bh / 2, "JSON response:")
pdf.text(rx + 1, (top + y3) / 2 + bh / 2 + 4, "probability, %,")
pdf.text(rx + 1, (top + y3) / 2 + bh / 2 + 8, "risk_level")

# ----- Training pipeline (horizontal) -----
pdf.set_y(y3 + bh + 12)
sub(pdf, "B. Training pipeline (offline, run once via train.py)")
ty = pdf.get_y() + 2
tbw = (CW - 3 * 6) / 4
tbh = 20
labels = [
    ["Pima Dataset", "diabetes.csv", "768 rows x 8"],
    ["Preprocess", "0 -> NaN,", "median impute"],
    ["Train & Select", "LogReg / RF / GBM", "best CV ROC-AUC"],
    ["Artifacts", "model.joblib", "metadata.json"],
]
colors = [GREY, SKYDK, SKYDK, GREEN]
xs = []
for i in range(4):
    x = LM + i * (tbw + 6)
    xs.append(x)
    box(pdf, x, ty, tbw, tbh, labels[i], colors[i], WHITE, fontsize=8.5)
    if i < 3:
        arrow_right(pdf, x + tbw, x + tbw + 6, ty + tbh / 2)

# dashed link: artifacts loaded by serving pipeline at startup
pdf.set_draw_color(*AMBER)
ax = xs[3] + tbw / 2
pdf.dashed_line(ax, ty, ax, y3 + bh + 3, 1.5, 1.5) if hasattr(pdf, "dashed_line") else pdf.line(ax, ty, ax, y3 + bh + 3)
pdf.set_font("Arial", "I", 7.5)
pdf.set_text_color(*AMBER)
pdf.set_xy(ax + 2, ty - 6)
pdf.cell(70, 4, "loaded at server startup", 0, 0, "L")

pdf.set_y(ty + tbh + 8)
pdf.set_text_color(*INK)
body(pdf, "Because training and serving share the exact same preprocessing (the medians learned "
          "at training time are stored in metadata.json and re-applied at inference), the model "
          "sees production inputs in precisely the form it was trained on - eliminating "
          "train/serve skew.")

# ---- 3. HOW IT WORKS -----------------------------------------------------
pdf.add_page()
section_bar(pdf, "3.", "How It Works - Request Lifecycle")
steps = [
    ("1. User input", "The user fills the web form (glucose, BMI, age, etc.) and clicks Predict. "
                       "JavaScript packages the values into a JSON object."),
    ("2. HTTP request", "A POST request is sent to /predict with the JSON body and "
                        "Content-Type: application/json."),
    ("3. Validation", "FastAPI parses the body into a Pydantic PatientData model. Out-of-range or "
                      "missing fields are rejected automatically with a 422 error and a clear message."),
    ("4. Preprocessing", "Biologically impossible zeros (e.g. Glucose=0) are replaced with the "
                         "training-set median stored in metadata.json - identical to training."),
    ("5. Inference", "The scikit-learn pipeline scales the features and calls predict_proba to get "
                     "the probability of the positive (diabetic) class."),
    ("6. Risk banding", "The probability is mapped to a band: Low (<30%), Moderate (30-60%), "
                        "High (>=60%), and a 0/1 label at a 0.5 threshold."),
    ("7. Response", "A JSON response (probability, risk_percent, risk_level, prediction, "
                    "model_name) is returned and rendered as a colour-coded result card."),
]
for k, v in steps:
    kv_bullet(pdf, k, v)

# ---- 4. APPROACH ---------------------------------------------------------
pdf.add_page()
section_bar(pdf, "4.", "Development Approach")
body(pdf, "The project was built in clear phases, each producing a testable artifact.")
phases = [
    ("Phase 1 - Data", "Acquire the Pima dataset, inspect distributions, and identify the "
                       "'impossible zero' problem in Glucose, BloodPressure, SkinThickness, Insulin and BMI."),
    ("Phase 2 - Modelling", "Split data (stratified 80/20). Impute missing values using medians "
                            "computed on the TRAINING split only. Standardise features."),
    ("Phase 3 - Selection", "Compare Logistic Regression, Random Forest and Gradient Boosting via "
                            "5-fold cross-validated ROC-AUC; keep the best pipeline."),
    ("Phase 4 - Evaluation", "Report accuracy, ROC-AUC and a full classification report on the "
                             "held-out test set. Persist model.joblib + metadata.json."),
    ("Phase 5 - API", "Wrap the model in FastAPI with Pydantic validation, a /predict endpoint, a "
                      "/health check, and auto-generated Swagger docs at /docs."),
    ("Phase 6 - Frontend", "Build a lightweight HTML/JS page that posts to the API and shows a "
                           "colour-coded risk card."),
    ("Phase 7 - Verify & document", "Test low/high/invalid inputs end-to-end; write the README "
                                    "and this design document."),
]
for k, v in phases:
    kv_bullet(pdf, k, v)

sub(pdf, "Engineering principles applied")
bullet(pdf, "No data leakage - imputation statistics learned on training data only.")
bullet(pdf, "Reproducibility - fixed random_state; artifacts versioned as files.")
bullet(pdf, "Train/serve parity - one shared preprocessing definition in metadata.json.")
bullet(pdf, "Separation of concerns - training, model, API and UI are independent modules.")

# ---- 5. TECH STACK -------------------------------------------------------
pdf.add_page()
section_bar(pdf, "5.", "Technology Stack")
rows = [
    ("Language", "Python 3.13", "Rich ML + web ecosystem; single language end to end."),
    ("ML library", "scikit-learn", "Pipelines, imputation, scaling, model selection, metrics."),
    ("Data handling", "pandas / NumPy", "Loading the CSV and vectorised feature processing."),
    ("Persistence", "joblib + JSON", "Serialise the fitted pipeline and preprocessing metadata."),
    ("API framework", "FastAPI", "Async, type-safe REST API with automatic OpenAPI/Swagger docs."),
    ("Validation", "Pydantic v2", "Declarative request schemas with range checks and errors."),
    ("Web server", "Uvicorn", "High-performance ASGI server to run FastAPI."),
    ("Frontend", "HTML + CSS + JS", "Zero-build single page using the native fetch() API."),
    ("Dataset", "Pima Indians Diabetes", "768 labelled clinical records, 8 features."),
]
# table header
col1, col2 = 34, 40
col3 = CW - col1 - col2
pdf.set_font("Arial", "B", 9.5)
pdf.set_fill_color(*NAVY)
pdf.set_text_color(*WHITE)
pdf.cell(col1, 8, "  Layer", 1, 0, "L", True)
pdf.cell(col2, 8, "  Technology", 1, 0, "L", True)
pdf.cell(col3, 8, "  Purpose", 1, 1, "L", True)
fill = False
for layer, tech, why in rows:
    pdf.set_text_color(*INK)
    pdf.set_fill_color(*(LIGHT if fill else WHITE))
    x0, y0 = pdf.get_x(), pdf.get_y()
    # measure height by wrapping purpose
    pdf.set_font("Arial", "", 9)
    # compute number of lines for col3
    line_h = 5
    # draw purpose in a temp to get height
    pdf.set_xy(x0 + col1 + col2, y0)
    before = pdf.get_y()
    pdf.multi_cell(col3, line_h, "  " + why, 0, "L", fill)
    after = pdf.get_y()
    row_h = max(after - before, 8)
    # left cells
    pdf.set_xy(x0, y0)
    pdf.set_font("Arial", "B", 9)
    pdf.cell(col1, row_h, "  " + layer, 0, 0, "L", fill)
    pdf.set_font("Arial", "", 9)
    pdf.set_text_color(*SKYDK)
    pdf.cell(col2, row_h, "  " + tech, 0, 0, "L", fill)
    pdf.set_text_color(*INK)
    # borders
    pdf.set_draw_color(220, 226, 235)
    pdf.rect(x0, y0, col1, row_h)
    pdf.rect(x0 + col1, y0, col2, row_h)
    pdf.rect(x0 + col1 + col2, y0, col3, row_h)
    pdf.set_xy(x0, y0 + row_h)
    fill = not fill

pdf.ln(4)
sub(pdf, "Model performance (this build)")
bullet(pdf, "Selected model: Logistic Regression (best cross-validated ROC-AUC).")
bullet(pdf, "Cross-validated ROC-AUC ~ 0.84;  held-out test ROC-AUC ~ 0.81;  accuracy ~ 0.71.")

# ---- 6. API + FUTURE -----------------------------------------------------
pdf.add_page()
section_bar(pdf, "6.", "API Endpoints & Future Work")
sub(pdf, "REST endpoints")
kv_bullet(pdf, "POST /predict", "Accepts patient JSON, returns probability, risk_percent, risk_level and prediction.")
kv_bullet(pdf, "GET /health", "Liveness check; returns model name and evaluation metrics.")
kv_bullet(pdf, "GET /", "Serves the web UI (index.html).")
kv_bullet(pdf, "GET /docs", "Interactive Swagger UI (auto-generated by FastAPI).")

sub(pdf, "Future enhancements")
bullet(pdf, "Explainability: add SHAP values so each prediction shows which factors drove the risk.")
bullet(pdf, "Containerisation: Dockerfile + docker-compose for one-command, portable deployment.")
bullet(pdf, "Model registry & retraining: schedule periodic retraining and track model versions.")
bullet(pdf, "Persistence & auth: store predictions in a database and secure the API with API keys.")
bullet(pdf, "Testing & CI: pytest suite for the API plus a GitHub Actions pipeline.")
bullet(pdf, "Monitoring: log prediction distributions to detect data drift over time.")

pdf.ln(4)
pdf.set_draw_color(*SKY)
pdf.set_line_width(0.4)
pdf.line(LM, pdf.get_y(), PW - RM, pdf.get_y())
pdf.ln(3)
pdf.set_font("Arial", "I", 9)
pdf.set_text_color(*GREY)
pdf.multi_cell(CW, 5, "This document describes the design of an educational machine-learning "
                     "project. It is not intended for clinical use.")

pdf.output(str(OUT))
print("Wrote", OUT)

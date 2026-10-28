# CareerCast — Milestone 1

An AI-assisted career guidance web app (Flask). Users register, build a profile, upload a resume (PDF/DOCX), and receive ranked career recommendations, a skill-gap view and a CSV export.

This repository contains **Milestone 1 (Weeks 1–2): Parsing & Baseline Prediction**.

## Features

- User registration / login with hashed passwords and CSRF protection
- Structured profile form with validation
- Resume analyzer: PDF/DOCX text extraction, skill / certification / education matching (Infosys, Guvi, upGrad and other common courses recognised)
- Optional spaCy named-entity extraction (`en_core_web_sm`)
- Career recommendations ranked by a TF-IDF + Logistic Regression model, with a keyword-based fallback if the model file is missing
- Skill-gap analysis with suggested learning resources
- CSV export of recommendations

## Project structure

```
app.py                  Flask app (routes, resume parsing, recommendation logic)
ml/predict_multifeature.py   Trained model inference (skills + description + certifications)
ml/predict.py           Keyword-based fallback ranker
models/                 Trained .joblib models and evaluation reports (JSON/CSV)
templates/, static/     HTML templates and CSS
inspect_model.py        Prints details of the saved model
test_career_model.py    Quick sanity test of the saved model
```

## Model results (Milestone 1)

Evaluated on a held-out 20% split (random_state=42) of a job-listings dataset (`jobs.csv`, 118 job-title classes). Full numbers are in `models/`.

| Model | Top-1 accuracy | Macro F1 | Top-3 coverage |
|---|---|---|---|
| TF-IDF + Logistic Regression (skills only, baseline) | 57.6% | 61.2% | 82.3% |
| Skills + job description + certifications (**used by the app**) | **72.1%** | **74.4%** | **91.4%** |

Milestone 1 target: ≥65% Top-1 accuracy for the baseline classifier. The skills-only baseline reaches 57.6% and the multi-feature model reaches 72.1%.

**Limitations**
- The evaluation uses job-listing records, not labelled candidate resumes, so it does not measure real resume-to-career accuracy.
- Scores are model confidence values, **not** calibrated probabilities of getting a job.
- Scanned (image-only) PDFs need OCR, which is not included.
- If the spaCy English model is not installed, entity extraction is disabled and the rest of the app still works.
- The dataset (`jobs.csv`) and the training script are not included in this repository.

## Setup

Requires Python 3.10+ (developed on 3.13).

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm      # optional, enables entity extraction
```

Copy `.env.example` to `.env` and set `SECRET_KEY` to a long random value:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Run:

```bash
python app.py
```

Open http://127.0.0.1:5000, then register, fill in your profile, try the resume analyzer, recommendations, skill gap and CSV export.

To check the saved model:

```bash
python inspect_model.py
python test_career_model.py
```

## Database

A local SQLite file (`career_intelligence.db`) is created automatically on first run. It is git-ignored because it holds user data; do not commit it.

## Security notes

- Never commit `.env`, credentials or database files.
- Rotate any `SECRET_KEY` that has been shared outside your machine.

## Roadmap

- Milestone 2: Random Forest / XGBoost, Sentence-BERT skill embeddings, Top-K ranking with skill-alignment metrics
- Milestone 3: FastAPI service, MLflow registry, CI accuracy gate, Streamlit UI
- Milestone 4: pip packaging, tests, cohort analytics, model/dataset cards
"""Career prediction using the saved TF-IDF + Logistic Regression model."""

from pathlib import Path
import joblib

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "career_multifeature_model.joblib"
)

_model = None


def load_model():
    """Load the saved model once and reuse it."""
    global _model

    if _model is None and MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)

    return _model


def predict_careers(
    skills="",
    job_description="",
    certifications="",
    top_k=5,
    **kwargs
):
    """Return ranked career predictions and model confidence scores."""

    combined_text = " ".join(
        str(value or "")
        for value in (skills, job_description, certifications)
    ).strip()

    if not combined_text:
        return []

    model = load_model()

    if model is None:
        return []

    probabilities = model.predict_proba([combined_text])[0]
    classes = model.classes_

    top_indices = probabilities.argsort()[::-1][
        :max(1, int(top_k))
    ]

    results = []

    for index in top_indices:
        results.append({
            "career": str(classes[index]),
            "score": round(float(probabilities[index]) * 100, 2),
        })

    return results
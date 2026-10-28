import joblib
from pathlib import Path

model_path = Path(__file__).resolve().parent / "models" / "career_multifeature_model.joblib"

if not model_path.exists():
    print("ERROR: Model file not found.")
else:
    try:
        model = joblib.load(model_path)

        sample_text = (
            "Python, machine learning, deep learning, "
            "TensorFlow, data analysis, SQL, pandas"
        )

        print("Testing the trained career model...")
        print("Input:", sample_text)

        predictions = model.predict([sample_text])

        print("\nPredicted career:", predictions[0])

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([sample_text])[0]
            classes = model.classes_

            top_indices = probabilities.argsort()[::-1][:5]

            print("\nTop 5 career predictions:")

            for index in top_indices:
                print(
                    f"{classes[index]}: "
                    f"{probabilities[index] * 100:.2f}%"
                )

        print("\nModel test completed successfully.")

    except Exception as error:
        print("Model test failed:", repr(error))
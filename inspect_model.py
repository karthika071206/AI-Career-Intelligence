import joblib
from pathlib import Path

model_path = Path(__file__).resolve().parent / "models" / "career_multifeature_model.joblib"

print("Checking model:", model_path.resolve())

if not model_path.exists():
    print("Model file not found. Check the folder name.")

else:
    try:
        model = joblib.load(model_path)

        print("\nModel loaded successfully!")
        print("Model type:", type(model).__name__)

        if hasattr(model, "steps"):
            print("\nPipeline steps:")

            for name, step in model.steps:
                print(f"{name}: {type(step).__name__}")

        if hasattr(model, "classes_"):
            print(
                "\nNumber of career categories:",
                len(model.classes_)
            )
            print("First 10 categories:", model.classes_[:10])

        if hasattr(model, "n_features_in_"):
            print(
                "\nNumber of input features:",
                model.n_features_in_
            )

    except Exception as error:
        print("\nError loading model:", repr(error))
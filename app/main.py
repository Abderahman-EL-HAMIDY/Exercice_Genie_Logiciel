

from pipeline.trainer import train_pipeline
from pipeline.evaluator import evaluate_model
from core.model import load_model
from typing import List
from typing import Tuple, Any

# Configuration
DATA_PATH = 'data/patient_data.csv'
MODEL_PATH = 'app/trained_model.pkl'


def predict_infection(model: Any, symptoms: List[float]) -> str:
    
    prediction = model.predict([symptoms])
    return 'Infected' if prediction[0] == 1 else 'Not Infected'


def main() -> None:
    """Execute main application workflow."""
    # Training phase
    print("=" * 50)
    print("TRAINING PHASE")
    print("=" * 50)
    model, X_test, y_test = train_pipeline(DATA_PATH, MODEL_PATH)
    
    # Evaluation phase
    print("\n" + "=" * 50)
    print("EVALUATION PHASE")
    print("=" * 50)
    evaluate_model(model, X_test, y_test)
    
    # Prediction example
    print("\n" + "=" * 50)
    print("PREDICTION EXAMPLE")
    print("=" * 50)
    new_symptoms = [38.5, 1, 1]  # temperature, cough, fatigue
    result = predict_infection(model, new_symptoms)
    print(f"Symptoms: {new_symptoms}")
    print(f"Prediction: {result}")


if __name__ == "__main__":
    main()
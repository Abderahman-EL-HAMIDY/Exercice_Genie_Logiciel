
from sklearn.metrics import accuracy_score, classification_report
from typing import Any
import pandas as pd


def evaluate_model(model: Any, X_test: pd.DataFrame, 
                   y_test: pd.Series) -> float:
    
    
    # Evaluate the trained model on test data
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    _print_evaluation_results(accuracy, y_test, y_pred)
    
    return accuracy


def _print_evaluation_results(accuracy: float, y_test: pd.Series, 
                              y_pred: Any) -> None:
    """Display evaluation metrics.
    
    Args:
        accuracy: Model accuracy score
        y_test: True labels
        y_pred: Predicted labels
    """
    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

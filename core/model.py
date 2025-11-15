
from sklearn.tree import DecisionTreeClassifier
import pickle
from pathlib import Path

RANDOM_STATE = 42


def create_model() -> DecisionTreeClassifier:


    # create and return a Decision Tree Classifier
    return DecisionTreeClassifier(random_state=RANDOM_STATE)


def save_model(model: DecisionTreeClassifier, filepath: str) -> None:
    

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_model(filepath: str) -> DecisionTreeClassifier:
    # load the model 
    with open(filepath, 'rb') as f:
        return pickle.load(f)

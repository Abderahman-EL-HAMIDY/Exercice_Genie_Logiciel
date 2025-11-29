import pytest
import os
from sklearn.tree import DecisionTreeClassifier
from core.model import create_model, save_model, load_model

def test_create_model_type():
    """
    Test that the model created is actually a Decision Tree.
    """
    model = create_model()
    assert isinstance(model, DecisionTreeClassifier)

def test_save_and_load_model(tmp_path):
    """
    Test the full cycle of saving and loading a model.
    uses 'tmp_path' fixture to avoid creating real files.
    """
    # 1. Create and Save
    model = create_model()
    # tmp_path is a pathlib object provided by pytest
    save_location = tmp_path / "subdir" / "test_model.pkl"
    
    # Convert path to string as your code expects a string path
    save_model(model, str(save_location))

    # 2. Check file existence
    assert os.path.exists(save_location)

    # 3. Load and Verify
    loaded_model = load_model(str(save_location))
    assert isinstance(loaded_model, DecisionTreeClassifier)

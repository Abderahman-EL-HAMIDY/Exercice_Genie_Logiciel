

from core.dataset import load_data, prepare_data, split_data
from core.model import create_model, save_model
from typing import Tuple
import pandas as pd



def train_pipeline(data_path: str, model_path: str) -> Tuple:
    
    # Data preparation
    df = load_data(data_path)
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Model training
    model = create_model()
    model.fit(X_train, y_train)
    
    # Model persistence
    save_model(model, model_path)
    
    return model, X_test, y_test

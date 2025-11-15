

import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

# Constants
FEATURE_COLUMNS = ['temperature', 'toux', 'fatigue']
TARGET_COLUMN = 'infecte'
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(filepath: str) -> pd.DataFrame:
    
    return pd.read_csv(filepath)


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, 
               test_size: float = TEST_SIZE) -> Tuple:
    
    return train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=RANDOM_STATE
    )

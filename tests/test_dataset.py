import pytest
import pandas as pd
from core.dataset import prepare_data, split_data

def test_prepare_data_structure():
    """
    Test that prepare_data correctly separates features (X) and target (y).
    """
    # Create dummy data
    mock_df = pd.DataFrame({
        'temperature': [37.5, 38.1, 36.6],
        'toux': [0, 1, 0],
        'fatigue': [1, 1, 0],
        'infecte': [0, 1, 0], # Target
        'extra_column': [1, 2, 3] # Should be ignored
    })

    X, y = prepare_data(mock_df)

    # Assertions
    expected_cols = ['temperature', 'toux', 'fatigue']
    assert list(X.columns) == expected_cols
    assert len(y) == 3
    assert 'infecte' not in X.columns

def test_split_data_ratio():
    """
    Test that split_data respects the test_size ratio.
    """
    # Create 10 rows of dummy data
    X = pd.DataFrame({'feature': range(10)})
    y = pd.Series([0, 1] * 5)

    # Split with 20% test size
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

    # Assertions
    assert len(X_train) == 8
    assert len(X_test) == 2
    assert len(y_train) == len(X_train)

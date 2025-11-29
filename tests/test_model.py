import pytest
from core.dataset import DataLoader # Assuming Class name based on file name
from core.model import MLModel      # Assuming Class name based on file name

def test_imports():
    """Simple test to ensure modules are accessible."""
    assert DataLoader is not None
    assert MLModel is not None

def test_basic_math():
    """Sanity check."""
    assert 1 + 1 == 2
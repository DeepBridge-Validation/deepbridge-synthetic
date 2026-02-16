"""
Fixtures comuns para testes de deepbridge-synthetic.
"""

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def sample_data():
    """Dataset de exemplo para testes."""
    np.random.seed(42)
    n_samples = 500

    df = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.exponential(2, n_samples),
        'feature3': np.random.uniform(0, 100, n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples),
        'target': np.random.randint(0, 2, n_samples),
    })

    return df


@pytest.fixture
def numeric_data(sample_data):
    """Apenas features numéricas."""
    return sample_data[['feature1', 'feature2', 'feature3', 'target']]


@pytest.fixture
def mixed_data(sample_data):
    """Features numéricas e categóricas."""
    return sample_data

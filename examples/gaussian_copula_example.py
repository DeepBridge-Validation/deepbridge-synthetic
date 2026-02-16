"""
Example: Synthetic Data Generation with Gaussian Copula

Shows how to generate synthetic data that preserves statistical properties.
"""

import pandas as pd
import numpy as np

from deepbridge_synthetic import Synthesize


# Load or create original data
np.random.seed(42)
n_samples = 1000

original_df = pd.DataFrame({
    'age': np.random.randint(18, 80, n_samples),
    'income': np.random.exponential(50000, n_samples),
    'credit_score': np.random.randint(300, 850, n_samples),
    'category': np.random.choice(['A', 'B', 'C'], n_samples),
})

print(f"Original data shape: {original_df.shape}")
print(f"\nOriginal statistics:\n{original_df.describe()}")

# Generate synthetic data
print("\nGenerating synthetic data...")
synthesizer = Synthesize(
    data=original_df,
    method='gaussian_copula'
)

synthetic_df = synthesizer.generate(n_samples=2000)

print(f"\n✅ Synthetic data generated!")
print(f"Synthetic data shape: {synthetic_df.shape}")
print(f"\nSynthetic statistics:\n{synthetic_df.describe()}")

# Save
synthetic_df.to_csv('synthetic_data.csv', index=False)
print(f"\n💾 Synthetic data saved to: synthetic_data.csv")

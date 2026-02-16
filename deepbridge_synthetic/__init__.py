"""
DeepBridge Synthetic - Privacy-Preserving Synthetic Data Generation

This package provides tools for generating high-quality synthetic data
while preserving statistical properties and privacy.

Note: This is a standalone library and does NOT require deepbridge core.
"""

__version__ = '2.0.0'
__author__ = 'Team DeepBridge'

from deepbridge_synthetic.synthesizer import Synthesize
from deepbridge_synthetic.base_generator import BaseGenerator
from deepbridge_synthetic.standard_generator import StandardGenerator

__all__ = [
    'Synthesize',
    'BaseGenerator',
    'StandardGenerator',
]

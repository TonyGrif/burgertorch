"""This module holds the Neural Networks for Burgers' Equation

Attributes:
    ContinuousInferenceNetwork: Neural Network for solving continuous inference
    ContinuousIdentificationNetwork: Neural Network for solving continuous identification
"""

from .models import ContinuousIdentificationNetwork, ContinuousInferenceNetwork
from .plotting import plot_results
from .preprocess import prepare_continuous_inference

__all__ = [
    "ContinuousInferenceNetwork",
    "ContinuousIdentificationNetwork",
    "prepare_continuous_inference",
    "plot_results",
]

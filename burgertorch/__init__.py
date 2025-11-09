"""This module holds the Neural Networks for Burgers' Equation

Attributes:
    ContinuousInferenceNetwork: Neural Network for solving continuous inference
    ContinuousIdentificationNetwork: Neural Network for solving continuous identification
"""

from .continuous_identification import ContinuousIdentificationNetwork
from .continuous_inference import ContinuousInferenceNetwork

__all__ = ["ContinuousInferenceNetwork", "ContinuousIdentificationNetwork"]

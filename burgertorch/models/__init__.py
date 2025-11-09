"""This module contains models for solving Burgers' Equation

Attributes:
    ContinuousInferenceNetwork: Neural Network for solving continuous inference
    ContinuousIdentificationNetwork: Neural Network for solving continuous identification
"""

from .continuous_inference import ContinuousInferenceNetwork
from .continuous_identification import ContinuousIdentificationNetwork

__all__ = ["ContinuousInferenceNetwork", "ContinuousIdentificationNetwork"]

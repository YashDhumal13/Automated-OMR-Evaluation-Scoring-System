"""
Automated OMR Evaluation and Scoring System - Image Processing Engine

This module provides comprehensive image processing capabilities for optical mark recognition (OMR).
It handles the complete pipeline from raw photographed OMR sheets to final scored results.

Author: Image Processing Team
Created: September 2025
"""

from .pipeline import OMRProcessor
from .modules.perspective_correction import PerspectiveCorrector
from .modules.bubble_detection import BubbleDetector
from .modules.bubble_classification import BubbleClassifier
from .modules.scoring import ScoreCalculator

__version__ = "1.0.0"
__all__ = [
    "OMRProcessor",
    "PerspectiveCorrector", 
    "BubbleDetector",
    "BubbleClassifier",
    "ScoreCalculator"
]
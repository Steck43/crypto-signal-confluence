"""
Technical Analysis Module

EXPLORATORY: pattern recognition, indicators, and multi-timeframe tools.
Not on the validated ablation path.
"""

from .pattern_recognition import PatternRecognizer
from .indicators import CustomIndicators
from .multitimeframe import MultiTimeframeAnalyzer

__all__ = [
    "PatternRecognizer",
    "CustomIndicators", 
    "MultiTimeframeAnalyzer"
]
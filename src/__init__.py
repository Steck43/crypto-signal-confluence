"""
AI-Driven Cryptocurrency Trading System

A comprehensive trading system that combines advanced volume anomaly detection,
sentiment analysis, and machine learning for cryptocurrency trading.

Mathematical foundations:
- Isolation Forest: score = 2^(-E(h(x))/c(n))
- Mahalanobis Distance: D² = (x-μ)ᵀ Σ⁻¹ (x-μ)
- Attention Mechanism: Attention(Q,K,V) = softmax(QKᵀ/√dk)V
- Kelly Criterion: f* = (μ-r)/σ² - λVar(μ)/2σ²

Author: AI Trading Research Team
License: MIT
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "AI Trading Research Team"

# Core modules
from . import config
from . import data
from . import technical_analysis
from . import sentiment_analysis
from . import trading_strategies
from . import machine_learning
from . import utils

__all__ = [
    "config",
    "data",
    "technical_analysis", 
    "sentiment_analysis",
    "trading_strategies",
    "machine_learning",
    "utils"
]
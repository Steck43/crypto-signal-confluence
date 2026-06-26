"""
crypto-signal-confluence research package.

Validated ablation path: run_sentiment_ablation.py with PYTHONPATH=src.
"""

__version__ = "1.0.0"

from . import technical_analysis
from . import sentiment_analysis
from . import trading_strategies
from . import machine_learning
from . import utils

__all__ = [
    "technical_analysis",
    "sentiment_analysis",
    "trading_strategies",
    "machine_learning",
    "utils",
]

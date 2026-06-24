"""
Trading Strategies Module

This module contains various trading strategies for the AI crypto trading system.
"""

# Try to import advanced strategies (may require additional dependencies)
try:
    from .ensemble_strategy import EnsembleStrategy
    _ENSEMBLE_AVAILABLE = True
except ImportError as e:
    print(f"Note: EnsembleStrategy not available (freqtrade not installed): {e}")
    EnsembleStrategy = None
    _ENSEMBLE_AVAILABLE = False

__all__ = []

# Add EnsembleStrategy to __all__ only if available
if _ENSEMBLE_AVAILABLE:
    __all__.append("EnsembleStrategy")
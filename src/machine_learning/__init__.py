"""
Machine Learning Module for AI Crypto Trading System
Integrates XGBoost and GRU with existing institutional components
"""

from .xgboost_predictor import XGBoostSignalPredictor

try:
    from .ml_ensemble import MLEnsembleManager
except ImportError:
    MLEnsembleManager = None

try:
    from .meta_learning import MAMLTrader
    from .gru_forecaster import GRUTradingPredictor, GRUPriceForecaster
except ImportError:
    MAMLTrader = None
    GRUTradingPredictor = None
    GRUPriceForecaster = None

__all__ = [
    'MAMLTrader',
    'XGBoostSignalPredictor',
    'GRUTradingPredictor',
    'GRUPriceForecaster',
    'MLEnsembleManager',
]

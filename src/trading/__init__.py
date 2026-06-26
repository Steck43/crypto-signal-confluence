"""
Trading module for AI Crypto Trading System
"""

from .signal_generator import SimplifiedInstitutionalSignalGenerator

try:
    from .risk_manager import RiskManager
    from .order_executor import OrderExecutor
    from .trading_bot import TradingBot
except ImportError:
    RiskManager = None
    OrderExecutor = None
    TradingBot = None

__all__ = [
    'SimplifiedInstitutionalSignalGenerator',
    'RiskManager',
    'OrderExecutor',
    'TradingBot',
]

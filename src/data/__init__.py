"""
Data Management Module

Handles all data collection, preprocessing, and storage operations:
- Market data collection from multiple exchanges
- Social media data aggregation
- Data preprocessing and feature engineering
- Real-time data streaming
"""

from .market_data import MarketDataCollector
from .social_data import SocialDataCollector
from .preprocessor import DataPreprocessor
from .streaming import RealTimeDataStreamer

__all__ = [
    "MarketDataCollector",
    "SocialDataCollector", 
    "DataPreprocessor",
    "RealTimeDataStreamer"
]
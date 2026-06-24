"""
Data Preprocessing Module

Handles data preprocessing, feature engineering, and data cleaning
for the institutional AI crypto trading system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression


class DataPreprocessor:
    """Data preprocessing and feature engineering."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize data preprocessor."""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Scalers
        self.price_scaler = StandardScaler()
        self.volume_scaler = StandardScaler()
        self.feature_scaler = StandardScaler()
        
        # Feature selection
        self.feature_selector = None
        self.selected_features = []
        
        self.logger.info("Data preprocessor initialized")
    
    def preprocess_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess market data."""
        if df.empty:
            return df
        
        # Clean data
        df = self._clean_data(df)
        
        # Engineer features
        df = self._engineer_features(df)
        
        # Scale features
        df = self._scale_features(df)
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data."""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')
        
        # Remove outliers (simple approach)
        for col in ['price', 'volume', 'high', 'low', 'open', 'close']:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer technical features."""
        if df.empty:
            return df
        
        # Price-based features
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['volume_price_ratio'] = df['volume'] / df['close']
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ma_ratio_{window}'] = df['close'] / df[f'ma_{window}']
        
        # Volatility features
        df['volatility'] = df['price_change'].rolling(window=20).std()
        df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(window=50).mean()
        
        # Momentum features
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
            df[f'rsi_{period}'] = self._calculate_rsi(df['close'], period)
        
        # Remove NaN values
        df = df.dropna()
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features."""
        # Identify numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove timestamp and identifier columns
        exclude_cols = ['timestamp', 'symbol', 'source', 'exchange']
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        if numerical_cols:
            # Scale features
            scaled_features = self.feature_scaler.fit_transform(df[numerical_cols])
            df[numerical_cols] = scaled_features
        
        return df
    
    def select_features(self, df: pd.DataFrame, target_col: str, n_features: int = 20) -> pd.DataFrame:
        """Select most important features."""
        if df.empty or target_col not in df.columns:
            return df
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'symbol', 'source', 'exchange', target_col]]
        X = df[feature_cols]
        y = df[target_col]
        
        # Remove any remaining NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            return df
        
        # Select features
        self.feature_selector = SelectKBest(score_func=f_regression, k=min(n_features, len(feature_cols)))
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_indices = self.feature_selector.get_support(indices=True)
        self.selected_features = [feature_cols[i] for i in selected_indices]
        
        # Create new dataframe with selected features
        result_df = df[['timestamp', 'symbol', 'source', 'exchange'] + self.selected_features + [target_col]].copy()
        
        return result_df
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.feature_selector is None:
            return {}
        
        feature_scores = self.feature_selector.scores_
        return dict(zip(self.selected_features, feature_scores))


# Factory function
def create_data_preprocessor(config: Optional[Dict[str, Any]] = None) -> DataPreprocessor:
    """Create a data preprocessor."""
    return DataPreprocessor(config) 
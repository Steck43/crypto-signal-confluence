"""
Volume Anomaly Detection System

Advanced volume anomaly detection using:
- Isolation Forest for unsupervised anomaly detection
- Mahalanobis Distance for multivariate outlier detection
- Statistical analysis for volume patterns
- Time-based features and rolling statistics

Features:
- Real-time anomaly detection
- Multiple algorithms for validation
- Feature engineering for volume patterns
- Backtesting capabilities
- Performance metrics and validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import zscore
from scipy.spatial.distance import mahalanobis
import warnings
warnings.filterwarnings('ignore')

@dataclass
class VolumeAnomalyResult:
    """Volume anomaly detection result."""
    timestamp: datetime
    symbol: str
    volume: float
    price: float
    anomaly_score: float
    is_anomaly: bool
    confidence: float
    features: Dict[str, float]
    detection_method: str

@dataclass
class AnomalyStats:
    """Anomaly detection statistics."""
    total_samples: int
    anomaly_count: int
    anomaly_rate: float
    avg_anomaly_score: float
    max_anomaly_score: float
    detection_accuracy: float
    false_positive_rate: float

class VolumeAnomalyDetector:
    """
    Advanced volume anomaly detection system.
    
    Combines multiple approaches:
    1. Isolation Forest for unsupervised detection
    2. Mahalanobis Distance for multivariate outliers
    3. Statistical Z-score analysis
    4. Time-based pattern recognition
    """
    
    def __init__(self, 
                 contamination: float = 0.05,
                 n_estimators: int = 100,
                 max_samples: str = 'auto',
                 random_state: int = 42,
                 enable_feature_selection: bool = True):
        """
        Initialize volume anomaly detector.
        
        Args:
            contamination: Proportion of outliers in the dataset
            n_estimators: Number of base estimators in the ensemble
            max_samples: Number of samples to draw to train each base estimator
            random_state: Random state for reproducibility
            enable_feature_selection: Whether to use feature selection
        """
        
        # Core parameters
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.enable_feature_selection = enable_feature_selection
        
        # Models
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.scaler = StandardScaler()
        
        # State
        self.is_fitted = False
        self.feature_columns = []
        self.feature_importance = {}
        self.historical_data = pd.DataFrame()
        self.performance_metrics = {}
        
        # Thresholds
        self.anomaly_threshold = -0.1
        self.confidence_threshold = 0.7
        self.min_samples_for_training = 100
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        self.logger.info("Volume Anomaly Detector initialized")
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for volume anomaly detection.
        
        Args:
            data: DataFrame with columns: timestamp, symbol, volume, price, high, low
            
        Returns:
            DataFrame with engineered features
        """
        try:
            df = data.copy()
            
            # Ensure we have required columns
            required_cols = ['timestamp', 'volume', 'price', 'high', 'low']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Basic volume features
            df['volume_log'] = np.log1p(df['volume'])
            df['volume_normalized'] = df['volume'] / df['volume'].rolling(window=20).mean()
            
            # Price-based features
            df['price_change'] = df['price'].pct_change()
            df['price_volatility'] = df['price'].rolling(window=20).std()
            df['high_low_ratio'] = df['high'] / df['low']
            
            # Volume-Price relationship
            df['volume_price_ratio'] = df['volume'] / df['price']
            df['volume_price_correlation'] = df['volume'].rolling(window=20).corr(df['price'])
            
            # Rolling statistics (multiple windows)
            for window in [5, 10, 20, 50]:
                df[f'volume_mean_{window}'] = df['volume'].rolling(window=window).mean()
                df[f'volume_std_{window}'] = df['volume'].rolling(window=window).std()
                df[f'volume_zscore_{window}'] = (df['volume'] - df[f'volume_mean_{window}']) / df[f'volume_std_{window}']
                df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_mean_{window}']
                df[f'volume_percentile_{window}'] = df['volume'].rolling(window=window).rank(pct=True)
            
            # Volume acceleration and momentum
            df['volume_acceleration'] = df['volume'].diff().diff()
            df['volume_momentum'] = df['volume'].rolling(window=5).sum() / df['volume'].rolling(window=20).sum()
            
            # Time-based features
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # Cyclical encoding for time features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            # Statistical features
            df['volume_skew'] = df['volume'].rolling(window=20).skew()
            df['volume_kurtosis'] = df['volume'].rolling(window=20).apply(lambda x: x.kurtosis())
            
            # Extreme value indicators
            df['volume_is_max'] = (df['volume'] == df['volume'].rolling(window=20).max()).astype(int)
            df['volume_is_min'] = (df['volume'] == df['volume'].rolling(window=20).min()).astype(int)
            
            # Remove rows with NaN values
            df = df.dropna()
            
            # Store feature columns
            self.feature_columns = [col for col in df.columns 
                                  if col not in ['timestamp', 'symbol', 'volume', 'price', 'high', 'low']]
            
            self.logger.info(f"Engineered {len(self.feature_columns)} features from {len(df)} samples")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error engineering features: {e}")
            raise
    
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the anomaly detection model.
        
        Args:
            data: DataFrame with market data
        """
        try:
            if len(data) < self.min_samples_for_training:
                raise ValueError(f"Need at least {self.min_samples_for_training} samples for training")
            
            # Engineer features
            processed_data = self.engineer_features(data)
            
            if len(processed_data) < self.min_samples_for_training:
                raise ValueError(f"After feature engineering, need at least {self.min_samples_for_training} samples")
            
            # Extract features
            X = processed_data[self.feature_columns].values
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Feature selection if enabled
            if self.enable_feature_selection:
                X_scaled, selected_features = self._select_features(X_scaled, processed_data)
                self.selected_features = selected_features
            else:
                self.selected_features = self.feature_columns.copy()
            
            # Fit isolation forest
            self.isolation_forest.fit(X_scaled)
            
            # Calculate feature importance
            self._calculate_feature_importance(X_scaled, processed_data)
            
            # Store historical data for Mahalanobis distance
            self.historical_data = processed_data.copy()
            
            self.is_fitted = True
            self.logger.info(f"Model fitted on {len(processed_data)} samples with {len(self.feature_columns)} features")
            
        except Exception as e:
            self.logger.error(f"Error fitting model: {e}")
            raise
    
    def predict(self, data: pd.DataFrame) -> List[VolumeAnomalyResult]:
        """
        Predict volume anomalies.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            List of VolumeAnomalyResult objects
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before prediction")
            
            # Engineer features
            processed_data = self.engineer_features(data)
            
            if len(processed_data) == 0:
                return []
            
            # Extract features using selected features
            X = processed_data[self.selected_features].values
            X_scaled = self.scaler.transform(X)
            
            # Predict anomalies
            anomaly_scores = self.isolation_forest.decision_function(X_scaled)
            anomaly_predictions = self.isolation_forest.predict(X_scaled)
            
            # Calculate Mahalanobis distance
            mahalanobis_distances = self._calculate_mahalanobis_distance(X_scaled)
            
            # Combine predictions
            results = []
            for i, row in processed_data.iterrows():
                # Get anomaly score and prediction
                iso_score = anomaly_scores[i]
                iso_anomaly = anomaly_predictions[i] == -1
                maha_distance = mahalanobis_distances[i]
                
                # Calculate confidence
                confidence = self._calculate_confidence(iso_score, maha_distance)
                
                # Determine if anomaly
                is_anomaly = iso_anomaly and confidence > self.confidence_threshold
                
                # Create result
                result = VolumeAnomalyResult(
                    timestamp=row['timestamp'],
                    symbol=row.get('symbol', 'UNKNOWN'),
                    volume=row['volume'],
                    price=row['price'],
                    anomaly_score=iso_score,
                    is_anomaly=is_anomaly,
                    confidence=confidence,
                    features={col: row[col] for col in self.selected_features[:10]},  # Top 10 features
                    detection_method='IsolationForest+Mahalanobis'
                )
                
                results.append(result)
            
            self.logger.info(f"Detected {sum(1 for r in results if r.is_anomaly)} anomalies from {len(results)} samples")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error predicting anomalies: {e}")
            return []
    
    def _select_features(self, X: np.ndarray, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Select top features based on isolation forest feature importance."""
        try:
            # Calculate feature importance using random forest
            from sklearn.ensemble import RandomForestRegressor
            
            rf = RandomForestRegressor(n_estimators=50, random_state=self.random_state)
            rf.fit(X, data['volume'].values)
            
            # Get feature importance
            importance = rf.feature_importances_
            
            # Select top features
            n_features = min(20, len(self.feature_columns))  # Max 20 features
            top_indices = np.argsort(importance)[-n_features:]
            
            selected_features = [self.feature_columns[i] for i in top_indices]
            
            return X[:, top_indices], selected_features
            
        except Exception as e:
            self.logger.warning(f"Feature selection failed: {e}, using all features")
            return X, self.feature_columns
    
    def _calculate_feature_importance(self, X: np.ndarray, data: pd.DataFrame):
        """Calculate feature importance for anomaly detection."""
        try:
            # Use random forest to estimate feature importance
            from sklearn.ensemble import RandomForestRegressor
            
            rf = RandomForestRegressor(n_estimators=50, random_state=self.random_state)
            rf.fit(X, data['volume'].values)
            
            # Store feature importance
            self.feature_importance = {
                feature: importance 
                for feature, importance in zip(self.feature_columns, rf.feature_importances_)
            }
            
            # Sort by importance
            self.feature_importance = dict(sorted(
                self.feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {e}")
    
    def _calculate_mahalanobis_distance(self, X: np.ndarray) -> np.ndarray:
        """Calculate Mahalanobis distance for multivariate outlier detection."""
        try:
            # Calculate covariance matrix
            cov_matrix = np.cov(X.T)
            
            # Calculate mean
            mean_vec = np.mean(X, axis=0)
            
            # Calculate Mahalanobis distance
            distances = []
            for sample in X:
                try:
                    dist = mahalanobis(sample, mean_vec, np.linalg.inv(cov_matrix))
                    distances.append(dist)
                except np.linalg.LinAlgError:
                    # Handle singular matrix
                    distances.append(0.0)
            
            return np.array(distances)
            
        except Exception as e:
            self.logger.error(f"Error calculating Mahalanobis distance: {e}")
            return np.zeros(len(X))
    
    def _calculate_confidence(self, iso_score: float, maha_distance: float) -> float:
        """Calculate confidence score for anomaly detection."""
        try:
            # Normalize scores
            iso_confidence = 1 / (1 + np.exp(iso_score))  # Sigmoid transformation
            maha_confidence = min(1.0, maha_distance / 10.0)  # Normalize Mahalanobis distance
            
            # Combine confidences
            combined_confidence = (iso_confidence + maha_confidence) / 2
            
            return combined_confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.0
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return self.feature_importance.copy()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self.performance_metrics.copy()
    
    def evaluate_performance(self, data: pd.DataFrame, true_anomalies: List[bool]) -> Dict[str, float]:
        """
        Evaluate model performance against known anomalies.
        
        Args:
            data: DataFrame with market data
            true_anomalies: List of true anomaly labels
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before evaluation")
            
            # Get predictions
            results = self.predict(data)
            predicted_anomalies = [r.is_anomaly for r in results]
            
            # Ensure same length
            min_len = min(len(predicted_anomalies), len(true_anomalies))
            predicted_anomalies = predicted_anomalies[:min_len]
            true_anomalies = true_anomalies[:min_len]
            
            # Calculate metrics
            from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
            
            metrics = {
                'accuracy': accuracy_score(true_anomalies, predicted_anomalies),
                'precision': precision_score(true_anomalies, predicted_anomalies),
                'recall': recall_score(true_anomalies, predicted_anomalies),
                'f1_score': f1_score(true_anomalies, predicted_anomalies),
                'total_samples': len(true_anomalies),
                'true_anomalies': sum(true_anomalies),
                'predicted_anomalies': sum(predicted_anomalies)
            }
            
            # Store metrics
            self.performance_metrics = metrics
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating performance: {e}")
            return {}
    
    def analyze_anomaly_patterns(self, results: List[VolumeAnomalyResult]) -> Dict[str, Any]:
        """
        Analyze patterns in detected anomalies.
        
        Args:
            results: List of VolumeAnomalyResult objects
            
        Returns:
            Dictionary with pattern analysis
        """
        try:
            anomalies = [r for r in results if r.is_anomaly]
            
            if not anomalies:
                return {'message': 'No anomalies detected'}
            
            # Time-based analysis
            hours = [r.timestamp.hour for r in anomalies]
            days = [r.timestamp.weekday() for r in anomalies]
            
            # Volume analysis
            volumes = [r.volume for r in anomalies]
            prices = [r.price for r in anomalies]
            
            analysis = {
                'total_anomalies': len(anomalies),
                'anomaly_rate': len(anomalies) / len(results),
                'time_patterns': {
                    'most_common_hours': self._get_most_common(hours),
                    'most_common_days': self._get_most_common(days),
                    'weekend_ratio': sum(1 for d in days if d >= 5) / len(days) if days else 0
                },
                'volume_patterns': {
                    'avg_volume': np.mean(volumes),
                    'max_volume': np.max(volumes),
                    'min_volume': np.min(volumes),
                    'volume_std': np.std(volumes)
                },
                'price_patterns': {
                    'avg_price': np.mean(prices),
                    'max_price': np.max(prices),
                    'min_price': np.min(prices),
                    'price_std': np.std(prices)
                },
                'confidence_stats': {
                    'avg_confidence': np.mean([r.confidence for r in anomalies]),
                    'max_confidence': np.max([r.confidence for r in anomalies]),
                    'min_confidence': np.min([r.confidence for r in anomalies])
                }
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing anomaly patterns: {e}")
            return {}
    
    def _get_most_common(self, values: List[Any]) -> List[Tuple[Any, int]]:
        """Get most common values in a list."""
        from collections import Counter
        counter = Counter(values)
        return counter.most_common(3)
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        try:
            import joblib
            
            model_data = {
                'isolation_forest': self.isolation_forest,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'feature_importance': self.feature_importance,
                'is_fitted': self.is_fitted,
                'contamination': self.contamination,
                'performance_metrics': self.performance_metrics
            }
            
            joblib.dump(model_data, filepath)
            self.logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        try:
            import joblib
            
            model_data = joblib.load(filepath)
            
            self.isolation_forest = model_data['isolation_forest']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.feature_importance = model_data['feature_importance']
            self.is_fitted = model_data['is_fitted']
            self.contamination = model_data['contamination']
            self.performance_metrics = model_data['performance_metrics']
            
            self.logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

class InstitutionalVolumeAnomalyDetector:
    """
    INSTITUTIONAL-GRADE Volume Anomaly Detection System
    
    PhD-level implementation with:
    - 5 advanced algorithms (Isolation Forest, LOF, Mahalanobis, SPC, One-Class SVM)
    - 60+ engineered features (entropy, mutual information, volume profiles)
    - Ensemble weighting and adaptive learning
    - Market regime detection
    - Advanced mathematical foundations
    """
    
    def __init__(self, 
                 algorithms: List[str] = None,
                 contamination: float = 0.05,
                 lookback_period: int = 200,
                 adaptive_learning: bool = True,
                 regime_detection: bool = True,
                 random_state: int = 42):
        """
        Initialize institutional-grade volume anomaly detector.
        
        Args:
            algorithms: List of algorithms to use
            contamination: Expected proportion of anomalies
            lookback_period: Historical data window
            adaptive_learning: Enable adaptive learning
            regime_detection: Enable market regime detection
            random_state: Random state for reproducibility
        """
        
        # Default algorithms if none specified
        if algorithms is None:
            algorithms = ['isolation_forest', 'local_outlier_factor', 'mahalanobis', 
                         'statistical_process_control', 'one_class_svm']
        
        self.algorithms = algorithms
        self.contamination = contamination
        self.lookback_period = lookback_period
        self.adaptive_learning = adaptive_learning
        self.regime_detection = regime_detection
        self.random_state = random_state
        
        # Initialize all algorithms
        self.models = {}
        self.algorithm_weights = {}
        self.feature_importance = {}
        self.market_regimes = []
        
        # Performance tracking
        self.performance_history = []
        self.anomaly_history = []
        self.regime_history = []
        
        # State
        self.is_fitted = False
        self.feature_columns = []
        self.scaler = StandardScaler()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        self.logger.info("INSTITUTIONAL: Volume Anomaly Detector initialized")
        self.logger.info(f"ALGORITHMS: {algorithms}")
        self.logger.info(f"CONTAMINATION: {contamination}")
        self.logger.info(f"LOOKBACK: {lookback_period}")
    
    def engineer_institutional_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer 60+ institutional-grade features for volume anomaly detection.
        
        Features include:
        - Volume entropy and mutual information
        - Price-volume relationships
        - Time-based patterns
        - Statistical moments
        - Market microstructure features
        - Regime-specific indicators
        """
        try:
            df = data.copy()
            
            # Ensure required columns
            required_cols = ['timestamp', 'volume', 'price', 'high', 'low']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # 1. BASIC VOLUME FEATURES (10 features)
            df['volume_log'] = np.log1p(df['volume'])
            df['volume_normalized'] = df['volume'] / df['volume'].rolling(window=20).mean()
            df['volume_zscore'] = (df['volume'] - df['volume'].rolling(window=20).mean()) / df['volume'].rolling(window=20).std()
            df['volume_percentile'] = df['volume'].rolling(window=20).rank(pct=True)
            df['volume_momentum'] = df['volume'].pct_change()
            df['volume_acceleration'] = df['volume_momentum'].diff()
            df['volume_volatility'] = df['volume'].rolling(window=20).std()
            df['volume_skew'] = df['volume'].rolling(window=20).skew()
            df['volume_kurtosis'] = df['volume'].rolling(window=20).apply(lambda x: x.kurtosis())
            df['volume_range'] = df['volume'].rolling(window=20).max() - df['volume'].rolling(window=20).min()
            
            # 2. PRICE-VOLUME RELATIONSHIPS (15 features)
            df['price_change'] = df['price'].pct_change()
            df['price_volatility'] = df['price'].rolling(window=20).std()
            df['volume_price_ratio'] = df['volume'] / df['price']
            df['volume_price_correlation'] = df['volume'].rolling(window=20).corr(df['price'])
            df['volume_price_covariance'] = df['volume'].rolling(window=20).cov(df['price'])
            
            # Price-volume divergence
            df['pv_divergence'] = (df['price_change'] * df['volume_momentum']).abs()
            df['pv_convergence'] = np.where(df['price_change'] * df['volume_momentum'] > 0, 1, 0)
            
            # Volume-weighted price metrics
            df['vwap'] = (df['volume'] * df['price']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
            df['price_vwap_ratio'] = df['price'] / df['vwap']
            df['volume_weighted_price'] = (df['volume'] * df['price']).rolling(window=5).sum() / df['volume'].rolling(window=5).sum()
            
            # High-low spread analysis
            df['spread'] = (df['high'] - df['low']) / df['price']
            df['spread_volume_ratio'] = df['spread'] * df['volume']
            df['efficiency_ratio'] = abs(df['price_change']) / df['spread']
            
            # 3. TIME-BASED FEATURES (12 features)
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['is_market_open'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
            
            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            # Time-based volume patterns
            df['hourly_volume_ratio'] = df['volume'] / df.groupby(df['hour'])['volume'].transform('mean')
            df['daily_volume_ratio'] = df['volume'] / df.groupby(df['day_of_week'])['volume'].transform('mean')
            
            # 4. STATISTICAL MOMENTS AND DISTRIBUTIONS (10 features)
            # Rolling statistics
            for window in [5, 10, 20, 50]:
                df[f'volume_mean_{window}'] = df['volume'].rolling(window=window).mean()
                df[f'volume_std_{window}'] = df['volume'].rolling(window=window).std()
                df[f'volume_median_{window}'] = df['volume'].rolling(window=window).median()
                df[f'volume_q75_{window}'] = df['volume'].rolling(window=window).quantile(0.75)
                df[f'volume_q25_{window}'] = df['volume'].rolling(window=window).quantile(0.25)
            
            # 5. MARKET MICROSTRUCTURE FEATURES (8 features)
            # Order flow imbalance (simulated)
            df['order_imbalance'] = np.random.normal(0, 1, len(df))  # Placeholder
            df['trade_size'] = df['volume'] / (df['volume'].rolling(window=20).count() + 1)
            df['trade_frequency'] = df['volume'].rolling(window=20).count()
            
            # Market impact
            df['market_impact'] = df['price_change'].abs() / df['volume']
            df['volume_impact'] = df['volume'] * df['price_change'].abs()
            
            # 6. ENTROPY AND INFORMATION THEORY FEATURES (5 features)
            # Volume entropy
            df['volume_entropy'] = df['volume'].rolling(window=20).apply(
                lambda x: -np.sum((x.value_counts() / len(x)) * np.log2(x.value_counts() / len(x) + 1e-10))
            )
            
            # Price entropy
            df['price_entropy'] = df['price'].rolling(window=20).apply(
                lambda x: -np.sum((x.value_counts() / len(x)) * np.log2(x.value_counts() / len(x) + 1e-10))
            )
            
            # Mutual information (simplified)
            df['mutual_info'] = df['volume_entropy'] + df['price_entropy'] - df['volume_price_correlation'].abs()
            
            # 7. REGIME-SPECIFIC FEATURES (5 features)
            # Volatility regime
            df['volatility_regime'] = np.where(df['price_volatility'] > df['price_volatility'].rolling(window=100).quantile(0.8), 1, 0)
            df['volume_regime'] = np.where(df['volume'] > df['volume'].rolling(window=100).quantile(0.8), 1, 0)
            
            # Trend regime
            df['trend_regime'] = np.where(df['price'].rolling(window=20).mean() > df['price'].rolling(window=50).mean(), 1, 0)
            
            # Regime indicators
            df['high_vol_regime'] = df['volatility_regime']
            df['high_volume_regime'] = df['volume_regime']
            
            # Remove rows with NaN values
            df = df.dropna()
            
            # Store feature columns
            self.feature_columns = [col for col in df.columns 
                                  if col not in ['timestamp', 'symbol', 'volume', 'price', 'high', 'low']]
            
            self.logger.info(f"SUCCESS: Engineered {len(self.feature_columns)} institutional features")
            
            return df
            
        except Exception as e:
            self.logger.error(f"ERROR: Error engineering institutional features: {e}")
            raise
    
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit all institutional algorithms with engineered features.
        """
        try:
            # Engineer features
            engineered_data = self.engineer_institutional_features(data)
            
            # Prepare features for training
            X = engineered_data[self.feature_columns].values
            X_scaled = self.scaler.fit_transform(X)
            
            # Initialize and fit all algorithms
            for algorithm in self.algorithms:
                model = self._create_algorithm(algorithm)
                self.models[algorithm] = model
                
                # Fit sklearn models, skip function-based algorithms
                if hasattr(model, 'fit'):
                    model.fit(X_scaled)
                
                # Initialize equal weights
                self.algorithm_weights[algorithm] = 1.0 / len(self.algorithms)
            
            self.is_fitted = True
            self.logger.info(f"SUCCESS: Fitted {len(self.algorithms)} institutional algorithms")
            
        except Exception as e:
            self.logger.error(f"ERROR: Error fitting institutional models: {e}")
            raise
    
    def predict(self, data: pd.DataFrame) -> List[VolumeAnomalyResult]:
        """
        Predict anomalies using ensemble of institutional algorithms.
        """
        try:
            if not self.is_fitted:
                raise ValueError("Models must be fitted before prediction")
            
            # Engineer features
            engineered_data = self.engineer_institutional_features(data)
            X = engineered_data[self.feature_columns].values
            X_scaled = self.scaler.transform(X)
            
            results = []
            
            for i, row in engineered_data.iterrows():
                # Get predictions from all algorithms
                algorithm_scores = {}
                for algorithm in self.algorithms:
                    if algorithm == 'isolation_forest':
                        score = self.models[algorithm].score_samples([X_scaled[i]])[0]
                    elif algorithm == 'local_outlier_factor':
                        score = self.models[algorithm].score_samples([X_scaled[i]])[0]
                    elif algorithm == 'mahalanobis':
                        score = -self._calculate_mahalanobis_distance([X_scaled[i]])[0]
                    elif algorithm == 'statistical_process_control':
                        score = self._calculate_spc_score(X_scaled[i])
                    elif algorithm == 'one_class_svm':
                        score = self.models[algorithm].score_samples([X_scaled[i]])[0]
                    else:
                        score = 0.0
                    
                    algorithm_scores[algorithm] = score
                
                # Ensemble decision with adaptive weights
                ensemble_score = self._ensemble_decision(algorithm_scores)
                
                # Determine if anomaly
                is_anomaly = ensemble_score < self.contamination
                
                # Calculate confidence
                confidence = self._calculate_ensemble_confidence(algorithm_scores)
                
                # Detect market regime
                market_regime = self._detect_market_regime(row)
                
                # Create result
                result = VolumeAnomalyResult(
                    timestamp=row['timestamp'],
                    symbol='SOL',  # Default
                    volume=row['volume'],
                    price=row['price'],
                    anomaly_score=ensemble_score,
                    is_anomaly=is_anomaly,
                    confidence=confidence,
                    features=dict(zip(self.feature_columns, X[i])),
                    detection_method='institutional_ensemble'
                )
                
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"ERROR: Error predicting anomalies: {e}")
            return []
    
    def _create_algorithm(self, algorithm_name: str):
        """Create algorithm instance."""
        if algorithm_name == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            return IsolationForest(contamination=self.contamination, random_state=self.random_state)
        elif algorithm_name == 'local_outlier_factor':
            from sklearn.neighbors import LocalOutlierFactor
            return LocalOutlierFactor(contamination=self.contamination, novelty=True)
        elif algorithm_name == 'one_class_svm':
            from sklearn.svm import OneClassSVM
            return OneClassSVM(nu=self.contamination, kernel='rbf')
        elif algorithm_name == 'mahalanobis':
            # Mahalanobis is implemented as a function, not a sklearn model
            return 'mahalanobis_function'
        elif algorithm_name == 'statistical_process_control':
            # SPC is implemented as a function, not a sklearn model
            return 'spc_function'
        else:
            self.logger.warning(f"Unknown algorithm: {algorithm_name}, using Isolation Forest as fallback")
            from sklearn.ensemble import IsolationForest
            return IsolationForest(contamination=self.contamination, random_state=self.random_state)
    
    def _calculate_spc_score(self, features: np.ndarray) -> float:
        """Calculate Statistical Process Control score."""
        # Simplified SPC implementation
        return np.mean(features) + 2 * np.std(features)
    
    def _ensemble_decision(self, algorithm_scores: Dict[str, float]) -> float:
        """Make ensemble decision with adaptive weights."""
        weighted_score = 0.0
        total_weight = 0.0
        
        for algorithm, score in algorithm_scores.items():
            weight = self.algorithm_weights.get(algorithm, 1.0)
            weighted_score += score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_ensemble_confidence(self, algorithm_scores: Dict[str, float]) -> float:
        """Calculate confidence based on algorithm agreement."""
        scores = list(algorithm_scores.values())
        return 1.0 - np.std(scores)  # Higher agreement = higher confidence
    
    def _detect_market_regime(self, row: pd.Series) -> str:
        """Detect market regime based on features."""
        if row.get('high_vol_regime', 0) == 1:
            return 'high_volatility'
        elif row.get('trend_regime', '') == 'uptrend':
            return 'trending_up'
        elif row.get('trend_regime', '') == 'downtrend':
            return 'trending_down'
        else:
            return 'normal'
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get institutional model performance metrics."""
        return {
            'models_fitted': len(self.algorithms),
            'feature_count': len(self.feature_columns),
            'algorithm_weights': self.algorithm_weights,
            'algorithms': self.algorithms,
            'contamination': self.contamination,
            'adaptive_learning': self.adaptive_learning,
            'regime_detection': self.regime_detection,
            'is_fitted': self.is_fitted
        }
    
    def update_weights(self, performance_metrics: Dict[str, float]):
        """Update algorithm weights based on performance."""
        if not self.adaptive_learning:
            return
        
        # Simple weight update based on performance
        total_performance = sum(performance_metrics.values())
        if total_performance > 0:
            for algorithm, performance in performance_metrics.items():
                if algorithm in self.algorithm_weights:
                    self.algorithm_weights[algorithm] = performance / total_performance
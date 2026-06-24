"""
Advanced Volume Anomaly Detection System

Mathematical foundations:
- Isolation Forest: score = 2^(-E(h(x))/c(n))
- Mahalanobis Distance: D² = (x-μ)ᵀ Σ⁻¹ (x-μ)
- Local Outlier Factor (LOF): density-based anomaly detection
- Information-theoretic feature selection

References:
- Liu et al. (2008): "Isolation Forest"
- Mahalanobis (1936): "On the generalized distance in statistics"
- Breunig et al. (2000): "LOF: Identifying density-based local outliers"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance
from sklearn.feature_selection import mutual_info_regression
from scipy import stats
from scipy.spatial.distance import mahalanobis
import logging
from numba import jit
import warnings

warnings.filterwarnings('ignore')

@dataclass
class AnomalyResult:
    """Anomaly detection result container."""
    timestamp: pd.Timestamp
    anomaly_score: float
    is_anomaly: bool
    method: str
    confidence: float
    feature_contributions: Dict[str, float]

class VolumeAnomalyDetector:
    """
    Advanced Volume Anomaly Detection using multiple mathematical methods.
    
    Combines Isolation Forest, Mahalanobis Distance, and Local Outlier Factor
    for robust anomaly detection in cryptocurrency volume patterns.
    
    Mathematical Foundations:
    1. Isolation Forest Score: score = 2^(-E(h(x))/c(n))
       where E(h(x)) is expected path length, c(n) is average path length
    
    2. Mahalanobis Distance: D² = (x-μ)ᵀ Σ⁻¹ (x-μ)
       where x is observation, μ is mean, Σ is covariance matrix
    
    3. LOF Score: measures local density deviation
       LOF(x) = Σ(lrd(y)/lrd(x)) / |N_k(x)|
    """
    
    def __init__(self, 
                 contamination: float = 0.1,
                 n_estimators: int = 100,
                 random_state: int = 42,
                 n_neighbors: int = 20,
                 enable_feature_selection: bool = True):
        """
        Initialize Volume Anomaly Detector.
        
        Args:
            contamination: Expected proportion of anomalies
            n_estimators: Number of isolation trees
            random_state: Random seed for reproducibility
            n_neighbors: Number of neighbors for LOF
            enable_feature_selection: Use information-theoretic feature selection
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.enable_feature_selection = enable_feature_selection
        
        # Initialize models
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True,
            n_jobs=-1
        )
        
        self.scaler = StandardScaler()
        self.covariance = EmpiricalCovariance()
        
        # Feature importance tracking
        self.feature_importance_ = {}
        self.selected_features_ = []
        
        # Model state
        self.is_fitted = False
        self.feature_names_ = []
        
        # Performance tracking
        self.detection_history = []
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def engineer_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer comprehensive volume-based features.
        
        Features include:
        - Volume rate of change patterns
        - Volume-price relationship metrics
        - Rolling volume statistics
        - Volume clustering features
        - Microstructure indicators
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with engineered features
        """
        features = pd.DataFrame(index=data.index)
        
        # Basic volume metrics
        features['volume'] = data['volume']
        features['volume_log'] = np.log1p(data['volume'])
        features['volume_zscore'] = stats.zscore(data['volume'])
        
        # Volume rate of change
        for period in [3, 5, 10, 15, 20]:
            features[f'volume_roc_{period}'] = data['volume'].pct_change(period)
            features[f'volume_ratio_{period}'] = (
                data['volume'] / data['volume'].rolling(period).mean()
            )
        
        # Volume-price relationships
        features['volume_price_trend'] = (
            data['volume'].rolling(10).corr(data['close'])
        )
        features['vwap'] = (
            (data['volume'] * (data['high'] + data['low'] + data['close']) / 3).cumsum() 
            / data['volume'].cumsum()
        )
        features['volume_weighted_return'] = (
            data['close'].pct_change() * data['volume']
        )
        
        # Rolling statistics
        for window in [5, 10, 20, 50]:
            vol_rolling = data['volume'].rolling(window)
            features[f'volume_mean_{window}'] = vol_rolling.mean()
            features[f'volume_std_{window}'] = vol_rolling.std()
            features[f'volume_skew_{window}'] = vol_rolling.skew()
            features[f'volume_kurt_{window}'] = vol_rolling.kurt()
            features[f'volume_range_{window}'] = (
                vol_rolling.max() - vol_rolling.min()
            )
        
        # Volume clustering (density features)
        features['volume_density_5'] = (
            data['volume'].rolling(5).apply(
                lambda x: len(np.unique(np.round(x, 2))) / len(x)
            )
        )
        
        # Microstructure indicators
        features['volume_acceleration'] = (
            features['volume_roc_3'].diff()
        )
        features['volume_momentum'] = (
            features['volume_roc_5'].rolling(3).mean()
        )
        
        # Volume regime indicators
        features['volume_regime'] = self._detect_volume_regime(data['volume'])
        
        # Fill NaN values
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_isolation_score(path_length: float, n_samples: int) -> float:
        """
        Calculate Isolation Forest anomaly score.
        
        Mathematical formula: score = 2^(-E(h(x))/c(n))
        where c(n) = 2*H(n-1) - (2*(n-1)/n) for average path length
        
        Args:
            path_length: Expected path length for isolation
            n_samples: Number of training samples
            
        Returns:
            Anomaly score between 0 and 1
        """
        if n_samples <= 1:
            return 0.0
        
        # Calculate c(n) - average path length of unsuccessful search in BST
        harmonic = np.sum(1.0 / np.arange(1, n_samples))
        c_n = 2 * harmonic - (2 * (n_samples - 1) / n_samples)
        
        # Calculate isolation score
        score = 2 ** (-path_length / c_n)
        return score
    
    def _detect_volume_regime(self, volume: pd.Series) -> pd.Series:
        """
        Detect volume regimes using statistical analysis.
        
        Args:
            volume: Volume time series
            
        Returns:
            Series with regime indicators (0: low, 1: normal, 2: high)
        """
        # Calculate volume percentiles
        volume_pct = volume.rolling(50).rank(pct=True)
        
        regimes = pd.Series(1, index=volume.index)  # Default: normal
        regimes[volume_pct < 0.2] = 0  # Low volume regime
        regimes[volume_pct > 0.8] = 2  # High volume regime
        
        return regimes
    
    def _select_features(self, X: np.ndarray, y: np.ndarray = None) -> List[int]:
        """
        Information-theoretic feature selection.
        
        Args:
            X: Feature matrix
            y: Target variable (optional, uses volume if None)
            
        Returns:
            List of selected feature indices
        """
        if not self.enable_feature_selection or X.shape[1] < 10:
            return list(range(X.shape[1]))
        
        if y is None:
            # Use volume as target for unsupervised selection
            y = X[:, 0]  # Assuming first feature is volume
        
        # Calculate mutual information
        mi_scores = mutual_info_regression(X, y, random_state=self.random_state)
        
        # Select top features
        n_select = min(int(X.shape[1] * 0.7), 20)  # Select 70% or max 20 features
        selected_indices = np.argsort(mi_scores)[-n_select:]
        
        # Store feature importance
        for i, score in enumerate(mi_scores):
            if i < len(self.feature_names_):
                self.feature_importance_[self.feature_names_[i]] = score
        
        return selected_indices.tolist()
    
    def fit(self, data: pd.DataFrame) -> 'VolumeAnomalyDetector':
        """
        Fit anomaly detection models on historical data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Fitted detector instance
        """
        self.logger.info("Engineering volume features...")
        features_df = self.engineer_volume_features(data)
        self.feature_names_ = features_df.columns.tolist()
        
        # Convert to numpy array
        X = features_df.values
        
        # Feature selection
        if self.enable_feature_selection:
            selected_indices = self._select_features(X)
            self.selected_features_ = selected_indices
            X = X[:, selected_indices]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        self.logger.info("Fitting Isolation Forest...")
        self.isolation_forest.fit(X_scaled)
        
        self.logger.info("Fitting Local Outlier Factor...")
        self.lof.fit(X_scaled)
        
        self.logger.info("Fitting covariance matrix for Mahalanobis distance...")
        self.covariance.fit(X_scaled)
        
        self.is_fitted = True
        self.logger.info("Volume anomaly detector fitted successfully!")
        
        return self
    
    def calculate_mahalanobis_distance(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate Mahalanobis distance for each sample.
        
        Mathematical formula: D² = (x-μ)ᵀ Σ⁻¹ (x-μ)
        
        Args:
            X: Scaled feature matrix
            
        Returns:
            Array of Mahalanobis distances
        """
        mean = np.mean(X, axis=0)
        cov_inv = self.covariance.precision_
        
        distances = []
        for x in X:
            diff = x - mean
            distance = np.sqrt(np.dot(np.dot(diff, cov_inv), diff))
            distances.append(distance)
        
        return np.array(distances)
    
    def detect_anomalies(self, 
                        data: pd.DataFrame,
                        method: str = 'ensemble',
                        threshold: float = 0.7) -> List[AnomalyResult]:
        """
        Detect volume anomalies using specified method(s).
        
        Args:
            data: DataFrame with OHLCV data
            method: Detection method ('isolation', 'mahalanobis', 'lof', 'ensemble')
            threshold: Anomaly threshold for classification
            
        Returns:
            List of anomaly detection results
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before detecting anomalies")
        
        # Engineer features
        features_df = self.engineer_volume_features(data)
        X = features_df.values
        
        # Apply feature selection
        if self.selected_features_:
            X = X[:, self.selected_features_]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        results = []
        
        for i, timestamp in enumerate(data.index):
            x_sample = X_scaled[i:i+1]
            
            scores = {}
            
            # Isolation Forest
            if method in ['isolation', 'ensemble']:
                iso_score = self.isolation_forest.decision_function(x_sample)[0]
                # Convert to 0-1 scale (higher = more anomalous)
                iso_score_norm = (iso_score + 0.5) / 1.0  # Rough normalization
                scores['isolation'] = max(0, min(1, iso_score_norm))
            
            # Local Outlier Factor
            if method in ['lof', 'ensemble']:
                lof_score = self.lof.decision_function(x_sample)[0]
                # Convert to 0-1 scale
                lof_score_norm = max(0, min(1, -lof_score / 2 + 0.5))
                scores['lof'] = lof_score_norm
            
            # Mahalanobis Distance
            if method in ['mahalanobis', 'ensemble']:
                maha_dist = self.calculate_mahalanobis_distance(x_sample)[0]
                # Convert to anomaly score (normalize by threshold)
                maha_score = min(1, maha_dist / 5.0)  # Adjust threshold as needed
                scores['mahalanobis'] = maha_score
            
            # Ensemble scoring
            if method == 'ensemble':
                final_score = np.mean(list(scores.values()))
                final_method = 'ensemble'
            else:
                final_score = scores[method]
                final_method = method
            
            # Determine if anomaly
            is_anomaly = final_score > threshold
            
            # Calculate confidence (distance from threshold)
            confidence = abs(final_score - threshold) / threshold
            
            # Feature contributions (simplified)
            feature_contributions = {}
            if i < len(self.feature_names_):
                for j, fname in enumerate(self.feature_names_):
                    if j in self.selected_features_:
                        feature_contributions[fname] = abs(X_scaled[i, self.selected_features_.index(j)])
            
            result = AnomalyResult(
                timestamp=timestamp,
                anomaly_score=final_score,
                is_anomaly=is_anomaly,
                method=final_method,
                confidence=confidence,
                feature_contributions=feature_contributions
            )
            
            results.append(result)
        
        # Store detection history
        self.detection_history.extend(results)
        
        return results
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return self.feature_importance_
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics from detection history."""
        if not self.detection_history:
            return {}
        
        anomaly_rate = sum(1 for r in self.detection_history if r.is_anomaly) / len(self.detection_history)
        avg_confidence = np.mean([r.confidence for r in self.detection_history])
        avg_score = np.mean([r.anomaly_score for r in self.detection_history])
        
        return {
            'anomaly_rate': anomaly_rate,
            'average_confidence': avg_confidence,
            'average_score': avg_score,
            'total_detections': len(self.detection_history)
        }
    
    def plot_anomaly_timeline(self, save_path: Optional[str] = None):
        """Plot anomaly detection timeline."""
        import matplotlib.pyplot as plt
        
        if not self.detection_history:
            self.logger.warning("No detection history available for plotting")
            return
        
        timestamps = [r.timestamp for r in self.detection_history]
        scores = [r.anomaly_score for r in self.detection_history]
        anomalies = [r.is_anomaly for r in self.detection_history]
        
        plt.figure(figsize=(15, 8))
        
        # Plot scores
        plt.subplot(2, 1, 1)
        plt.plot(timestamps, scores, label='Anomaly Score', alpha=0.7)
        plt.scatter([t for t, a in zip(timestamps, anomalies) if a], 
                   [s for s, a in zip(scores, anomalies) if a], 
                   color='red', label='Anomalies', s=50)
        plt.ylabel('Anomaly Score')
        plt.title('Volume Anomaly Detection Timeline')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot anomaly indicators
        plt.subplot(2, 1, 2)
        plt.plot(timestamps, [1 if a else 0 for a in anomalies], 
                'r-', linewidth=2, label='Anomaly Detected')
        plt.ylabel('Anomaly Flag')
        plt.xlabel('Time')
        plt.ylim(-0.1, 1.1)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# Example usage and testing
if __name__ == "__main__":
    # Generate sample data for testing
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    
    # Create synthetic OHLCV data with some volume anomalies
    volume_base = np.random.lognormal(10, 0.5, 1000)
    
    # Inject anomalies
    anomaly_indices = np.random.choice(1000, 50, replace=False)
    volume_base[anomaly_indices] *= np.random.uniform(3, 10, 50)
    
    sample_data = pd.DataFrame({
        'open': np.random.uniform(40000, 50000, 1000),
        'high': np.random.uniform(40000, 50000, 1000),
        'low': np.random.uniform(40000, 50000, 1000),
        'close': np.random.uniform(40000, 50000, 1000),
        'volume': volume_base
    }, index=dates)
    
    # Test the detector
    detector = VolumeAnomalyDetector(contamination=0.05)
    detector.fit(sample_data[:800])  # Fit on first 800 samples
    
    # Detect anomalies in test set
    results = detector.detect_anomalies(sample_data[800:], method='ensemble')
    
    print(f"Detected {sum(1 for r in results if r.is_anomaly)} anomalies")
    print(f"Performance metrics: {detector.get_performance_metrics()}")
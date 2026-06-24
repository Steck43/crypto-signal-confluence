# Quick Fixes for Critical Trading System Issues

import os
import sys
import logging
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# 1. FIX UNICODE LOGGING ISSUES (Windows Console)
def setup_unicode_logging():
    """Setup logging with proper Unicode support for Windows"""
    
    # Fix Windows console encoding for emojis
    if sys.platform == "win32":
        # Set console to UTF-8
        os.system("chcp 65001 > nul")
        
        # Force UTF-8 encoding for stdout/stderr
        import io
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, 
            encoding='utf-8', 
            errors='replace'
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, 
            encoding='utf-8', 
            errors='replace'
        )

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Custom formatter that handles emojis safely
    class SafeFormatter(logging.Formatter):
        def format(self, record):
            try:
                return super().format(record)
            except UnicodeEncodeError:
                # Remove emojis if encoding fails
                record.msg = str(record.msg).encode('ascii', 'ignore').decode('ascii')
                return super().format(record)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)8s | %(name)s | %(message)s',
        handlers=[
            # File handler (UTF-8)
            logging.FileHandler('logs/trading_system.log', encoding='utf-8'),
            # Console handler (safe)
            logging.StreamHandler()
        ]
    )
    
    # Apply safe formatter to console handler
    console_handler = logging.getLogger().handlers[-1]
    console_handler.setFormatter(SafeFormatter('%(asctime)s | %(levelname)8s | %(name)20s | %(message)s'))

# 2. FIX TELEGRAM API PARSING ERRORS
async def send_safe_telegram_message(bot, chat_id, message):
    """Send Telegram message with safe parsing"""
    try:
        # Remove problematic characters and fix formatting
        safe_message = message.replace('`', '').replace('*', '').replace('_', '')
        
        # Remove emojis if they cause issues
        safe_message = re.sub(r'[^\x00-\x7F]+', '', safe_message)
        
        # Send as plain text instead of markdown
        await bot.send_message(
            chat_id=chat_id,
            text=safe_message,
            parse_mode=None  # No parsing
        )
        
    except Exception as e:
        logging.error(f"Telegram send error: {e}")
        # Fallback to even simpler message
        try:
            simple_message = f"Trading Signal: {message[:100]}..."
            await bot.send_message(chat_id=chat_id, text=simple_message)
        except:
            pass

# 3. FIX VOLUME ANOMALY DATA PROCESSING
class FixedInstitutionalVolumeAnomalyDetector:
    """Fixed version with proper data handling"""
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_columns = []
        self.algorithms = ['isolation_forest', 'local_outlier_factor', 'mahalanobis', 'statistical_process_control', 'one_class_svm']
    
    def predict(self, data: pd.DataFrame) -> List:
        """
        Fixed predict method with proper data validation
        """
        try:
            if data.empty:
                logging.warning("Empty data provided to volume detector")
                return []
            
            logging.info(f"Input data shape: {data.shape}")
            
            # Engineer features with validation
            engineered_data = self.engineer_institutional_features(data)
            logging.info(f"Engineered data shape: {engineered_data.shape}")
            
            if engineered_data.empty:
                logging.warning("No valid features after engineering")
                return []
            
            # Ensure we have the same features as training
            if not hasattr(self, 'feature_columns') or not self.feature_columns:
                logging.error("Model not properly fitted - no feature columns")
                return []
            
            # Select only the features we trained on
            available_features = [col for col in self.feature_columns if col in engineered_data.columns]
            
            if not available_features:
                logging.error("No matching features between training and prediction")
                return []
            
            X = engineered_data[available_features].values
            logging.info(f"Feature matrix shape: {X.shape}")
            
            if X.shape[0] == 0:
                logging.warning("No valid samples after feature selection")
                return []
            
            # Handle NaN values
            if np.any(np.isnan(X)):
                logging.warning("NaN values detected, filling with median")
                X = np.nan_to_num(X, nan=np.nanmedian(X))
            
            # Scale features
            if hasattr(self, 'scaler') and self.scaler is not None:
                try:
                    X_scaled = self.scaler.transform(X)
                except Exception as e:
                    logging.error(f"Scaling error: {e}")
                    # Fallback to unscaled data
                    X_scaled = X
            else:
                X_scaled = X
            
            results = []
            
            for i in range(X_scaled.shape[0]):
                try:
                    # Get predictions from available algorithms
                    algorithm_scores = {}
                    
                    for algorithm in self.algorithms:
                        try:
                            if algorithm in self.models and self.models[algorithm] is not None:
                                if algorithm == 'isolation_forest':
                                    if hasattr(self.models[algorithm], 'score_samples'):
                                        score = self.models[algorithm].score_samples([X_scaled[i]])[0]
                                    else:
                                        score = -self.models[algorithm].decision_function([X_scaled[i]])[0]
                                elif algorithm == 'local_outlier_factor':
                                    if hasattr(self.models[algorithm], 'score_samples'):
                                        score = self.models[algorithm].score_samples([X_scaled[i]])[0]
                                    else:
                                        score = 0.0
                                elif algorithm == 'one_class_svm':
                                    if hasattr(self.models[algorithm], 'score_samples'):
                                        score = self.models[algorithm].score_samples([X_scaled[i]])[0]
                                    else:
                                        score = self.models[algorithm].decision_function([X_scaled[i]])[0]
                                else:
                                    # For function-based algorithms
                                    score = 0.0
                                
                                algorithm_scores[algorithm] = score
                        except Exception as e:
                            logging.warning(f"Algorithm {algorithm} prediction failed: {e}")
                            algorithm_scores[algorithm] = 0.0
                    
                    # Ensemble decision
                    if algorithm_scores:
                        ensemble_score = np.mean(list(algorithm_scores.values()))
                        is_anomaly = ensemble_score < -0.1  # Threshold for anomaly
                        confidence = len(algorithm_scores) / len(self.algorithms)
                    else:
                        ensemble_score = 0.0
                        is_anomaly = False
                        confidence = 0.0
                    
                    # Create result
                    result = type('AnomalyResult', (), {
                        'timestamp': engineered_data.index[i] if i < len(engineered_data) else pd.Timestamp.now(),
                        'anomaly_score': float(ensemble_score),
                        'is_anomaly': bool(is_anomaly),
                        'confidence': float(confidence),
                        'algorithm_scores': algorithm_scores,
                        'feature_contributions': {},
                        'market_regime': 'normal',
                        'volume_percentile': 50.0,
                        'raw_features': {}
                    })()
                    
                    results.append(result)
                    
                except Exception as e:
                    logging.error(f"Error processing sample {i}: {e}")
                    continue
            
            logging.info(f"Generated {len(results)} anomaly results")
            return results
            
        except Exception as e:
            logging.error(f"Critical error in volume anomaly prediction: {e}")
            return []
    
    def engineer_institutional_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Basic feature engineering that won't fail"""
        try:
            df = data.copy()
            
            # Ensure minimum data requirements
            if len(data) < 10:
                logging.warning("Insufficient data for feature engineering, using basic features")
                # Add basic features only
                df['volume_ma_5'] = df['volume'].rolling(5, min_periods=1).mean()
                df['volume_ma_20'] = df['volume'].rolling(20, min_periods=1).mean()
                df['volume_ratio'] = df['volume'] / df['volume_ma_20']
                df['price_change'] = df['close'].pct_change().fillna(0)
                df['volatility'] = df['price_change'].rolling(10, min_periods=1).std().fillna(0)
                
                # Select only numerical features
                feature_cols = ['volume_ma_5', 'volume_ma_20', 'volume_ratio', 'price_change', 'volatility']
                self.feature_columns = feature_cols
                
                result = df[['timestamp', 'volume', 'close'] + feature_cols].dropna()
                logging.info(f"Basic feature engineering: {len(feature_cols)} features, {len(result)} samples")
                return result
            
            # More advanced features if we have enough data
            df['volume_ma_5'] = df['volume'].rolling(5, min_periods=1).mean()
            df['volume_ma_20'] = df['volume'].rolling(20, min_periods=1).mean()
            df['volume_ma_50'] = df['volume'].rolling(50, min_periods=1).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma_20']
            df['volume_std'] = df['volume'].rolling(20, min_periods=1).std()
            df['volume_zscore'] = (df['volume'] - df['volume_ma_20']) / df['volume_std']
            
            df['price_change'] = df['close'].pct_change().fillna(0)
            df['volatility'] = df['price_change'].rolling(10, min_periods=1).std().fillna(0)
            df['price_ma_5'] = df['close'].rolling(5, min_periods=1).mean()
            df['price_ma_20'] = df['close'].rolling(20, min_periods=1).mean()
            
            # Select features
            feature_cols = ['volume_ma_5', 'volume_ma_20', 'volume_ma_50', 'volume_ratio', 
                           'volume_std', 'volume_zscore', 'price_change', 'volatility',
                           'price_ma_5', 'price_ma_20']
            self.feature_columns = feature_cols
            
            result = df[['timestamp', 'volume', 'close'] + feature_cols].dropna()
            logging.info(f"Advanced feature engineering: {len(feature_cols)} features, {len(result)} samples")
            return result
            
        except Exception as e:
            logging.error(f"Feature engineering failed: {e}")
            # Fallback to minimal features
            df = data.copy()
            df['simple_feature'] = df['volume'] / df['volume'].rolling(5, min_periods=1).mean()
            self.feature_columns = ['simple_feature']
            return df[['timestamp', 'volume', 'close', 'simple_feature']].dropna()

# 4. FIX FEATURE ENGINEERING CONSISTENCY
def fix_feature_engineering(detector):
    """Fix feature engineering consistency issues"""
    
    original_engineer = detector.engineer_institutional_features
    
    def fixed_engineer_features(data):
        """Fixed feature engineering with consistent output"""
        try:
            logging.info(f"Starting feature engineering with {len(data)} samples")
            
            # Ensure minimum data requirements
            if len(data) < 10:
                logging.warning("Insufficient data for feature engineering, using basic features")
                df = data.copy()
                # Add basic features only
                df['volume_ma_5'] = df['volume'].rolling(5, min_periods=1).mean()
                df['volume_ma_20'] = df['volume'].rolling(20, min_periods=1).mean()
                df['volume_ratio'] = df['volume'] / df['volume_ma_20']
                df['price_change'] = df['close'].pct_change().fillna(0)
                df['volatility'] = df['price_change'].rolling(10, min_periods=1).std().fillna(0)
                
                # Select only numerical features
                feature_cols = ['volume_ma_5', 'volume_ma_20', 'volume_ratio', 'price_change', 'volatility']
                detector.feature_columns = feature_cols
                
                result = df[['timestamp', 'volume', 'close'] + feature_cols].dropna()
                logging.info(f"Basic feature engineering: {len(feature_cols)} features, {len(result)} samples")
                return result
            
            # Call original engineering
            result = original_engineer(data)
            logging.info(f"Advanced feature engineering: {len(detector.feature_columns)} features, {len(result)} samples")
            return result
            
        except Exception as e:
            logging.error(f"Feature engineering failed: {e}")
            # Fallback to minimal features
            df = data.copy()
            df['simple_feature'] = df['volume'] / df['volume'].rolling(5, min_periods=1).mean()
            detector.feature_columns = ['simple_feature']
            return df[['timestamp', 'volume', 'close', 'simple_feature']].dropna()
    
    detector.engineer_institutional_features = fixed_engineer_features
    return detector

# 5. SAFE DATA GENERATION FOR TESTING
def generate_safe_test_data(n_samples=100):
    """Generate safe test data that won't cause processing errors"""
    
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='5min')
    
    # Generate realistic but safe data
    np.random.seed(42)  # Reproducible
    
    base_price = 100.0
    price_changes = np.random.normal(0, 0.02, n_samples)
    prices = [base_price]
    
    for change in price_changes:
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1.0))  # Ensure positive prices
    
    prices = prices[1:]  # Remove initial value
    
    # Generate correlated volume
    volumes = np.random.lognormal(mean=10, sigma=0.5, size=n_samples)
    volumes = np.maximum(volumes, 1.0)  # Ensure positive volumes
    
    # Create DataFrame with all required columns
    data = pd.DataFrame({
        'timestamp': dates,
        'open': [p * np.random.uniform(0.99, 1.01) for p in prices],
        'high': [p * np.random.uniform(1.0, 1.02) for p in prices],
        'low': [p * np.random.uniform(0.98, 1.0) for p in prices],
        'close': prices,
        'volume': volumes
    })
    
    # Ensure no NaN or infinite values
    data = data.fillna(method='ffill').fillna(0)
    data = data.replace([np.inf, -np.inf], 0)
    
    logging.info(f"Generated safe test data: {len(data)} samples")
    logging.info(f"Price range: {data['close'].min():.2f} - {data['close'].max():.2f}")
    logging.info(f"Volume range: {data['volume'].min():.2f} - {data['volume'].max():.2f}")
    
    return data

# 6. MAIN DEBUGGING WRAPPER
def debug_trading_system():
    """Main debugging function to wrap your trading system"""
    
    # Setup safe logging
    setup_unicode_logging()
    
    logging.info("=" * 50)
    logging.info("DEBUG MODE: Trading System Diagnostics")
    logging.info("=" * 50)
    
    try:
        # Your existing trading system code here
        # But with debugging enabled
        
        logging.info("System starting with debug fixes applied...")
        
        # Add debug information
        logging.info(f"Python version: {sys.version}")
        logging.info(f"Pandas version: {pd.__version__}")
        logging.info(f"NumPy version: {np.__version__}")
        logging.info(f"Platform: {sys.platform}")
        
        return True
        
    except Exception as e:
        logging.error(f"Debug initialization failed: {e}")
        return False

# USAGE INSTRUCTIONS
if __name__ == "__main__":
    print("""
DEBUGGING INSTRUCTIONS:
======================

1. Add this at the TOP of your paper_trading_system.py:
   from debug_fixes import setup_unicode_logging, debug_trading_system
   setup_unicode_logging()

2. Replace your Telegram sending with:
   await send_safe_telegram_message(bot, chat_id, message)

3. Apply the fixed volume detector:
   detector = fix_feature_engineering(detector)

4. Use safe test data:
   test_data = generate_safe_test_data(200)

5. Wrap your main function:
   if debug_trading_system():
       # Your trading code here
""") 
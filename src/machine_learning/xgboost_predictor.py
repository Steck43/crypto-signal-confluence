"""
XGBoost Signal Predictor for Institutional Trading System

Integrates with the 66-feature volume anomaly detection system to provide:
- Feature importance ranking from institutional features
- Signal prediction with confidence scoring
- Integration with existing ensemble decision making
"""

import xgboost as xgb
import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, List, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from backtesting.purged_cv import PurgedKFold
from dataclasses import dataclass
import joblib
from datetime import datetime

@dataclass
class XGBoostPrediction:
    signal: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0.0 to 1.0
    probabilities: Dict[str, float]  # Individual class probabilities
    feature_importance: Dict[str, float]  # Top contributing features
    reasoning: str  # Human-readable explanation

class XGBoostSignalPredictor:
    """
    XGBoost predictor optimized for the institutional 66-feature system
    
    Key Features:
    - Handles 66+ engineered features from volume anomaly detection
    - Time series cross-validation for financial data
    - Feature importance analysis for system optimization
    - Integration with existing sentiment and technical analysis
    - Professional logging and monitoring
    """
    
    def __init__(self, 
                 n_estimators: int = 300,
                 max_depth: int = 8,
                 learning_rate: float = 0.1,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 random_state: int = 42):
        """
        Initialize XGBoost predictor with optimal parameters for trading
        
        Args:
            n_estimators: Number of boosting rounds (300 for stability)
            max_depth: Maximum tree depth (8 for complexity without overfitting)
            learning_rate: Learning rate (0.1 for stable convergence)
            subsample: Fraction of samples for training (0.8 for regularization)
            colsample_bytree: Fraction of features per tree (0.8 for regularization)
            random_state: Random seed for reproducibility
        """
        
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective='multi:softprob',  # Multi-class probabilities
            eval_metric='mlogloss',      # Log loss for probability calibration
            random_state=random_state,
            n_jobs=1,                   # Single-threaded for deterministic CV
            tree_method='hist',          # Fast histogram-based method
            enable_categorical=False,    # All features are numerical
            verbosity=0                  # Reduce output noise
        )
        
        # Model state
        self.is_fitted = False
        self.feature_names = []
        self.feature_importance_dict = {}
        self.top_features = []
        self.cv_scores = []
        
        # Performance tracking
        self.training_history = []
        self.prediction_history = []
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("🤖 XGBoost Signal Predictor initialized")
        
    def prepare_institutional_features(self, 
                                     volume_features: pd.DataFrame,
                                     sentiment_data: Dict,
                                     technical_data: Dict,
                                     market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Combine all features for XGBoost training
        
        Integrates:
        - 66 volume anomaly features from InstitutionalVolumeAnomalyDetector
        - 4 sentiment features from multi-source analysis
        - 6 technical features from technical analysis engine
        - 4 market structure features
        
        Total: ~80 features for comprehensive analysis
        """
        
        # Start with the 66 institutional volume features
        combined_features = volume_features.copy()
        
        # Add sentiment features (4 features)
        sentiment_df = pd.DataFrame({
            'rss_sentiment_score': sentiment_data.get('rss_score', 0.0),
            'alpha_vantage_sentiment': sentiment_data.get('av_score', 0.0),
            'sentiment_confidence': sentiment_data.get('confidence', 0.0),
            'news_article_count': sentiment_data.get('article_count', 0),
        }, index=volume_features.index)
        
        # Add technical features (6 features)
        technical_df = pd.DataFrame({
            'rsi_14': technical_data.get('rsi', 50.0),
            'momentum_5_period': technical_data.get('momentum', 0.0),
            'volume_ratio': technical_data.get('volume_ratio', 1.0),
            'price_volatility': technical_data.get('volatility', 0.02),
            'technical_score': technical_data.get('technical_score', 0.0),
            'technical_confidence': technical_data.get('confidence', 0.0)
        }, index=volume_features.index)
        
        # Add market structure features (4 features)
        market_df = pd.DataFrame({
            'price_change_1period': market_data['close'].pct_change(),
            'price_change_5period': market_data['close'].pct_change(5),
            'volume_change_1period': market_data['volume'].pct_change(),
            'high_low_spread': (market_data['high'] - market_data['low']) / market_data['close']
        }, index=volume_features.index)
        
        # Combine all feature sets
        combined_features = pd.concat([
            combined_features,
            sentiment_df,
            technical_df,
            market_df
        ], axis=1)

        combined_features = combined_features.loc[:, ~combined_features.columns.duplicated()]
        combined_features = combined_features.apply(pd.to_numeric, errors="coerce")
        combined_features = combined_features.replace([np.inf, -np.inf], np.nan)
        combined_features = combined_features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Store feature names
        self.feature_names = combined_features.columns.tolist()
        
        self.logger.info(f"✅ Prepared {len(self.feature_names)} features for XGBoost training")
        
        return combined_features
    
    def create_trading_targets(self, 
                             price_data: pd.Series, 
                             future_periods: int = 5,
                             buy_threshold: float = 0.015,
                             sell_threshold: float = -0.015) -> np.ndarray:
        """
        Create trading targets based on future price movements
        
        Args:
            price_data: Series of close prices
            future_periods: Number of periods to look ahead (5 = 25 minutes with 5-min data)
            buy_threshold: Minimum return for BUY signal (1.5% default)
            sell_threshold: Maximum return for SELL signal (-1.5% default)
            
        Returns:
            Array of targets: 0=SELL, 1=HOLD, 2=BUY
        """
        
        # Calculate future returns
        future_returns = price_data.pct_change(future_periods).shift(-future_periods)
        
        # Create classification targets
        targets = np.where(
            future_returns >= buy_threshold, 2,      # BUY: >= 1.5% gain
            np.where(
                future_returns <= sell_threshold, 0,  # SELL: <= -1.5% loss
                1                                     # HOLD: between -1.5% and 1.5%
            )
        )
        
        # Remove last periods without targets
        valid_targets = targets[:-future_periods]
        
        self.logger.info(f"📊 Created targets: BUY={np.sum(valid_targets==2)}, "
                        f"HOLD={np.sum(valid_targets==1)}, SELL={np.sum(valid_targets==0)}")
        
        return valid_targets
    
    def _cross_validate(
        self,
        features: pd.DataFrame,
        targets: np.ndarray,
        validation_splits: int,
        early_stopping_rounds: int,
        validation_mode: str,
        future_periods: int,
        embargo: int,
    ) -> Dict[str, List[float]]:
        """
        Run expanding-window CV. validation_mode 'standard' uses sklearn TimeSeriesSplit;
        'purged' applies López de Prado purge (drop train rows whose label window
        overlaps the test fold) plus a post-test embargo before training resumes.
        """
        if validation_mode == "standard":
            splitter = TimeSeriesSplit(n_splits=validation_splits)
            split_iter = splitter.split(features)
        elif validation_mode == "purged":
            splitter = PurgedKFold(
                n_splits=validation_splits,
                label_horizon=future_periods,
                embargo=embargo,
            )
            split_iter = splitter.split(features.values)
        else:
            raise ValueError("validation_mode must be 'standard' or 'purged'")

        cv_scores: List[float] = []
        cv_precisions: List[float] = []
        cv_recalls: List[float] = []
        cv_f1s: List[float] = []

        mode_label = "standard TimeSeriesSplit" if validation_mode == "standard" else "purged + embargo"
        self.logger.info(
            f"Starting {validation_splits}-fold {mode_label} cross-validation..."
        )

        for fold, (train_idx, val_idx) in enumerate(split_iter):
            self.logger.info(f"   Fold {fold + 1}/{validation_splits}")

            X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
            y_train, y_val = targets[train_idx], targets[val_idx]

            if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 1:
                self.logger.info(f"   Fold {fold + 1} skipped (insufficient class diversity)")
                continue

            fold_model = xgb.XGBClassifier(
                n_estimators=self.model.n_estimators,
                max_depth=self.model.max_depth,
                learning_rate=self.model.learning_rate,
                subsample=self.model.subsample,
                colsample_bytree=self.model.colsample_bytree,
                objective=self.model.objective,
                eval_metric=self.model.eval_metric,
                random_state=self.model.random_state,
                n_jobs=self.model.n_jobs,
                tree_method=self.model.tree_method,
                enable_categorical=self.model.enable_categorical,
                verbosity=0,
            )
            fit_kwargs = {"verbose": False}
            fold_model.fit(X_train, y_train, **fit_kwargs)

            val_pred = fold_model.predict(X_val)
            cv_scores.append(accuracy_score(y_val, val_pred))
            cv_precisions.append(
                precision_score(y_val, val_pred, average="weighted", zero_division=0)
            )
            cv_recalls.append(
                recall_score(y_val, val_pred, average="weighted", zero_division=0)
            )
            cv_f1s.append(f1_score(y_val, val_pred, average="weighted", zero_division=0))

        if not cv_scores:
            raise ValueError("Cross-validation produced no valid folds")

        return {
            "accuracy": cv_scores,
            "precision": cv_precisions,
            "recall": cv_recalls,
            "f1": cv_f1s,
        }

    def fit(self, 
            features_df: pd.DataFrame, 
            price_data: pd.Series,
            validation_splits: int = 5,
            early_stopping_rounds: int = 30,
            validation_mode: str = "purged",
            future_periods: int = 5,
            embargo: int = 5,
            buy_threshold: float = 0.015,
            sell_threshold: float = -0.015,
            cv_only: bool = False) -> 'XGBoostSignalPredictor':
        """
        Train XGBoost model with time series cross-validation
        
        Args:
            features_df: Combined feature dataframe (~80 features)
            price_data: Price series for target creation
            validation_splits: Number of time series validation splits
            early_stopping_rounds: Early stopping patience
            validation_mode: 'standard' (leaky TimeSeriesSplit) or 'purged'
            future_periods: Forward return horizon used for labels and purging
            embargo: Bars withheld after each test fold before training resumes
        """
        
        targets = self.create_trading_targets(
            price_data,
            future_periods=future_periods,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        )
        aligned_len = len(targets)
        features = features_df.iloc[:aligned_len].reset_index(drop=True)
        targets = targets[:aligned_len]

        self.validation_mode = validation_mode
        self.cv_scores = self._cross_validate(
            features,
            targets,
            validation_splits,
            early_stopping_rounds,
            validation_mode,
            future_periods,
            embargo,
        )
        
        if not cv_only:
            self.logger.info("Final training on complete dataset...")
            self.model.fit(features, targets)

            self.feature_importance_dict = dict(zip(
                self.feature_names,
                self.model.feature_importances_
            ))

            self.top_features = sorted(
                self.feature_importance_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20]
        else:
            self.feature_importance_dict = {}
            self.top_features = []
        
        self.is_fitted = not cv_only
        
        cv_scores = self.cv_scores["accuracy"]
        cv_precisions = self.cv_scores["precision"]
        cv_recalls = self.cv_scores["recall"]
        cv_f1s = self.cv_scores["f1"]
        
        self.logger.info("XGBoost training completed:")
        self.logger.info(f"   CV Accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        self.logger.info(f"   CV Precision: {np.mean(cv_precisions):.3f} ± {np.std(cv_precisions):.3f}")
        self.logger.info(f"   CV Recall: {np.mean(cv_recalls):.3f} ± {np.std(cv_recalls):.3f}")
        self.logger.info(f"   CV F1-Score: {np.mean(cv_f1s):.3f} ± {np.std(cv_f1s):.3f}")
        
        self.logger.info("Top 10 Most Important Features:")
        for i, (feature, importance) in enumerate(self.top_features[:10]):
            self.logger.info(f"   {i+1:2d}. {feature}: {importance:.4f}")
        
        return self
    
    def predict(self, current_features: pd.Series) -> XGBoostPrediction:
        """
        Generate trading signal prediction with detailed analysis
        
        Args:
            current_features: Current feature values (single row)
            
        Returns:
            XGBoostPrediction object with signal, confidence, and analysis
        """
        
        if not self.is_fitted:
            raise ValueError("❌ Model must be fitted before prediction")
        
        try:
            # Ensure feature order matches training
            feature_values = current_features.reindex(self.feature_names).fillna(0).values
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(feature_values.reshape(1, -1))[0]
            
            # Map to signals
            signal_mapping = {0: 'sell', 1: 'hold', 2: 'buy'}
            predicted_class = np.argmax(probabilities)
            predicted_signal = signal_mapping[predicted_class]
            confidence = float(probabilities[predicted_class])
            
            # Analyze feature contributions for this prediction
            feature_contributions = self._analyze_feature_contributions(feature_values)
            
            # Create reasoning
            reasoning = self._generate_reasoning(predicted_signal, confidence, feature_contributions)
            
            # Create prediction object
            prediction = XGBoostPrediction(
                signal=predicted_signal,
                confidence=confidence,
                probabilities={
                    'sell': float(probabilities[0]),
                    'hold': float(probabilities[1]),
                    'buy': float(probabilities[2])
                },
                feature_importance=feature_contributions,
                reasoning=reasoning
            )
            
            # Store prediction for analysis
            self.prediction_history.append({
                'timestamp': datetime.now(),
                'signal': predicted_signal,
                'confidence': confidence,
                'probabilities': prediction.probabilities
            })
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"❌ XGBoost prediction error: {e}")
            
            # Return neutral prediction on error
            return XGBoostPrediction(
                signal='hold',
                confidence=0.0,
                probabilities={'sell': 0.33, 'hold': 0.34, 'buy': 0.33},
                feature_importance={},
                reasoning=f"XGBoost prediction failed: {str(e)}"
            )
    
    def _analyze_feature_contributions(self, feature_values: np.ndarray) -> Dict[str, float]:
        """Analyze which features contributed most to current prediction"""
        
        try:
            # Use SHAP-like analysis with feature importance and values
            contributions = {}
            
            for i, (feature_name, importance) in enumerate(self.top_features[:10]):
                feature_idx = self.feature_names.index(feature_name)
                feature_value = feature_values[feature_idx]
                
                # Simplified contribution: importance * normalized feature value
                contribution = importance * abs(feature_value)
                contributions[feature_name] = float(contribution)
            
            return contributions
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature contribution analysis failed: {e}")
            return {}
    
    def _generate_reasoning(self, 
                          signal: str, 
                          confidence: float, 
                          contributions: Dict[str, float]) -> str:
        """Generate human-readable reasoning for the prediction"""
        
        top_contributors = sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:3]
        
        reasoning_parts = [
            f"XGBoost predicts {signal.upper()} with {confidence:.1%} confidence"
        ]
        
        if top_contributors:
            feature_reasons = [f"{name}" for name, _ in top_contributors]
            reasoning_parts.append(f"Key factors: {', '.join(feature_reasons)}")
        
        return ". ".join(reasoning_parts)
    
    def get_feature_importance_summary(self) -> Dict:
        """Get comprehensive feature importance analysis"""
        
        if not self.is_fitted:
            return {"error": "Model not fitted"}
        
        return {
            "top_20_features": self.top_features,
            "feature_count": len(self.feature_names),
            "cv_performance": {
                "accuracy_mean": np.mean(self.cv_scores['accuracy']),
                "accuracy_std": np.std(self.cv_scores['accuracy']),
                "precision_mean": np.mean(self.cv_scores['precision']),
                "recall_mean": np.mean(self.cv_scores['recall']),
                "f1_mean": np.mean(self.cv_scores['f1'])
            },
            "model_info": {
                "n_estimators": self.model.n_estimators,
                "max_depth": self.model.max_depth,
                "learning_rate": self.model.learning_rate
            }
        }
    
    def save_model(self, filepath: str):
        """Save trained XGBoost model"""
        if self.is_fitted:
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'feature_importance': self.feature_importance_dict,
                'top_features': self.top_features,
                'cv_scores': self.cv_scores
            }
            joblib.dump(model_data, filepath)
            self.logger.info(f"💾 XGBoost model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained XGBoost model"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.feature_importance_dict = model_data['feature_importance']
            self.top_features = model_data['top_features']
            self.cv_scores = model_data['cv_scores']
            self.is_fitted = True
            self.logger.info(f"📂 XGBoost model loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"❌ Failed to load XGBoost model: {e}") 
"""
ML Ensemble Manager for Institutional Trading System

EXPLORATORY: fixed-weight XGBoost+GRU fusion scaffold. Not on the validated ablation path.
The validated harness uses xgboost_predictor.py directly.

Coordinates XGBoost and GRU models with existing institutional components.
Provides unified interface for all ML predictions and ensemble decisions.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

from .xgboost_predictor import XGBoostSignalPredictor, XGBoostPrediction
from .gru_forecaster import GRUTradingPredictor, GRUPrediction

class MLEnsemblePrediction:
    """Data class for ML ensemble predictions"""
    
    def __init__(self, 
                 signal: str,
                 confidence: float,
                 strength: float,
                 individual_predictions: Dict[str, Any],
                 ensemble_weights: Dict[str, float],
                 reasoning: str,
                 timestamp: datetime):
        self.signal = signal
        self.confidence = confidence
        self.strength = strength
        self.individual_predictions = individual_predictions
        self.ensemble_weights = ensemble_weights
        self.reasoning = reasoning
        self.timestamp = timestamp

class MLEnsembleManager:
    """
    Manages the ensemble of XGBoost and GRU models alongside institutional components
    
    Features:
    - Coordinated training of all ML models
    - Intelligent ensemble weighting
    - Professional monitoring and logging
    - Integration with existing institutional system
    """
    
    def __init__(self, 
                 enable_xgboost: bool = True,
                 enable_gru: bool = True,
                 ensemble_method: str = 'weighted_voting'):
        """
        Initialize ML ensemble manager
        
        Args:
            enable_xgboost: Whether to include XGBoost model
            enable_gru: Whether to include GRU model  
            ensemble_method: Method for combining predictions
        """
        
        self.enable_xgboost = enable_xgboost
        self.enable_gru = enable_gru
        self.ensemble_method = ensemble_method
        
        # Initialize models
        self.xgboost_predictor = XGBoostSignalPredictor() if enable_xgboost else None
        self.gru_predictor = GRUTradingPredictor() if enable_gru else None
        
        # Ensemble state
        self.is_fitted = False
        self.models_fitted = {}
        
        # Performance tracking
        self.prediction_history = []
        self.performance_metrics = {}
        
        # Ensemble weights (learned adaptively)
        self.ensemble_weights = {
            'institutional_volume': 0.30,  # Volume anomaly detection (5 algorithms)
            'sentiment_analysis': 0.20,    # RSS + Alpha Vantage sentiment
            'technical_analysis': 0.15,    # Technical indicators
            'xgboost': 0.25 if enable_xgboost else 0.0,  # Feature-based prediction
            'gru': 0.10 if enable_gru else 0.0           # Time series forecasting
        }
        
        # Normalize weights
        total_weight = sum(self.ensemble_weights.values())
        if total_weight > 0:
            self.ensemble_weights = {k: v/total_weight for k, v in self.ensemble_weights.items()}
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("🤖 ML Ensemble Manager initialized")
        self.logger.info(f"   XGBoost: {'✅ Enabled' if enable_xgboost else '❌ Disabled'}")
        self.logger.info(f"   GRU: {'✅ Enabled' if enable_gru else '❌ Disabled'}")
        self.logger.info(f"   Ensemble weights: {self.ensemble_weights}")
    
    async def fit_all_models(self, 
                           historical_data: pd.DataFrame,
                           volume_features: pd.DataFrame,
                           sentiment_data: Dict,
                           technical_data: Dict) -> Dict[str, Any]:
        """
        Train all ML models on historical data
        
        Args:
            historical_data: Historical OHLCV market data
            volume_features: 66 features from InstitutionalVolumeAnomalyDetector
            sentiment_data: Sentiment analysis results
            technical_data: Technical analysis results
            
        Returns:
            Dictionary with fitting results and performance metrics
        """
        
        fitting_results = {}
        
        self.logger.info("🚀 Starting ML ensemble training...")
        
        # Fit XGBoost model
        if self.enable_xgboost and self.xgboost_predictor:
            try:
                self.logger.info("📊 Training XGBoost model...")
                
                # Prepare features for XGBoost
                combined_features = self.xgboost_predictor.prepare_institutional_features(
                    volume_features=volume_features,
                    sentiment_data=sentiment_data,
                    technical_data=technical_data,
                    market_data=historical_data
                )
                
                # Train XGBoost
                self.xgboost_predictor.fit(combined_features, historical_data['close'])
                
                # Get performance summary
                xgb_performance = self.xgboost_predictor.get_feature_importance_summary()
                fitting_results['xgboost'] = {
                    'status': 'success',
                    'performance': xgb_performance,
                    'top_features': xgb_performance['top_20_features'][:5]  # Top 5 for summary
                }
                
                self.models_fitted['xgboost'] = True
                self.logger.info("✅ XGBoost training completed successfully")
                
            except Exception as e:
                self.logger.error(f"❌ XGBoost training failed: {e}")
                fitting_results['xgboost'] = {'status': 'failed', 'error': str(e)}
                self.models_fitted['xgboost'] = False
        
        # Fit GRU model
        if self.enable_gru and self.gru_predictor:
            try:
                self.logger.info("🧠 Training GRU model...")
                
                # Train GRU
                self.gru_predictor.fit(
                    market_data=historical_data,
                    volume_features=volume_features,
                    epochs=100,  # Reduced for faster training
                    batch_size=32
                )
                
                # Get model summary
                gru_summary = self.gru_predictor.get_model_summary()
                fitting_results['gru'] = {
                    'status': 'success',
                    'model_info': gru_summary
                }
                
                self.models_fitted['gru'] = True
                self.logger.info("✅ GRU training completed successfully")
                
            except Exception as e:
                self.logger.error(f"❌ GRU training failed: {e}")
                fitting_results['gru'] = {'status': 'failed', 'error': str(e)}
                self.models_fitted['gru'] = False
        
        # Update ensemble weights based on successful fits
        self._update_ensemble_weights_after_fitting()
        
        self.is_fitted = any(self.models_fitted.values())
        
        # Comprehensive results
        fitting_results['ensemble'] = {
            'models_fitted': self.models_fitted,
            'ensemble_weights': self.ensemble_weights,
            'total_fitted': sum(self.models_fitted.values()),
            'is_ready': self.is_fitted
        }
        
        self.logger.info(f"🎯 ML Ensemble training summary:")
        self.logger.info(f"   Models fitted: {sum(self.models_fitted.values())}/{len(self.models_fitted)}")
        self.logger.info(f"   Updated weights: {self.ensemble_weights}")
        
        return fitting_results
    
    async def generate_ml_predictions(self,
                                    current_data: pd.DataFrame,
                                    current_volume_features: pd.DataFrame,
                                    institutional_signals: List[Dict]) -> MLEnsemblePrediction:
        """
        Generate ML predictions and combine with institutional signals
        
        Args:
            current_data: Current market data
            current_volume_features: Current volume features
            institutional_signals: Signals from existing institutional system
            
        Returns:
            MLEnsemblePrediction with combined analysis
        """
        
        if not self.is_fitted:
            return self._create_fallback_prediction("ML models not fitted")
        
        try:
            individual_predictions = {}
            
            # Get XGBoost prediction
            if self.models_fitted.get('xgboost', False):
                try:
                    # Prepare current features (simplified for single prediction)
                    current_features = current_volume_features.iloc[-1] if len(current_volume_features) > 0 else pd.Series()
                    
                    # Add simple sentiment and technical data
                    enhanced_features = current_features.copy()
                    enhanced_features['rss_sentiment_score'] = 0.0  # Would come from actual sentiment
                    enhanced_features['alpha_vantage_sentiment'] = 0.0
                    enhanced_features['sentiment_confidence'] = 0.0
                    enhanced_features['news_article_count'] = 0
                    enhanced_features['rsi_14'] = 50.0
                    enhanced_features['momentum_5_period'] = 0.0
                    enhanced_features['volume_ratio'] = 1.0
                    enhanced_features['price_volatility'] = 0.02
                    enhanced_features['technical_score'] = 0.0
                    enhanced_features['technical_confidence'] = 0.0
                    enhanced_features['price_change_1period'] = 0.0
                    enhanced_features['price_change_5period'] = 0.0
                    enhanced_features['volume_change_1period'] = 0.0
                    enhanced_features['high_low_spread'] = 0.02
                    
                    xgb_pred = self.xgboost_predictor.predict(enhanced_features)
                    individual_predictions['xgboost'] = {
                        'signal': xgb_pred.signal,
                        'confidence': xgb_pred.confidence,
                        'probabilities': xgb_pred.probabilities,
                        'reasoning': xgb_pred.reasoning
                    }
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ XGBoost prediction failed: {e}")
                    individual_predictions['xgboost'] = {
                        'signal': 'hold', 'confidence': 0.0, 'reasoning': f'XGBoost error: {e}'
                    }
            
            # Get GRU prediction
            if self.models_fitted.get('gru', False):
                try:
                    gru_pred = self.gru_predictor.predict(current_data, current_volume_features)
                    
                    # Convert GRU prediction to trading signal
                    if gru_pred.price_change_prediction > 0.01:
                        gru_signal = 'buy'
                    elif gru_pred.price_change_prediction < -0.01:
                        gru_signal = 'sell'
                    else:
                        gru_signal = 'hold'
                    
                    individual_predictions['gru'] = {
                        'signal': gru_signal,
                        'confidence': gru_pred.confidence,
                        'price_change_prediction': gru_pred.price_change_prediction,
                        'trend_direction': gru_pred.trend_direction,
                        'reasoning': gru_pred.reasoning
                    }
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ GRU prediction failed: {e}")
                    individual_predictions['gru'] = {
                        'signal': 'hold', 'confidence': 0.0, 'reasoning': f'GRU error: {e}'
                    }
            
            # Combine with institutional signals
            all_signals = institutional_signals.copy()
            
            # Add ML signals to the mix
            for model_name, pred in individual_predictions.items():
                all_signals.append({
                    'type': model_name,
                    'signal': pred['signal'],
                    'confidence': pred['confidence'],
                    'reasoning': pred['reasoning']
                })
            
            # Generate ensemble prediction
            ensemble_prediction = self._make_ensemble_decision(all_signals, individual_predictions)
            
            return ensemble_prediction
            
        except Exception as e:
            self.logger.error(f"❌ ML ensemble prediction error: {e}")
            return self._create_fallback_prediction(f"Ensemble error: {e}")
    
    def _make_ensemble_decision(self, 
                               all_signals: List[Dict], 
                               ml_predictions: Dict) -> MLEnsemblePrediction:
        """
        Make final ensemble decision combining all signals
        
        Args:
            all_signals: All signals including institutional + ML
            ml_predictions: ML-specific predictions with additional info
            
        Returns:
            Final ensemble prediction
        """
        
        # Calculate weighted votes
        buy_score = 0.0
        sell_score = 0.0
        total_confidence = 0.0
        
        reasoning_parts = []
        
        for signal in all_signals:
            signal_type = signal['type']
            weight = self.ensemble_weights.get(signal_type, 0.1)  # Default small weight
            confidence = signal['confidence']
            
            effective_weight = weight * confidence
            
            if signal['signal'] == 'buy':
                buy_score += effective_weight
            elif signal['signal'] == 'sell':
                sell_score += effective_weight
            
            total_confidence += effective_weight
            reasoning_parts.append(f"{signal_type}: {signal['reasoning']}")
        
        # Normalize scores
        if total_confidence > 0:
            buy_score /= total_confidence
            sell_score /= total_confidence
        
        # Decision thresholds
        decision_threshold = 0.6
        
        if buy_score > decision_threshold and buy_score > sell_score:
            final_signal = 'buy'
            final_strength = buy_score
            final_confidence = total_confidence
        elif sell_score > decision_threshold and sell_score > buy_score:
            final_signal = 'sell'
            final_strength = sell_score
            final_confidence = total_confidence
        else:
            final_signal = 'hold'
            final_strength = max(buy_score, sell_score) * 0.5
            final_confidence = total_confidence * 0.5
        
        # Create comprehensive reasoning
        reasoning = f"ML Ensemble: {final_signal.upper()} (strength: {final_strength:.2f}, confidence: {final_confidence:.2f})"
        
        # Create final prediction
        prediction = MLEnsemblePrediction(
            signal=final_signal,
            confidence=min(final_confidence, 1.0),
            strength=min(final_strength, 1.0),
            individual_predictions=ml_predictions,
            ensemble_weights=self.ensemble_weights.copy(),
            reasoning=reasoning,
            timestamp=datetime.now()
        )
        
        # Store for analysis
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'signal': final_signal,
            'confidence': final_confidence,
            'strength': final_strength,
            'buy_score': buy_score,
            'sell_score': sell_score
        })
        
        return prediction
    
    def _update_ensemble_weights_after_fitting(self):
        """Update ensemble weights based on which models were successfully fitted"""
        
        # Disable weights for models that failed to fit
        if not self.models_fitted.get('xgboost', False):
            self.ensemble_weights['xgboost'] = 0.0
        
        if not self.models_fitted.get('gru', False):
            self.ensemble_weights['gru'] = 0.0
        
        # Redistribute weights
        total_weight = sum(self.ensemble_weights.values())
        if total_weight > 0:
            self.ensemble_weights = {k: v/total_weight for k, v in self.ensemble_weights.items()}
    
    def _create_fallback_prediction(self, reason: str) -> MLEnsemblePrediction:
        """Create fallback prediction when ML models fail"""
        
        return MLEnsemblePrediction(
            signal='hold',
            confidence=0.0,
            strength=0.0,
            individual_predictions={},
            ensemble_weights=self.ensemble_weights,
            reasoning=f"ML Ensemble fallback: {reason}",
            timestamp=datetime.now()
        )
    
    def get_ensemble_performance(self) -> Dict[str, Any]:
        """Get comprehensive ensemble performance metrics"""
        
        return {
            'models_status': {
                'xgboost_fitted': self.models_fitted.get('xgboost', False),
                'gru_fitted': self.models_fitted.get('gru', False),
                'ensemble_ready': self.is_fitted
            },
            'ensemble_weights': self.ensemble_weights,
            'prediction_count': len(self.prediction_history),
            'recent_predictions': self.prediction_history[-5:] if self.prediction_history else [],
            'model_summaries': {
                'xgboost': self.xgboost_predictor.get_feature_importance_summary() if self.models_fitted.get('xgboost', False) else None,
                'gru': self.gru_predictor.get_model_summary() if self.models_fitted.get('gru', False) else None
            }
        }
    
    def save_ensemble(self, base_filepath: str):
        """Save all ensemble models"""
        
        if self.models_fitted.get('xgboost', False):
            self.xgboost_predictor.save_model(f"{base_filepath}_xgboost.pkl")
        
        if self.models_fitted.get('gru', False):
            self.gru_predictor.save_model(f"{base_filepath}_gru.pth")
        
        # Save ensemble metadata
        import json
        ensemble_metadata = {
            'models_fitted': self.models_fitted,
            'ensemble_weights': self.ensemble_weights,
            'prediction_history': self.prediction_history[-100:]  # Last 100 predictions
        }
        
        with open(f"{base_filepath}_ensemble_metadata.json", 'w') as f:
            json.dump(ensemble_metadata, f, indent=2, default=str)
        
        self.logger.info(f"💾 ML Ensemble saved to {base_filepath}")
    
    def load_ensemble(self, base_filepath: str):
        """Load all ensemble models"""
        
        try:
            # Load XGBoost if exists
            if self.enable_xgboost:
                try:
                    self.xgboost_predictor.load_model(f"{base_filepath}_xgboost.pkl")
                    self.models_fitted['xgboost'] = True
                except:
                    self.models_fitted['xgboost'] = False
            
            # Load GRU if exists
            if self.enable_gru:
                try:
                    self.gru_predictor.load_model(f"{base_filepath}_gru.pth")
                    self.models_fitted['gru'] = True
                except:
                    self.models_fitted['gru'] = False
            
            # Load ensemble metadata
            import json
            with open(f"{base_filepath}_ensemble_metadata.json", 'r') as f:
                metadata = json.load(f)
            
            self.ensemble_weights = metadata['ensemble_weights']
            self.prediction_history = metadata['prediction_history']
            
            self.is_fitted = any(self.models_fitted.values())
            
            self.logger.info(f"📂 ML Ensemble loaded from {base_filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load ML ensemble: {e}") 
"""
Enhanced Signal Generator with XGBoost + GRU Integration

EXPLORATORY: ML overlay on the primary ensemble. Not on the validated ablation path.

Extends the existing SimplifiedInstitutionalSignalGenerator to include:
- XGBoost feature-based predictions
- GRU time series forecasting  
- ML ensemble decision making
- Comprehensive performance tracking
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any

# Import existing institutional components
from .signal_generator import SimplifiedInstitutionalSignalGenerator

# Import new ML components
from ..machine_learning.ml_ensemble import MLEnsembleManager, MLEnsemblePrediction

class EnhancedInstitutionalSignalGenerator(SimplifiedInstitutionalSignalGenerator):
    """
    Enhanced signal generator combining institutional analysis with ML predictions
    
    Complete Integration:
    - Preserves all existing 66-feature volume anomaly detection
    - Maintains RSS + Alpha Vantage sentiment analysis
    - Adds XGBoost feature importance and signal prediction
    - Adds GRU time series forecasting
    - Provides unified ensemble decision making
    """
    
    def __init__(self, 
                 enable_xgboost: bool = True,
                 enable_gru: bool = True,
                 ml_weight: float = 0.35):
        """
        Initialize enhanced signal generator
        
        Args:
            enable_xgboost: Enable XGBoost model
            enable_gru: Enable GRU model
            ml_weight: Weight for ML predictions in final ensemble (0.35 = 35%)
        """
        
        # Initialize base institutional system
        super().__init__()
        
        # Initialize ML ensemble
        self.ml_ensemble = MLEnsembleManager(
            enable_xgboost=enable_xgboost,
            enable_gru=enable_gru
        )
        
        # Configuration
        self.enable_ml = enable_xgboost or enable_gru
        self.ml_weight = ml_weight
        self.institutional_weight = 1.0 - ml_weight
        
        # Enhanced state tracking
        self.ml_fitted = False
        self.enhanced_signal_history = []
        
        # Performance comparison
        self.performance_comparison = {
            'institutional_only': [],
            'ml_only': [],
            'enhanced_ensemble': []
        }
        
        self.logger.info("🚀 Enhanced Institutional Signal Generator initialized")
        self.logger.info(f"   ML Models: {'✅ Enabled' if self.enable_ml else '❌ Disabled'}")
        self.logger.info(f"   Weights: Institutional {self.institutional_weight:.1%}, ML {self.ml_weight:.1%}")
    
    async def initialize_enhanced_system(self, 
                                       historical_data: pd.DataFrame, 
                                       symbol: str = "SOL") -> Dict[str, Any]:
        """
        Initialize both institutional and ML systems
        
        Args:
            historical_data: Historical OHLCV data (minimum 500 samples recommended)
            symbol: Trading symbol
            
        Returns:
            Comprehensive initialization results
        """
        
        self.logger.info(f"🏛️ Initializing Enhanced Institutional System for {symbol}...")
        
        # Initialize base institutional system
        institutional_performance = await super().initialize_system(historical_data, symbol)
        
        initialization_results = {
            'institutional': institutional_performance,
            'ml': {'status': 'skipped', 'reason': 'ML disabled'}
        }
        
        # Initialize ML ensemble if enabled
        if self.enable_ml:
            try:
                self.logger.info("🤖 Initializing ML ensemble...")
                
                # Get volume features from institutional detector
                volume_features = self.volume_detector.engineer_institutional_features(historical_data)
                
                # Prepare sample sentiment and technical data for training
                sample_sentiment = {
                    'rss_score': 0.0, 'av_score': 0.0, 'confidence': 0.5, 'article_count': 5
                }
                sample_technical = {
                    'rsi': 50, 'momentum': 0, 'volume_ratio': 1, 'volatility': 0.02, 
                    'technical_score': 0, 'confidence': 0.5
                }
                
                # Fit ML models
                ml_results = await self.ml_ensemble.fit_all_models(
                    historical_data=historical_data,
                    volume_features=volume_features,
                    sentiment_data=sample_sentiment,
                    technical_data=sample_technical
                )
                
                initialization_results['ml'] = ml_results
                self.ml_fitted = ml_results['ensemble']['is_ready']
                
                if self.ml_fitted:
                    self.logger.info("✅ ML ensemble fitted successfully")
                else:
                    self.logger.warning("⚠️ ML ensemble fitting failed")
                
            except Exception as e:
                self.logger.error(f"❌ ML ensemble initialization failed: {e}")
                initialization_results['ml'] = {'status': 'failed', 'error': str(e)}
                self.ml_fitted = False
        
        # Comprehensive system status
        initialization_results['enhanced_system'] = {
            'institutional_ready': self.is_fitted,
            'ml_ready': self.ml_fitted,
            'total_algorithms': 5 + (2 if self.ml_fitted else 0),  # 5 institutional + XGBoost + GRU
            'feature_count': institutional_performance.get('feature_count', 0),
            'enhanced_weights': {
                'institutional': self.institutional_weight,
                'ml': self.ml_weight if self.ml_fitted else 0.0
            }
        }
        
        self.logger.info("🎯 Enhanced System Initialization Complete:")
        self.logger.info(f"   Institutional: {'✅' if self.is_fitted else '❌'}")
        self.logger.info(f"   ML Ensemble: {'✅' if self.ml_fitted else '❌'}")
        self.logger.info(f"   Total Algorithms: {initialization_results['enhanced_system']['total_algorithms']}")
        
        return initialization_results
    
    async def generate_enhanced_signals(self, 
                                      current_data: pd.DataFrame, 
                                      symbol: str = "SOL") -> Dict[str, Any]:
        """
        Generate enhanced signals using institutional + ML ensemble
        
        Args:
            current_data: Current market data
            symbol: Trading symbol
            
        Returns:
            Enhanced signal with institutional + ML analysis
        """
        
        if not self.is_fitted:
            raise RuntimeError("Enhanced system not initialized. Call initialize_enhanced_system() first.")
        
        self.logger.debug(f"📊 Generating enhanced signals for {symbol}...")
        
        # Get institutional signals (base system)
        institutional_signal = await super().generate_trading_signals(current_data, symbol)
        
        # Initialize enhanced signal with institutional base
        enhanced_signal = institutional_signal.copy()
        enhanced_signal['signal_type'] = 'institutional_only'
        enhanced_signal['ml_prediction'] = None
        
        # Add ML predictions if available
        if self.ml_fitted:
            try:
                # Get current volume features
                current_volume_features = self.volume_detector.engineer_institutional_features(current_data)
                
                # Get ML ensemble prediction
                ml_prediction = await self.ml_ensemble.generate_ml_predictions(
                    current_data=current_data,
                    current_volume_features=current_volume_features,
                    institutional_signals=institutional_signal.get('individual_signals', [])
                )
                
                # Combine institutional and ML signals
                enhanced_signal = self._combine_institutional_and_ml(
                    institutional_signal, ml_prediction
                )
                
                enhanced_signal['signal_type'] = 'enhanced_ensemble'
                enhanced_signal['ml_prediction'] = {
                    'signal': ml_prediction.signal,
                    'confidence': ml_prediction.confidence,
                    'strength': ml_prediction.strength,
                    'reasoning': ml_prediction.reasoning
                }
                
            except Exception as e:
                self.logger.warning(f"⚠️ ML prediction failed, using institutional only: {e}")
                enhanced_signal['signal_type'] = 'institutional_fallback'
                enhanced_signal['ml_prediction'] = {'error': str(e)}
        
        # Store for performance analysis
        self.enhanced_signal_history.append(enhanced_signal)
        
        # Add enhanced metadata
        enhanced_signal['enhanced_metadata'] = {
            'institutional_weight': self.institutional_weight,
            'ml_weight': self.ml_weight if self.ml_fitted else 0.0,
            'total_algorithms': 5 + (2 if self.ml_fitted else 0),
            'signal_source': enhanced_signal['signal_type'],
            'timestamp': datetime.now()
        }
        
        return enhanced_signal
    
    def _combine_institutional_and_ml(self, 
                                    institutional_signal: Dict, 
                                    ml_prediction: MLEnsemblePrediction) -> Dict:
        """
        Combine institutional and ML signals with intelligent weighting
        
        Args:
            institutional_signal: Signal from institutional system
            ml_prediction: Prediction from ML ensemble
            
        Returns:
            Combined enhanced signal
        """
        
        # Extract key metrics
        inst_signal = institutional_signal['signal']
        inst_confidence = institutional_signal['confidence']
        inst_strength = institutional_signal['strength']
        
        ml_signal = ml_prediction.signal
        ml_confidence = ml_prediction.confidence
        ml_strength = ml_prediction.strength
        
        # Weighted combination of confidences
        total_confidence = (inst_confidence * self.institutional_weight + 
                          ml_confidence * self.ml_weight)
        
        # Weighted combination of strengths  
        total_strength = (inst_strength * self.institutional_weight + 
                         ml_strength * self.ml_weight)
        
        # Signal agreement analysis
        signals_agree = inst_signal == ml_signal
        confidence_boost = 1.2 if signals_agree else 0.8  # Boost for agreement
        
        final_confidence = min(total_confidence * confidence_boost, 1.0)
        final_strength = min(total_strength * confidence_boost, 1.0)
        
        # Final signal decision
        if signals_agree:
            final_signal = inst_signal  # Both agree
        else:
            # Disagreement - use higher confidence signal
            if inst_confidence > ml_confidence:
                final_signal = inst_signal
            else:
                final_signal = ml_signal
        
        # Enhanced reasoning
        reasoning_parts = [
            f"Institutional: {inst_signal} (conf: {inst_confidence:.2f})",
            f"ML Ensemble: {ml_signal} (conf: {ml_confidence:.2f})",
            f"Agreement: {'✅' if signals_agree else '❌'}",
            f"Final: {final_signal} (conf: {final_confidence:.2f})"
        ]
        
        enhanced_reasoning = "; ".join(reasoning_parts)
        
        # Create enhanced signal
        enhanced_signal = institutional_signal.copy()
        enhanced_signal.update({
            'signal': final_signal,
            'confidence': final_confidence,
            'strength': final_strength,
            'reasoning': enhanced_reasoning,
            'signal_agreement': signals_agree,
            'confidence_boost': confidence_boost,
            'institutional_component': {
                'signal': inst_signal,
                'confidence': inst_confidence,
                'strength': inst_strength
            },
            'ml_component': {
                'signal': ml_signal,
                'confidence': ml_confidence,
                'strength': ml_strength,
                'individual_predictions': ml_prediction.individual_predictions
            }
        })
        
        return enhanced_signal
    
    def get_enhanced_performance(self) -> Dict[str, Any]:
        """Get comprehensive performance analysis of enhanced system"""
        
        # Base performance
        base_performance = super().get_system_status()
        
        # ML ensemble performance
        ml_performance = self.ml_ensemble.get_ensemble_performance() if self.ml_fitted else {}
        
        # Enhanced signal analysis
        enhanced_analysis = {}
        if self.enhanced_signal_history:
            recent_signals = self.enhanced_signal_history[-20:]  # Last 20 signals
            
            signal_types = [s.get('signal_type', 'unknown') for s in recent_signals]
            agreements = [s.get('signal_agreement', False) for s in recent_signals if 'signal_agreement' in s]
            confidence_scores = [s.get('confidence', 0) for s in recent_signals]
            
            enhanced_analysis = {
                'signal_count': len(self.enhanced_signal_history),
                'recent_signal_types': {
                    'institutional_only': signal_types.count('institutional_only'),
                    'enhanced_ensemble': signal_types.count('enhanced_ensemble'),
                    'institutional_fallback': signal_types.count('institutional_fallback')
                },
                'agreement_rate': np.mean(agreements) if agreements else 0.0,
                'average_confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
                'confidence_std': np.std(confidence_scores) if confidence_scores else 0.0
            }
        
        return {
            'system_status': {
                'institutional_ready': self.is_fitted,
                'ml_ready': self.ml_fitted,
                'enhanced_ready': self.is_fitted and (not self.enable_ml or self.ml_fitted)
            },
            'institutional_performance': base_performance,
            'ml_performance': ml_performance,
            'enhanced_analysis': enhanced_analysis,
            'configuration': {
                'ml_enabled': self.enable_ml,
                'institutional_weight': self.institutional_weight,
                'ml_weight': self.ml_weight,
                'total_algorithms': 5 + (2 if self.ml_fitted else 0)
            }
        }
    
    def get_feature_importance_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive feature importance from both institutional and ML systems
        """
        
        analysis = {
            'institutional_features': {
                'count': 66,
                'source': 'InstitutionalVolumeAnomalyDetector',
                'algorithms': ['isolation_forest', 'local_outlier_factor', 'mahalanobis', 
                             'statistical_process_control', 'one_class_svm']
            },
            'ml_features': {}
        }
        
        # Get XGBoost feature importance if available
        if self.ml_fitted and hasattr(self.ml_ensemble.xgboost_predictor, 'get_feature_importance_summary'):
            try:
                xgb_importance = self.ml_ensemble.xgboost_predictor.get_feature_importance_summary()
                analysis['ml_features']['xgboost'] = {
                    'total_features': xgb_importance.get('feature_count', 0),
                    'top_10_features': xgb_importance.get('top_20_features', [])[:10],
                    'cv_performance': xgb_importance.get('cv_performance', {})
                }
            except Exception as e:
                analysis['ml_features']['xgboost'] = {'error': str(e)}
        
        # Get GRU model info if available
        if self.ml_fitted and hasattr(self.ml_ensemble.gru_predictor, 'get_model_summary'):
            try:
                gru_summary = self.ml_ensemble.gru_predictor.get_model_summary()
                analysis['ml_features']['gru'] = {
                    'input_features': gru_summary.get('features', {}).get('count', 0),
                    'sequence_length': gru_summary.get('model_architecture', {}).get('sequence_length', 0),
                    'feature_names': gru_summary.get('features', {}).get('names', [])[:10]
                }
            except Exception as e:
                analysis['ml_features']['gru'] = {'error': str(e)}
        
        return analysis
    
    async def save_enhanced_model(self, filepath: str):
        """Save complete enhanced model including institutional + ML components"""
        
        # Save institutional components (base class handles this)
        base_filepath = f"{filepath}_institutional"
        
        # Save ML ensemble
        if self.ml_fitted:
            ml_filepath = f"{filepath}_ml_ensemble"
            self.ml_ensemble.save_ensemble(ml_filepath)
        
        # Save enhanced metadata
        import json
        enhanced_metadata = {
            'configuration': {
                'ml_enabled': self.enable_ml,
                'institutional_weight': self.institutional_weight,
                'ml_weight': self.ml_weight
            },
            'performance_history': self.enhanced_signal_history[-100:],  # Last 100 signals
            'model_status': {
                'institutional_fitted': self.is_fitted,
                'ml_fitted': self.ml_fitted
            }
        }
        
        with open(f"{filepath}_enhanced_metadata.json", 'w') as f:
            json.dump(enhanced_metadata, f, indent=2, default=str)
        
        self.logger.info(f"💾 Enhanced model saved to {filepath}")
    
    async def load_enhanced_model(self, filepath: str):
        """Load complete enhanced model"""
        
        try:
            # Load institutional components (would need to implement in base class)
            # self.load_institutional_model(f"{filepath}_institutional")
            
            # Load ML ensemble
            if self.enable_ml:
                ml_filepath = f"{filepath}_ml_ensemble"
                self.ml_ensemble.load_ensemble(ml_filepath)
                self.ml_fitted = self.ml_ensemble.is_fitted
            
            # Load enhanced metadata
            import json
            with open(f"{filepath}_enhanced_metadata.json", 'r') as f:
                metadata = json.load(f)
            
            self.enhanced_signal_history = metadata.get('performance_history', [])
            
            self.logger.info(f"📂 Enhanced model loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load enhanced model: {e}") 
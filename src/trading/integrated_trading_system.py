"""
Integrated Trading System with All Enhancements

This module brings together all improvements to fix the 100% HOLD signal issue
and integrate the enhanced components.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

# Import all enhanced components
from ..machine_learning.enhanced_ml_ensemble import EnhancedMLEnsembleManager
from ..sentiment_analysis.enhanced_sentiment_analyzer import EnhancedSentimentAnalyzer, SentimentAnalysisIntegration
from ..utils.robust_volume_detector import RobustVolumeAnomalyDetector, safe_volume_analysis
from ..utils.telegram_notifier import TelegramNotifier

logger = logging.getLogger(__name__)


@dataclass
class TradingSystemConfig:
    """Configuration for the integrated trading system"""
    # Trading parameters
    symbol: str = "SOL"
    trading_interval: int = 300  # 5 minutes
    position_size: float = 0.02  # 2% of portfolio per trade
    max_positions: int = 5
    
    # Signal generation thresholds (DYNAMIC)
    initial_buy_threshold: float = 0.6
    initial_sell_threshold: float = 0.6
    hold_zone: float = 0.2  # Hold if signal strength within this range of 0.5
    
    # Risk management
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.05  # 5% take profit
    max_drawdown: float = 0.15  # 15% max drawdown
    
    # ML configuration
    enable_ml: bool = True
    ml_device: str = "auto"  # Will use GPU if available
    continuous_learning: bool = True
    retrain_interval_hours: int = 24
    
    # Performance tuning
    enable_performance_monitoring: bool = True
    enable_telegram_alerts: bool = True
    
    # Market condition adaptation
    adapt_to_volatility: bool = True
    volatility_window: int = 20  # periods


class MarketConditionAnalyzer:
    """Analyzes market conditions to adapt trading behavior"""
    
    def __init__(self, window: int = 20):
        self.window = window
        self.volatility_history = []
        self.trend_history = []
        
    def analyze(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current market conditions"""
        
        if len(price_data) < self.window:
            return self._default_conditions()
        
        # Calculate volatility
        returns = price_data['close'].pct_change().dropna()
        current_volatility = returns.rolling(self.window).std().iloc[-1]
        
        # Calculate trend
        sma_short = price_data['close'].rolling(self.window // 2).mean().iloc[-1]
        sma_long = price_data['close'].rolling(self.window).mean().iloc[-1]
        trend = (sma_short - sma_long) / sma_long
        
        # Determine market regime
        if current_volatility < 0.01:
            regime = "low_volatility"
        elif current_volatility < 0.03:
            regime = "normal"
        else:
            regime = "high_volatility"
        
        # Trend classification
        if abs(trend) < 0.01:
            trend_type = "ranging"
        elif trend > 0.01:
            trend_type = "bullish"
        else:
            trend_type = "bearish"
        
        return {
            'volatility': float(current_volatility),
            'regime': regime,
            'trend': float(trend),
            'trend_type': trend_type,
            'confidence': min(len(price_data) / 100, 1.0)  # Confidence based on data
        }
    
    def _default_conditions(self) -> Dict[str, Any]:
        """Default market conditions when insufficient data"""
        return {
            'volatility': 0.02,
            'regime': 'normal',
            'trend': 0.0,
            'trend_type': 'ranging',
            'confidence': 0.0
        }


class SignalAggregator:
    """
    Aggregates signals from multiple sources and generates diverse trading signals
    This fixes the 100% HOLD issue by implementing dynamic thresholds
    """
    
    def __init__(self, config: TradingSystemConfig):
        self.config = config
        self.signal_history = []
        self.threshold_adjuster = DynamicThresholdAdjuster(config)
        
    def aggregate_signals(self, 
                         ml_prediction: Any,
                         sentiment_data: Dict[str, Any],
                         technical_signals: Dict[str, Any],
                         volume_anomaly: Dict[str, Any],
                         market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate all signals and generate final trading decision
        """
        
        # Collect all signal components
        signal_components = []
        
        # ML signals (if available)
        if ml_prediction and hasattr(ml_prediction, 'signal'):
            signal_components.append({
                'source': 'ml_ensemble',
                'signal': ml_prediction.signal,
                'strength': ml_prediction.strength,
                'confidence': ml_prediction.confidence,
                'weight': 0.35
            })
        
        # Sentiment signal
        if sentiment_data and 'signal' in sentiment_data:
            signal_components.append({
                'source': 'sentiment',
                'signal': sentiment_data['signal']['type'].lower(),
                'strength': sentiment_data['signal']['strength'],
                'confidence': sentiment_data['confidence'],
                'weight': 0.25
            })
        
        # Technical signals
        if technical_signals:
            signal_components.append({
                'source': 'technical',
                'signal': technical_signals.get('signal', 'hold'),
                'strength': technical_signals.get('strength', 0.5),
                'confidence': technical_signals.get('confidence', 0.5),
                'weight': 0.20
            })
        
        # Volume anomaly signal
        if volume_anomaly and volume_anomaly.get('is_anomaly'):
            # Volume anomalies often precede price movements
            signal_components.append({
                'source': 'volume_anomaly',
                'signal': 'buy' if volume_anomaly.get('volume_zscore', 0) > 0 else 'sell',
                'strength': min(abs(volume_anomaly.get('anomaly_score', 0)) / 3, 1.0),
                'confidence': volume_anomaly.get('confidence', 0.5),
                'weight': 0.20
            })
        
        # Aggregate signals
        aggregated_signal = self._weighted_signal_aggregation(signal_components)
        
        # Apply market condition adjustments
        adjusted_signal = self.threshold_adjuster.adjust_signal(
            aggregated_signal, market_conditions
        )
        
        # Add reasoning
        adjusted_signal['reasoning'] = self._generate_reasoning(
            adjusted_signal, signal_components, market_conditions
        )
        
        # Store in history
        self.signal_history.append({
            'timestamp': datetime.now(),
            'signal': adjusted_signal,
            'components': signal_components,
            'market_conditions': market_conditions
        })
        
        return adjusted_signal
    
    def _weighted_signal_aggregation(self, components: List[Dict]) -> Dict[str, Any]:
        """Perform weighted aggregation of signal components"""
        
        if not components:
            return {'signal': 'hold', 'strength': 0.0, 'confidence': 0.0}
        
        # Calculate weighted scores
        buy_score = 0.0
        sell_score = 0.0
        hold_score = 0.0
        total_weight = 0.0
        total_confidence = 0.0
        
        for component in components:
            weight = component['weight'] * component['confidence']
            signal = component['signal'].lower()
            strength = component['strength']
            
            if signal == 'buy':
                buy_score += strength * weight
            elif signal == 'sell':
                sell_score += strength * weight
            else:  # hold
                hold_score += strength * weight
            
            total_weight += weight
            total_confidence += component['confidence'] * component['weight']
        
        # Normalize
        if total_weight > 0:
            buy_score /= total_weight
            sell_score /= total_weight
            hold_score /= total_weight
            total_confidence /= sum(c['weight'] for c in components)
        
        # Determine signal
        scores = {'buy': buy_score, 'sell': sell_score, 'hold': hold_score}
        max_signal = max(scores, key=scores.get)
        
        return {
            'signal': max_signal,
            'strength': scores[max_signal],
            'confidence': total_confidence,
            'scores': scores
        }
    
    def _generate_reasoning(self, signal: Dict, components: List[Dict], 
                          market_conditions: Dict) -> str:
        """Generate human-readable reasoning for the signal"""
        
        reasons = []
        
        # Signal strength
        reasons.append(f"{signal['signal'].upper()} signal with {signal['strength']:.2f} strength")
        
        # Market conditions
        regime = market_conditions.get('regime', 'normal')
        trend = market_conditions.get('trend_type', 'ranging')
        reasons.append(f"Market: {regime} volatility, {trend} trend")
        
        # Component breakdown
        component_summary = []
        for comp in components:
            component_summary.append(
                f"{comp['source']}: {comp['signal']} ({comp['confidence']:.2f})"
            )
        reasons.append(f"Components: {', '.join(component_summary)}")
        
        return ". ".join(reasons)


class DynamicThresholdAdjuster:
    """
    Dynamically adjusts signal thresholds based on market conditions
    This is key to fixing the 100% HOLD issue
    """
    
    def __init__(self, config: TradingSystemConfig):
        self.config = config
        self.base_buy_threshold = config.initial_buy_threshold
        self.base_sell_threshold = config.initial_sell_threshold
        self.performance_history = []
        
    def adjust_signal(self, signal: Dict, market_conditions: Dict) -> Dict:
        """
        Adjust signal based on market conditions and performance
        """
        
        # Get dynamic thresholds
        buy_threshold, sell_threshold = self._calculate_dynamic_thresholds(market_conditions)
        
        # Get signal scores
        scores = signal.get('scores', {})
        buy_score = scores.get('buy', 0)
        sell_score = scores.get('sell', 0)
        hold_score = scores.get('hold', 0)
        
        # Apply thresholds with market-adaptive logic
        regime = market_conditions.get('regime', 'normal')
        
        # In high volatility, be more decisive
        if regime == 'high_volatility':
            # Lower thresholds to generate more signals
            buy_threshold *= 0.8
            sell_threshold *= 0.8
            
            # Reduce hold zone
            if abs(buy_score - sell_score) > 0.1:
                if buy_score > sell_score and buy_score > buy_threshold * 0.7:
                    signal['signal'] = 'buy'
                    signal['strength'] = buy_score
                elif sell_score > buy_score and sell_score > sell_threshold * 0.7:
                    signal['signal'] = 'sell'
                    signal['strength'] = sell_score
        
        # In low volatility, look for smaller edges
        elif regime == 'low_volatility':
            # Even lower thresholds for low volatility
            buy_threshold *= 0.6
            sell_threshold *= 0.6
            
            if buy_score > buy_threshold and buy_score > sell_score * 1.2:
                signal['signal'] = 'buy'
                signal['strength'] = buy_score
            elif sell_score > sell_threshold and sell_score > buy_score * 1.2:
                signal['signal'] = 'sell'
                signal['strength'] = sell_score
        
        # Normal conditions
        else:
            if buy_score > buy_threshold and buy_score > sell_score:
                signal['signal'] = 'buy'
                signal['strength'] = buy_score
            elif sell_score > sell_threshold and sell_score > buy_score:
                signal['signal'] = 'sell'
                signal['strength'] = sell_score
        
        # Add threshold info to signal
        signal['thresholds'] = {
            'buy': buy_threshold,
            'sell': sell_threshold,
            'regime': regime
        }
        
        return signal
    
    def _calculate_dynamic_thresholds(self, market_conditions: Dict) -> Tuple[float, float]:
        """
        Calculate dynamic thresholds based on market conditions
        """
        
        volatility = market_conditions.get('volatility', 0.02)
        trend_type = market_conditions.get('trend_type', 'ranging')
        
        # Base thresholds
        buy_threshold = self.base_buy_threshold
        sell_threshold = self.base_sell_threshold
        
        # Adjust for volatility
        if volatility < 0.01:  # Low volatility
            # Lower thresholds to catch smaller moves
            buy_threshold *= 0.8
            sell_threshold *= 0.8
        elif volatility > 0.03:  # High volatility
            # Slightly lower thresholds to be more active
            buy_threshold *= 0.9
            sell_threshold *= 0.9
        
        # Adjust for trend
        if trend_type == 'bullish':
            buy_threshold *= 0.9  # Easier to buy in uptrend
            sell_threshold *= 1.1  # Harder to sell
        elif trend_type == 'bearish':
            buy_threshold *= 1.1  # Harder to buy in downtrend
            sell_threshold *= 0.9  # Easier to sell
        
        # Performance-based adjustment
        if len(self.performance_history) > 10:
            recent_performance = self.performance_history[-10:]
            win_rate = sum(1 for p in recent_performance if p['profitable']) / len(recent_performance)
            
            if win_rate < 0.4:  # Poor performance
                # Be more selective
                buy_threshold *= 1.1
                sell_threshold *= 1.1
            elif win_rate > 0.6:  # Good performance
                # Be more aggressive
                buy_threshold *= 0.95
                sell_threshold *= 0.95
        
        return buy_threshold, sell_threshold
    
    def update_performance(self, signal_time: datetime, signal_type: str, 
                         entry_price: float, exit_price: float):
        """Update performance history for threshold adjustment"""
        
        self.performance_history.append({
            'timestamp': signal_time,
            'signal': signal_type,
            'entry': entry_price,
            'exit': exit_price,
            'profitable': (exit_price > entry_price) if signal_type == 'buy' else (exit_price < entry_price)
        })
        
        # Keep last 100 trades
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]


class IntegratedTradingSystem:
    """
    Main integrated trading system that brings everything together
    """
    
    def __init__(self, config: TradingSystemConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
        
        # Trading state
        self.is_running = False
        self.current_position = None
        self.trade_history = []
        
        self.logger.info("Integrated Trading System initialized")
        
    def _initialize_components(self):
        """Initialize all system components"""
        
        # ML Ensemble
        self.ml_ensemble = EnhancedMLEnsembleManager(
            enable_xgboost=True,
            enable_gru=True,
            enable_maml=True,
            device=self.config.ml_device,
            continuous_learning=self.config.continuous_learning,
            retrain_interval_hours=self.config.retrain_interval_hours
        )
        
        # Enhanced sentiment analyzer
        self.sentiment_analyzer = EnhancedSentimentAnalyzer(
            device=self.config.ml_device,
            use_ml=self.config.enable_ml
        )
        
        # Robust volume detector
        self.volume_detector = RobustVolumeAnomalyDetector(
            min_samples=100,
            warmup_mode=True
        )
        
        # Multi-exchange manager (configure with your credentials)
        self.exchange_manager = None  # Initialize with your exchange configs
        
        # Signal aggregator
        self.signal_aggregator = SignalAggregator(self.config)
        
        # Market condition analyzer
        self.market_analyzer = MarketConditionAnalyzer()
        
        # Telegram notifier
        self.notifier = TelegramNotifier() if self.config.enable_telegram_alerts else None
    
    async def start_trading(self):
        """Start the main trading loop"""
        
        self.is_running = True
        self.logger.info("Trading system started")
        
        # Initial model training
        await self._initial_training()
        
        # Main trading loop
        while self.is_running:
            try:
                await self._trading_iteration()
                await asyncio.sleep(self.config.trading_interval)
                
            except Exception as e:
                self.logger.error(f"Trading iteration error: {e}")
                await self._handle_error(e)
                await asyncio.sleep(60)  # Wait before retry
    
    async def _trading_iteration(self):
        """Single trading iteration"""
        
        start_time = datetime.now()
        
        # 1. Fetch market data
        market_data = await self._fetch_market_data()
        if not market_data:
            return
        
        # 2. Extract volume features
        volume_analysis = await safe_volume_analysis(market_data, self.volume_detector)
        volume_features = volume_analysis.get('features', pd.DataFrame())
        
        # 3. Get sentiment analysis
        news_data = await self._fetch_news_data()
        sentiment_data = await self.sentiment_analyzer.analyze_news_sentiment(news_data)
        
        # 4. Get ML predictions
        ml_prediction = None
        if self.config.enable_ml and len(volume_features) > 0:
            ml_prediction = await self.ml_ensemble.generate_ml_predictions(
                market_data, 
                volume_features,
                []  # Institutional signals placeholder
            )
        
        # 5. Get technical signals
        technical_signals = self._calculate_technical_signals(market_data)
        
        # 6. Analyze market conditions
        market_conditions = self.market_analyzer.analyze(market_data)
        
        # 7. Aggregate all signals
        final_signal = self.signal_aggregator.aggregate_signals(
            ml_prediction,
            {'signal': sentiment_data, 'confidence': sentiment_data.confidence},
            technical_signals,
            volume_analysis.get('anomalies', {}),
            market_conditions
        )
        
        # 8. Log signal
        self.logger.info(
            f"Signal: {final_signal['signal'].upper()} "
            f"(strength={final_signal['strength']:.2f}, "
            f"confidence={final_signal['confidence']:.2f})"
        )
        
        # 9. Execute trade if needed
        if final_signal['signal'] != 'hold' and final_signal['confidence'] > 0.6:
            await self._execute_trade(final_signal, market_data)
    
    async def _initial_training(self):
        """Perform initial model training"""
        
        self.logger.info("Starting initial model training...")
        
        # Fetch historical data
        historical_data = await self._fetch_historical_data(periods=1000)
        
        if len(historical_data) < 200:
            self.logger.warning("Insufficient historical data for training")
            return
        
        # Extract features
        volume_features = self.volume_detector.extract_features(historical_data)
        
        # Placeholder sentiment/technical data
        sentiment_data = {'rss_score': 0.0, 'av_score': 0.0, 'confidence': 0.5, 'article_count': 0}
        technical_data = {'rsi': 50.0, 'momentum': 0.0, 'volume_ratio': 1.0, 'volatility': 0.02}
        
        # Train models
        training_results = await self.ml_ensemble.fit_all_models(
            historical_data,
            volume_features,
            sentiment_data,
            technical_data
        )
        
        self.logger.info(f"Training completed: {training_results['ensemble']}")
    
    async def _fetch_market_data(self) -> pd.DataFrame:
        """Fetch current market data"""
        # Placeholder - implement with your exchange manager
        return pd.DataFrame()
    
    async def _fetch_historical_data(self, periods: int) -> pd.DataFrame:
        """Fetch historical market data"""
        # Placeholder - implement with your exchange manager
        return pd.DataFrame()
    
    async def _fetch_news_data(self) -> List[Dict]:
        """Fetch news data for sentiment analysis"""
        # Placeholder - implement with your news sources
        return []
    
    def _calculate_technical_signals(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators and signals"""
        
        if len(market_data) < 50:
            return {'signal': 'hold', 'strength': 0.5, 'confidence': 0.0}
        
        # RSI
        rsi = self._calculate_rsi(market_data['close'])
        
        # Moving averages
        sma_20 = market_data['close'].rolling(20).mean().iloc[-1]
        sma_50 = market_data['close'].rolling(50).mean().iloc[-1]
        current_price = market_data['close'].iloc[-1]
        
        # Generate signal
        signal = 'hold'
        strength = 0.5
        
        if rsi < 30 and current_price > sma_20:
            signal = 'buy'
            strength = 0.7
        elif rsi > 70 and current_price < sma_20:
            signal = 'sell'
            strength = 0.7
        elif sma_20 > sma_50 and current_price > sma_20:
            signal = 'buy'
            strength = 0.6
        elif sma_20 < sma_50 and current_price < sma_20:
            signal = 'sell'
            strength = 0.6
        
        return {
            'signal': signal,
            'strength': strength,
            'confidence': 0.7,
            'rsi': rsi,
            'sma_20': sma_20,
            'sma_50': sma_50
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1]
    
    async def _execute_trade(self, signal: Dict, market_data: pd.DataFrame):
        """Execute trade based on signal"""
        
        current_price = market_data['close'].iloc[-1]
        
        # Log trade attempt
        self.logger.info(f"Executing {signal['signal']} trade at {current_price}")
        
        # Send notification
        if self.config.enable_telegram_alerts and self.notifier:
            self.notifier.send_trade_notification(
                signal['signal'].upper(),
                self.config.symbol,
                current_price,
                self.config.position_size
            )
        
        # Record trade
        self.trade_history.append({
            'timestamp': datetime.now(),
            'signal': signal,
            'price': current_price,
            'size': self.config.position_size
        })
    
    async def _handle_error(self, error: Exception):
        """Handle system errors"""
        
        error_msg = f"System error: {error}"
        self.logger.error(error_msg)
        
        if self.config.enable_telegram_alerts and self.notifier:
            self.notifier.send_alert(
                "Trading System Error",
                error_msg,
                "ERROR"
            )
    
    async def stop_trading(self):
        """Stop trading system"""
        
        self.is_running = False
        self.logger.info("Trading system stopped")


# Main entry point
async def main():
    """Main entry point for the trading system"""
    
    # Configure logging
    from ..utils.logging_config import setup_logging
    setup_logging(log_level=logging.INFO)
    
    # Create configuration
    config = TradingSystemConfig(
        symbol="SOL",
        trading_interval=300,  # 5 minutes
        enable_ml=True,
        ml_device="auto",  # Will use GPU if available
        continuous_learning=True,
        adapt_to_volatility=True
    )
    
    # Create and start system
    trading_system = IntegratedTradingSystem(config)
    
    try:
        await trading_system.start_trading()
    except KeyboardInterrupt:
        logger.info("Shutdown requested...")
        await trading_system.stop_trading()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        await trading_system.stop_trading()


if __name__ == "__main__":
    asyncio.run(main()) 
"""
Ensemble Trading Strategy for Freqtrade

Multi-signal trading strategy combining:
- Volume anomaly detection
- Sentiment analysis signals
- Technical indicators
- Dynamic risk management with Kelly Criterion
- Regime detection using Hidden Markov Models

Mathematical foundations:
- Kelly Criterion: f* = (μ-r)/σ² - λVar(μ)/2σ²
- Sharpe Ratio optimization
- Information Ratio calculation
- Regime probability estimation

References:
- Kelly Jr. (1956): "A New Interpretation of Information Rate"
- Markowitz (1952): "Portfolio Selection"
- Hamilton (1989): "A New Approach to Economic Analysis of Nonstationary Time Series"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import ta
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, BooleanParameter
from freqtrade.strategy.interface import SellCheckTuple
from pandas import DataFrame
import logging
from datetime import datetime, timedelta
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy import stats
from hmmlearn import hmm
import warnings

warnings.filterwarnings('ignore')

# Import our custom modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from technical_analysis.volume_anomaly_detection import VolumeAnomalyDetector
    from sentiment_analysis.twitter_collector import TwitterSentimentCollector
except ImportError:
    # Fallback for testing
    VolumeAnomalyDetector = None
    TwitterSentimentCollector = None

class EnsembleStrategy(IStrategy):
    """
    Advanced ensemble trading strategy for cryptocurrency markets.
    
    Combines multiple signal sources:
    1. Volume anomaly detection using mathematical models
    2. Sentiment analysis from social media
    3. Traditional technical indicators
    4. Market regime detection
    5. Dynamic position sizing using Kelly Criterion
    
    Mathematical Framework:
    - Signal fusion using weighted ensemble
    - Risk-adjusted position sizing
    - Regime-aware parameter adaptation
    - Information-theoretic signal validation
    """
    
    # Strategy metadata
    INTERFACE_VERSION = 3
    can_short = False
    
    # Minimal ROI table
    minimal_roi = {
        "0": 0.05,    # 5% initial target
        "40": 0.03,   # 3% after 40 minutes
        "120": 0.01,  # 1% after 2 hours
        "240": 0.005  # 0.5% after 4 hours
    }
    
    # Stoploss
    stoploss = -0.15  # -15% stop loss
    
    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.05
    trailing_only_offset_is_reached = True
    
    # Timeframe
    timeframe = '5m'
    
    # Startup candle count
    startup_candle_count: int = 200
    
    # Strategy parameters (optimizable)
    # Volume anomaly parameters
    volume_anomaly_threshold = DecimalParameter(0.5, 0.9, default=0.7, decimals=2, space="buy")
    volume_anomaly_weight = DecimalParameter(0.1, 0.5, default=0.3, decimals=2, space="buy")
    
    # Sentiment parameters
    sentiment_threshold = DecimalParameter(0.1, 0.8, default=0.4, decimals=2, space="buy")
    sentiment_weight = DecimalParameter(0.1, 0.5, default=0.25, decimals=2, space="buy")
    
    # Technical indicator parameters
    rsi_buy = IntParameter(20, 40, default=30, space="buy")
    rsi_sell = IntParameter(70, 90, default=80, space="sell")
    ema_short = IntParameter(8, 20, default=12, space="buy")
    ema_long = IntParameter(21, 50, default=26, space="buy")
    
    # Risk management parameters
    max_position_size = DecimalParameter(0.05, 0.30, default=0.15, decimals=2, space="buy")
    kelly_fraction = DecimalParameter(0.1, 0.8, default=0.4, decimals=2, space="buy")
    
    # Ensemble parameters
    signal_threshold = DecimalParameter(0.3, 0.8, default=0.5, decimals=2, space="buy")
    confidence_threshold = DecimalParameter(0.6, 0.9, default=0.75, decimals=2, space="buy")
    
    def __init__(self, config: dict) -> None:
        """Initialize the ensemble strategy."""
        super().__init__(config)
        
        # Initialize custom analyzers
        self.volume_detector = None
        self.sentiment_collector = None
        
        # Initialize ML models
        self.regime_model = hmm.GaussianHMM(n_components=3, random_state=42)
        self.signal_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
        # Model states
        self.models_fitted = False
        self.regime_fitted = False
        
        # Signal history for model training
        self.signal_history = []
        self.performance_history = []
        
        # Current market regime (0: bear, 1: sideways, 2: bull)
        self.current_regime = 1
        self.regime_probabilities = [0.33, 0.34, 0.33]
        
        # Performance tracking
        self.trade_history = []
        self.cumulative_returns = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize external data sources
        self._setup_external_sources()
    
    def _setup_external_sources(self):
        """Setup external data sources for sentiment and volume analysis."""
        try:
            # Initialize volume anomaly detector
            if VolumeAnomalyDetector:
                self.volume_detector = VolumeAnomalyDetector(
                    contamination=0.05,
                    enable_feature_selection=True
                )
            
            # Initialize sentiment collector (requires API keys)
            # This would need to be configured with actual credentials
            # self.sentiment_collector = TwitterSentimentCollector(...)
            
        except Exception as e:
            self.logger.warning(f"Could not initialize external sources: {e}")
    
    def informative_pairs(self):
        """Define additional informative pairs for analysis."""
        return [
            ("BTC/USDT", "1h"),
            ("ETH/USDT", "1h"),
            # Add more pairs for cross-asset analysis
        ]
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate all indicators including custom ensemble signals.
        
        This is where we combine volume anomalies, sentiment, and technical indicators.
        """
        
        # === BASIC TECHNICAL INDICATORS ===
        
        # Moving averages
        dataframe['ema_short'] = ta.trend.ema_indicator(dataframe['close'], window=self.ema_short.value)
        dataframe['ema_long'] = ta.trend.ema_indicator(dataframe['close'], window=self.ema_long.value)
        dataframe['sma_20'] = ta.trend.sma_indicator(dataframe['close'], window=20)
        dataframe['sma_50'] = ta.trend.sma_indicator(dataframe['close'], window=50)
        
        # Momentum indicators
        dataframe['rsi'] = ta.momentum.rsi(dataframe['close'], window=14)
        dataframe['macd'] = ta.trend.macd(dataframe['close'])
        dataframe['macdsignal'] = ta.trend.macd_signal(dataframe['close'])
        dataframe['macdhist'] = ta.trend.macd_diff(dataframe['close'])
        dataframe['stoch_k'] = ta.momentum.stoch(dataframe['high'], dataframe['low'], dataframe['close'])
        dataframe['stoch_d'] = ta.momentum.stoch_signal(dataframe['high'], dataframe['low'], dataframe['close'])
        
        # Volatility indicators
        bb = ta.volatility.BollingerBands(dataframe['close'])
        dataframe['bb_upper'] = bb.bollinger_hband()
        dataframe['bb_middle'] = bb.bollinger_mavg()
        dataframe['bb_lower'] = bb.bollinger_lband()
        dataframe['atr'] = ta.volatility.average_true_range(dataframe['high'], dataframe['low'], dataframe['close'], window=14)
        
        # Volume indicators
        dataframe['volume_sma'] = ta.trend.sma_indicator(dataframe['volume'], window=20)
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma']
        dataframe['ad'] = ta.volume.acc_dist_index(dataframe['high'], dataframe['low'], dataframe['close'], dataframe['volume'])
        dataframe['obv'] = ta.volume.on_balance_volume(dataframe['close'], dataframe['volume'])
        
        # === CUSTOM ENSEMBLE INDICATORS ===
        
        # Volume anomaly detection signal
        dataframe['volume_anomaly_score'] = self._calculate_volume_anomaly_signal(dataframe)
        
        # Sentiment signal
        dataframe['sentiment_score'] = self._calculate_sentiment_signal(dataframe, metadata)
        
        # Market regime detection
        dataframe['market_regime'] = self._detect_market_regime(dataframe)
        
        # Technical signal fusion
        dataframe['technical_signal'] = self._calculate_technical_signal(dataframe)
        
        # === ENSEMBLE SIGNAL CALCULATION ===
        
        # Weighted ensemble score
        dataframe['ensemble_score'] = (
            self.volume_anomaly_weight.value * dataframe['volume_anomaly_score'] +
            self.sentiment_weight.value * dataframe['sentiment_score'] +
            (1 - self.volume_anomaly_weight.value - self.sentiment_weight.value) * dataframe['technical_signal']
        )
        
        # Signal confidence calculation
        dataframe['signal_confidence'] = self._calculate_signal_confidence(dataframe)
        
        # Dynamic position sizing using Kelly Criterion
        dataframe['kelly_position_size'] = self._calculate_kelly_position_size(dataframe)
        
        # === RISK MANAGEMENT INDICATORS ===
        
        # Volatility-adjusted signals
        dataframe['vol_adjusted_signal'] = dataframe['ensemble_score'] / (dataframe['atr'] / dataframe['close'])
        
        # Trend strength
        dataframe['trend_strength'] = abs(dataframe['ema_short'] - dataframe['ema_long']) / dataframe['ema_long']
        
        return dataframe
    
    def _calculate_volume_anomaly_signal(self, dataframe: DataFrame) -> pd.Series:
        """
        Calculate volume anomaly signal using mathematical detection methods.
        
        Returns normalized signal between -1 and 1.
        """
        if self.volume_detector is None or len(dataframe) < 100:
            return pd.Series(0.0, index=dataframe.index)
        
        try:
            # Fit detector if not already done
            if not self.volume_detector.is_fitted and len(dataframe) >= 200:
                self.volume_detector.fit(dataframe[['open', 'high', 'low', 'close', 'volume']])
            
            if self.volume_detector.is_fitted:
                # Detect anomalies
                recent_data = dataframe[['open', 'high', 'low', 'close', 'volume']].tail(50)
                anomaly_results = self.volume_detector.detect_anomalies(recent_data, method='ensemble')
                
                # Convert to signal
                signal_values = []
                for result in anomaly_results:
                    if result.is_anomaly:
                        # Positive anomalies (unusual high volume) can indicate buying opportunity
                        signal_values.append(min(1.0, result.anomaly_score))
                    else:
                        signal_values.append(0.0)
                
                # Pad with zeros for earlier periods
                full_signal = [0.0] * (len(dataframe) - len(signal_values)) + signal_values
                return pd.Series(full_signal, index=dataframe.index)
            
        except Exception as e:
            self.logger.warning(f"Volume anomaly calculation failed: {e}")
        
        return pd.Series(0.0, index=dataframe.index)
    
    def _calculate_sentiment_signal(self, dataframe: DataFrame, metadata: dict) -> pd.Series:
        """
        Calculate sentiment signal from social media data.
        
        Returns normalized signal between -1 and 1.
        """
        if self.sentiment_collector is None:
            return pd.Series(0.0, index=dataframe.index)
        
        try:
            # Extract cryptocurrency symbol from metadata
            pair = metadata.get('pair', '')
            crypto = pair.split('/')[0].lower() if '/' in pair else 'bitcoin'
            
            # Get recent sentiment summary
            sentiment_summary = self.sentiment_collector.get_sentiment_summary(
                crypto=crypto, 
                hours=4  # Look back 4 hours for sentiment
            )
            
            if 'average_sentiment' in sentiment_summary:
                # Use average sentiment as signal
                avg_sentiment = sentiment_summary['average_sentiment']
                
                # Apply confidence weighting
                total_tweets = sentiment_summary.get('total_tweets', 0)
                confidence_multiplier = min(1.0, total_tweets / 100)  # More tweets = higher confidence
                
                signal_value = avg_sentiment * confidence_multiplier
                return pd.Series(signal_value, index=dataframe.index)
            
        except Exception as e:
            self.logger.warning(f"Sentiment calculation failed: {e}")
        
        return pd.Series(0.0, index=dataframe.index)
    
    def _detect_market_regime(self, dataframe: DataFrame) -> pd.Series:
        """
        Detect market regime using Hidden Markov Model.
        
        Returns regime indicator: 0 (bear), 1 (sideways), 2 (bull)
        """
        if len(dataframe) < 100:
            return pd.Series(1, index=dataframe.index)  # Default to sideways
        
        try:
            # Prepare features for regime detection
            returns = dataframe['close'].pct_change().fillna(0)
            volatility = returns.rolling(20).std().fillna(0)
            volume_ratio = (dataframe['volume'] / dataframe['volume'].rolling(20).mean()).fillna(1)
            
            # Combine features
            features = np.column_stack([returns, volatility, volume_ratio])
            features = features[~np.isnan(features).any(axis=1)]  # Remove NaN rows
            
            if len(features) < 50:
                return pd.Series(1, index=dataframe.index)
            
            # Fit HMM if not already done
            if not self.regime_fitted and len(features) >= 100:
                self.regime_model.fit(features)
                self.regime_fitted = True
            
            if self.regime_fitted:
                # Predict regimes
                regimes = self.regime_model.predict(features)
                
                # Map to our regime convention (0: bear, 1: sideways, 2: bull)
                regime_mapping = self._interpret_hmm_regimes(features, regimes)
                mapped_regimes = [regime_mapping[r] for r in regimes]
                
                # Pad with default regime for earlier periods
                full_regimes = [1] * (len(dataframe) - len(mapped_regimes)) + mapped_regimes
                
                # Update current regime
                if mapped_regimes:
                    self.current_regime = mapped_regimes[-1]
                
                return pd.Series(full_regimes, index=dataframe.index)
            
        except Exception as e:
            self.logger.warning(f"Regime detection failed: {e}")
        
        return pd.Series(1, index=dataframe.index)
    
    def _interpret_hmm_regimes(self, features: np.ndarray, regimes: np.ndarray) -> Dict[int, int]:
        """
        Interpret HMM regimes based on feature characteristics.
        
        Args:
            features: Feature matrix used for HMM
            regimes: HMM predicted regimes
            
        Returns:
            Mapping from HMM regime to our regime convention
        """
        regime_stats = {}
        
        for regime_id in np.unique(regimes):
            regime_mask = regimes == regime_id
            regime_features = features[regime_mask]
            
            avg_return = np.mean(regime_features[:, 0])
            avg_volatility = np.mean(regime_features[:, 1])
            
            regime_stats[regime_id] = {
                'return': avg_return,
                'volatility': avg_volatility
            }
        
        # Sort regimes by average return
        sorted_regimes = sorted(regime_stats.keys(), key=lambda x: regime_stats[x]['return'])
        
        # Map to our convention: lowest return = bear (0), middle = sideways (1), highest = bull (2)
        mapping = {}
        for i, regime_id in enumerate(sorted_regimes):
            mapping[regime_id] = i
        
        return mapping
    
    def _calculate_technical_signal(self, dataframe: DataFrame) -> pd.Series:
        """
        Calculate technical analysis signal using multiple indicators.
        
        Returns normalized signal between -1 and 1.
        """
        signals = []
        
        # RSI signal
        rsi_signal = np.where(
            dataframe['rsi'] < self.rsi_buy.value, 1.0,
            np.where(dataframe['rsi'] > self.rsi_sell.value, -1.0, 0.0)
        )
        signals.append(rsi_signal)
        
        # Moving average crossover signal
        ma_signal = np.where(
            dataframe['ema_short'] > dataframe['ema_long'], 0.5,
            np.where(dataframe['ema_short'] < dataframe['ema_long'], -0.5, 0.0)
        )
        signals.append(ma_signal)
        
        # MACD signal
        macd_signal = np.where(
            (dataframe['macd'] > dataframe['macdsignal']) & (dataframe['macd'].shift(1) <= dataframe['macdsignal'].shift(1)), 0.3,
            np.where((dataframe['macd'] < dataframe['macdsignal']) & (dataframe['macd'].shift(1) >= dataframe['macdsignal'].shift(1)), -0.3, 0.0)
        )
        signals.append(macd_signal)
        
        # Bollinger Bands signal
        bb_signal = np.where(
            dataframe['close'] < dataframe['bb_lower'], 0.4,
            np.where(dataframe['close'] > dataframe['bb_upper'], -0.4, 0.0)
        )
        signals.append(bb_signal)
        
        # Volume confirmation
        volume_conf = np.where(dataframe['volume_ratio'] > 1.2, 0.2, 0.0)
        signals.append(volume_conf)
        
        # Combine signals
        combined_signal = np.mean(signals, axis=0)
        return pd.Series(combined_signal, index=dataframe.index)
    
    def _calculate_signal_confidence(self, dataframe: DataFrame) -> pd.Series:
        """
        Calculate confidence in ensemble signal based on agreement between indicators.
        
        Returns confidence score between 0 and 1.
        """
        # Calculate agreement between different signal components
        signals = [
            dataframe['volume_anomaly_score'],
            dataframe['sentiment_score'],
            dataframe['technical_signal']
        ]
        
        # Remove any NaN or infinite values
        clean_signals = []
        for signal in signals:
            clean_signal = signal.fillna(0).replace([np.inf, -np.inf], 0)
            clean_signals.append(clean_signal)
        
        # Calculate standard deviation of signals (lower = higher agreement)
        signal_matrix = np.column_stack(clean_signals)
        signal_std = np.std(signal_matrix, axis=1)
        
        # Convert to confidence (inverse of standard deviation, normalized)
        max_std = 2.0  # Maximum expected standard deviation
        confidence = 1.0 - np.clip(signal_std / max_std, 0, 1)
        
        return pd.Series(confidence, index=dataframe.index)
    
    def _calculate_kelly_position_size(self, dataframe: DataFrame) -> pd.Series:
        """
        Calculate position size using Kelly Criterion.
        
        Kelly formula: f* = (μ-r)/σ² where:
        - μ = expected return
        - r = risk-free rate
        - σ² = variance of returns
        
        Modified for trading: f* = (win_rate * avg_win - loss_rate * avg_loss) / avg_win
        """
        if len(self.performance_history) < 10:
            # Use default position size if insufficient history
            return pd.Series(self.max_position_size.value * 0.5, index=dataframe.index)
        
        try:
            # Calculate historical performance metrics
            returns = np.array(self.performance_history)
            wins = returns[returns > 0]
            losses = returns[returns < 0]
            
            if len(wins) == 0 or len(losses) == 0:
                return pd.Series(self.max_position_size.value * 0.5, index=dataframe.index)
            
            win_rate = len(wins) / len(returns)
            avg_win = np.mean(wins)
            avg_loss = abs(np.mean(losses))
            
            # Kelly fraction calculation
            kelly_f = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            
            # Apply safety factor and constraints
            safe_kelly = kelly_f * self.kelly_fraction.value
            position_size = np.clip(safe_kelly, 0.01, self.max_position_size.value)
            
            return pd.Series(position_size, index=dataframe.index)
            
        except Exception as e:
            self.logger.warning(f"Kelly calculation failed: {e}")
            return pd.Series(self.max_position_size.value * 0.5, index=dataframe.index)
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate buy signals based on ensemble analysis.
        
        Entry conditions:
        1. Ensemble score above threshold
        2. Signal confidence above threshold
        3. Regime-appropriate conditions
        4. Risk management filters
        """
        
        # Base ensemble condition
        ensemble_condition = (
            (dataframe['ensemble_score'] > self.signal_threshold.value) &
            (dataframe['signal_confidence'] > self.confidence_threshold.value)
        )
        
        # Technical confirmation
        technical_condition = (
            (dataframe['rsi'] < 70) &  # Not overbought
            (dataframe['ema_short'] > dataframe['ema_long']) &  # Uptrend
            (dataframe['volume_ratio'] > 0.8)  # Decent volume
        )
        
        # Regime-based conditions
        regime_condition = dataframe['market_regime'] != 0  # Not in bear market
        
        # Risk management filters
        risk_condition = (
            (dataframe['atr'] / dataframe['close'] < 0.05) &  # Not too volatile
            (dataframe['kelly_position_size'] > 0.02)  # Positive Kelly sizing
        )
        
        # Combine all conditions
        dataframe.loc[
            ensemble_condition & 
            technical_condition & 
            regime_condition & 
            risk_condition, 
            'enter_long'
        ] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate sell signals based on ensemble analysis.
        
        Exit conditions:
        1. Ensemble score turns negative
        2. Technical indicators suggest exit
        3. Risk management triggers
        """
        
        # Ensemble-based exit
        ensemble_exit = (
            (dataframe['ensemble_score'] < -self.signal_threshold.value) |
            (dataframe['signal_confidence'] < 0.3)
        )
        
        # Technical exit conditions
        technical_exit = (
            (dataframe['rsi'] > self.rsi_sell.value) |  # Overbought
            (dataframe['ema_short'] < dataframe['ema_long']) |  # Trend reversal
            (dataframe['close'] > dataframe['bb_upper'])  # Above upper Bollinger Band
        )
        
        # Risk management exit
        risk_exit = (
            (dataframe['atr'] / dataframe['close'] > 0.08) |  # High volatility
            (dataframe['market_regime'] == 0)  # Bear market
        )
        
        # Combine exit conditions
        dataframe.loc[
            ensemble_exit | technical_exit | risk_exit,
            'exit_long'
        ] = 1
        
        return dataframe
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                          proposed_stake: float, **kwargs) -> float:
        """
        Customize stake amount based on Kelly Criterion and risk management.
        """
        try:
            # Get current dataframe
            dataframe = self.dp.get_pair_dataframe(pair, self.timeframe)
            
            if dataframe.empty:
                return proposed_stake * 0.5
            
            # Get Kelly position size for current candle
            kelly_size = dataframe['kelly_position_size'].iloc[-1]
            
            # Get current confidence
            confidence = dataframe['signal_confidence'].iloc[-1]
            
            # Adjust stake based on Kelly sizing and confidence
            adjusted_stake = proposed_stake * kelly_size * confidence
            
            # Apply safety limits
            max_stake = proposed_stake * self.max_position_size.value
            final_stake = min(adjusted_stake, max_stake)
            
            return max(final_stake, proposed_stake * 0.01)  # Minimum 1% of proposed
            
        except Exception as e:
            self.logger.warning(f"Custom stake calculation failed: {e}")
            return proposed_stake * 0.5
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                          time_in_force: str, current_time: datetime, **kwargs) -> bool:
        """
        Confirm trade entry with additional risk checks.
        """
        try:
            # Get current market data
            dataframe = self.dp.get_pair_dataframe(pair, self.timeframe)
            
            if dataframe.empty:
                return False
            
            # Final ensemble score check
            current_score = dataframe['ensemble_score'].iloc[-1]
            current_confidence = dataframe['signal_confidence'].iloc[-1]
            
            # Confirm entry only if signals are still strong
            if (current_score > self.signal_threshold.value and 
                current_confidence > self.confidence_threshold.value):
                
                # Log trade decision
                self.logger.info(f"Confirming entry for {pair}: score={current_score:.3f}, confidence={current_confidence:.3f}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Trade confirmation failed: {e}")
            return False
    
    def confirm_trade_exit(self, pair: str, trade, order_type: str, amount: float,
                         rate: float, time_in_force: str, exit_reason: str,
                         current_time: datetime, **kwargs) -> bool:
        """
        Confirm trade exit and update performance history.
        """
        try:
            # Calculate trade return
            entry_price = trade.open_rate
            exit_price = rate
            trade_return = (exit_price - entry_price) / entry_price
            
            # Store performance for Kelly calculation
            self.performance_history.append(trade_return)
            
            # Keep only recent history (last 100 trades)
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]
            
            # Log trade exit
            self.logger.info(f"Confirming exit for {pair}: return={trade_return:.3f}, reason={exit_reason}")
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Trade exit confirmation failed: {e}")
            return True  # Allow exit to proceed
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
               proposed_leverage: float, **kwargs) -> float:
        """
        Dynamic leverage based on market regime and volatility.
        """
        try:
            dataframe = self.dp.get_pair_dataframe(pair, self.timeframe)
            
            if dataframe.empty:
                return 1.0  # No leverage if no data
            
            # Get current regime and volatility
            current_regime = dataframe['market_regime'].iloc[-1]
            current_vol = dataframe['atr'].iloc[-1] / dataframe['close'].iloc[-1]
            
            # Adjust leverage based on regime
            regime_multiplier = {0: 0.5, 1: 0.8, 2: 1.0}  # Lower leverage in bear/sideways
            base_leverage = regime_multiplier.get(current_regime, 0.8)
            
            # Adjust for volatility (lower leverage in high vol)
            vol_adjustment = max(0.3, 1.0 - current_vol * 10)
            
            final_leverage = base_leverage * vol_adjustment
            return min(final_leverage, 3.0)  # Max 3x leverage
            
        except Exception as e:
            self.logger.warning(f"Leverage calculation failed: {e}")
            return 1.0
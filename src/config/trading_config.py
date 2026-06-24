"""
Trading Configuration Module

Defines all trading-related configuration parameters for the institutional
AI crypto trading system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import os


class TradingMode(Enum):
    """Trading modes."""
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"


class RiskLevel(Enum):
    """Risk management levels."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class PositionSizing:
    """Position sizing configuration."""
    # Kelly Criterion parameters
    kelly_fraction: float = 0.20  # Conservative fraction of Kelly to use
    max_position_size: float = 0.10  # Maximum 10% per trade (conservative)
    min_position_size: float = 0.03  # Minimum 3% per trade
    
    # Dynamic sizing
    volatility_adjustment: bool = True
    market_regime_adjustment: bool = True
    
    # Risk-based sizing
    max_drawdown_limit: float = 0.15  # 15% max drawdown
    target_volatility: float = 0.20  # 20% annualized volatility


@dataclass
class RiskManagement:
    """Risk management configuration."""
    # Position limits
    max_concurrent_positions: int = 3
    max_sector_exposure: float = 0.30  # 30% max per sector
    
    # Stop-loss and take-profit
    default_stop_loss: float = 0.05  # 5% stop loss
    default_take_profit: float = 0.15  # 15% take profit
    trailing_stop: bool = True
    trailing_stop_distance: float = 0.03  # 3% trailing stop
    
    # Risk metrics
    max_var_95: float = 0.02  # 2% VaR at 95% confidence
    max_correlation: float = 0.70  # Maximum correlation between positions
    
    # Circuit breakers
    daily_loss_limit: float = 0.05  # 5% daily loss limit
    weekly_loss_limit: float = 0.12  # 12% weekly loss limit
    monthly_loss_limit: float = 0.20  # 20% monthly loss limit


@dataclass
class PerformanceTargets:
    """Performance target configuration."""
    # Return targets (realistic institutional targets)
    target_monthly_return: float = 0.15  # 15% monthly (achievable with our edge)
    target_annual_return: float = 1.5  # 150% annual
    min_sharpe_ratio: float = 1.2  # Realistic minimum Sharpe ratio
    
    # Risk-adjusted metrics
    target_sortino_ratio: float = 2.0
    max_calmar_ratio: float = 0.15  # Maximum drawdown for Calmar
    
    # Win rate and consistency (realistic targets)
    target_win_rate: float = 0.60  # 60% win rate
    target_profit_factor: float = 1.8  # 1.8:1 profit factor
    
    # Volatility targets
    target_volatility: float = 0.25  # 25% annualized
    max_volatility: float = 0.45  # 45% maximum


@dataclass
class TradingPairs:
    """Trading pairs configuration."""
    # Primary pairs
    primary_pairs: List[str] = field(default_factory=lambda: [
        "SOL/USDT",
        "BTC/USDT", 
        "ETH/USDT",
        "HYPE/USDT"
    ])
    
    # Secondary pairs
    secondary_pairs: List[str] = field(default_factory=lambda: [
        "ADA/USDT",
        "DOT/USDT",
        "LINK/USDT",
        "MATIC/USDT"
    ])
    
    # Meme coins (high risk)
    meme_pairs: List[str] = field(default_factory=lambda: [
        "DOGE/USDT",
        "SHIB/USDT",
        "PEPE/USDT"
    ])
    
    # Maximum pairs to trade simultaneously
    max_active_pairs: int = 5


@dataclass
class ExchangeConfig:
    """Exchange-specific configuration."""
    # Primary exchange
    primary_exchange: str = "binance"
    primary_exchange_weight: float = 0.60  # 60% of volume
    
    # Secondary exchanges
    secondary_exchanges: List[str] = field(default_factory=lambda: [
        "okx",
        "hyperliquid"
    ])
    
    # Exchange weights
    exchange_weights: Dict[str, float] = field(default_factory=lambda: {
        "binance": 0.60,
        "okx": 0.25,
        "hyperliquid": 0.15
    })
    
    # Order execution
    use_smart_order_routing: bool = True
    max_slippage: float = 0.002  # 0.2% max slippage
    order_timeout: int = 30  # 30 seconds


@dataclass
class SignalConfig:
    """Signal generation configuration."""
    # Signal sources
    technical_analysis_weight: float = 0.40
    sentiment_analysis_weight: float = 0.25
    volume_anomaly_weight: float = 0.20
    macro_regime_weight: float = 0.15
    
    # Signal thresholds
    min_signal_strength: float = 0.60  # Minimum signal strength
    signal_confirmation_period: int = 3  # 3 periods for confirmation
    
    # Ensemble parameters
    ensemble_method: str = "weighted_average"
    adaptive_weights: bool = True
    weight_update_frequency: int = 24  # Update weights every 24 hours


@dataclass
class ModelConfig:
    """Machine learning model configuration."""
    # Model types
    use_ensemble_models: bool = True
    primary_model: str = "xgboost"
    secondary_models: List[str] = field(default_factory=lambda: [
        "lightgbm",
        "random_forest",
        "neural_network"
    ])
    
    # Training parameters
    retrain_frequency: int = 168  # Retrain every week (168 hours)
    min_training_samples: int = 1000
    validation_split: float = 0.20
    
    # Feature engineering
    feature_selection_method: str = "mutual_info"
    max_features: int = 50
    feature_importance_threshold: float = 0.01


@dataclass
class TradingConfig:
    """Main trading configuration class."""
    
    # Basic settings
    trading_mode: TradingMode = TradingMode.PAPER
    risk_level: RiskLevel = RiskLevel.MODERATE
    
    # Configuration objects
    position_sizing: PositionSizing = field(default_factory=PositionSizing)
    risk_management: RiskManagement = field(default_factory=RiskManagement)
    performance_targets: PerformanceTargets = field(default_factory=PerformanceTargets)
    trading_pairs: TradingPairs = field(default_factory=TradingPairs)
    exchange_config: ExchangeConfig = field(default_factory=ExchangeConfig)
    signal_config: SignalConfig = field(default_factory=SignalConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    
    # System settings
    data_update_frequency: int = 60  # 60 seconds
    signal_update_frequency: int = 300  # 5 minutes
    position_update_frequency: int = 60  # 1 minute
    
    # Logging and monitoring
    log_level: str = "INFO"
    enable_telegram_notifications: bool = True
    enable_performance_monitoring: bool = True
    
    # Advanced settings
    enable_adaptive_learning: bool = True
    enable_regime_detection: bool = True
    enable_anomaly_detection: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Validate position sizing
        if self.position_sizing.max_position_size > 0.20:
            raise ValueError("Maximum position size cannot exceed 20%")
        
        if self.position_sizing.min_position_size < 0.01:
            raise ValueError("Minimum position size cannot be less than 1%")
        
        # Validate risk management
        if self.risk_management.max_concurrent_positions > 10:
            raise ValueError("Maximum concurrent positions cannot exceed 10")
        
        if self.risk_management.max_sector_exposure > 0.50:
            raise ValueError("Maximum sector exposure cannot exceed 50%")
        
        # Validate performance targets
        if self.performance_targets.target_monthly_return > 1.0:
            raise ValueError("Target monthly return cannot exceed 100%")
        
        if self.performance_targets.min_sharpe_ratio < 0.5:
            raise ValueError("Minimum Sharpe ratio cannot be less than 0.5")
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "trading_mode": self.trading_mode.value,
            "risk_level": self.risk_level.value,
            "position_sizing": {
                "kelly_fraction": self.position_sizing.kelly_fraction,
                "max_position_size": self.position_sizing.max_position_size,
                "min_position_size": self.position_sizing.min_position_size
            },
            "risk_management": {
                "max_concurrent_positions": self.risk_management.max_concurrent_positions,
                "default_stop_loss": self.risk_management.default_stop_loss,
                "default_take_profit": self.risk_management.default_take_profit
            },
            "performance_targets": {
                "target_monthly_return": self.performance_targets.target_monthly_return,
                "min_sharpe_ratio": self.performance_targets.min_sharpe_ratio,
                "target_win_rate": self.performance_targets.target_win_rate
            },
            "trading_pairs": {
                "primary_pairs": self.trading_pairs.primary_pairs,
                "max_active_pairs": self.trading_pairs.max_active_pairs
            },
            "exchange_config": {
                "primary_exchange": self.exchange_config.primary_exchange,
                "exchange_weights": self.exchange_config.exchange_weights
            }
        }
    
    def update_from_env(self):
        """Update configuration from environment variables."""
        # Trading mode
        trading_mode = os.getenv("TRADING_MODE")
        if trading_mode:
            self.trading_mode = TradingMode(trading_mode)
        
        # Risk level
        risk_level = os.getenv("RISK_LEVEL")
        if risk_level:
            self.risk_level = RiskLevel(risk_level)
        
        # Position sizing
        max_pos_size = os.getenv("MAX_POSITION_SIZE")
        if max_pos_size:
            self.position_sizing.max_position_size = float(max_pos_size)
        
        min_pos_size = os.getenv("MIN_POSITION_SIZE")
        if min_pos_size:
            self.position_sizing.min_position_size = float(min_pos_size)
        
        # Risk management
        max_concurrent = os.getenv("MAX_CONCURRENT_POSITIONS")
        if max_concurrent:
            self.risk_management.max_concurrent_positions = int(max_concurrent)
        
        stop_loss = os.getenv("DEFAULT_STOP_LOSS")
        if stop_loss:
            self.risk_management.default_stop_loss = float(stop_loss)
        
        # Performance targets
        monthly_return = os.getenv("TARGET_MONTHLY_RETURN")
        if monthly_return:
            self.performance_targets.target_monthly_return = float(monthly_return)
        
        sharpe_ratio = os.getenv("MIN_SHARPE_RATIO")
        if sharpe_ratio:
            self.performance_targets.min_sharpe_ratio = float(sharpe_ratio)


# Default configuration instance
default_config = TradingConfig() 
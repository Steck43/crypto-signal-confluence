"""
Exchange Types and Data Structures

Common types used across all exchange connectors
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionSide(Enum):
    """Position side enumeration"""
    LONG = "long"
    SHORT = "short"


@dataclass
class Position:
    """Position information across exchanges"""
    exchange: str
    symbol: str
    size: float
    side: PositionSide
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    margin_used: float = 0.0
    leverage: float = 1.0
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class OrderResult:
    """Order execution result"""
    exchange: str
    symbol: str
    order_id: str
    status: OrderStatus
    side: OrderSide
    order_type: OrderType
    filled_size: float
    remaining_size: float
    avg_fill_price: float
    fee: float
    fee_currency: str = "USDT"
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class MarketData:
    """Market data structure"""
    exchange: str
    symbol: str
    price: float
    bid: float
    ask: float
    volume_24h: float
    high_24h: float
    low_24h: float
    price_change_24h: float
    price_change_percent_24h: float
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class AccountInfo:
    """Account information"""
    exchange: str
    total_balance: float
    available_balance: float
    margin_balance: float
    unrealized_pnl: float
    realized_pnl: float
    margin_ratio: float
    positions: Dict[str, Position]
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ExchangeConfig:
    """Exchange configuration"""
    name: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    passphrase: Optional[str] = None  # For OKX
    private_key: Optional[str] = None  # For Hyperliquid
    wallet_address: Optional[str] = None  # For Hyperliquid
    sandbox: bool = True
    testnet: bool = True
    rate_limit: int = 1200
    timeout: int = 30
    
    def is_configured(self) -> bool:
        """Check if exchange has required credentials"""
        if self.name == "hyperliquid":
            return bool(self.private_key and self.wallet_address)
        elif self.name == "okx":
            return bool(self.api_key and self.api_secret and self.passphrase)
        else:  # binance and others
            return bool(self.api_key and self.api_secret)


@dataclass
class ArbitrageOpportunity:
    """Arbitrage opportunity between exchanges"""
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    profit_bps: float  # Profit in basis points
    volume_available: float
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    @property
    def profit_percent(self) -> float:
        """Get profit as percentage"""
        return self.profit_bps / 100


@dataclass
class TradingSignal:
    """Trading signal for multi-exchange execution"""
    symbol: str
    signal_type: OrderSide
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    total_size: float
    exchange_allocation: Dict[str, float]  # Exchange name -> allocation percentage
    order_type: OrderType = OrderType.MARKET
    price: Optional[float] = None  # For limit orders
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        
        # Validate allocation percentages sum to 1.0
        total_allocation = sum(self.exchange_allocation.values())
        if abs(total_allocation - 1.0) > 0.01:
            raise ValueError(f"Exchange allocations must sum to 1.0, got {total_allocation}")


@dataclass
class ExchangeStatus:
    """Exchange connection status"""
    exchange: str
    connected: bool
    last_heartbeat: Optional[datetime] = None
    error_count: int = 0
    last_error: Optional[str] = None
    latency_ms: Optional[float] = None
    
    def __post_init__(self):
        if self.last_heartbeat is None:
            self.last_heartbeat = datetime.now()


class ExchangeError(Exception):
    """Base exception for exchange errors"""
    def __init__(self, exchange: str, message: str, error_code: Optional[str] = None):
        self.exchange = exchange
        self.error_code = error_code
        super().__init__(f"[{exchange}] {message}")


class AuthenticationError(ExchangeError):
    """Authentication failed"""
    pass


class RateLimitError(ExchangeError):
    """Rate limit exceeded"""
    pass


class InsufficientFundsError(ExchangeError):
    """Insufficient funds for order"""
    pass


class OrderRejectedError(ExchangeError):
    """Order was rejected by exchange"""
    pass


class ConnectionError(ExchangeError):
    """Connection to exchange failed"""
    pass 
"""
Institutional-Grade Data Validation System
Provides comprehensive validation for trading data, API responses, signals,
and risk parameters to prevent bad trades and system failures.
"""

import re
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass
import pandas as pd
import numpy as np
from enum import Enum

# Setup logger
logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a validation check"""
    is_valid: bool
    level: ValidationLevel
    message: str
    field: Optional[str] = None
    value: Optional[Any] = None
    suggestion: Optional[str] = None


class DataValidator:
    """Core data validation functionality"""
    
    @staticmethod
    def validate_price(price: Union[float, int], min_price: float = 0.000001, 
                      max_price: float = 1000000) -> ValidationResult:
        """
        Validate price data
        
        Args:
            price: Price value to validate
            min_price: Minimum acceptable price
            max_price: Maximum acceptable price
            
        Returns:
            ValidationResult
        """
        if not isinstance(price, (int, float)):
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Price must be numeric, got {type(price)}",
                field="price",
                value=price
            )
        
        if pd.isna(price) or np.isnan(price):
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message="Price cannot be NaN",
                field="price",
                value=price
            )
        
        if price <= 0:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Price must be positive, got {price}",
                field="price",
                value=price
            )
        
        if price < min_price:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.WARNING,
                message=f"Price {price} below minimum threshold {min_price}",
                field="price",
                value=price
            )
        
        if price > max_price:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.WARNING,
                message=f"Price {price} above maximum threshold {max_price}",
                field="price",
                value=price
            )
        
        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message="Price validation passed",
            field="price",
            value=price
        )
    
    @staticmethod
    def validate_volume(volume: Union[float, int], min_volume: float = 0) -> ValidationResult:
        """Validate volume data"""
        if not isinstance(volume, (int, float)):
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Volume must be numeric, got {type(volume)}",
                field="volume",
                value=volume
            )
        
        if pd.isna(volume) or np.isnan(volume):
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message="Volume cannot be NaN",
                field="volume",
                value=volume
            )
        
        if volume < 0:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Volume cannot be negative, got {volume}",
                field="volume",
                value=volume
            )
        
        if volume < min_volume:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.WARNING,
                message=f"Volume {volume} below minimum threshold {min_volume}",
                field="volume",
                value=volume
            )
        
        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message="Volume validation passed",
            field="volume",
            value=volume
        )
    
    @staticmethod
    def validate_timestamp(timestamp: Union[datetime, str, float]) -> ValidationResult:
        """Validate timestamp data"""
        try:
            if isinstance(timestamp, str):
                # Try to parse string timestamp
                parsed_timestamp = pd.to_datetime(timestamp)
            elif isinstance(timestamp, (int, float)):
                # Unix timestamp
                parsed_timestamp = pd.to_datetime(timestamp, unit='s')
            elif isinstance(timestamp, datetime):
                parsed_timestamp = timestamp
            else:
                return ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"Invalid timestamp type: {type(timestamp)}",
                    field="timestamp",
                    value=timestamp
                )
            
            # Check if timestamp is reasonable (not too old or in future)
            now = datetime.now()
            if parsed_timestamp < now - timedelta(days=365 * 10):  # 10 years ago
                return ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.WARNING,
                    message=f"Timestamp too old: {parsed_timestamp}",
                    field="timestamp",
                    value=timestamp
                )
            
            if parsed_timestamp > now + timedelta(hours=1):  # 1 hour in future
                return ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.WARNING,
                    message=f"Timestamp in future: {parsed_timestamp}",
                    field="timestamp",
                    value=timestamp
                )
            
            return ValidationResult(
                is_valid=True,
                level=ValidationLevel.INFO,
                message="Timestamp validation passed",
                field="timestamp",
                value=parsed_timestamp
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Cannot parse timestamp: {e}",
                field="timestamp",
                value=timestamp
            )


class MarketDataValidator:
    """Validate market data structures"""
    
    @staticmethod
    def validate_ohlcv(data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate OHLCV candlestick data"""
        results = []
        required_fields = ['open', 'high', 'low', 'close', 'volume']
        
        # Check required fields
        for field in required_fields:
            if field not in data:
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"Missing required field: {field}",
                    field=field
                ))
                continue
            
            value = data[field]
            
            # Validate based on field type
            if field == 'volume':
                results.append(DataValidator.validate_volume(value))
            else:
                results.append(DataValidator.validate_price(value))
        
        # Check OHLC relationships if all fields present
        if all(field in data for field in ['open', 'high', 'low', 'close']):
            o, h, l, c = data['open'], data['high'], data['low'], data['close']
            
            if h < max(o, c) or h < min(o, c):
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"High {h} is not the highest price (O:{o}, L:{l}, C:{c})",
                    field="high"
                ))
            
            if l > min(o, c) or l > max(o, c):
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"Low {l} is not the lowest price (O:{o}, H:{h}, C:{c})",
                    field="low"
                ))
        
        return results
    
    @staticmethod
    def validate_orderbook(orderbook: Dict[str, Any]) -> List[ValidationResult]:
        """Validate order book data"""
        results = []
        
        # Check structure
        if 'bids' not in orderbook or 'asks' not in orderbook:
            results.append(ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message="Order book missing bids or asks",
                field="structure"
            ))
            return results
        
        # Validate bids
        bids = orderbook['bids']
        if not isinstance(bids, list) or len(bids) == 0:
            results.append(ValidationResult(
                is_valid=False,
                level=ValidationLevel.WARNING,
                message="No bids in order book",
                field="bids"
            ))
        else:
            # Check bid order (descending prices)
            for i in range(len(bids) - 1):
                if len(bids[i]) < 2 or len(bids[i+1]) < 2:
                    continue
                if bids[i][0] <= bids[i+1][0]:  # Price should be descending
                    results.append(ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.ERROR,
                        message=f"Bids not in descending order: {bids[i][0]} <= {bids[i+1][0]}",
                        field="bids"
                    ))
                    break
        
        # Validate asks
        asks = orderbook['asks']
        if not isinstance(asks, list) or len(asks) == 0:
            results.append(ValidationResult(
                is_valid=False,
                level=ValidationLevel.WARNING,
                message="No asks in order book",
                field="asks"
            ))
        else:
            # Check ask order (ascending prices)
            for i in range(len(asks) - 1):
                if len(asks[i]) < 2 or len(asks[i+1]) < 2:
                    continue
                if asks[i][0] >= asks[i+1][0]:  # Price should be ascending
                    results.append(ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.ERROR,
                        message=f"Asks not in ascending order: {asks[i][0]} >= {asks[i+1][0]}",
                        field="asks"
                    ))
                    break
        
        # Check bid-ask spread
        if bids and asks and len(bids[0]) >= 2 and len(asks[0]) >= 2:
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            
            if best_bid >= best_ask:
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"Crossed book: best_bid {best_bid} >= best_ask {best_ask}",
                    field="spread"
                ))
            
            spread_pct = (best_ask - best_bid) / best_bid * 100
            if spread_pct > 5:  # 5% spread seems excessive
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.WARNING,
                    message=f"Wide spread: {spread_pct:.2f}%",
                    field="spread"
                ))
        
        return results


class TradingValidator:
    """Validate trading-related data and parameters"""
    
    @staticmethod
    def validate_signal(signal: Dict[str, Any]) -> List[ValidationResult]:
        """Validate trading signal"""
        results = []
        required_fields = ['symbol', 'signal_type', 'strength', 'timestamp']
        
        # Check required fields
        for field in required_fields:
            if field not in signal:
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"Missing required signal field: {field}",
                    field=field
                ))
        
        # Validate signal type
        if 'signal_type' in signal:
            valid_types = ['BUY', 'SELL', 'HOLD', 'buy', 'sell', 'hold']
            if signal['signal_type'] not in valid_types:
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"Invalid signal type: {signal['signal_type']}",
                    field="signal_type",
                    suggestion="Use BUY, SELL, or HOLD"
                ))
        
        # Validate strength
        if 'strength' in signal:
            strength = signal['strength']
            if not isinstance(strength, (int, float)):
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"Signal strength must be numeric, got {type(strength)}",
                    field="strength"
                ))
            elif strength < 0 or strength > 1:
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.WARNING,
                    message=f"Signal strength {strength} outside 0-1 range",
                    field="strength"
                ))
        
        # Validate symbol
        if 'symbol' in signal:
            symbol = signal['symbol']
            if not isinstance(symbol, str) or len(symbol) < 2:
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"Invalid symbol: {symbol}",
                    field="symbol"
                ))
        
        # Validate timestamp
        if 'timestamp' in signal:
            results.append(DataValidator.validate_timestamp(signal['timestamp']))
        
        return results
    
    @staticmethod
    def validate_trade_parameters(params: Dict[str, Any]) -> List[ValidationResult]:
        """Validate trade execution parameters"""
        results = []
        
        # Validate position size
        if 'position_size' in params:
            size = params['position_size']
            if not isinstance(size, (int, float)) or size <= 0:
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"Position size must be positive, got {size}",
                    field="position_size"
                ))
        
        # Validate stop loss
        if 'stop_loss' in params:
            stop_loss = params['stop_loss']
            if 'entry_price' in params:
                entry_price = params['entry_price']
                signal_type = params.get('signal_type', '').upper()
                
                if signal_type == 'BUY' and stop_loss >= entry_price:
                    results.append(ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.ERROR,
                        message=f"Buy stop loss {stop_loss} should be below entry {entry_price}",
                        field="stop_loss"
                    ))
                elif signal_type == 'SELL' and stop_loss <= entry_price:
                    results.append(ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.ERROR,
                        message=f"Sell stop loss {stop_loss} should be above entry {entry_price}",
                        field="stop_loss"
                    ))
        
        # Validate take profit
        if 'take_profit' in params:
            take_profit = params['take_profit']
            if 'entry_price' in params:
                entry_price = params['entry_price']
                signal_type = params.get('signal_type', '').upper()
                
                if signal_type == 'BUY' and take_profit <= entry_price:
                    results.append(ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.ERROR,
                        message=f"Buy take profit {take_profit} should be above entry {entry_price}",
                        field="take_profit"
                    ))
                elif signal_type == 'SELL' and take_profit >= entry_price:
                    results.append(ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.ERROR,
                        message=f"Sell take profit {take_profit} should be below entry {entry_price}",
                        field="take_profit"
                    ))
        
        return results
    
    @staticmethod
    def validate_risk_parameters(risk_params: Dict[str, Any]) -> List[ValidationResult]:
        """Validate risk management parameters"""
        results = []
        
        # Max position size
        if 'max_position_size' in risk_params:
            max_size = risk_params['max_position_size']
            if not isinstance(max_size, (int, float)) or max_size <= 0 or max_size > 1:
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"Max position size must be between 0 and 1, got {max_size}",
                    field="max_position_size"
                ))
        
        # Max drawdown
        if 'max_drawdown' in risk_params:
            max_dd = risk_params['max_drawdown']
            if not isinstance(max_dd, (int, float)) or max_dd <= 0 or max_dd > 1:
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"Max drawdown must be between 0 and 1, got {max_dd}",
                    field="max_drawdown"
                ))
        
        # Risk per trade
        if 'risk_per_trade' in risk_params:
            risk = risk_params['risk_per_trade']
            if not isinstance(risk, (int, float)) or risk <= 0 or risk > 0.1:
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.WARNING,
                    message=f"Risk per trade {risk} seems high (>10%)",
                    field="risk_per_trade",
                    suggestion="Consider using <5% risk per trade"
                ))
        
        return results


class APIValidator:
    """Validate API responses and external data"""
    
    @staticmethod
    def validate_exchange_response(response: Dict[str, Any], expected_fields: List[str]) -> List[ValidationResult]:
        """Validate exchange API response"""
        results = []
        
        # Check if response is dict
        if not isinstance(response, dict):
            results.append(ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Response must be dict, got {type(response)}",
                field="response_type"
            ))
            return results
        
        # Check for error fields
        error_fields = ['error', 'message', 'code']
        for field in error_fields:
            if field in response:
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"API returned error: {response[field]}",
                    field=field
                ))
        
        # Check expected fields
        for field in expected_fields:
            if field not in response:
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.WARNING,
                    message=f"Missing expected field: {field}",
                    field=field
                ))
        
        return results
    
    @staticmethod
    def validate_api_rate_limit(requests_made: int, time_window: int, rate_limit: int) -> ValidationResult:
        """Validate API rate limiting"""
        if requests_made >= rate_limit:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Rate limit exceeded: {requests_made}/{rate_limit} in {time_window}s",
                field="rate_limit"
            )
        
        if requests_made > rate_limit * 0.8:  # 80% of rate limit
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.WARNING,
                message=f"Approaching rate limit: {requests_made}/{rate_limit}",
                field="rate_limit"
            )
        
        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message="Rate limit OK",
            field="rate_limit"
        )


class ValidationEngine:
    """Main validation engine that orchestrates all validators"""
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize validation engine
        
        Args:
            strict_mode: If True, warnings are treated as errors
        """
        self.strict_mode = strict_mode
        self.validation_history: List[ValidationResult] = []
    
    def validate(self, data: Any, validation_type: str, **kwargs) -> Tuple[bool, List[ValidationResult]]:
        """
        Main validation method
        
        Args:
            data: Data to validate
            validation_type: Type of validation to perform
            **kwargs: Additional validation parameters
            
        Returns:
            Tuple of (is_valid, validation_results)
        """
        results = []
        
        if validation_type == 'price':
            results.append(DataValidator.validate_price(data, **kwargs))
        elif validation_type == 'volume':
            results.append(DataValidator.validate_volume(data, **kwargs))
        elif validation_type == 'timestamp':
            results.append(DataValidator.validate_timestamp(data))
        elif validation_type == 'ohlcv':
            results.extend(MarketDataValidator.validate_ohlcv(data))
        elif validation_type == 'orderbook':
            results.extend(MarketDataValidator.validate_orderbook(data))
        elif validation_type == 'signal':
            results.extend(TradingValidator.validate_signal(data))
        elif validation_type == 'trade_params':
            results.extend(TradingValidator.validate_trade_parameters(data))
        elif validation_type == 'risk_params':
            results.extend(TradingValidator.validate_risk_parameters(data))
        elif validation_type == 'api_response':
            expected_fields = kwargs.get('expected_fields', [])
            results.extend(APIValidator.validate_exchange_response(data, expected_fields))
        else:
            results.append(ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Unknown validation type: {validation_type}",
                field="validation_type"
            ))
        
        # Store validation history
        self.validation_history.extend(results)
        
        # Keep only recent validations (last 1000)
        if len(self.validation_history) > 1000:
            self.validation_history = self.validation_history[-1000:]
        
        # Determine overall validity
        is_valid = True
        for result in results:
            if not result.is_valid:
                if result.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL]:
                    is_valid = False
                elif self.strict_mode and result.level == ValidationLevel.WARNING:
                    is_valid = False
        
        # Log validation results
        for result in results:
            if not result.is_valid:
                if result.level == ValidationLevel.ERROR:
                    logger.error(f"Validation failed: {result.message}")
                elif result.level == ValidationLevel.WARNING:
                    logger.warning(f"Validation warning: {result.message}")
                elif result.level == ValidationLevel.CRITICAL:
                    logger.critical(f"Critical validation failure: {result.message}")
        
        return is_valid, results
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of recent validations"""
        if not self.validation_history:
            return {"message": "No validations performed"}
        
        total = len(self.validation_history)
        passed = sum(1 for r in self.validation_history if r.is_valid)
        failed = total - passed
        
        # Count by level
        by_level = {}
        for level in ValidationLevel:
            by_level[level.value] = sum(1 for r in self.validation_history if r.level == level)
        
        return {
            "total_validations": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0,
            "by_level": by_level,
            "recent_failures": [
                {"message": r.message, "field": r.field, "level": r.level.value}
                for r in self.validation_history[-10:]
                if not r.is_valid
            ]
        }


# Global validation engine instance
_global_validator: Optional[ValidationEngine] = None

def get_validator() -> ValidationEngine:
    """Get global validation engine instance"""
    global _global_validator
    if _global_validator is None:
        _global_validator = ValidationEngine()
    return _global_validator

# Convenience functions
def validate_price(price: float, **kwargs) -> Tuple[bool, List[ValidationResult]]:
    """Validate price using global validator"""
    return get_validator().validate(price, 'price', **kwargs)

def validate_signal(signal: Dict[str, Any]) -> Tuple[bool, List[ValidationResult]]:
    """Validate trading signal using global validator"""
    return get_validator().validate(signal, 'signal')

def validate_ohlcv(data: Dict[str, Any]) -> Tuple[bool, List[ValidationResult]]:
    """Validate OHLCV data using global validator"""
    return get_validator().validate(data, 'ohlcv')

def validate_trade_params(params: Dict[str, Any]) -> Tuple[bool, List[ValidationResult]]:
    """Validate trade parameters using global validator"""
    return get_validator().validate(params, 'trade_params') 
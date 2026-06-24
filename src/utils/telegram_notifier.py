"""
Telegram notification system for trading alerts and updates
Provides real-time notifications for trade execution, alerts, and system status
"""

import requests
import logging
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime
import json
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class NotificationLevel(Enum):
    """Notification priority levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    TRADE = "trade"


@dataclass
class TelegramConfig:
    """Telegram bot configuration"""
    bot_token: Optional[str] = None
    chat_id: Optional[str] = None
    enabled: bool = False


class TelegramNotifier:
    """Telegram notification system for trading alerts"""
    
    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        """
        Initialize Telegram notifier
        
        Args:
            bot_token: Telegram bot token (get from @BotFather)
            chat_id: Telegram chat ID to send messages to
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = bool(bot_token and chat_id)
        
        if not self.enabled:
            logger.warning("Telegram notifications disabled - no bot token or chat ID provided")
        else:
            logger.info("Telegram notifications enabled")
    
    def send_message(self, message: str, level: NotificationLevel = NotificationLevel.INFO,
                    parse_mode: str = "Markdown") -> bool:
        """
        Send message to Telegram
        
        Args:
            message: Message text
            level: Notification level
            parse_mode: Telegram parse mode (Markdown or HTML)
            
        Returns:
            True if sent successfully
        """
        if not self.enabled:
            return False
        
        # Format message with level indicator
        formatted_message = self._format_message(message, level)
        
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            
            payload = {
                'chat_id': self.chat_id,
                'text': formatted_message,
                'parse_mode': parse_mode,
                'disable_web_page_preview': True
            }
            
            response = requests.post(url, data=payload, timeout=10)
            
            if response.status_code == 200:
                logger.debug(f"Telegram message sent: {level.value}")
                return True
            else:
                logger.error(f"Telegram API error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    def _format_message(self, message: str, level: NotificationLevel) -> str:
        """Format message with level indicator"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        level_icons = {
            NotificationLevel.INFO: "ℹ️",
            NotificationLevel.WARNING: "⚠️", 
            NotificationLevel.ERROR: "❌",
            NotificationLevel.CRITICAL: "🚨",
            NotificationLevel.TRADE: "💰"
        }
        
        icon = level_icons.get(level, "📌")
        
        return f"{icon} *{level.value.upper()}* | {timestamp}\n{message}"
    
    def send_trade_notification(self, trade_type: str, symbol: str, price: float,
                              quantity: float, pnl: Optional[float] = None) -> bool:
        """
        Send trade execution notification
        
        Args:
            trade_type: BUY or SELL
            symbol: Trading symbol
            price: Execution price
            quantity: Trade quantity
            pnl: Profit/loss if available
            
        Returns:
            True if sent successfully
        """
        pnl_text = f"\nP&L: ${pnl:.2f}" if pnl is not None else ""
        
        message = f"""
**TRADE EXECUTED**
Symbol: {symbol}
Type: {trade_type}
Price: ${price:.4f}
Quantity: {quantity:.4f}{pnl_text}
"""
        
        return self.send_message(message, NotificationLevel.TRADE)
    
    def send_signal_notification(self, signal_type: str, symbol: str, strength: float,
                               confidence: float) -> bool:
        """
        Send trading signal notification
        
        Args:
            signal_type: Signal type (BUY/SELL/HOLD)
            symbol: Trading symbol
            strength: Signal strength (0-1)
            confidence: Signal confidence (0-1)
            
        Returns:
            True if sent successfully
        """
        message = f"""
**TRADING SIGNAL**
Symbol: {symbol}
Signal: {signal_type}
Strength: {strength:.2%}
Confidence: {confidence:.2%}
"""
        
        return self.send_message(message, NotificationLevel.INFO)
    
    def send_alert(self, title: str, description: str, 
                  level: NotificationLevel = NotificationLevel.WARNING) -> bool:
        """
        Send system alert
        
        Args:
            title: Alert title
            description: Alert description
            level: Alert level
            
        Returns:
            True if sent successfully
        """
        message = f"""
**{title}**
{description}
"""
        
        return self.send_message(message, level)
    
    def send_system_status(self, status: Dict[str, Any]) -> bool:
        """
        Send system status update
        
        Args:
            status: System status dictionary
            
        Returns:
            True if sent successfully
        """
        uptime = status.get('uptime_hours', 0)
        total_trades = status.get('total_trades', 0)
        win_rate = status.get('win_rate', 0)
        total_pnl = status.get('total_pnl', 0)
        
        message = f"""
**SYSTEM STATUS**
Uptime: {uptime:.1f} hours
Total Trades: {total_trades}
Win Rate: {win_rate:.1%}
Total P&L: ${total_pnl:.2f}
"""
        
        return self.send_message(message, NotificationLevel.INFO)
    
    def send_performance_report(self, report: Dict[str, Any]) -> bool:
        """
        Send performance report
        
        Args:
            report: Performance report dictionary
            
        Returns:
            True if sent successfully
        """
        message = "**DAILY PERFORMANCE REPORT**\n"
        
        for key, value in report.items():
            if isinstance(value, float):
                if 'rate' in key.lower() or 'ratio' in key.lower():
                    message += f"{key.replace('_', ' ').title()}: {value:.2%}\n"
                elif 'pnl' in key.lower() or 'value' in key.lower():
                    message += f"{key.replace('_', ' ').title()}: ${value:.2f}\n"
                else:
                    message += f"{key.replace('_', ' ').title()}: {value:.2f}\n"
            else:
                message += f"{key.replace('_', ' ').title()}: {value}\n"
        
        return self.send_message(message, NotificationLevel.INFO)
    
    async def send_async(self, message: str, level: NotificationLevel = NotificationLevel.INFO) -> bool:
        """
        Send message asynchronously
        
        Args:
            message: Message text
            level: Notification level
            
        Returns:
            True if sent successfully
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.send_message, message, level)
    
    def test_connection(self) -> bool:
        """
        Test Telegram connection
        
        Returns:
            True if connection successful
        """
        if not self.enabled:
            logger.warning("Cannot test Telegram - not configured")
            return False
        
        test_message = f"""
**TELEGRAM TEST**
Connection successful!
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        
        success = self.send_message(test_message, NotificationLevel.INFO)
        
        if success:
            logger.info("Telegram connection test successful")
        else:
            logger.error("Telegram connection test failed")
        
        return success


# Global notifier instance
_global_notifier: Optional[TelegramNotifier] = None

def get_telegram_notifier() -> TelegramNotifier:
    """Get global Telegram notifier instance"""
    global _global_notifier
    if _global_notifier is None:
        # Initialize with empty config (disabled by default)
        _global_notifier = TelegramNotifier()
    return _global_notifier

def setup_telegram_notifications(bot_token: str, chat_id: str) -> TelegramNotifier:
    """
    Setup global Telegram notifications
    
    Args:
        bot_token: Telegram bot token
        chat_id: Telegram chat ID
        
    Returns:
        Configured notifier
    """
    global _global_notifier
    _global_notifier = TelegramNotifier(bot_token, chat_id)
    return _global_notifier

# Convenience functions
def send_trade_alert(trade_type: str, symbol: str, price: float, quantity: float, pnl: Optional[float] = None):
    """Send trade notification using global notifier"""
    notifier = get_telegram_notifier()
    return notifier.send_trade_notification(trade_type, symbol, price, quantity, pnl)

def send_signal_alert(signal_type: str, symbol: str, strength: float, confidence: float):
    """Send signal notification using global notifier"""
    notifier = get_telegram_notifier()
    return notifier.send_signal_notification(signal_type, symbol, strength, confidence)

def send_system_alert(title: str, description: str, level: NotificationLevel = NotificationLevel.WARNING):
    """Send system alert using global notifier"""
    notifier = get_telegram_notifier()
    return notifier.send_alert(title, description, level)

def send_error_alert(error_message: str):
    """Send error alert using global notifier"""
    notifier = get_telegram_notifier()
    return notifier.send_alert("System Error", error_message, NotificationLevel.ERROR)

def send_critical_alert(critical_message: str):
    """Send critical alert using global notifier"""
    notifier = get_telegram_notifier()
    return notifier.send_alert("CRITICAL ALERT", critical_message, NotificationLevel.CRITICAL)


# Example usage
if __name__ == "__main__":
    # Test the notification system
    notifier = TelegramNotifier()  # Will be disabled without tokens
    
    # Test with fake data
    print("Testing notifications (will fail without tokens):")
    notifier.send_trade_notification("BUY", "SOL/USDT", 180.50, 10.0, 25.50)
    notifier.send_signal_notification("BUY", "SOL/USDT", 0.85, 0.75)
    notifier.send_alert("Test Alert", "This is a test alert message") 
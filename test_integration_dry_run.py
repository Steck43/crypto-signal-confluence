#!/usr/bin/env python3
"""
Dry-Run Integration Test for Secure Config

This script tests that the main trading system and order executor
can initialize and load all secure credentials without errors.
No real trades or Telegram messages are sent.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from exchanges.exchange_config import get_exchange_config
from trading.order_executor import OrderExecutor
from utils.telegram_notifier import TelegramNotifier


def dry_run():
    print("\n🚦 Starting dry-run integration test...")
    config = get_exchange_config()
    status = config.validate_config()
    print("\n🔒 Secure Config Status:")
    for k, v in status.items():
        print(f"  {k}: {'✅' if v else '❌'}")

    # Test Telegram notifier
    telegram_cfg = config.get_telegram_config()
    if telegram_cfg and telegram_cfg.bot_token and telegram_cfg.chat_id:
        print("\n✅ Telegram credentials loaded.")
        try:
            notifier = TelegramNotifier(telegram_cfg.bot_token, telegram_cfg.chat_id)
            print("✅ TelegramNotifier initialized (no message sent)")
        except Exception as e:
            print(f"❌ TelegramNotifier failed to initialize: {e}")
    else:
        print("❌ Telegram credentials missing.")

    # Test OrderExecutor and exchange clients
    try:
        executor = OrderExecutor()
        print("\n✅ OrderExecutor initialized.")
        if hasattr(executor, 'hyperliquid') and executor.hyperliquid:
            print("  - Hyperliquid client: initialized")
        if hasattr(executor, 'binance') and executor.binance:
            print("  - Binance client: initialized")
        if hasattr(executor, 'okx') and executor.okx:
            print("  - OKX client: initialized")
    except Exception as e:
        print(f"❌ OrderExecutor failed to initialize: {e}")

    print("\n🎉 Dry-run integration test completed. If all components show as initialized, you are ready to go!")

if __name__ == "__main__":
    dry_run() 
#!/usr/bin/env python3
"""
Secure Credentials Setup Script

Loads API credentials from environment variables and stores them in the
encrypted exchange configuration system.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / "src"))

from exchanges.exchange_config import get_exchange_config


def _env(name: str) -> str:
    return os.getenv(name, "") or ""


def main() -> int:
    print("Setting up secure API credentials from environment...")

    config = get_exchange_config()

    try:
        hyperliquid_wallet = _env("HYPERLIQUID_WALLET_ADDRESS")
        hyperliquid_key = _env("HYPERLIQUID_PRIVATE_KEY")
        # Set HYPERLIQUID_WALLET_ADDRESS and HYPERLIQUID_PRIVATE_KEY in your .env file
        if hyperliquid_wallet and hyperliquid_key:
            print("\nSetting up Hyperliquid API...")
            config.set_hyperliquid_credentials(
                wallet_address=hyperliquid_wallet,
                private_key=hyperliquid_key,
                testnet=True,
            )
            print("Hyperliquid credentials stored securely")

        binance_key = _env("BINANCE_API_KEY")
        binance_secret = _env("BINANCE_API_SECRET")
        # Set BINANCE_API_KEY and BINANCE_API_SECRET in your .env file
        if binance_key and binance_secret:
            print("\nSetting up Binance API...")
            config.set_binance_credentials(
                api_key=binance_key,
                secret_key=binance_secret,
                testnet=True,
            )
            print("Binance credentials stored securely")

        okx_key = _env("OKX_API_KEY")
        okx_secret = _env("OKX_API_SECRET")
        if okx_key and okx_secret:
            print("\nSetting up OKX API...")
            config.set_okx_credentials(
                api_key=okx_key,
                secret_key=okx_secret,
                testnet=True,
            )
            print("OKX credentials stored securely")

        coingecko_key = _env("COINGECKO_API_KEY")
        if coingecko_key:
            print("\nSetting up CoinGecko API...")
            config.set_coingecko_credentials(demo_api_key=coingecko_key)
            print("CoinGecko credentials stored securely")

        alpha_key = _env("ALPHA_VANTAGE_API_KEY")
        if alpha_key:
            print("\nSetting up Alpha Vantage API...")
            config.set_alpha_vantage_credentials(api_key=alpha_key)
            print("Alpha Vantage credentials stored securely")

        fred_key = _env("FRED_API_KEY")
        if fred_key:
            print("\nSetting up FRED API...")
            config.set_fred_credentials(api_key=fred_key)
            print("FRED credentials stored securely")

        telegram_token = _env("TELEGRAM_BOT_TOKEN")
        telegram_chat = _env("TELEGRAM_CHAT_ID")
        if telegram_token and telegram_chat:
            print("\nSetting up Telegram Bot...")
            config.set_telegram_credentials(
                bot_token=telegram_token,
                chat_id=telegram_chat,
            )
            print("Telegram credentials stored securely")

        validation = config.validate_config()

        print("\nConfiguration Status:")
        for service, is_configured in validation.items():
            status = "Configured" if is_configured else "Not configured"
            print(f"  {service.capitalize()}: {status}")

        print("\nSecurity Information:")
        print(f"  - Encryption key: {config.key_file}")
        print(f"  - Encrypted config: {config.encrypted_config_file}")

        if not any(validation.values()):
            print(
                "\nNo credentials found in environment. "
                "Copy .env.example to .env and fill in the keys you need."
            )
            return 0

        print("\nAPI credentials stored from environment.")
        return 0

    except Exception as e:
        print(f"Error setting up credentials: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Test Secure Configuration System

This script tests the secure exchange configuration system
to ensure credentials are properly stored and retrieved.
"""

import sys
import os
import types
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from exchanges.exchange_config import get_exchange_config

# Client extras are unused by _resolve_salt. Stub them so the helper can be
# imported without aiohttp and without opening any vault path.
for _name, _attrs in (
    ("api.coingecko", {"get_coingecko_client": None, "CoinGeckoClient": object}),
    ("api.fred", {"get_fred_client": None, "FREDClient": object}),
    ("api.exchanges", {}),
    ("api.exchanges.binance", {"BinanceClient": object}),
    ("api.exchanges.okx", {"OKXClient": object}),
    ("api.exchanges.hyperliquid", {"HyperliquidClient": object}),
):
    _module = types.ModuleType(_name)
    for _key, _value in _attrs.items():
        setattr(_module, _key, _value)
    sys.modules[_name] = _module

from api.secure_manager import SecureAPIManager


def test_resolve_salt_is_per_deployment(tmp_path):
    """README claim: encryption derives its key with a per-deployment salt.

    This test calls _resolve_salt. A helper that always returns one fixed
    salt must fail. Temporary files only; no vault path is opened.
    """
    first = object.__new__(SecureAPIManager)
    first.salt_file = tmp_path / "deploy-a.salt"
    second = object.__new__(SecureAPIManager)
    second.salt_file = tmp_path / "deploy-b.salt"

    salt_a = first._resolve_salt()
    salt_b = second._resolve_salt()

    assert isinstance(salt_a, bytes) and len(salt_a) == 16
    assert isinstance(salt_b, bytes) and len(salt_b) == 16
    assert salt_a != salt_b, "per-deployment salt must not be a fixed source constant"

    assert first.salt_file.read_bytes() == salt_a
    assert first._resolve_salt() == salt_a

def test_secure_config():
    """Test the secure configuration system."""
    print("🧪 Testing secure configuration system...")
    
    # Get the exchange configuration manager
    config = get_exchange_config()
    
    try:
        # Test retrieving all configurations
        print("\n📋 Testing configuration retrieval...")
        
        # Test each service
        services = [
            ("Hyperliquid", config.get_hyperliquid_config),
            ("Binance", config.get_binance_config),
            ("OKX", config.get_okx_config),
            ("CoinGecko", config.get_coingecko_config),
            ("Alpha Vantage", config.get_alpha_vantage_config),
            ("FRED", config.get_fred_config),
            ("Telegram", config.get_telegram_config)
        ]
        
        for service_name, getter_func in services:
            config_obj = getter_func()
            if config_obj:
                print(f"✅ {service_name}: Configured")
                # Show partial info for verification (without exposing full credentials)
                if hasattr(config_obj, 'api_key') and config_obj.api_key:
                    masked_key = config_obj.api_key[:8] + "..." + config_obj.api_key[-4:] if len(config_obj.api_key) > 12 else "***"
                    print(f"   API Key: {masked_key}")
                if hasattr(config_obj, 'demo_api_key') and getattr(config_obj, 'demo_api_key', None):
                    demo_key = config_obj.demo_api_key
                    masked_demo = demo_key[:8] + "..." + demo_key[-4:] if len(demo_key) > 12 else "***"
                    print(f"   Demo API Key: {masked_demo}")
                if hasattr(config_obj, 'wallet_address') and config_obj.wallet_address:
                    masked_wallet = config_obj.wallet_address[:10] + "..." + config_obj.wallet_address[-4:]
                    print(f"   Wallet: {masked_wallet}")
                if hasattr(config_obj, 'bot_token') and config_obj.bot_token:
                    masked_token = config_obj.bot_token[:8] + "..." + config_obj.bot_token[-4:] if len(config_obj.bot_token) > 12 else "***"
                    print(f"   Bot Token: {masked_token}")
                if hasattr(config_obj, 'chat_id') and config_obj.chat_id:
                    masked_chat = config_obj.chat_id[:2] + "..." + config_obj.chat_id[-2:] if len(config_obj.chat_id) > 4 else "***"
                    print(f"   Chat ID: {masked_chat}")
            else:
                print(f"❌ {service_name}: Not configured")
        
        # Test validation
        print("\n🔍 Testing configuration validation...")
        validation = config.validate_config()
        
        configured_count = sum(validation.values())
        total_count = len(validation)
        
        print(f"Configured: {configured_count}/{total_count} services")
        
        for service, is_configured in validation.items():
            status = "✅" if is_configured else "❌"
            print(f"  {status} {service.capitalize()}")
        
        # Test getting all configs at once
        print("\n📦 Testing bulk configuration retrieval...")
        all_configs = config.get_all_configs()
        
        config_count = sum([
            all_configs.hyperliquid is not None,
            all_configs.binance is not None,
            all_configs.okx is not None,
            all_configs.coingecko is not None,
            all_configs.alpha_vantage is not None,
            all_configs.fred is not None,
            all_configs.telegram is not None
        ])
        
        print(f"Retrieved {config_count} configurations successfully")
        
        # Security check
        print("\n🔒 Security verification...")
        if config.key_file.exists():
            print(f"✅ Encryption key exists: {config.key_file}")
        else:
            print(f"❌ Encryption key missing: {config.key_file}")
            
        if config.encrypted_config_file.exists():
            print(f"✅ Encrypted config exists: {config.encrypted_config_file}")
            file_size = config.encrypted_config_file.stat().st_size
            print(f"   File size: {file_size} bytes")
        else:
            print(f"❌ Encrypted config missing: {config.encrypted_config_file}")
        
        print("\n🎉 Secure configuration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_fallback():
    """Test environment variable fallback."""
    print("\n🌍 Testing environment variable fallback...")
    
    # Set a test environment variable
    os.environ['TEST_API_KEY'] = 'test_key_12345'
    
    # Create a new config instance to test env fallback
    config = get_exchange_config()
    
    # Test that we can get the env var
    test_key = config._get_env_credential('TEST_API_KEY')
    if test_key == 'test_key_12345':
        print("✅ Environment variable fallback works")
    else:
        print("❌ Environment variable fallback failed")
    
    # Clean up
    del os.environ['TEST_API_KEY']

if __name__ == "__main__":
    print("🚀 Starting secure configuration tests...")
    
    # Run tests
    success = test_secure_config()
    test_environment_fallback()
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1) 
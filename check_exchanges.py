#!/usr/bin/env python3
"""
Simple script to check exchange configuration status
"""

import os
import json
from pathlib import Path

def check_exchange_config():
    """Check what exchanges are configured."""
    
    # Check if config directory exists
    config_dir = Path("config")
    config_file = config_dir / "api_config.json"
    
    print("=== Exchange Configuration Status ===")
    
    if not config_dir.exists():
        print("❌ Config directory does not exist")
        print("   Creating default configuration...")
        config_dir.mkdir(exist_ok=True)
        
        # Create default config
        default_config = {
            "exchanges": {
                "binance": {
                    "name": "binance",
                    "api_key": None,
                    "api_secret": None,
                    "sandbox": True,
                    "rate_limit": 1200
                },
                "okx": {
                    "name": "okx", 
                    "api_key": None,
                    "api_secret": None,
                    "sandbox": True,
                    "rate_limit": 20
                },
                "hyperliquid": {
                    "name": "hyperliquid",
                    "api_key": None,
                    "api_secret": None,
                    "sandbox": True,
                    "rate_limit": 100
                }
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        print("✅ Created default configuration")
        return default_config["exchanges"]
    
    elif not config_file.exists():
        print("❌ Config file does not exist")
        return {}
    
    else:
        print("✅ Config file exists")
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            exchanges = config.get("exchanges", {})
            print(f"📋 Found {len(exchanges)} exchange(s) in config:")
            
            for name, details in exchanges.items():
                has_key = bool(details.get("api_key"))
                has_secret = bool(details.get("api_secret"))
                sandbox = details.get("sandbox", True)
                
                status = "✅ Configured" if (has_key and has_secret) else "❌ Not configured"
                mode = "SANDBOX" if sandbox else "LIVE"
                
                print(f"   {name.upper()}: {status} ({mode})")
            
            return exchanges
            
        except Exception as e:
            print(f"❌ Error reading config: {e}")
            return {}

def check_environment_variables():
    """Check for exchange API keys in environment variables."""
    print("\n=== Environment Variables ===")
    
    exchange_vars = [
        "BINANCE_API_KEY", "BINANCE_API_SECRET",
        "OKX_API_KEY", "OKX_API_SECRET", 
        "HYPERLIQUID_API_KEY", "HYPERLIQUID_API_SECRET"
    ]
    
    found_vars = []
    for var in exchange_vars:
        value = os.getenv(var)
        if value:
            found_vars.append(var)
            print(f"✅ {var}: {'*' * min(len(value), 8)}...")
        else:
            print(f"❌ {var}: Not set")
    
    return found_vars

def check_exchange_clients():
    """Check if exchange client modules exist."""
    print("\n=== Exchange Client Modules ===")
    
    clients = [
        ("src/api/exchanges/binance.py", "Binance"),
        ("src/api/exchanges/okx.py", "OKX"),
        ("src/api/exchanges/hyperliquid.py", "Hyperliquid")
    ]
    
    available_clients = []
    for client_path, name in clients:
        if Path(client_path).exists():
            print(f"✅ {name} client: Available")
            available_clients.append(name)
        else:
            print(f"❌ {name} client: Missing")
    
    return available_clients

if __name__ == "__main__":
    exchanges = check_exchange_config()
    env_vars = check_environment_variables()
    clients = check_exchange_clients()
    
    print(f"\n=== Summary ===")
    print(f"Configured exchanges: {len(exchanges)}")
    print(f"Environment variables: {len(env_vars)}")
    print(f"Available clients: {len(clients)}")
    
    if len(exchanges) > 0 and len(clients) > 0:
        print("\n🎯 Ready for trading setup!")
        print("   Next steps:")
        print("   1. Set API keys via environment variables or config file")
        print("   2. Test exchange connections")
        print("   3. Configure trading parameters")
    else:
        print("\n⚠️  System needs configuration before trading") 
# Exchange Configuration Guide

## Overview

Your AI crypto trading system supports three major exchanges:

1. **Binance** - Largest crypto exchange with extensive liquidity
2. **OKX** - Professional trading platform with advanced features  
3. **Hyperliquid** - Decentralized exchange with unique trading opportunities

## Current Status

✅ **Exchange Clients**: All three exchange clients are implemented and ready
✅ **Configuration System**: Secure credential management with encryption
❌ **API Keys**: Not configured (need to be set up)

## Exchange Setup Instructions

### 1. Binance Setup

**Features Available:**
- Market data for SOL/USDT and other pairs
- Order placement and management
- Account information and balance tracking
- Rate limiting: 1200 requests/minute

**Setup Steps:**
1. Create account at [Binance](https://www.binance.com)
2. Enable API access in account settings
3. Create API key with trading permissions
4. Set environment variables:
   ```bash
   set BINANCE_API_KEY=your_api_key_here
   set BINANCE_API_SECRET=your_api_secret_here
   ```

### 2. OKX Setup

**Features Available:**
- Professional trading interface
- Advanced order types
- Comprehensive market data
- Rate limiting: 600 requests/minute

**Setup Steps:**
1. Create account at [OKX](https://www.okx.com)
2. Enable API access in account settings
3. Create API key with trading permissions
4. Note your API passphrase
5. Set environment variables:
   ```bash
   set OKX_API_KEY=your_api_key_here
   set OKX_API_SECRET=your_api_secret_here
   set OKX_PASSPHRASE=your_passphrase_here
   ```

### 3. Hyperliquid Setup

**Features Available:**
- Decentralized trading
- Unique HYPE token trading
- On-chain order management
- Rate limiting: 300 requests/minute

**Setup Steps:**
1. Connect wallet to [Hyperliquid](https://hyperliquid.xyz)
2. Generate API key in settings
3. Set environment variables:
   ```bash
   set HYPERLIQUID_API_KEY=your_api_key_here
   set HYPERLIQUID_WALLET_ADDRESS=your_wallet_address_here
   ```

## Configuration Methods

### Method 1: Environment Variables (Recommended)

Set API keys as environment variables for security:

```bash
# Windows PowerShell
$env:BINANCE_API_KEY="your_key"
$env:BINANCE_API_SECRET="your_secret"

# Or permanent (Windows)
setx BINANCE_API_KEY "your_key"
setx BINANCE_API_SECRET "your_secret"
```

### Method 2: Configuration File

Update `config/api_config.json`:

```json
{
  "exchanges": {
    "binance": {
      "name": "binance",
      "api_key": "your_encrypted_key",
      "api_secret": "your_encrypted_secret",
      "sandbox": true,
      "rate_limit": 1200
    }
  }
}
```

### Method 3: Programmatic Setup

Use the API configuration system:

```python
from src.config.api_config import get_api_config

config = get_api_config()
config.set_exchange_credentials(
    exchange_name="binance",
    api_key="your_key",
    api_secret="your_secret",
    sandbox=True
)
```

## Security Best Practices

1. **Use Sandbox Mode**: Always test with sandbox/testnet first
2. **Limit Permissions**: Only grant necessary trading permissions
3. **Secure Storage**: Use environment variables or encrypted config
4. **IP Whitelisting**: Restrict API access to your trading server IP
5. **Regular Rotation**: Rotate API keys periodically

## Testing Exchange Connections

Run the exchange checker:

```bash
python check_exchanges.py
```

This will show:
- ✅ Available exchange clients
- ❌ Missing API credentials
- 📋 Configuration status

## Paper Trading vs Live Trading

### Paper Trading (Default)
- Uses simulated orders
- No real money at risk
- Perfect for testing strategies
- All exchanges work in paper mode

### Live Trading
- Requires valid API credentials
- Real money transactions
- Additional risk management required
- Set `sandbox: false` in config

## Exchange-Specific Features

### Binance
- **SOL/USDT Trading**: Primary trading pair
- **High Liquidity**: Best for large orders
- **Advanced Orders**: Stop-loss, take-profit
- **Futures Trading**: Available with separate API

### OKX  
- **Professional Tools**: Advanced charting
- **Multiple Order Types**: IOC, FOK, conditional
- **Cross-Margin**: Flexible margin management
- **Options Trading**: Available with separate API

### Hyperliquid
- **Decentralized**: No KYC required
- **HYPE Token**: Native exchange token
- **On-Chain**: All trades on blockchain
- **Unique Markets**: Specialized trading pairs

## Troubleshooting

### Common Issues

1. **"Invalid API Key"**
   - Check API key format
   - Verify permissions are enabled
   - Ensure IP is whitelisted

2. **"Rate Limit Exceeded"**
   - System automatically handles rate limits
   - Reduce trading frequency if needed
   - Check exchange-specific limits

3. **"Connection Failed"**
   - Check internet connection
   - Verify exchange API endpoints
   - Check firewall settings

### Support Resources

- **Binance**: [API Documentation](https://binance-docs.github.io/apidocs/)
- **OKX**: [API Documentation](https://www.okx.com/docs-v5/)
- **Hyperliquid**: [API Documentation](https://hyperliquid.gitbook.io/hyperliquid/)

## Next Steps

1. **Choose Primary Exchange**: Select based on your trading needs
2. **Set Up API Keys**: Follow exchange-specific instructions
3. **Test Connection**: Use `check_exchanges.py`
4. **Configure Trading Parameters**: Set risk limits and strategies
5. **Start Paper Trading**: Test with simulated orders
6. **Monitor Performance**: Use built-in monitoring tools

## Safety Reminder

⚠️ **Never share your API keys or secrets**
⚠️ **Start with paper trading to test the system**
⚠️ **Use small amounts when transitioning to live trading**
⚠️ **Monitor your positions and risk management**

---

For additional help, check the main `README.md` and `SETUP_GUIDE.md` files. 
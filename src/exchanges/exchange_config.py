"""
Secure Exchange Configuration Management

Handles secure storage and retrieval of exchange API credentials:
- Hyperliquid API
- Binance API  
- OKX API
- CoinGecko API
- Alpha Vantage API
- FRED API
- Telegram Bot
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
from cryptography.fernet import Fernet
import base64

@dataclass
class HyperliquidConfig:
    """Hyperliquid API configuration."""
    wallet_address: str
    private_key: str
    testnet: bool = True
    
@dataclass
class BinanceConfig:
    """Binance API configuration."""
    api_key: str
    secret_key: str
    testnet: bool = True
    
@dataclass
class OKXConfig:
    """OKX API configuration."""
    api_key: str
    secret_key: str
    passphrase: Optional[str] = None
    testnet: bool = True
    
@dataclass
class CoinGeckoConfig:
    """CoinGecko API configuration."""
    api_key: Optional[str] = None
    demo_api_key: Optional[str] = None
    base_url: str = "https://api.coingecko.com/api/v3"
    
@dataclass
class AlphaVantageConfig:
    """Alpha Vantage API configuration."""
    api_key: str
    base_url: str = "https://www.alphavantage.co/query"
    
@dataclass
class FREDConfig:
    """FRED API configuration."""
    api_key: str
    base_url: str = "https://api.stlouisfed.org/fred"
    
@dataclass
class TelegramConfig:
    """Telegram Bot configuration."""
    bot_token: str
    chat_id: str
    
@dataclass
class ExchangeCredentials:
    """Complete exchange credentials container."""
    hyperliquid: Optional[HyperliquidConfig] = None
    binance: Optional[BinanceConfig] = None
    okx: Optional[OKXConfig] = None
    coingecko: Optional[CoinGeckoConfig] = None
    alpha_vantage: Optional[AlphaVantageConfig] = None
    fred: Optional[FREDConfig] = None
    telegram: Optional[TelegramConfig] = None

class SecureExchangeConfig:
    """
    Secure exchange configuration manager with encryption.
    
    Features:
    - Encrypted credential storage
    - Environment variable fallback
    - Configuration validation
    - Secure key management
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize secure exchange configuration.
        
        Args:
            config_dir: Directory for configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Security files
        self.encrypted_config_file = self.config_dir / "exchange_credentials.enc"
        self.key_file = self.config_dir / ".encryption_key"
        
        # Setup logging first
        self.logger = logging.getLogger(__name__)
        
        # Initialize encryption
        self._encryption_key = self._get_or_create_encryption_key()
        self._cipher = Fernet(self._encryption_key)
        
        # Load credentials
        self._credentials = self._load_credentials()
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for sensitive data."""
        if self.key_file.exists():
            with open(self.key_file, 'rb') as f:
                return f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
            self.logger.info(f"Generated new encryption key: {self.key_file}")
            return key
    
    def _encrypt_data(self, data: Dict) -> bytes:
        """Encrypt configuration data."""
        json_data = json.dumps(data, indent=2)
        return self._cipher.encrypt(json_data.encode())
    
    def _decrypt_data(self, encrypted_data: bytes) -> Dict:
        """Decrypt configuration data."""
        try:
            decrypted = self._cipher.decrypt(encrypted_data)
            return json.loads(decrypted.decode())
        except Exception as e:
            self.logger.error(f"Failed to decrypt data: {e}")
            return {}
    
    def _load_credentials(self) -> ExchangeCredentials:
        """Load encrypted credentials from file."""
        if self.encrypted_config_file.exists():
            try:
                with open(self.encrypted_config_file, 'rb') as f:
                    encrypted_data = f.read()
                data = self._decrypt_data(encrypted_data)
                return self._dict_to_credentials(data)
            except Exception as e:
                self.logger.warning(f"Error loading encrypted config: {e}")
        
        # Return empty credentials
        return ExchangeCredentials()
    
    def _save_credentials(self):
        """Save credentials to encrypted file."""
        try:
            data = self._credentials_to_dict(self._credentials)
            encrypted_data = self._encrypt_data(data)
            with open(self.encrypted_config_file, 'wb') as f:
                f.write(encrypted_data)
            self.logger.info(f"Saved encrypted credentials to: {self.encrypted_config_file}")
        except Exception as e:
            self.logger.error(f"Error saving credentials: {e}")
    
    def _get_env_credential(self, env_var: str, default: Optional[str] = None) -> Optional[str]:
        """Get credential from environment variable."""
        return os.getenv(env_var, default)
    
    def _credentials_to_dict(self, credentials: ExchangeCredentials) -> Dict:
        """Convert credentials to dictionary."""
        data = {}
        
        if credentials.hyperliquid:
            data['hyperliquid'] = asdict(credentials.hyperliquid)
        if credentials.binance:
            data['binance'] = asdict(credentials.binance)
        if credentials.okx:
            data['okx'] = asdict(credentials.okx)
        if credentials.coingecko:
            data['coingecko'] = asdict(credentials.coingecko)
        if credentials.alpha_vantage:
            data['alpha_vantage'] = asdict(credentials.alpha_vantage)
        if credentials.fred:
            data['fred'] = asdict(credentials.fred)
        if credentials.telegram:
            data['telegram'] = asdict(credentials.telegram)
        
        return data
    
    def _dict_to_credentials(self, data: Dict) -> ExchangeCredentials:
        """Convert dictionary to credentials."""
        credentials = ExchangeCredentials()
        
        if 'hyperliquid' in data:
            credentials.hyperliquid = HyperliquidConfig(**data['hyperliquid'])
        if 'binance' in data:
            credentials.binance = BinanceConfig(**data['binance'])
        if 'okx' in data:
            credentials.okx = OKXConfig(**data['okx'])
        if 'coingecko' in data:
            credentials.coingecko = CoinGeckoConfig(**data['coingecko'])
        if 'alpha_vantage' in data:
            credentials.alpha_vantage = AlphaVantageConfig(**data['alpha_vantage'])
        if 'fred' in data:
            credentials.fred = FREDConfig(**data['fred'])
        if 'telegram' in data:
            credentials.telegram = TelegramConfig(**data['telegram'])
        
        return credentials
    
    def set_hyperliquid_credentials(self, wallet_address: str, private_key: str, testnet: bool = True):
        """Set Hyperliquid credentials."""
        self._credentials.hyperliquid = HyperliquidConfig(
            wallet_address=wallet_address,
            private_key=private_key,
            testnet=testnet
        )
        self._save_credentials()
    
    def set_binance_credentials(self, api_key: str, secret_key: str, testnet: bool = True):
        """Set Binance credentials."""
        self._credentials.binance = BinanceConfig(
            api_key=api_key,
            secret_key=secret_key,
            testnet=testnet
        )
        self._save_credentials()
    
    def set_okx_credentials(self, api_key: str, secret_key: str, passphrase: Optional[str] = None, testnet: bool = True):
        """Set OKX credentials."""
        self._credentials.okx = OKXConfig(
            api_key=api_key,
            secret_key=secret_key,
            passphrase=passphrase,
            testnet=testnet
        )
        self._save_credentials()
    
    def set_coingecko_credentials(self, api_key: Optional[str] = None, demo_api_key: Optional[str] = None):
        """Set CoinGecko credentials."""
        self._credentials.coingecko = CoinGeckoConfig(
            api_key=api_key,
            demo_api_key=demo_api_key
        )
        self._save_credentials()
    
    def set_alpha_vantage_credentials(self, api_key: str):
        """Set Alpha Vantage credentials."""
        self._credentials.alpha_vantage = AlphaVantageConfig(api_key=api_key)
        self._save_credentials()
    
    def set_fred_credentials(self, api_key: str):
        """Set FRED credentials."""
        self._credentials.fred = FREDConfig(api_key=api_key)
        self._save_credentials()
    
    def set_telegram_credentials(self, bot_token: str, chat_id: str):
        """Set Telegram credentials."""
        self._credentials.telegram = TelegramConfig(
            bot_token=bot_token,
            chat_id=chat_id
        )
        self._save_credentials()
    
    def get_hyperliquid_config(self) -> Optional[HyperliquidConfig]:
        """Get Hyperliquid configuration with environment fallback."""
        if self._credentials.hyperliquid:
            return self._credentials.hyperliquid
        
        # Try environment variables
        wallet = self._get_env_credential("HYPERLIQUID_WALLET")
        private_key = self._get_env_credential("HYPERLIQUID_PRIVATE_KEY")
        
        if wallet and private_key:
            return HyperliquidConfig(
                wallet_address=wallet,
                private_key=private_key,
                testnet=True
            )
        
        return None
    
    def get_binance_config(self) -> Optional[BinanceConfig]:
        """Get Binance configuration with environment fallback."""
        if self._credentials.binance:
            return self._credentials.binance
        
        # Try environment variables
        api_key = self._get_env_credential("BINANCE_API_KEY")
        secret_key = self._get_env_credential("BINANCE_SECRET_KEY")
        
        if api_key and secret_key:
            return BinanceConfig(
                api_key=api_key,
                secret_key=secret_key,
                testnet=True
            )
        
        return None
    
    def get_okx_config(self) -> Optional[OKXConfig]:
        """Get OKX configuration with environment fallback."""
        if self._credentials.okx:
            return self._credentials.okx
        
        # Try environment variables
        api_key = self._get_env_credential("OKX_API_KEY")
        secret_key = self._get_env_credential("OKX_SECRET_KEY")
        passphrase = self._get_env_credential("OKX_PASSPHRASE")
        
        if api_key and secret_key:
            return OKXConfig(
                api_key=api_key,
                secret_key=secret_key,
                passphrase=passphrase,
                testnet=True
            )
        
        return None
    
    def get_coingecko_config(self) -> Optional[CoinGeckoConfig]:
        """Get CoinGecko configuration with environment fallback."""
        if self._credentials.coingecko:
            return self._credentials.coingecko
        
        # Try environment variables
        api_key = self._get_env_credential("COINGECKO_API_KEY")
        demo_api_key = self._get_env_credential("COINGECKO_DEMO_API_KEY")
        
        if api_key or demo_api_key:
            return CoinGeckoConfig(
                api_key=api_key,
                demo_api_key=demo_api_key
            )
        
        return None
    
    def get_alpha_vantage_config(self) -> Optional[AlphaVantageConfig]:
        """Get Alpha Vantage configuration with environment fallback."""
        if self._credentials.alpha_vantage:
            return self._credentials.alpha_vantage
        
        # Try environment variables
        api_key = self._get_env_credential("ALPHA_VANTAGE_API_KEY")
        
        if api_key:
            return AlphaVantageConfig(api_key=api_key)
        
        return None
    
    def get_fred_config(self) -> Optional[FREDConfig]:
        """Get FRED configuration with environment fallback."""
        if self._credentials.fred:
            return self._credentials.fred
        
        # Try environment variables
        api_key = self._get_env_credential("FRED_API_KEY")
        
        if api_key:
            return FREDConfig(api_key=api_key)
        
        return None
    
    def get_telegram_config(self) -> Optional[TelegramConfig]:
        """Get Telegram configuration with environment fallback."""
        if self._credentials.telegram:
            return self._credentials.telegram
        
        # Try environment variables
        bot_token = self._get_env_credential("TELEGRAM_BOT_TOKEN")
        chat_id = self._get_env_credential("TELEGRAM_CHAT_ID")
        
        if bot_token and chat_id:
            return TelegramConfig(
                bot_token=bot_token,
                chat_id=chat_id
            )
        
        return None
    
    def validate_config(self) -> Dict[str, bool]:
        """Validate all configurations."""
        validation = {}
        
        # Check each exchange
        validation['hyperliquid'] = self.get_hyperliquid_config() is not None
        validation['binance'] = self.get_binance_config() is not None
        validation['okx'] = self.get_okx_config() is not None
        validation['coingecko'] = self.get_coingecko_config() is not None
        validation['alpha_vantage'] = self.get_alpha_vantage_config() is not None
        validation['fred'] = self.get_fred_config() is not None
        validation['telegram'] = self.get_telegram_config() is not None
        
        return validation
    
    def get_all_configs(self) -> ExchangeCredentials:
        """Get all configurations with environment fallbacks."""
        return ExchangeCredentials(
            hyperliquid=self.get_hyperliquid_config(),
            binance=self.get_binance_config(),
            okx=self.get_okx_config(),
            coingecko=self.get_coingecko_config(),
            alpha_vantage=self.get_alpha_vantage_config(),
            fred=self.get_fred_config(),
            telegram=self.get_telegram_config()
        )
    
    def clear_credentials(self):
        """Clear all stored credentials."""
        self._credentials = ExchangeCredentials()
        if self.encrypted_config_file.exists():
            self.encrypted_config_file.unlink()
        self.logger.info("Cleared all stored credentials")

# Global instance
_exchange_config = None

def get_exchange_config() -> SecureExchangeConfig:
    """Get global exchange configuration instance."""
    global _exchange_config
    if _exchange_config is None:
        _exchange_config = SecureExchangeConfig()
    return _exchange_config 
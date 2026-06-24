"""
API Configuration Management

Centralized configuration for all external APIs:
- Twitter API credentials
- Exchange API keys
- Database connections
- Model service endpoints
"""

import os
from typing import Dict, Optional, List
from dataclasses import dataclass
from pathlib import Path
import json
import logging
from cryptography.fernet import Fernet

@dataclass
class TwitterConfig:
    """Twitter API configuration."""
    bearer_token: Optional[str] = None
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    access_token: Optional[str] = None
    access_token_secret: Optional[str] = None

@dataclass  
class ExchangeConfig:
    """Exchange API configuration."""
    name: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    sandbox: bool = True
    rate_limit: int = 1200  # requests per minute

@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str = "sqlite:///crypto_trading.db"
    echo: bool = False
    pool_size: int = 10
    max_overflow: int = 20

class APIConfig:
    """
    Centralized API configuration manager with encryption support.
    
    Features:
    - Encrypted credential storage
    - Environment variable fallback
    - Configuration validation
    - Hot-reloading of settings
    """
    
    def __init__(self, config_file: str = "config/api_config.json"):
        """
        Initialize API configuration manager.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = Path(config_file)
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize encryption key
        self._encryption_key = self._get_or_create_encryption_key()
        self._cipher = Fernet(self._encryption_key)
        
        # Load configuration
        self._config = self._load_config()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for sensitive data."""
        key_file = Path("config/.encryption_key")
        key_file.parent.mkdir(parents=True, exist_ok=True)
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            return key
    
    def _encrypt_value(self, value: str) -> str:
        """Encrypt a sensitive value."""
        if not value:
            return value
        return self._cipher.encrypt(value.encode()).decode()
    
    def _decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a sensitive value."""
        if not encrypted_value:
            return encrypted_value
        try:
            return self._cipher.decrypt(encrypted_value.encode()).decode()
        except Exception:
            # If decryption fails, assume it's not encrypted
            return encrypted_value
    
    def _load_config(self) -> Dict:
        """Load configuration from file or create default."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Error loading config: {e}, using defaults")
        
        # Return default configuration
        return {
            "twitter": {
                "bearer_token": None,
                "api_key": None,
                "api_secret": None,
                "access_token": None,
                "access_token_secret": None
            },
            "exchanges": {
                "binance": {
                    "name": "binance",
                    "api_key": None,
                    "api_secret": None,
                    "sandbox": True,
                    "rate_limit": 1200
                },
                "coinbase": {
                    "name": "coinbase",
                    "api_key": None,
                    "api_secret": None,
                    "sandbox": True,
                    "rate_limit": 10
                }
            },
            "database": {
                "url": "sqlite:///crypto_trading.db",
                "echo": False,
                "pool_size": 10,
                "max_overflow": 20
            },
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 0,
                "password": None
            },
            "telegram": {
                "bot_token": None,
                "chat_id": None
            }
        }
    
    def _save_config(self):
        """Save current configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self._config, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
    
    def get_twitter_config(self) -> TwitterConfig:
        """Get Twitter API configuration."""
        twitter_config = self._config.get("twitter", {})
        
        return TwitterConfig(
            bearer_token=self._get_credential("TWITTER_BEARER_TOKEN", 
                                           twitter_config.get("bearer_token")),
            api_key=self._get_credential("TWITTER_API_KEY", 
                                       twitter_config.get("api_key")),
            api_secret=self._get_credential("TWITTER_API_SECRET", 
                                          twitter_config.get("api_secret")),
            access_token=self._get_credential("TWITTER_ACCESS_TOKEN", 
                                            twitter_config.get("access_token")),
            access_token_secret=self._get_credential("TWITTER_ACCESS_TOKEN_SECRET", 
                                                   twitter_config.get("access_token_secret"))
        )
    
    def get_exchange_config(self, exchange_name: str) -> Optional[ExchangeConfig]:
        """Get exchange API configuration."""
        exchanges = self._config.get("exchanges", {})
        exchange_config = exchanges.get(exchange_name)
        
        if not exchange_config:
            return None
        
        return ExchangeConfig(
            name=exchange_name,
            api_key=self._get_credential(f"{exchange_name.upper()}_API_KEY",
                                       exchange_config.get("api_key")),
            api_secret=self._get_credential(f"{exchange_name.upper()}_API_SECRET",
                                          exchange_config.get("api_secret")),
            sandbox=exchange_config.get("sandbox", True),
            rate_limit=exchange_config.get("rate_limit", 1200)
        )
    
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration."""
        db_config = self._config.get("database", {})
        
        return DatabaseConfig(
            url=os.getenv("DATABASE_URL", db_config.get("url", "sqlite:///crypto_trading.db")),
            echo=db_config.get("echo", False),
            pool_size=db_config.get("pool_size", 10),
            max_overflow=db_config.get("max_overflow", 20)
        )
    
    def _get_credential(self, env_var: str, config_value: Optional[str]) -> Optional[str]:
        """Get credential from environment variable or config file."""
        # Try environment variable first
        env_value = os.getenv(env_var)
        if env_value:
            return env_value
        
        # Try config file value
        if config_value:
            return self._decrypt_value(config_value)
        
        return None
    
    def set_twitter_credentials(self, bearer_token: str = None, api_key: str = None,
                              api_secret: str = None, access_token: str = None,
                              access_token_secret: str = None, encrypt: bool = True):
        """Set Twitter API credentials."""
        twitter_config = self._config.setdefault("twitter", {})
        
        if bearer_token:
            twitter_config["bearer_token"] = self._encrypt_value(bearer_token) if encrypt else bearer_token
        if api_key:
            twitter_config["api_key"] = self._encrypt_value(api_key) if encrypt else api_key
        if api_secret:
            twitter_config["api_secret"] = self._encrypt_value(api_secret) if encrypt else api_secret
        if access_token:
            twitter_config["access_token"] = self._encrypt_value(access_token) if encrypt else access_token
        if access_token_secret:
            twitter_config["access_token_secret"] = self._encrypt_value(access_token_secret) if encrypt else access_token_secret
        
        self._save_config()
    
    def set_exchange_credentials(self, exchange_name: str, api_key: str, 
                               api_secret: str, sandbox: bool = True, encrypt: bool = True):
        """Set exchange API credentials."""
        exchanges = self._config.setdefault("exchanges", {})
        exchange_config = exchanges.setdefault(exchange_name, {"name": exchange_name})
        
        exchange_config["api_key"] = self._encrypt_value(api_key) if encrypt else api_key
        exchange_config["api_secret"] = self._encrypt_value(api_secret) if encrypt else api_secret
        exchange_config["sandbox"] = sandbox
        
        self._save_config()
    
    def validate_config(self) -> Dict[str, bool]:
        """Validate configuration completeness."""
        validation_results = {}
        
        # Validate Twitter config
        twitter_config = self.get_twitter_config()
        validation_results["twitter"] = bool(
            twitter_config.bearer_token or 
            (twitter_config.api_key and twitter_config.api_secret)
        )
        
        # Validate exchange configs
        exchanges = self._config.get("exchanges", {})
        for exchange_name in exchanges:
            exchange_config = self.get_exchange_config(exchange_name)
            validation_results[f"exchange_{exchange_name}"] = bool(
                exchange_config and 
                exchange_config.api_key and 
                exchange_config.api_secret
            )
        
        # Validate database config
        db_config = self.get_database_config()
        validation_results["database"] = bool(db_config.url)
        
        return validation_results
    
    def get_available_exchanges(self) -> List[str]:
        """Get list of configured exchanges."""
        return list(self._config.get("exchanges", {}).keys())
    
    def reload_config(self):
        """Reload configuration from file."""
        self._config = self._load_config()
        self.logger.info("Configuration reloaded")

# Singleton instance
_api_config = None

def get_api_config() -> APIConfig:
    """Get singleton API configuration instance."""
    global _api_config
    if _api_config is None:
        _api_config = APIConfig()
    return _api_config
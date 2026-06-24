"""
Secure API Credential Manager
Manages encrypted storage and retrieval of API credentials
"""

import asyncio
import logging
import os
import secrets
import sys
from pathlib import Path
from typing import Dict, Optional, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import json
import time

sys.path.append(str(Path(__file__).parent.parent))

from api.coingecko import get_coingecko_client, CoinGeckoClient
from api.fred import get_fred_client, FREDClient
from api.exchanges.binance import BinanceClient
from api.exchanges.okx import OKXClient
from api.exchanges.hyperliquid import HyperliquidClient

logger = logging.getLogger(__name__)

PBKDF2_ITERATIONS = 100_000


class SecureAPIManager:
    """Professional-grade secure API credential manager"""

    def __init__(self, encryption_password: Optional[str] = None):
        self.credentials_file = Path("secure_credentials.enc")
        self.salt_file = Path("secure_credentials.salt")
        self.credentials: Dict[str, Dict] = {}
        self.clients: Dict[str, Any] = {}
        self.rate_limits: Dict[str, Dict] = {}

        if encryption_password is not None:
            self.encryption_password = encryption_password
        else:
            # Set ENCRYPTION_PASSWORD in your .env file
            self.encryption_password = os.getenv("ENCRYPTION_PASSWORD")
        if not self.encryption_password:
            raise ValueError(
                "ENCRYPTION_PASSWORD is required. Set it in your .env file before "
                "initializing the credential manager."
            )

        self._initialize_encryption()
        self._load_credentials()

    def _resolve_salt(self) -> bytes:
        if self.salt_file.exists():
            stored = self.salt_file.read_bytes()
            if len(stored) != 16:
                raise ValueError(
                    f"Invalid salt file {self.salt_file}; expected 16 bytes."
                )
            return stored

        # Per-deployment salt: a fixed salt in source lets anyone with the repo
        # precompute keys offline if the password ever leaks elsewhere.
        salt = secrets.token_bytes(16)
        self.salt_file.write_bytes(salt)
        return salt

    def _initialize_encryption(self):
        try:
            salt = self._resolve_salt()
            # 100k PBKDF2 rounds: slow enough to blunt brute force on a stolen
            # blob, still tolerable for occasional credential read/write.
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=PBKDF2_ITERATIONS,
            )
            key = base64.urlsafe_b64encode(
                kdf.derive(self.encryption_password.encode())
            )
            self.cipher = Fernet(key)
            logger.info("Encryption initialized successfully")
        except Exception as e:
            logger.error(f"Encryption initialization failed: {e}")
            raise

    def _encrypt_data(self, data: str) -> bytes:
        return self.cipher.encrypt(data.encode())

    def _decrypt_data(self, encrypted_data: bytes) -> str:
        return self.cipher.decrypt(encrypted_data).decode()

    def _load_credentials(self):
        try:
            if self.credentials_file.exists():
                encrypted_data = self.credentials_file.read_bytes()
                decrypted_data = self._decrypt_data(encrypted_data)
                self.credentials = json.loads(decrypted_data)
                logger.info(f"Loaded {len(self.credentials)} API credentials")
            else:
                logger.info("No existing credentials found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading credentials: {e}")
            self.credentials = {}

    def _save_credentials(self):
        try:
            data = json.dumps(self.credentials)
            encrypted_data = self._encrypt_data(data)
            self.credentials_file.write_bytes(encrypted_data)
            logger.info("Credentials saved securely")
        except Exception as e:
            logger.error(f"Error saving credentials: {e}")

    def add_api_credentials(
        self,
        name: str,
        api_key: str,
        api_secret: Optional[str] = None,
        passphrase: Optional[str] = None,
        rate_limit: int = 60,
    ) -> bool:
        try:
            self.credentials[name] = {
                "api_key": api_key,
                "api_secret": api_secret,
                "passphrase": passphrase,
                "rate_limit": rate_limit,
                "added_at": time.time(),
            }

            self.rate_limits[name] = {
                "remaining": rate_limit,
                "reset_time": time.time() + 60,
                "last_request": 0,
            }

            self._save_credentials()
            logger.info(f"Added credentials for {name}")
            return True

        except Exception as e:
            logger.error(f"Error adding credentials for {name}: {e}")
            return False

    def get_api_credentials(self, name: str) -> Optional[Dict]:
        return self.credentials.get(name)

    def get_active_apis(self) -> list:
        return list(self.credentials.keys())

    async def validate_api_connection(self, api_name: str) -> bool:
        try:
            if api_name == "alpha_vantage":
                return True
            elif api_name == "coingecko":
                client = self.get_coingecko_client()
                if client:
                    data = await client.get_global_market_data()
                    return data is not None
            elif api_name == "fred":
                client = self.get_fred_client()
                if client:
                    data = await client.get_series("FEDFUNDS", limit=1)
                    return data is not None
            elif api_name == "binance":
                client = self.get_binance_client()
                if client:
                    await client.connect()
                    return client.connected
            elif api_name == "okx":
                client = self.get_okx_client()
                if client:
                    await client.connect()
                    return client.connected
            elif api_name == "hyperliquid":
                client = self.get_hyperliquid_client()
                if client:
                    await client.connect()
                    return client.connected
            elif api_name == "telegram":
                return True

            return False

        except Exception as e:
            logger.error(f"API validation failed for {api_name}: {e}")
            return False

    def get_coingecko_client(self) -> Optional[CoinGeckoClient]:
        try:
            creds = self.get_api_credentials("coingecko")
            if creds:
                return get_coingecko_client(api_key=creds["api_key"])
            return None
        except Exception as e:
            logger.error(f"Error creating CoinGecko client: {e}")
            return None

    def get_fred_client(self) -> Optional[FREDClient]:
        try:
            creds = self.get_api_credentials("fred")
            if creds:
                return get_fred_client(api_key=creds["api_key"])
            return None
        except Exception as e:
            logger.error(f"Error creating FRED client: {e}")
            return None

    def get_binance_client(self) -> Optional[BinanceClient]:
        try:
            creds = self.get_api_credentials("binance")
            if creds:
                return BinanceClient(
                    api_key=creds["api_key"],
                    api_secret=creds["api_secret"],
                )
            return None
        except Exception as e:
            logger.error(f"Error creating Binance client: {e}")
            return None

    def get_okx_client(self) -> Optional[OKXClient]:
        try:
            creds = self.get_api_credentials("okx")
            if creds:
                return OKXClient(
                    api_key=creds["api_key"],
                    api_secret=creds["api_secret"],
                    passphrase=creds.get("passphrase"),
                )
            return None
        except Exception as e:
            logger.error(f"Error creating OKX client: {e}")
            return None

    def get_hyperliquid_client(self) -> Optional[HyperliquidClient]:
        try:
            creds = self.get_api_credentials("hyperliquid")
            if creds:
                return HyperliquidClient(
                    api_key=creds["api_key"],
                    wallet_address=creds.get("wallet_address"),
                )
            return None
        except Exception as e:
            logger.error(f"Error creating Hyperliquid client: {e}")
            return None

    async def test_all_connections(self) -> Dict[str, bool]:
        results = {}

        for api_name in self.get_active_apis():
            try:
                is_valid = await self.validate_api_connection(api_name)
                results[api_name] = is_valid
                status = "✅" if is_valid else "❌"
                logger.info(f"{status} {api_name}: {'Connected' if is_valid else 'Failed'}")
            except Exception as e:
                results[api_name] = False
                logger.error(f"{api_name} test error: {e}")

        return results

    def get_api_status(self) -> Dict[str, Dict]:
        status = {}

        for api_name in self.get_active_apis():
            creds = self.get_api_credentials(api_name)
            rate_limit = self.rate_limits.get(api_name, {})

            status[api_name] = {
                "has_credentials": bool(creds),
                "rate_limit_remaining": rate_limit.get("remaining", 0),
                "rate_limit_reset": rate_limit.get("reset_time", 0),
                "last_request": rate_limit.get("last_request", 0),
                "configured_at": creds.get("added_at", 0) if creds else 0,
            }

        return status

    def remove_api_credentials(self, name: str) -> bool:
        try:
            if name in self.credentials:
                del self.credentials[name]
                if name in self.rate_limits:
                    del self.rate_limits[name]
                self._save_credentials()
                logger.info(f"Removed credentials for {name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing credentials for {name}: {e}")
            return False

    def update_api_credentials(self, name: str, **kwargs) -> bool:
        try:
            if name in self.credentials:
                self.credentials[name].update(kwargs)
                self.credentials[name]["updated_at"] = time.time()
                self._save_credentials()
                logger.info(f"Updated credentials for {name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error updating credentials for {name}: {e}")
            return False

    def get_credentials_summary(self) -> Dict[str, Dict]:
        summary = {}

        for name, creds in self.credentials.items():
            summary[name] = {
                "has_api_key": bool(creds.get("api_key")),
                "has_api_secret": bool(creds.get("api_secret")),
                "has_passphrase": bool(creds.get("passphrase")),
                "rate_limit": creds.get("rate_limit", 0),
                "added_at": creds.get("added_at", 0),
                "updated_at": creds.get("updated_at", 0),
            }

        return summary


_secure_manager: Optional[SecureAPIManager] = None


def get_secure_api_manager() -> SecureAPIManager:
    global _secure_manager
    if _secure_manager is None:
        _secure_manager = SecureAPIManager()
    return _secure_manager


def initialize_secure_manager(encryption_password: Optional[str] = None) -> SecureAPIManager:
    global _secure_manager
    _secure_manager = SecureAPIManager(encryption_password)
    return _secure_manager

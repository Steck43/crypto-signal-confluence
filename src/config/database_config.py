"""
Database Configuration Module

Defines configuration parameters for database connections and data storage
in the institutional AI crypto trading system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class DatabaseType(Enum):
    """Database types."""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MONGODB = "mongodb"
    REDIS = "redis"
    INFLUXDB = "influxdb"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    
    # Database type
    database_type: DatabaseType = DatabaseType.SQLITE
    
    # Connection parameters
    host: str = "localhost"
    port: int = 5432
    database: str = "trading_system"
    username: str = ""
    password: str = ""
    
    # SQLite specific
    sqlite_path: str = "data/trading_system.db"
    
    # Connection pool
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    # Performance settings
    enable_connection_pooling: bool = True
    enable_query_cache: bool = True
    cache_size: int = 1000
    cache_ttl: int = 300  # 5 minutes
    
    # Data retention
    enable_data_retention: bool = True
    retention_days: int = 365  # 1 year
    enable_data_archiving: bool = True
    archive_frequency: str = "monthly"  # daily, weekly, monthly
    
    # Backup settings
    enable_backups: bool = True
    backup_frequency: str = "daily"  # hourly, daily, weekly
    backup_retention: int = 30  # days
    backup_compression: bool = True
    
    # Security
    enable_ssl: bool = False
    ssl_cert: str = ""
    ssl_key: str = ""
    ssl_ca: str = ""
    
    # Monitoring
    enable_query_logging: bool = False
    enable_performance_monitoring: bool = True
    slow_query_threshold: float = 1.0  # seconds
    
    # Redis specific
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    
    # InfluxDB specific
    influxdb_url: str = "http://localhost:8086"
    influxdb_token: str = ""
    influxdb_org: str = ""
    influxdb_bucket: str = "trading_data"
    
    def get_connection_string(self) -> str:
        """Get database connection string."""
        if self.database_type == DatabaseType.SQLITE:
            return f"sqlite:///{self.sqlite_path}"
        elif self.database_type == DatabaseType.POSTGRESQL:
            return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif self.database_type == DatabaseType.MYSQL:
            return f"mysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        else:
            raise ValueError(f"Unsupported database type: {self.database_type}")
    
    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration."""
        return {
            "host": self.redis_host,
            "port": self.redis_port,
            "db": self.redis_db,
            "password": self.redis_password if self.redis_password else None,
            "decode_responses": True
        }
    
    def get_influxdb_config(self) -> Dict[str, Any]:
        """Get InfluxDB configuration."""
        return {
            "url": self.influxdb_url,
            "token": self.influxdb_token,
            "org": self.influxdb_org,
            "bucket": self.influxdb_bucket
        }
    
    def validate_config(self) -> bool:
        """Validate database configuration."""
        try:
            if self.database_type == DatabaseType.SQLITE:
                # Ensure SQLite directory exists
                import os
                os.makedirs(os.path.dirname(self.sqlite_path), exist_ok=True)
                return True
            
            elif self.database_type in [DatabaseType.POSTGRESQL, DatabaseType.MYSQL]:
                # Validate required parameters
                if not self.host or not self.database:
                    return False
                return True
            
            elif self.database_type == DatabaseType.REDIS:
                # Validate Redis parameters
                if not self.redis_host:
                    return False
                return True
            
            elif self.database_type == DatabaseType.INFLUXDB:
                # Validate InfluxDB parameters
                if not self.influxdb_url or not self.influxdb_token:
                    return False
                return True
            
            return False
            
        except Exception:
            return False


# Default database configuration
default_database_config = DatabaseConfig() 
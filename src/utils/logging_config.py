"""
Windows-safe logging configuration for trading system
Handles Unicode encoding issues and provides clean console output
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logging(log_level=logging.INFO, log_to_file=True, log_to_console=True):
    """
    Setup clean logging without Unicode issues
    Args:
        log_level: Logging level (default: INFO)
        log_to_file: Write logs to file (default: True)
        log_to_console: Write logs to console (default: True)
    """
    if log_to_file:
        Path("logs").mkdir(exist_ok=True)
    class CleanFormatter(logging.Formatter):
        def format(self, record):
            formatted = super().format(record)
            clean_message = formatted.encode('ascii', 'ignore').decode('ascii')
            return clean_message
    handlers = []
    if log_to_file:
        file_handler = logging.FileHandler(
            f'logs/trading_{datetime.now().strftime("%Y%m%d")}.log',
            encoding='utf-8'
        )
        file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s | %(levelname)8s | %(name)20s | %(message)s'
            )
        )
        handlers.append(file_handler)
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            CleanFormatter(
                '%(asctime)s | %(levelname)8s | %(name)20s | %(message)s'
            )
        )
        handlers.append(console_handler)
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized successfully")
    logger.info(f"Log level: {logging.getLevelName(log_level)}")
    logger.info(f"Handlers: File={log_to_file}, Console={log_to_console}")


def get_trading_logger() -> logging.Logger:
    """Get the trading system logger"""
    return logging.getLogger('trading_system')
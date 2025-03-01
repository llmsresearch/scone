"""Logging utilities for SCONE."""

import logging
import os
import sys
from typing import Optional


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True,
) -> logging.Logger:
    """Get a logger with the specified name and configuration.
    
    Args:
        name: Name of the logger.
        level: Logging level.
        log_file: Path to log file. If None, no file logging is performed.
        console_output: Whether to output logs to console.
        
    Returns:
        Configured logger.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Add file handler if log_file is specified
    if log_file is not None:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler if console_output is True
    if console_output:
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console_output: bool = True,
) -> None:
    """Set up logging for the entire application.
    
    Args:
        log_file: Path to log file. If None, no file logging is performed.
        level: Logging level.
        console_output: Whether to output logs to console.
    """
    # Configure root logger
    root_logger = get_logger(
        name="scone",
        level=level,
        log_file=log_file,
        console_output=console_output,
    )
    
    # Configure library loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("numpy").setLevel(logging.WARNING)
    
    root_logger.info("Logging configured successfully.") 
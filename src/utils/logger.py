"""
Logging functionality for the trading project
"""

import os
import logging
import datetime
from typing import Optional

def setup_logger(log_dir: str = "logs", log_level: int = logging.INFO, 
                name: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with file and console handlers
    
    Args:
        log_dir (str): Directory for log files
        log_level (int): Logging level
        name (str, optional): Logger name (if None, use root logger)
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logs directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"trading_{timestamp}.log")
    
    # Get logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Remove existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Create formatter and add to handlers
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Log initial message
    logger.info(f"Logger initialized. Log file: {log_file}")
    
    return logger


# Create a default logger
logger = setup_logger()

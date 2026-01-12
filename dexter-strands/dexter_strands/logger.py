"""
Logging module for Dexter Strands agent.

Provides structured logging with console and file handlers for debugging
and monitoring agent behavior.
"""

import logging
import sys
from pathlib import Path
from typing import Any


# Global logger instance
_logger: logging.Logger | None = None


def setup_logger(
    log_level: str = "INFO",
    log_file: str | None = None,
    logger_name: str = "dexter_strands"
) -> logging.Logger:
    """
    Configure logger with console and file handlers.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file. If None, logs only to console
        logger_name: Name for the logger instance
        
    Returns:
        Configured logger instance
        
    Requirements:
        - 9.4: Support different log levels (DEBUG, INFO, WARNING, ERROR)
        - 9.5: Output logs to both console and file
    """
    global _logger
    
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logger.level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file specified)
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logger.level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    """
    Get the global logger instance.
    
    Returns:
        Logger instance
        
    Raises:
        RuntimeError: If logger has not been initialized with setup_logger()
    """
    global _logger
    if _logger is None:
        raise RuntimeError(
            "Logger not initialized. Call setup_logger() first."
        )
    return _logger


def log_query(query: str) -> None:
    """
    Log user query.
    
    Args:
        query: User query string
        
    Requirements:
        - 9.1: Log key events including query receipt
    """
    logger = get_logger()
    logger.info(f"User Query: {query}")


def log_tool_call(tool_name: str, args: dict[str, Any]) -> None:
    """
    Log tool invocation.
    
    Args:
        tool_name: Name of the tool being called
        args: Tool arguments/parameters
        
    Requirements:
        - 9.1: Log key events including tool calls
        - 9.3: Log tool names and parameters
    """
    logger = get_logger()
    logger.info(f"Tool Call: {tool_name} with args: {args}")


def log_response(response: str) -> None:
    """
    Log agent response.
    
    Args:
        response: Agent response string
        
    Requirements:
        - 9.1: Log key events including response generation
    """
    logger = get_logger()
    logger.info(f"Agent Response: {response[:200]}{'...' if len(response) > 200 else ''}")


def log_error(error: Exception) -> None:
    """
    Log error with stack trace.
    
    Args:
        error: Exception to log
        
    Requirements:
        - 9.2: Log error details with stack traces
    """
    logger = get_logger()
    logger.error(f"Error occurred: {error}", exc_info=True)


def log_info(message: str) -> None:
    """
    Log informational message.
    
    Args:
        message: Message to log
    """
    logger = get_logger()
    logger.info(message)


def log_debug(message: str) -> None:
    """
    Log debug message.
    
    Args:
        message: Message to log
    """
    logger = get_logger()
    logger.debug(message)


def log_warning(message: str) -> None:
    """
    Log warning message.
    
    Args:
        message: Message to log
    """
    logger = get_logger()
    logger.warning(message)

"""Dexter financial research agent built with Strands SDK."""

__version__ = "0.1.0"

# Export main components
from .agent_single import DexterAgent
from .cli import main as cli_main
from .config import Config
from .logger import setup_logger
from .model import create_bedrock_model
from .safety import SafetyHook, validate_user_input, handle_api_error, handle_tool_error

__all__ = [
    "DexterAgent",
    "cli_main",
    "Config",
    "setup_logger",
    "create_bedrock_model",
    "SafetyHook",
    "validate_user_input",
    "handle_api_error",
    "handle_tool_error",
]

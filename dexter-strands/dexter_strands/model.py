"""AWS Bedrock model integration for Dexter Strands agent.

This module provides functions to create and configure AWS Bedrock models
using the Strands SDK. It handles model initialization, credential management,
and cross-region inference configuration.
"""

import logging
from typing import Optional

import boto3
from botocore.config import Config as BotocoreConfig
from strands.models import BedrockModel

from .config import Config

logger = logging.getLogger(__name__)


def create_bedrock_model(
    config: Config,
    boto_session: Optional[boto3.Session] = None,
) -> BedrockModel:
    """Create and configure AWS Bedrock model with Claude 4.5 Haiku.
    
    This function creates a BedrockModel instance configured for the Dexter agent.
    It uses Claude 4.5 Haiku via the US cross-region inference profile.
    
    Model IDs:
    - US cross-region: us.anthropic.claude-haiku-4-5-20251001-v1:0 (recommended)
    - Global cross-region: global.anthropic.claude-haiku-4-5-20251001-v1:0
    - Single-region: anthropic.claude-haiku-4-5-20251001-v1:0
    
    The US cross-region inference profile automatically routes requests between
    US regions (us-east-1, us-east-2, us-west-2) for optimal performance while
    keeping data within the United States. This is recommended for most US-based
    use cases as it provides higher throughput and better availability.
    
    The model is configured with:
    - Temperature 0 for deterministic outputs
    - US cross-region inference (routes between US regions only)
    - AWS credentials from environment or credential chain
    - Automatic retry logic with exponential backoff (provided by Strands SDK)
    
    Args:
        config: Configuration object containing AWS and model settings
        boto_session: Optional boto3 Session. If None, creates a new session
            with credentials from environment or AWS credential chain.
    
    Returns:
        Configured BedrockModel instance ready for use with Strands Agent
        
    Raises:
        ValueError: If configuration is invalid
        ClientError: If AWS credentials are invalid or model access is denied
        
    Example:
        >>> config = Config.from_env()
        >>> model = create_bedrock_model(config)
        >>> # Use with Strands Agent
        >>> agent = Agent(model=model, ...)
    """
    logger.info(
        f"Creating Bedrock model: {config.bedrock_model_id} "
        f"in region {config.aws_region} with temperature {config.model_temperature}"
    )
    
    # Create boto3 session with credentials if provided
    if boto_session is None:
        session_kwargs = {}
        
        # Use explicit credentials if provided in config
        if config.aws_access_key_id and config.aws_secret_access_key:
            session_kwargs["aws_access_key_id"] = config.aws_access_key_id
            session_kwargs["aws_secret_access_key"] = config.aws_secret_access_key
            logger.debug("Using explicit AWS credentials from configuration")
        else:
            logger.debug("Using AWS credential chain (environment, IAM role, etc.)")
        
        # Set region
        if config.aws_region:
            session_kwargs["region_name"] = config.aws_region
        
        boto_session = boto3.Session(**session_kwargs)
    
    # Configure boto client with appropriate timeouts and retry settings
    # The Strands SDK will add its own user agent and handle retries
    boto_client_config = BotocoreConfig(
        read_timeout=120,  # 2 minutes for long-running inference
        connect_timeout=10,  # 10 seconds to establish connection
        retries={
            "max_attempts": 3,  # Retry up to 3 times
            "mode": "adaptive",  # Adaptive retry mode for better handling
        },
    )
    
    # Create BedrockModel with Strands SDK
    # The SDK handles:
    # - Automatic retry logic with exponential backoff
    # - Request formatting and response parsing
    # - Streaming support
    # - Tool calling integration
    # Note: We pass boto_session which already has region configured,
    # so we don't pass region_name separately (SDK doesn't allow both)
    model = BedrockModel(
        model_id=config.bedrock_model_id,
        temperature=config.model_temperature,
        boto_session=boto_session,
        boto_client_config=boto_client_config,
        # Note: max_tokens is optional and will use model defaults if not specified
        # For Claude 4.5 Haiku, the default is typically sufficient
    )
    
    logger.info("Bedrock model created successfully")
    logger.debug(f"Model configuration: {model.get_config()}")
    
    return model


def create_bedrock_model_simple(
    model_id: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    temperature: float = 0.0,
    region: str = "us-east-1",
    max_tokens: Optional[int] = None,
) -> BedrockModel:
    """Create a Bedrock model with simple parameters (convenience function).
    
    This is a simplified version of create_bedrock_model() for quick setup
    without requiring a full Config object. It uses AWS credentials from
    the environment or credential chain.
    
    Args:
        model_id: Bedrock model ID. Defaults to Claude 4.5 Haiku US cross-region
            inference profile for best availability and throughput within US regions.
        temperature: Model temperature (0.0 = deterministic, 2.0 = creative).
            Defaults to 0.0 for consistent outputs.
        region: AWS region for Bedrock. Defaults to us-east-1.
        max_tokens: Maximum tokens to generate. If None, uses model default.
    
    Returns:
        Configured BedrockModel instance
        
    Example:
        >>> model = create_bedrock_model_simple()
        >>> # Or with custom settings
        >>> model = create_bedrock_model_simple(
        ...     temperature=0.7,
        ...     region="us-west-2",
        ...     max_tokens=4096
        ... )
    """
    logger.info(f"Creating Bedrock model (simple): {model_id} in {region}")
    
    # Configure boto client
    boto_client_config = BotocoreConfig(
        read_timeout=120,
        connect_timeout=10,
        retries={"max_attempts": 3, "mode": "adaptive"},
    )
    
    # Build model config
    model_config = {
        "model_id": model_id,
        "temperature": temperature,
        "boto_client_config": boto_client_config,
        "region_name": region,
    }
    
    # Add max_tokens if specified
    if max_tokens is not None:
        model_config["max_tokens"] = max_tokens
    
    model = BedrockModel(**model_config)
    
    logger.info("Bedrock model created successfully")
    return model

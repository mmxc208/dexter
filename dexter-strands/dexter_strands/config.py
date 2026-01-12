"""Configuration management for Dexter Strands agent.

This module provides centralized configuration loading from environment variables
with validation and clear error messages.
"""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


class ConfigurationError(Exception):
    """Raised when configuration is missing or invalid."""

    pass


@dataclass
class Config:
    """Application configuration loaded from environment variables.
    
    This dataclass holds all configuration needed for the Dexter Strands agent,
    including AWS Bedrock settings, Financial API credentials, agent parameters,
    and logging configuration.
    """

    # AWS Bedrock Configuration (required fields first)
    aws_region: str
    bedrock_model_id: str
    model_temperature: float

    # Financial Datasets API
    financial_api_key: str
    financial_api_base_url: str

    # Agent Settings
    max_iterations: int
    max_steps_per_task: int

    # Logging Configuration
    log_level: str
    log_file: str

    # Context Management Settings (Phase 4)
    context_dir: str
    context_threshold_kb: int
    summary_ratio: float
    preserve_recent_messages: int

    # Langfuse Tracing Configuration (Phase 6)
    langfuse_enabled: bool
    langfuse_public_key: Optional[str]
    langfuse_secret_key: Optional[str]
    langfuse_host: str
    trace_environment: str
    trace_session_id: Optional[str]

    # AWS credentials (optional - can use IAM role or credential chain)
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None

    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "Config":
        """Load configuration from environment variables.
        
        Args:
            env_file: Optional path to .env file. If None, looks for .env in current directory.
            
        Returns:
            Config instance with all settings loaded and validated.
            
        Raises:
            ConfigurationError: If required configuration is missing or invalid.
        """
        # Load environment variables from .env file if it exists
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()

        # Collect all configuration errors to report them together
        errors = []

        # AWS Bedrock Configuration
        aws_region = os.getenv("AWS_REGION")
        if not aws_region:
            errors.append("AWS_REGION is required")

        bedrock_model_id = os.getenv("BEDROCK_MODEL_ID")
        if not bedrock_model_id:
            errors.append("BEDROCK_MODEL_ID is required")

        model_temperature_str = os.getenv("MODEL_TEMPERATURE")
        model_temperature = 0.0
        if model_temperature_str:
            try:
                model_temperature = float(model_temperature_str)
                if model_temperature < 0.0 or model_temperature > 2.0:
                    errors.append("MODEL_TEMPERATURE must be between 0.0 and 2.0")
            except ValueError:
                errors.append(f"MODEL_TEMPERATURE must be a valid float, got: {model_temperature_str}")
        else:
            errors.append("MODEL_TEMPERATURE is required")

        # AWS credentials (optional - can use IAM role or credential chain)
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

        # Financial Datasets API
        financial_api_key = os.getenv("FINANCIAL_DATASETS_API_KEY")
        if not financial_api_key:
            errors.append("FINANCIAL_DATASETS_API_KEY is required")

        financial_api_base_url = os.getenv(
            "FINANCIAL_API_BASE_URL", "https://api.financialdatasets.ai"
        )

        # Agent Settings
        max_iterations_str = os.getenv("MAX_ITERATIONS")
        max_iterations = 20  # default
        if max_iterations_str:
            try:
                max_iterations = int(max_iterations_str)
                if max_iterations <= 0:
                    errors.append("MAX_ITERATIONS must be a positive integer")
            except ValueError:
                errors.append(f"MAX_ITERATIONS must be a valid integer, got: {max_iterations_str}")
        else:
            errors.append("MAX_ITERATIONS is required")

        max_steps_per_task_str = os.getenv("MAX_STEPS_PER_TASK")
        max_steps_per_task = 5  # default
        if max_steps_per_task_str:
            try:
                max_steps_per_task = int(max_steps_per_task_str)
                if max_steps_per_task <= 0:
                    errors.append("MAX_STEPS_PER_TASK must be a positive integer")
            except ValueError:
                errors.append(
                    f"MAX_STEPS_PER_TASK must be a valid integer, got: {max_steps_per_task_str}"
                )
        else:
            errors.append("MAX_STEPS_PER_TASK is required")

        # Logging Configuration
        log_level = os.getenv("LOG_LEVEL", "INFO")
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if log_level.upper() not in valid_log_levels:
            errors.append(
                f"LOG_LEVEL must be one of {valid_log_levels}, got: {log_level}"
            )
        log_level = log_level.upper()

        log_file = os.getenv("LOG_FILE", "dexter-strands.log")

        # Context Management Settings (Phase 4)
        context_dir = os.getenv("CONTEXT_DIR", ".dexter-strands/context")

        context_threshold_kb_str = os.getenv("CONTEXT_THRESHOLD_KB")
        context_threshold_kb = 5  # default
        if context_threshold_kb_str:
            try:
                context_threshold_kb = int(context_threshold_kb_str)
                if context_threshold_kb <= 0:
                    errors.append("CONTEXT_THRESHOLD_KB must be a positive integer")
            except ValueError:
                errors.append(
                    f"CONTEXT_THRESHOLD_KB must be a valid integer, got: {context_threshold_kb_str}"
                )

        summary_ratio_str = os.getenv("SUMMARY_RATIO")
        summary_ratio = 0.3  # default
        if summary_ratio_str:
            try:
                summary_ratio = float(summary_ratio_str)
                if summary_ratio < 0.1 or summary_ratio > 0.8:
                    errors.append("SUMMARY_RATIO must be between 0.1 and 0.8")
            except ValueError:
                errors.append(
                    f"SUMMARY_RATIO must be a valid float, got: {summary_ratio_str}"
                )

        preserve_recent_messages_str = os.getenv("PRESERVE_RECENT_MESSAGES")
        preserve_recent_messages = 10  # default
        if preserve_recent_messages_str:
            try:
                preserve_recent_messages = int(preserve_recent_messages_str)
                if preserve_recent_messages <= 0:
                    errors.append("PRESERVE_RECENT_MESSAGES must be a positive integer")
            except ValueError:
                errors.append(
                    f"PRESERVE_RECENT_MESSAGES must be a valid integer, got: {preserve_recent_messages_str}"
                )

        # Langfuse Tracing Configuration (Phase 6)
        langfuse_enabled_str = os.getenv("LANGFUSE_ENABLED", "true")
        langfuse_enabled = langfuse_enabled_str.lower() in ["true", "1", "yes"]

        langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        
        # Validate Langfuse credentials if tracing is enabled
        if langfuse_enabled:
            if not langfuse_public_key or not langfuse_secret_key:
                # Log warning but don't fail - tracing will be disabled at runtime
                # This allows the agent to work without tracing configured
                pass

        langfuse_host = os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")

        trace_environment = os.getenv("TRACE_ENVIRONMENT", "dev")
        valid_environments = ["dev", "staging", "prod"]
        if trace_environment not in valid_environments:
            errors.append(
                f"TRACE_ENVIRONMENT must be one of {valid_environments}, got: {trace_environment}"
            )

        trace_session_id = os.getenv("TRACE_SESSION_ID")  # Optional - auto-generated if None

        # If there are any errors, raise them all together
        if errors:
            error_message = "Configuration errors:\n" + "\n".join(f"  - {err}" for err in errors)
            raise ConfigurationError(error_message)

        return cls(
            aws_region=aws_region,  # type: ignore
            bedrock_model_id=bedrock_model_id,  # type: ignore
            model_temperature=model_temperature,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            financial_api_key=financial_api_key,  # type: ignore
            financial_api_base_url=financial_api_base_url,
            max_iterations=max_iterations,
            max_steps_per_task=max_steps_per_task,
            log_level=log_level,
            log_file=log_file,
            context_dir=context_dir,
            context_threshold_kb=context_threshold_kb,
            summary_ratio=summary_ratio,
            preserve_recent_messages=preserve_recent_messages,
            langfuse_enabled=langfuse_enabled,
            langfuse_public_key=langfuse_public_key,
            langfuse_secret_key=langfuse_secret_key,
            langfuse_host=langfuse_host,
            trace_environment=trace_environment,
            trace_session_id=trace_session_id,
        )

    def validate(self) -> None:
        """Validate configuration values.
        
        Raises:
            ConfigurationError: If any configuration value is invalid.
        """
        errors = []

        # Validate temperature range
        if self.model_temperature < 0.0 or self.model_temperature > 2.0:
            errors.append(f"model_temperature must be between 0.0 and 2.0, got: {self.model_temperature}")

        # Validate positive integers
        if self.max_iterations <= 0:
            errors.append(f"max_iterations must be positive, got: {self.max_iterations}")

        if self.max_steps_per_task <= 0:
            errors.append(f"max_steps_per_task must be positive, got: {self.max_steps_per_task}")

        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            errors.append(f"log_level must be one of {valid_log_levels}, got: {self.log_level}")

        # Validate context management settings
        if self.context_threshold_kb <= 0:
            errors.append(f"context_threshold_kb must be positive, got: {self.context_threshold_kb}")

        if self.summary_ratio < 0.1 or self.summary_ratio > 0.8:
            errors.append(f"summary_ratio must be between 0.1 and 0.8, got: {self.summary_ratio}")

        if self.preserve_recent_messages <= 0:
            errors.append(f"preserve_recent_messages must be positive, got: {self.preserve_recent_messages}")

        # Validate Langfuse configuration
        valid_environments = ["dev", "staging", "prod"]
        if self.trace_environment not in valid_environments:
            errors.append(f"trace_environment must be one of {valid_environments}, got: {self.trace_environment}")

        # Note: We don't fail if credentials are missing - tracing will be disabled at runtime
        # This allows the agent to work without tracing configured

        if errors:
            error_message = "Configuration validation errors:\n" + "\n".join(f"  - {err}" for err in errors)
            raise ConfigurationError(error_message)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.validate()

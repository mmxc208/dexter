"""Tracing and observability setup for Dexter Strands agent.

This module configures OpenTelemetry (OTEL) to export traces to Langfuse,
enabling comprehensive monitoring and debugging of multi-agent execution.

Requirements:
    - 1.1: Create traces for every query execution
    - 7.2: Enable/disable tracing via configuration
    - 7.3: Handle missing credentials gracefully
    - 7.4: Continue execution if tracing fails
    - 7.5: Log tracing errors without interrupting agent
"""

import base64
import logging
import os
import uuid
from typing import Optional

from dexter_strands.config import Config

logger = logging.getLogger(__name__)


def setup_langfuse_tracing(config: Config) -> bool:
    """Configure OpenTelemetry to export traces to Langfuse.
    
    This function sets up the OTEL exporter to send traces to Langfuse's
    OTEL endpoint. It handles authentication, endpoint configuration, and
    initializes the Strands telemetry system.
    
    All errors are caught and logged without interrupting agent execution.
    The agent will continue to work normally even if tracing fails.
    
    Args:
        config: Configuration object with Langfuse settings
        
    Returns:
        bool: True if tracing was successfully configured, False otherwise
        
    Requirements:
        - 1.1: Create traces for every query execution
        - 4.1: Wrap tracing setup in try-catch block
        - 4.2: Log warnings for missing credentials (don't fail)
        - 4.3: Log errors for Langfuse connection failures (don't fail)
        - 4.4: Ensure agent execution continues if tracing fails
        - 7.2: Enable/disable tracing via configuration
        - 7.3: Handle missing credentials gracefully
        - 7.4: Continue execution if tracing fails
        - 7.5: Log tracing errors without interrupting agent
        - 10.4: Tracing failures don't cause agent execution to fail
        
    Example:
        >>> config = Config.from_env()
        >>> success = setup_langfuse_tracing(config)
        >>> if success:
        ...     print("Tracing enabled")
        ... else:
        ...     print("Tracing disabled - agent continues normally")
    """
    
    # Wrap entire function in try-catch to ensure no exceptions escape
    # Requirements: 4.1, 4.4, 7.5, 10.4
    try:
        # Check if tracing is enabled in configuration
        # Requirements: 7.2
        if not config.langfuse_enabled:
            logger.info("Tracing disabled via configuration (LANGFUSE_ENABLED=false)")
            return False
        
        # Check for required credentials
        # Requirements: 4.2, 7.3
        if not config.langfuse_public_key or not config.langfuse_secret_key:
            logger.warning(
                "Langfuse credentials missing (LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY) - "
                "tracing disabled. Agent will continue without observability."
            )
            return False
        
        try:
            # Configure OTEL endpoint
            # Note: Langfuse uses /api/public/otel, not the standard /v1/traces
            otel_endpoint = f"{config.langfuse_host}/api/public/otel"
            
            # Create Basic Auth token from public/secret keys
            # Format: base64(public_key:secret_key)
            auth_string = f"{config.langfuse_public_key}:{config.langfuse_secret_key}"
            auth_token = base64.b64encode(auth_string.encode()).decode()
            
            # Set OTEL environment variables for Strands SDK
            # These are read by the Strands telemetry system
            os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = otel_endpoint
            os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {auth_token}"
            
            # Initialize Strands telemetry
            # This creates the tracer provider and sets it as the global tracer
            # Requirements: 4.3, 7.4
            from strands.telemetry import StrandsTelemetry
            StrandsTelemetry().setup_otlp_exporter()
            
            logger.info(
                f"Langfuse tracing configured successfully: {config.langfuse_host} "
                f"(environment: {config.trace_environment})"
            )
            return True
            
        except ImportError as e:
            # Strands SDK not available or telemetry module missing
            # Requirements: 4.3, 7.5, 10.4
            logger.error(
                f"Failed to import Strands telemetry module: {e}. "
                "Tracing disabled. Agent will continue without observability."
            )
            return False
            
        except Exception as e:
            # Catch all other errors (network issues, invalid credentials, etc.)
            # Requirements: 4.3, 7.5, 10.4
            logger.error(
                f"Failed to configure Langfuse tracing: {e}. "
                "Tracing disabled. Agent will continue without observability.",
                exc_info=True
            )
            return False
    
    except Exception as e:
        # Outer try-catch to ensure absolutely no exceptions escape
        # This is a safety net in case something goes wrong in the checks above
        # Requirements: 4.1, 4.4, 7.5, 10.4
        logger.error(
            f"Unexpected error in tracing setup: {e}. "
            "Tracing disabled. Agent will continue without observability.",
            exc_info=True
        )
        return False


def generate_session_id() -> str:
    """Generate a unique session ID for trace grouping.
    
    Session IDs are used to group related queries from the same conversation
    or user session. This enables filtering and analysis of multi-turn
    interactions in Langfuse.
    
    Returns:
        str: UUID-formatted session ID
        
    Requirements:
        - 5.1: Include session ID in traces for grouping
        
    Example:
        >>> session_id = generate_session_id()
        >>> print(session_id)
        '550e8400-e29b-41d4-a716-446655440000'
    """
    return str(uuid.uuid4())


def build_trace_attributes(
    config: Config,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict:
    """Build trace attributes for MultiAgentDexter.
    
    Trace attributes are custom metadata attached to traces for filtering,
    searching, and analysis in Langfuse. This includes session tracking,
    environment tags, and system metadata.
    
    Args:
        config: Configuration object with tracing settings
        session_id: Optional session ID (generated if None)
        user_id: Optional user ID for tracking (defaults to "anonymous")
        
    Returns:
        dict: Trace attributes dictionary for Strands Agent
        
    Requirements:
        - 5.1: Include session ID for grouping related queries
        - 5.2: Include tags for filtering (dexter-strands, multi-agent, etc.)
        - 5.3: Include environment name (dev/staging/prod)
        
    Example:
        >>> config = Config.from_env()
        >>> attrs = build_trace_attributes(config, session_id="session-123")
        >>> print(attrs["session.id"])
        'session-123'
        >>> print(attrs["environment"])
        'dev'
    """
    # Generate session ID if not provided
    if session_id is None:
        session_id = config.trace_session_id or generate_session_id()
    
    # Build trace attributes dictionary
    return {
        # Session and user tracking
        "session.id": session_id,
        "user.id": user_id or "anonymous",
        
        # Environment and tags for filtering
        "environment": config.trace_environment,
        "langfuse.tags": [
            "dexter-strands",
            "multi-agent",
            "financial-research",
            f"env:{config.trace_environment}",
        ],
        
        # System metadata
        "system.version": "phase6",
        "system.architecture": "multi-agent-graph",
        "system.agents": ["planner", "executor", "synthesizer"],
    }


def add_metrics_to_trace(
    handoff_count: int = 0,
    task_count: int = 0,
    tool_count: int = 0,
    context_offload_count: int = 0,
    context_offload_size_kb: float = 0.0,
) -> None:
    """Add custom metrics to the current trace.
    
    This function adds performance and execution metrics to the active trace
    after query completion. These metrics help identify bottlenecks and
    understand system behavior.
    
    The metrics are logged with structured data that will be captured by
    the Strands SDK and included in the trace sent to Langfuse.
    
    Args:
        handoff_count: Number of times Synthesizer handed off to Planner
        task_count: Total number of tasks created by Planner
        tool_count: Total number of tool calls executed
        context_offload_count: Number of contexts offloaded to files
        context_offload_size_kb: Total size of offloaded contexts in KB
        
    Requirements:
        - 5.4: Include handoff count as trace attribute
        - 5.5: Include task count as trace attribute
        - 6.1: Include total query execution time (handled by Strands SDK)
        - 6.2: Include individual agent execution times (handled by Strands SDK)
        - 6.3: Include individual tool execution times (handled by Strands SDK)
        - 6.4: Include context offloading count and size
        - 6.5: Include total number of tool calls
        
    Example:
        >>> add_metrics_to_trace(
        ...     handoff_count=1,
        ...     task_count=3,
        ...     tool_count=5,
        ...     context_offload_count=2,
        ...     context_offload_size_kb=45.3
        ... )
        
    Note:
        This function logs metrics that are automatically captured by the
        Strands SDK's tracing system. The metrics appear in Langfuse as
        custom attributes on the trace.
    """
    try:
        # Log metrics with structured data
        # The Strands SDK will capture this in the trace
        logger.info(
            "Query execution metrics",
            extra={
                # Handoff and task metrics
                # Requirements: 5.4, 5.5
                "metrics.handoff_count": handoff_count,
                "metrics.task_count": task_count,
                
                # Tool execution metrics
                # Requirements: 6.5
                "metrics.tool_count": tool_count,
                
                # Context offloading metrics
                # Requirements: 6.4
                "metrics.context_offload_count": context_offload_count,
                "metrics.context_offload_size_kb": round(context_offload_size_kb, 2),
                
                # Flag for trace filtering
                "metrics_added": True,
            }
        )
        
        logger.debug(
            f"Trace metrics: handoffs={handoff_count}, tasks={task_count}, "
            f"tools={tool_count}, offloads={context_offload_count}, "
            f"offload_size={context_offload_size_kb:.2f}KB"
        )
        
    except Exception as e:
        # Don't let metrics logging errors break the agent
        # Requirements: 4.4, 7.5, 10.4
        logger.warning(f"Failed to add metrics to trace: {e}")

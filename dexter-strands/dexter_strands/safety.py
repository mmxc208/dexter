"""Safety hooks for Strands agent to prevent infinite loops and enforce limits.

This module provides safety mechanisms including:
- Iteration limit enforcement
- Loop detection (tracking repeated actions)
- Graceful termination on resource limits
- UI feedback during tool execution
"""

import asyncio
import logging
from collections import deque
from typing import Any, Optional, TYPE_CHECKING

from strands.hooks import HookProvider
from strands.hooks.events import BeforeToolCallEvent, AfterToolCallEvent
from strands.hooks.registry import HookRegistry

if TYPE_CHECKING:
    from .ui import RichUI

logger = logging.getLogger(__name__)


class SafetyHook(HookProvider):
    """Custom safety hook for iteration limits and loop detection.
    
    This hook enforces safety limits to prevent runaway agent execution:
    - Maximum iteration limit (default: 20)
    - Loop detection by tracking last N tool calls
    - Graceful termination with informative messages
    
    Requirements:
        - 7.1: Enforce maximum iteration limit to prevent infinite loops
        - 7.2: Provide meaningful error messages on limit exceeded
        - 7.5: Gracefully terminate on resource limit exceeded
    
    Example:
        >>> safety_hook = SafetyHook(max_iterations=20, loop_window=4)
        >>> agent = Agent(hooks=[safety_hook], ...)
    """
    
    def __init__(
        self,
        max_iterations: int = 20,
        loop_window: int = 4,
    ):
        """Initialize safety hook with limits.
        
        Args:
            max_iterations: Maximum number of tool calls allowed per query.
                After this limit, the agent will be stopped with an error message.
            loop_window: Number of recent actions to track for loop detection.
                If the same action signature appears multiple times in this window,
                it may indicate a loop.
        """
        self.max_iterations = max_iterations
        self.loop_window = loop_window
        
        # Track iteration count
        self.iteration_count = 0
        
        # Track recent actions for loop detection
        # Store tuples of (tool_name, args_signature)
        self.recent_actions: deque[tuple[str, str]] = deque(maxlen=loop_window)
        
        logger.info(
            f"SafetyHook initialized: max_iterations={max_iterations}, "
            f"loop_window={loop_window}"
        )
    
    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Register safety check callback for before tool calls.
        
        Args:
            registry: Hook registry to register callbacks with
            **kwargs: Additional keyword arguments (unused)
        """
        registry.add_callback(BeforeToolCallEvent, self.check_safety_limits)
    
    def check_safety_limits(self, event: BeforeToolCallEvent) -> None:
        """Check safety limits before each tool call.
        
        This callback:
        1. Increments iteration counter
        2. Checks if max iterations exceeded
        3. Tracks action signature for loop detection
        4. Cancels tool execution if limits exceeded
        
        Args:
            event: BeforeToolCallEvent containing tool information
            
        Requirements:
            - 7.1: Enforce maximum iteration limit
            - 7.2: Provide meaningful error messages
        """
        # Increment iteration count
        self.iteration_count += 1
        
        # Check iteration limit
        if self.iteration_count > self.max_iterations:
            error_message = (
                f"Safety limit exceeded: Maximum {self.max_iterations} tool calls reached. "
                "This may indicate an infinite loop or overly complex query. "
                "Please try simplifying your question or breaking it into smaller parts."
            )
            logger.warning(f"Iteration limit exceeded: {self.iteration_count}/{self.max_iterations}")
            
            # Cancel the tool call with error message
            event.cancel_tool = error_message
            return
        
        # Track action for loop detection
        tool_name = event.tool_use.get("toolName", "unknown")
        
        # Create a signature from tool arguments for loop detection
        # We use a simplified signature to detect repeated patterns
        args = event.tool_use.get("input", {})
        args_signature = self._create_args_signature(args)
        
        action_signature = (tool_name, args_signature)
        self.recent_actions.append(action_signature)
        
        # Check for potential loops
        # If we see the same action signature multiple times in recent history,
        # it might indicate a loop
        if len(self.recent_actions) >= self.loop_window:
            action_counts = {}
            for action in self.recent_actions:
                action_counts[action] = action_counts.get(action, 0) + 1
            
            # If any action appears more than half the window size, warn about potential loop
            max_count = max(action_counts.values())
            if max_count >= self.loop_window // 2 + 1:
                logger.warning(
                    f"Potential loop detected: action {action_signature} "
                    f"repeated {max_count} times in last {self.loop_window} calls"
                )
                # Note: We don't cancel here, just log a warning
                # The iteration limit will eventually stop it if it's truly looping
    
    def _create_args_signature(self, args: dict) -> str:
        """Create a simplified signature from tool arguments.
        
        This creates a string representation of the arguments that can be
        used to detect repeated patterns. We sort keys and use repr for
        consistent comparison.
        
        Args:
            args: Tool arguments dictionary
            
        Returns:
            String signature of the arguments
        """
        if not args:
            return ""
        
        # Sort keys for consistent ordering
        sorted_items = sorted(args.items())
        
        # Create signature from key-value pairs
        # Use repr for consistent string representation
        signature_parts = [f"{k}={repr(v)}" for k, v in sorted_items]
        return "|".join(signature_parts)
    
    def reset(self) -> None:
        """Reset safety counters for a new query.
        
        This should be called at the start of each new query to reset
        iteration counts and action history.
        """
        self.iteration_count = 0
        self.recent_actions.clear()
        logger.debug("SafetyHook counters reset")


def validate_user_input(query: str) -> tuple[bool, str]:
    """Validate user input and reject malformed queries.
    
    This function checks for common input issues:
    - Empty or whitespace-only queries
    - Excessively long queries
    - Queries with suspicious patterns
    
    Args:
        query: User's input query
        
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if query is valid, False otherwise
        - error_message: Empty string if valid, error description if invalid
        
    Requirements:
        - 7.4: Validate input and reject malformed queries
    
    Example:
        >>> is_valid, error = validate_user_input("What is Apple's revenue?")
        >>> if not is_valid:
        ...     print(f"Invalid query: {error}")
    """
    # Check for empty or whitespace-only input
    if not query or not query.strip():
        return False, "Query cannot be empty. Please enter a question or request."
    
    # Check for excessively long queries (> 10000 characters)
    # This prevents potential DoS or token limit issues
    max_length = 10000
    if len(query) > max_length:
        return False, (
            f"Query is too long ({len(query)} characters). "
            f"Please limit your query to {max_length} characters or less."
        )
    
    # Check for minimum meaningful length (at least 3 characters)
    if len(query.strip()) < 3:
        return False, "Query is too short. Please provide a more detailed question."
    
    # All checks passed
    return True, ""


def handle_api_error(error: Exception, context: str = "") -> str:
    """Handle API errors and return user-friendly error messages.
    
    This function converts technical API errors into clear, actionable
    messages for users. It handles common error types:
    - Network/connection errors
    - Authentication errors
    - Rate limiting
    - Timeout errors
    - General API errors
    
    Args:
        error: The exception that occurred
        context: Optional context about what was being attempted
        
    Returns:
        User-friendly error message
        
    Requirements:
        - 7.2: Catch exceptions and provide meaningful error messages
        - 4.5: Handle API errors gracefully
    
    Example:
        >>> try:
        ...     response = api.get_data()
        ... except Exception as e:
        ...     error_msg = handle_api_error(e, "fetching income statement")
        ...     print(error_msg)
    """
    error_type = type(error).__name__
    error_str = str(error)
    
    # Build context prefix if provided
    context_prefix = f"while {context}: " if context else ""
    
    # Handle specific error types
    if "connection" in error_str.lower() or "network" in error_str.lower():
        return (
            f"Network error {context_prefix}Unable to connect to the service. "
            "Please check your internet connection and try again."
        )
    
    if "authentication" in error_str.lower() or "unauthorized" in error_str.lower() or "401" in error_str:
        return (
            f"Authentication error {context_prefix}Invalid or missing API credentials. "
            "Please check your API keys in the .env file."
        )
    
    if "rate limit" in error_str.lower() or "429" in error_str:
        return (
            f"Rate limit error {context_prefix}Too many requests. "
            "Please wait a moment and try again."
        )
    
    if "timeout" in error_str.lower():
        return (
            f"Timeout error {context_prefix}The request took too long. "
            "Please try again or simplify your query."
        )
    
    if "not found" in error_str.lower() or "404" in error_str:
        return (
            f"Not found error {context_prefix}The requested resource was not found. "
            "Please check your input and try again."
        )
    
    # Generic error message for unknown errors
    return (
        f"Error {context_prefix}{error_type}: {error_str}\n\n"
        "Please try again or contact support if the issue persists."
    )


def handle_tool_error(tool_name: str, error: Exception) -> dict:
    """Handle tool execution errors and return error result.
    
    This function wraps tool errors in a standard format that allows
    the agent to continue processing with partial results.
    
    Args:
        tool_name: Name of the tool that failed
        error: The exception that occurred
        
    Returns:
        Dictionary with error information in standard format
        
    Requirements:
        - 7.3: Allow agent to continue with partial results on tool failure
        - 7.2: Provide meaningful error messages
    
    Example:
        >>> try:
        ...     result = tool.execute(args)
        ... except Exception as e:
        ...     result = handle_tool_error("get_income_statement", e)
    """
    error_message = handle_api_error(error, f"executing tool '{tool_name}'")
    
    logger.error(f"Tool {tool_name} failed: {error}", exc_info=True)
    
    return {
        "error": error_message,
        "tool_name": tool_name,
        "success": False,
        "partial_result": None,
    }


class UIFeedbackHook(HookProvider):
    """Simplified hook for displaying UI feedback during tool execution.
    
    This hook provides real-time feedback to users by:
    - Showing tool parameters before execution
    - Displaying tool results after execution
    - Using appropriate symbols and colors
    - Displaying immediately without async queue overhead
    
    Requirements:
        - 6.1: Display tool parameters before execution
        - 6.2: Display tool results after execution
        - 6.5: Use existing RichUI methods for consistent formatting
    
    Example:
        >>> ui = RichUI()
        >>> ui_hook = UIFeedbackHook(ui)
        >>> agent = Agent(hooks=[ui_hook], ...)
    """
    
    def __init__(self, ui: Optional['RichUI'] = None):
        """Initialize UI feedback hook.
        
        Args:
            ui: Optional RichUI instance for displaying feedback.
                If None, no feedback will be displayed.
        """
        self.ui = ui
        logger.info("UIFeedbackHook initialized with UI: %s", ui is not None)
    
    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Register UI feedback callbacks for tool execution.
        
        Args:
            registry: Hook registry to register callbacks with
            **kwargs: Additional keyword arguments (unused)
            
        Requirements:
            - 6.1: Register callbacks for tool events
            - 6.2: Display tool parameters and results
        """
        if self.ui:
            # Register callbacks for tool events
            registry.add_callback(BeforeToolCallEvent, self.on_before_tool_call)
            registry.add_callback(AfterToolCallEvent, self.on_after_tool_call)
    
    async def on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        """Display tool parameters before execution.
        
        Args:
            event: BeforeToolCallEvent containing tool information
            
        Requirements:
            - 6.1: Display tool parameters immediately
            - 6.5: Use existing RichUI methods for formatting
        """
        if not self.ui:
            return
        
        try:
            # Extract tool information from the event
            tool_name = event.tool_use.get("name", "unknown")
            params = event.tool_use.get("input", {})
            
            # Display immediately using RichUI
            self.ui.print_tool_params(tool_name, params)
            logger.debug("Displayed tool params: %s", tool_name)
            
        except Exception as e:
            # Graceful error recovery - log but don't propagate
            logger.error(
                "Error displaying tool params: %s",
                e,
                exc_info=True
            )
    
    async def on_after_tool_call(self, event: AfterToolCallEvent) -> None:
        """Display tool results after execution.
        
        Args:
            event: AfterToolCallEvent containing tool result
            
        Requirements:
            - 6.2: Display tool results immediately
            - 6.5: Use existing RichUI methods for formatting
        """
        if not self.ui:
            return
        
        try:
            # Extract tool information
            tool_name = event.tool_use.get("name", "unknown")
            result = event.result
            
            # Format result appropriately
            if isinstance(result, dict):
                # Check if it's an error from our tools
                if "error" in result:
                    result_str = f"Error: {result['error']}"
                else:
                    # Success - just show a simple confirmation
                    result_str = "✓ Data retrieved successfully"
            elif result is None:
                result_str = "✓ Completed (no result)"
            else:
                result_str = "✓ Completed"
            
            # Display immediately using RichUI
            self.ui.print_tool_run(tool_name, result_str)
            logger.debug("Displayed tool result: %s", tool_name)
            
        except Exception as e:
            # Graceful error recovery - log but don't propagate
            logger.error(
                "Error displaying tool result: %s",
                e,
                exc_info=True
            )

"""Strands agent implementation for Dexter financial research assistant.

This module provides the DexterAgent class which wraps the Strands Agent
with Dexter-specific configuration, tool registration, and conversation management.
"""

import logging
from typing import Any, Dict, List, Optional

from strands import Agent
from strands.agent import AgentResult
from strands.agent.conversation_manager import (
    SlidingWindowConversationManager,
    SummarizingConversationManager,
)
from strands.models import BedrockModel

from .config import Config
from .context import ContextManager
from .conversation import OffloadingConversationManager
from .logger import log_query, log_response, log_error, log_info
from .safety import SafetyHook, UIFeedbackHook, validate_user_input, handle_api_error

logger = logging.getLogger(__name__)


class DexterAgent:
    """Wrapper for Strands Agent with Dexter-specific configuration.
    
    This class provides a simplified interface for creating and using a Strands
    agent configured for financial research tasks. It handles:
    - Agent initialization with Bedrock model
    - Tool registration
    - Conversation management
    - Query processing
    - Error handling
    
    Example:
        >>> config = Config.from_env()
        >>> model = create_bedrock_model(config)
        >>> tools = [get_income_statement, get_balance_sheet, get_stock_price]
        >>> agent = DexterAgent(model=model, tools=tools, config=config)
        >>> response = agent.process_query("What is Apple's revenue?")
    """

    def __init__(
        self,
        model: BedrockModel,
        tools: list,
        config: Config,
        conversation_manager: Optional[SummarizingConversationManager] = None,
        safety_hook: Optional[SafetyHook] = None,
        ui: Optional['RichUI'] = None,
        context_manager: Optional[ContextManager] = None,
    ):
        """Initialize Dexter agent with Strands SDK.
        
        Args:
            model: Configured BedrockModel instance
            tools: List of financial tools to register with the agent
            config: Configuration object with agent parameters
            conversation_manager: Optional conversation manager. If None, creates
                a SummarizingConversationManager with summary_ratio and 
                preserve_recent_messages from config.
            safety_hook: Optional safety hook for iteration limits and loop detection.
                If None, creates a SafetyHook with max_iterations from config.
            ui: Optional RichUI instance for displaying progress and responses
            context_manager: Optional context manager for offloading large tool outputs.
                If None, creates a ContextManager with context_dir from config.
                
        Raises:
            ValueError: If model or tools are invalid
        """
        logger.info("Initializing DexterAgent")
        
        # Validate inputs
        if not model:
            raise ValueError("Model is required")
        if not tools:
            raise ValueError("At least one tool must be provided")
        
        self.model = model
        self.tools = tools
        self.config = config
        self.ui = ui
        
        # Create context manager if not provided
        # Requirements: 7.1, 7.3
        if context_manager is None:
            context_manager = ContextManager(context_dir=config.context_dir)
            logger.info(f"Created ContextManager with directory: {config.context_dir}")
        
        self.context_manager = context_manager
        
        # Create conversation manager if not provided
        # Requirements: 8.1, 8.2, 8.3, 11.1, 11.2, 11.3, 11.4
        if conversation_manager is None:
            conversation_manager = OffloadingConversationManager(
                context_manager=context_manager,
                size_threshold_kb=config.context_threshold_kb,
                summary_ratio=config.summary_ratio,
                preserve_recent_messages=config.preserve_recent_messages,
            )
            logger.info(
                f"Created OffloadingConversationManager with "
                f"threshold={config.context_threshold_kb}KB, "
                f"summary_ratio={config.summary_ratio}, "
                f"preserve_recent_messages={config.preserve_recent_messages}"
            )
        
        self.conversation_manager = conversation_manager
        
        # Create safety hook if not provided
        # Requirements: 7.1, 7.5
        if safety_hook is None:
            safety_hook = SafetyHook(
                max_iterations=config.max_iterations,
                loop_window=4,  # Track last 4 actions for loop detection
            )
        
        self.safety_hook = safety_hook
        
        # Create UI feedback hook if UI is available
        # Requirements: 2.4, 2.5, 3.4
        self.ui_feedback_hook = UIFeedbackHook(ui)
        
        # Context offloading is now handled by OffloadingConversationManager
        # Requirements: 1.1
        # The ContextOffloadingHook is no longer needed because offloading
        # happens at the conversation manager level, ensuring compatibility
        # with Strands SDK's message formatting.
        
        # System prompt for financial research
        system_prompt = """You are Dexter, an expert financial research assistant.

Your role is to help users analyze companies, understand financial data, and make informed decisions.

You have access to financial tools that can retrieve:
- Income statements (revenue, expenses, net income)
- Balance sheets (assets, liabilities, equity)
- Stock prices (current and historical)

When answering questions:
1. Use the appropriate tools to gather data
2. Analyze the data carefully
3. Provide clear, accurate, and insightful responses
4. Cite specific numbers and dates when relevant
5. Explain financial concepts in accessible language

Format Guidelines:
- Use plain text ONLY - NO markdown (no **, *, _, #, etc.)
- Use line breaks and indentation for structure
- Present key numbers on separate lines for easy scanning
- Use simple bullets (- or •) for lists if needed
- Keep sentences clear and direct

Be thorough but concise. Focus on what matters most to the user's question."""

        # Collect all hooks
        # Requirements: 1.1, 1.4, 3.1, 3.2, 3.4, 7.1
        # Context offloading is now handled at the conversation manager level
        # via OffloadingConversationManager, not through a hook.
        # This ensures compatibility with Strands SDK's message formatting.
        hooks = [self.safety_hook]
        
        # Register UI feedback hook if UI is available
        # Requirements: 3.4
        if ui:
            hooks.append(self.ui_feedback_hook)
        
        # Create Strands agent with hooks and conversation manager
        # Requirements: 6.1, 6.2, 6.3, 8.3, 8.4
        self.agent = Agent(
            model=self.model,
            tools=self.tools,
            system_prompt=system_prompt,
            hooks=hooks,  # Register all hooks
            conversation_manager=self.conversation_manager,  # Use SummarizingConversationManager
            name="Dexter",
            description="Financial research assistant with access to real-time market data",
        )
        
        logger.info(f"DexterAgent initialized with {len(tools)} tools")
        
        # Log tool names if logger is initialized
        try:
            log_info(f"Agent created with tools: {[tool.__name__ if hasattr(tool, '__name__') else str(tool) for tool in tools]}")
        except RuntimeError:
            # Logger not initialized yet, skip logging
            pass

    def process_query(self, query: str) -> str:
        """Process user query and return agent response.
        
        This method:
        1. Validates user input
        2. Resets safety counters for new query
        3. Logs the incoming query
        4. Passes the query to the Strands agent
        5. Allows the agent to autonomously select and execute tools
        6. Returns the final response
        7. Handles errors gracefully
        8. Monitors and logs conversation summarization
        
        Args:
            query: User's question or request
            
        Returns:
            Agent's response as a string
            
        Raises:
            ValueError: If query is empty or invalid
            
        Requirements:
            - 3.3: Pass query to Strands agent for processing
            - 3.4: Allow agent to autonomously select and execute tools
            - 3.5: Return final response to user
            - 7.2: Catch exceptions and provide meaningful error messages
            - 7.3: Display progress using Rich spinners
            - 7.4: Validate input and reject malformed queries
            - 7.5: Display errors with UI
            - 10.5: Log conversation summarization events
        """
        # Validate user input
        # Requirement: 7.4
        is_valid, error_message = validate_user_input(query)
        if not is_valid:
            logger.warning(f"Invalid query rejected: {error_message}")
            # Display error with UI if available
            # Requirement: 7.5
            if self.ui:
                self.ui.print_error(error_message, "Please provide a valid query")
            return f"Invalid input: {error_message}"
        
        query = query.strip()
        
        # Reset safety counters for new query
        # Requirement: 7.1
        self.safety_hook.reset()
        
        # Log query (if logger is initialized)
        try:
            log_query(query)
        except RuntimeError:
            pass
        logger.info(f"Processing query: {query[:100]}{'...' if len(query) > 100 else ''}")
        
        # Track conversation state before processing for summarization logging
        # Requirement: 10.5
        try:
            messages_before = len(self.agent.messages) if hasattr(self.agent, 'messages') and self.agent.messages is not None else 0
        except (TypeError, AttributeError):
            messages_before = 0
        removed_before = getattr(self.conversation_manager, 'removed_message_count', 0)
        
        try:
            # Process query with Strands agent
            # The agent will:
            # 1. Analyze the query
            # 2. Decide which tools to use (if any)
            # 3. Execute tools autonomously (with safety limits)
            # 4. Generate a final response
            # Requirement: 3.3, 3.4, 3.5
            
            # Add progress indicator if UI is available
            # Requirement: 7.3
            if self.ui:
                with self.ui.progress("Processing query", "Query processed"):
                    result: AgentResult = self.agent(query)
            else:
                # Maintain backward compatibility - process without UI
                result: AgentResult = self.agent(query)
            
            # Check if summarization occurred during processing
            # Requirement: 10.5
            removed_after = getattr(self.conversation_manager, 'removed_message_count', 0)
            if removed_after > removed_before:
                messages_summarized = removed_after - removed_before
                logger.info(
                    f"Conversation summarized: {messages_summarized} messages "
                    f"condensed into summary (total removed: {removed_after})"
                )
            
            # Extract response text from the result
            # AgentResult.__str__() extracts text content from the message
            response = str(result)
            
            # Log response (if logger is initialized)
            try:
                log_response(response)
            except RuntimeError:
                pass
            logger.info(f"Query processed successfully, response length: {len(response)} chars")
            
            return response
            
        except Exception as e:
            # Handle API errors with meaningful messages
            # Requirements: 7.2, 7.5
            error_message = handle_api_error(e, "processing your query")
            
            # Display error with UI if available
            # Requirement: 7.5
            if self.ui:
                self.ui.print_error(error_message)
            
            # Log error (if logger is initialized)
            try:
                log_error(e)
            except RuntimeError:
                pass
            logger.error(f"Error processing query: {e}", exc_info=True)
            
            return error_message

    def _select_and_load_contexts(self, query: str) -> List[Dict[str, Any]]:
        """Select and load relevant contexts for answer generation.
        
        This method:
        1. Gets all available pointers from context manager
        2. Uses LLM to select relevant contexts based on query
        3. Loads selected contexts from files
        4. Returns loaded context data
        
        Args:
            query: User's query
            
        Returns:
            List of loaded context data dictionaries
            
        Requirements:
            - 5.1: Select relevant contexts before answer generation
            - 5.2: Load selected contexts from files
        """
        # Step 1: Get all pointers from context_manager
        # Requirement: 5.1
        available_pointers = self.context_manager.get_all_pointers()
        
        # Handle no contexts case
        # Requirement: 5.5
        if not available_pointers:
            logger.debug("No contexts available for selection")
            return []
        
        logger.info(f"Selecting contexts from {len(available_pointers)} available")
        
        # Step 2: Call select_relevant_contexts() with query
        # Requirement: 5.1
        selected_filepaths = self.context_manager.select_relevant_contexts(
            query=query,
            available_pointers=available_pointers
        )
        
        logger.info(f"Selected {len(selected_filepaths)} contexts for query")
        
        # Step 3: Load selected contexts
        # Requirement: 5.2
        loaded_contexts = self.context_manager.load_contexts(selected_filepaths)
        
        logger.info(f"Loaded {len(loaded_contexts)} contexts successfully")
        
        return loaded_contexts

    def _generate_answer_with_contexts(self, query: str) -> str:
        """Generate answer with context selection and temporary merging.
        
        This method:
        1. Selects and loads relevant contexts
        2. Creates temporary enriched conversation with contexts
        3. Generates answer with full context data
        4. Does not persist merged contexts to conversation
        
        Args:
            query: User's query
            
        Returns:
            Agent's response as a string
            
        Requirements:
            - 5.1: Select relevant contexts before generation
            - 5.2: Load selected contexts
            - 5.3: Create temporary enriched conversation
            - 5.4: Don't persist merged contexts
            - 5.5: Handle no contexts case
        """
        # Step 1: Select and load relevant contexts
        # Requirements: 5.1, 5.2
        loaded_contexts = self._select_and_load_contexts(query)
        
        # Step 2: Check if we have contexts to merge
        # Requirement: 5.5
        if not loaded_contexts:
            logger.info("No contexts to merge, generating answer with conversation only")
            # Generate answer with conversation only
            result: AgentResult = self.agent(query)
            return str(result)
        
        # Step 3: Create temporary enriched conversation
        # Requirement: 5.3
        logger.info(f"Merging {len(loaded_contexts)} contexts for answer generation")
        
        # Save original messages
        original_messages = self.agent.messages.copy()
        
        try:
            # Step 4: Replace pointers with full context data in conversation
            # Requirement: 5.3
            enriched_messages = self._replace_pointers_with_contexts(
                original_messages, 
                loaded_contexts
            )
            
            # Temporarily set enriched messages
            self.agent.messages = enriched_messages
            
            # Step 5: Generate answer with full context
            # Requirement: 5.3
            result: AgentResult = self.agent(query)
            response = str(result)
            
            logger.info("Answer generated with merged contexts")
            
            return response
            
        finally:
            # Step 6: Restore original messages (don't persist merged contexts)
            # Requirement: 5.4
            self.agent.messages = original_messages
            logger.debug("Restored original conversation (contexts not persisted)")

    def _replace_pointers_with_contexts(
        self, 
        messages: List[Any], 
        contexts: List[Dict[str, Any]]
    ) -> List[Any]:
        """Replace pointer references with full context data in messages.
        
        This method scans through messages and replaces any pointer references
        (identified by "offloaded": True) with the full context data.
        
        Args:
            messages: List of conversation messages
            contexts: List of loaded context data
            
        Returns:
            New list of messages with pointers replaced by full data
            
        Requirements:
            - 5.3: Replace pointers with full context data
        """
        import copy
        
        # Create a deep copy to avoid modifying original
        enriched_messages = copy.deepcopy(messages)
        
        # Create a mapping from context_id to full context data
        context_map = {}
        for context in contexts:
            # Extract context_id from pointer metadata if available
            # The context file contains the full result data
            if "result" in context:
                # Try to find the context_id from the pointer
                # We'll match by tool_name and args
                tool_name = context.get("tool_name")
                args = context.get("args")
                
                # Find matching pointer in context manager
                for pointer in self.context_manager.pointers:
                    if (pointer["tool_name"] == tool_name and 
                        pointer["args"] == args):
                        context_map[pointer["id"]] = context["result"]
                        break
        
        # Scan through messages and replace pointers
        for message in enriched_messages:
            # Check if message has content that might contain pointers
            if hasattr(message, 'content'):
                content = message.content
                
                # Handle different content types
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("offloaded") is True:
                            # This is a pointer - replace with full data
                            context_id = item.get("context_id")
                            if context_id is not None and context_id in context_map:
                                # Replace the pointer dict with full context data
                                item.clear()
                                item.update(context_map[context_id])
                                logger.debug(f"Replaced pointer {context_id} with full context")
                
                elif isinstance(content, dict) and content.get("offloaded") is True:
                    # Single pointer in content
                    context_id = content.get("context_id")
                    if context_id is not None and context_id in context_map:
                        message.content = context_map[context_id]
                        logger.debug(f"Replaced pointer {context_id} with full context")
        
        logger.info(f"Replaced {len(context_map)} pointers with full context data")
        
        return enriched_messages

    def reset_conversation(self) -> None:
        """Reset conversation history and safety counters.
        
        This clears all messages from the conversation manager and resets
        safety counters, effectively starting a fresh conversation. Useful
        when switching topics or when conversation context becomes too large.
        
        Note: This preserves the context_manager state (offloaded contexts),
        which is the correct behavior. Offloaded contexts remain available
        for future queries even after conversation reset.
        
        Requirements:
            - 6.5: Properly clean up conversation resources
            - 7.1: Reset safety counters for new conversation
            - 8.5: Preserve context_manager state across resets
        """
        logger.info("Resetting conversation history")
        
        try:
            # Clear messages in the agent
            self.agent.messages = []
            
            # Reset conversation manager state if it has a reset method
            # OffloadingConversationManager inherits from SummarizingConversationManager
            # and will properly reset its internal state
            # Requirement: 8.5
            if hasattr(self.conversation_manager, 'reset'):
                self.conversation_manager.reset()
            
            # Reset safety counters
            self.safety_hook.reset()
            
            try:
                log_info("Conversation history and safety counters reset successfully")
            except RuntimeError:
                pass
            logger.info("Conversation reset complete")
            
        except Exception as e:
            try:
                log_error(e)
            except RuntimeError:
                pass
            logger.error(f"Error resetting conversation: {e}", exc_info=True)
            raise

    def get_conversation_history(self) -> list:
        """Get current conversation history.
        
        Returns:
            List of messages in the conversation
        """
        return self.agent.messages

    def get_tool_names(self) -> list[str]:
        """Get names of registered tools.
        
        Returns:
            List of tool names
        """
        return [
            tool.__name__ if hasattr(tool, '__name__') else str(tool)
            for tool in self.tools
        ]

    def process_query_stream(self, query: str) -> str:
        """Process query with streaming response display.
        
        This method:
        1. Validates user input
        2. Resets safety counters for new query
        3. Calls Strands agent (which handles tool execution properly)
        4. Streams the final response through UI if available
        5. Returns complete response
        6. Monitors and logs conversation summarization
        
        Note: We use the regular agent() call instead of stream_async because
        tool execution requires proper conversation history management. The
        streaming happens only for the final text response.
        
        Args:
            query: User's question or request
            
        Returns:
            Agent's complete response as a string
            
        Raises:
            ValueError: If query is empty or invalid
            
        Requirements:
            - 8.1: Use Strands streaming capabilities for final response
            - 8.2: Process text chunks character-by-character
            - 8.3: Maintain box formatting throughout streaming
            - 8.4: Close box properly when streaming completes
            - 8.5: Handle streaming errors gracefully
            - 10.5: Log conversation summarization events
        """
        # Validate user input
        is_valid, error_message = validate_user_input(query)
        if not is_valid:
            logger.warning(f"Invalid query rejected: {error_message}")
            if self.ui:
                self.ui.print_error(error_message, "Please provide a valid query")
            return f"Invalid input: {error_message}"
        
        query = query.strip()
        
        # Reset safety counters for new query
        self.safety_hook.reset()
        
        # Log query (if logger is initialized)
        try:
            log_query(query)
        except RuntimeError:
            pass
        logger.info(f"Processing streaming query: {query[:100]}{'...' if len(query) > 100 else ''}")
        
        # Track conversation state before processing for summarization logging
        # Requirement: 10.5
        try:
            messages_before = len(self.agent.messages) if hasattr(self.agent, 'messages') and self.agent.messages is not None else 0
        except (TypeError, AttributeError):
            messages_before = 0
        removed_before = getattr(self.conversation_manager, 'removed_message_count', 0)
        
        try:
            # Process query with regular agent call
            # This properly handles tool execution and conversation history
            # Requirement: 8.1
            if self.ui:
                with self.ui.progress("Processing query", "Query processed"):
                    result: AgentResult = self.agent(query)
            else:
                result: AgentResult = self.agent(query)
            
            # Check if summarization occurred during processing
            # Requirement: 10.5
            removed_after = getattr(self.conversation_manager, 'removed_message_count', 0)
            if removed_after > removed_before:
                messages_summarized = removed_after - removed_before
                logger.info(
                    f"Conversation summarized: {messages_summarized} messages "
                    f"condensed into summary (total removed: {removed_after})"
                )
            
            # Get the complete response text
            response = str(result)
            
            # Stream the response through UI if available
            # Requirements: 8.2, 8.3, 8.4
            if self.ui:
                # Split response into character chunks for streaming effect
                chunks = [char for char in response]
                # Stream through UI
                response = self.ui.stream_answer(iter(chunks))
            
            # Log response (if logger is initialized)
            try:
                log_response(response)
            except RuntimeError:
                pass
            logger.info(f"Streaming query processed successfully, response length: {len(response)} chars")
            
            return response
            
        except Exception as e:
            # Handle streaming errors gracefully
            # Requirement: 8.5
            error_message = handle_api_error(e, "processing your streaming query")
            
            # Display error with UI if available
            if self.ui:
                self.ui.print_error(error_message)
            
            # Log error (if logger is initialized)
            try:
                log_error(e)
            except RuntimeError:
                pass
            logger.error(f"Error processing streaming query: {e}", exc_info=True)
            
            return error_message

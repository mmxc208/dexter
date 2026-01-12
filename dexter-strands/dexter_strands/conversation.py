"""Conversation management with context offloading for Dexter Strands agent.

This module provides a custom conversation manager that extends
SummarizingConversationManager with intelligent context offloading:
- Detects large tool result messages
- Offloads them to files via ContextManager
- Replaces message content with lightweight pointers
- Preserves message structure and metadata

The offloading happens AFTER the SDK has properly formatted messages,
ensuring compatibility with Strands' internal message structure.
"""

import json
import logging
from typing import TYPE_CHECKING, Any, Optional

from strands.agent.conversation_manager import SummarizingConversationManager

from .context import ContextManager

if TYPE_CHECKING:
    from strands.agent.agent import Agent

logger = logging.getLogger(__name__)



class OffloadingConversationManager(SummarizingConversationManager):
    """Conversation manager that offloads large tool results to files.
    
    This manager wraps SummarizingConversationManager and adds context
    offloading functionality. It intercepts messages as they're added to
    the conversation and offloads large tool results to files, replacing
    them with lightweight pointers.
    
    The offloading happens AFTER the SDK has properly formatted the message,
    ensuring compatibility with Strands' internal message structure.
    
    Responsibilities:
    - Delegate conversation management to SummarizingConversationManager
    - Detect tool result messages
    - Calculate message sizes
    - Offload large tool results via ContextManager
    - Replace message content with pointers
    - Preserve message structure and metadata
    """
    
    def __init__(
        self,
        context_manager: ContextManager,
        size_threshold_kb: int = 5,
        summary_ratio: float = 0.3,
        preserve_recent_messages: int = 10,
    ):
        """Initialize offloading conversation manager.
        
        Args:
            context_manager: ContextManager instance for offloading
            size_threshold_kb: Size threshold in KB for offloading
            summary_ratio: Ratio of old messages to summarize (0.0-1.0)
            preserve_recent_messages: Number of recent messages to preserve
        """
        # Initialize parent SummarizingConversationManager
        super().__init__(
            summary_ratio=summary_ratio,
            preserve_recent_messages=preserve_recent_messages,
        )
        
        self.context_manager = context_manager
        self.threshold = size_threshold_kb * 1024  # Convert to bytes
        
        logger.info(
            f"OffloadingConversationManager initialized with "
            f"threshold={size_threshold_kb}KB, "
            f"summary_ratio={summary_ratio}, "
            f"preserve_recent={preserve_recent_messages}"
        )

    
    def apply_management(self, agent: "Agent", **kwargs: Any) -> None:
        """Apply management strategy with context offloading.
        
        This method is called by the agent during execution to manage
        conversation history. We override it to:
        1. Delegate to parent for summarization management
        2. Check recent messages for large tool results
        3. Offload large tool results to files
        
        Args:
            agent: The agent whose conversation history will be managed
            **kwargs: Additional keyword arguments
        """
        # Step 1: Delegate to parent for summarization management
        super().apply_management(agent, **kwargs)
        
        # Step 2: Check recent messages for large tool results
        # We only check the last few messages since offloading should happen
        # soon after a tool result is added
        messages_to_check = agent.messages[-5:] if len(agent.messages) > 5 else agent.messages
        
        for message in messages_to_check:
            # Step 3: Check if message is a tool result
            if not self._is_tool_result_message(message):
                continue
            
            # Step 4: Calculate message size
            try:
                message_size = self._calculate_message_size(message)
            except Exception as e:
                logger.warning(f"Failed to calculate message size: {e}")
                continue
            
            # Step 5: Check if offloading is needed
            if message_size <= self.threshold:
                logger.debug(
                    f"Message size {message_size/1024:.1f}KB below threshold "
                    f"{self.threshold/1024:.1f}KB, not offloading"
                )
                continue
            
            # Step 6: Check if already offloaded (avoid re-offloading)
            if self._is_already_offloaded(message):
                logger.debug("Message already offloaded, skipping")
                continue
            
            # Step 7: Offload large tool result
            try:
                self._offload_message(message, message_size)
            except Exception as e:
                logger.error(f"Failed to offload message: {e}", exc_info=True)
                # Leave message unchanged on error

    
    def _is_tool_result_message(self, message: Any) -> bool:
        """Check if message is a tool result.
        
        Args:
            message: Message to check
            
        Returns:
            True if message is a tool result, False otherwise
        """
        # Check for role attribute
        if hasattr(message, 'role'):
            if message.role == 'tool':
                return True
        
        # Check for tool_use_id attribute
        if hasattr(message, 'tool_use_id'):
            return True
        
        # Check if message is a dict with role
        if isinstance(message, dict):
            if message.get('role') == 'tool':
                return True
            if 'tool_use_id' in message:
                return True
            
            # Check content for toolResult
            content = message.get('content', [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and 'toolResult' in item:
                        return True
        
        return False

    
    def _calculate_message_size(self, message: Any) -> int:
        """Calculate size of message content in bytes.
        
        Args:
            message: Message to calculate size for
            
        Returns:
            Size in bytes
        """
        try:
            # Try to get content attribute
            if hasattr(message, 'content'):
                content = message.content
            elif isinstance(message, dict) and 'content' in message:
                content = message['content']
            else:
                # Fallback: serialize entire message
                content = message
            
            # Serialize to JSON to get accurate size
            content_json = json.dumps(
                content,
                default=self.context_manager._json_serializer
            )
            return len(content_json.encode('utf-8'))
            
        except Exception as e:
            logger.warning(f"Failed to calculate message size: {e}")
            # Fallback: use string length
            return len(str(message))

    
    def _offload_message(self, message: Any, message_size: int) -> None:
        """Offload message content to file and replace with pointer.
        
        Args:
            message: Message to offload
            message_size: Size of message in bytes
        """
        # Extract tool information from message
        tool_name = self._extract_tool_name(message)
        tool_args = self._extract_tool_args(message)
        content = self._extract_content(message)
        
        logger.info(
            f"Offloading {tool_name} result "
            f"({message_size/1024:.1f}KB > {self.threshold/1024:.1f}KB)"
        )
        
        # Save context to file
        filepath = self.context_manager.save_context(
            tool_name=tool_name,
            args=tool_args,
            result=content
        )
        
        # Get the pointer that was just created
        pointer = self.context_manager.pointers[-1]
        
        # Create pointer text string with summary, size, context_id
        pointer_text = (
            f"[Context offloaded to {filepath}]\n"
            f"Summary: {pointer['summary']}\n"
            f"Size: {pointer['size_kb']:.1f}KB\n"
            f"Context ID: {pointer['id']}"
        )
        
        # Replace message content with pointer
        self._replace_message_content(message, pointer_text)
        
        logger.info(
            f"Context offloaded: {tool_name} "
            f"({message_size/1024:.1f}KB) -> {filepath}"
        )

    
    def _extract_tool_name(self, message: Any) -> str:
        """Extract tool name from message.
        
        Args:
            message: Message to extract tool name from
            
        Returns:
            Tool name or "unknown_tool" if not found
        """
        # Try various attributes
        if hasattr(message, 'tool_name'):
            return message.tool_name
        if isinstance(message, dict) and 'tool_name' in message:
            return message['tool_name']
        if hasattr(message, 'name'):
            return message.name
        if isinstance(message, dict) and 'name' in message:
            return message['name']
        
        # Check content for toolResult with toolUseId
        if isinstance(message, dict):
            content = message.get('content', [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and 'toolResult' in item:
                        # Try to extract tool name from toolUseId
                        tool_use_id = item['toolResult'].get('toolUseId', '')
                        if tool_use_id:
                            # toolUseId often contains the tool name
                            return f"tool_from_{tool_use_id}"
        
        logger.debug("Could not extract tool name from message")
        return "unknown_tool"
    
    def _extract_tool_args(self, message: Any) -> dict:
        """Extract tool arguments from message.
        
        Args:
            message: Message to extract tool arguments from
            
        Returns:
            Tool arguments dict or empty dict if not found
        """
        # Try various attributes
        if hasattr(message, 'tool_args'):
            return message.tool_args
        if isinstance(message, dict) and 'tool_args' in message:
            return message['tool_args']
        if hasattr(message, 'input'):
            return message.input
        if isinstance(message, dict) and 'input' in message:
            return message['input']
        
        logger.debug("Could not extract tool args from message")
        return {}
    
    def _extract_content(self, message: Any) -> Any:
        """Extract content from message.
        
        Args:
            message: Message to extract content from
            
        Returns:
            Message content
        """
        if hasattr(message, 'content'):
            return message.content
        if isinstance(message, dict) and 'content' in message:
            return message['content']
        
        return message

    
    def _replace_message_content(self, message: Any, pointer_text: str) -> None:
        """Replace message content with pointer text.
        
        IMPORTANT: We must preserve the toolResult structure for Strands SDK compatibility.
        The SDK requires tool_result messages to maintain their structure, so we replace
        the content INSIDE the toolResult, not the toolResult itself.
        
        Args:
            message: Message to modify
            pointer_text: Pointer text to replace content with
        """
        if hasattr(message, 'content'):
            message.content = pointer_text
        elif isinstance(message, dict):
            # For Strands messages, content is typically a list of content blocks
            # We need to replace the content INSIDE toolResult, not remove toolResult
            if 'content' in message:
                content = message['content']
                if isinstance(content, list):
                    # Replace content inside toolResult blocks
                    for item in content:
                        if isinstance(item, dict) and 'toolResult' in item:
                            # Keep the toolResult structure but replace its content
                            tool_result = item['toolResult']
                            # Replace the content field inside toolResult
                            if 'content' in tool_result:
                                # If content is a list, replace with single text block
                                if isinstance(tool_result['content'], list):
                                    tool_result['content'] = [{'text': pointer_text}]
                                else:
                                    tool_result['content'] = pointer_text
                            else:
                                # Add content field if it doesn't exist
                                tool_result['content'] = [{'text': pointer_text}]
                else:
                    # Simple content, just replace
                    message['content'] = pointer_text
            else:
                message['content'] = pointer_text
        else:
            logger.warning(f"Could not replace content in message type: {type(message)}")
    
    def _is_already_offloaded(self, message: Any) -> bool:
        """Check if message has already been offloaded.
        
        Args:
            message: Message to check
            
        Returns:
            True if message contains offload pointer, False otherwise
        """
        try:
            content = self._extract_content(message)
            
            # Check if content is a string containing offload marker
            if isinstance(content, str):
                return '[Context offloaded to' in content
            
            # Check if content is a list with text blocks containing offload marker
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and 'text' in item:
                        if '[Context offloaded to' in item['text']:
                            return True
            
            return False
        except Exception:
            return False

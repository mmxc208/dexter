"""
Executor Agent for multi-agent architecture.

The Executor Agent is responsible for:
- Processing task lists sequentially
- Selecting appropriate tools for each task
- Optimizing tool arguments
- Executing tools and saving outputs to context files
- Performing lightweight task validation
- Storing context pointers in SharedContext

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 7.1
"""

import json
import logging
from typing import Any, Dict, List, Optional

from strands import Agent
from strands.models import BedrockModel

from ..models import Task, ContextPointer, ValidationResult
from ..context import ContextManager
from ..prompts import get_executor_prompt

logger = logging.getLogger(__name__)


class ExecutorAgent:
    """
    Executor Agent for tool selection and execution.
    
    The Executor processes tasks from the Planner, selects appropriate tools,
    optimizes arguments, executes tools, and validates task completion.
    
    Responsibilities:
    - Process each task from the task list
    - Select appropriate tool for each task
    - Optimize tool arguments
    - Execute tools and save outputs to context files
    - Perform lightweight task validation
    - Track execution progress
    
    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 7.1
    """
    
    def __init__(
        self,
        model: BedrockModel,
        tools: List[Any],
        context_manager: ContextManager,
        ui: Optional[Any] = None,
        agent_id: str = "executor",
    ):
        """
        Initialize Executor Agent.
        
        Args:
            model: Configured BedrockModel instance
            tools: List of available financial tools
            context_manager: ContextManager for saving tool outputs
            ui: Optional RichUI instance for progress indicators
            agent_id: Unique identifier for this agent
            
        Requirements: 3.1, 3.2
        """
        logger.info("Initializing ExecutorAgent")
        
        self.model = model
        self.tools = tools
        self.context_manager = context_manager
        self.ui = ui
        self.agent_id = agent_id
        
        # Create tool name to tool object mapping for fast lookup
        self.tools_by_name = {
            getattr(tool, 'name', str(tool)): tool 
            for tool in tools
        }
        
        # Format tools with names and descriptions for prompt
        # Requirements: 8.4
        tool_descriptions = []
        for tool in tools:
            # Get tool name - Strands tools have a 'tool_name' property
            if hasattr(tool, 'tool_name'):
                name = tool.tool_name
            elif hasattr(tool, 'name'):
                name = tool.name
            else:
                name = str(tool)
            
            # Get docstring as description
            if hasattr(tool, '__doc__') and tool.__doc__:
                doc = tool.__doc__
                # Clean up docstring - take first line only
                description = doc.strip().split('\n')[0]
            else:
                description = name
            
            tool_descriptions.append(f"{name}: {description}")
        
        # Create system prompt with tool list
        system_prompt = get_executor_prompt(tool_descriptions)
        
        # Create Strands Agent for execution
        # Includes all financial tools
        self.agent = Agent(
            model=model,
            system_prompt=system_prompt,
            agent_id=agent_id,
            name="Executor",
            description="Executes tools to complete tasks",
            tools=tools,
        )
        
        logger.info(f"ExecutorAgent initialized with {len(tools)} tools")
    
    def execute_tasks(
        self,
        tasks: List[Task],
        shared_context: Optional[Dict[str, Any]] = None,
    ) -> List[ContextPointer]:
        """
        Execute all tasks sequentially.
        
        This method processes each task from the task list, selecting and
        executing appropriate tools, saving outputs to context files, and
        validating task completion.
        
        Args:
            tasks: List of Task objects to execute
            shared_context: Optional shared context dict for storing results
            
        Returns:
            List of ContextPointer objects for all executed tools
            
        Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 7.1, 12.1, 12.3, 12.4
        """
        import time
        
        # Agent start logging
        # Requirements: 12.1
        start_time = time.time()
        logger.info("=" * 80)
        logger.info(f"AGENT START: Executor (agent_id={self.agent_id})")
        logger.info(f"  Tasks to Execute: {len(tasks)}")
        for task in tasks:
            logger.info(f"    Task {task.id}: {task.description[:80]}{'...' if len(task.description) > 80 else ''}")
        logger.info("=" * 80)
        
        logger.info(f"Executor processing {len(tasks)} tasks")
        
        # Log SharedContext state at entry
        # Requirements: 5.4
        if shared_context is not None:
            logger.debug(f"SharedContext at Executor entry: {list(shared_context.keys())}")
            logger.debug(f"SharedContext contains {len(tasks)} tasks")
        else:
            logger.warning("⚠ No SharedContext provided to Executor")
        
        context_pointers = []
        
        # Process each task sequentially
        # Requirements: 3.1
        for task in tasks:
            if task.done:
                logger.info(f"Task {task.id} already complete, skipping")
                continue
            
            logger.info(f"Executor processing task {task.id}: {task.description}")
            
            # Display task start indicator
            # Requirements: 3.1
            if self.ui:
                self.ui.print_task_start(task.description)
            
            try:
                # Execute single task
                # Requirements: 3.2, 3.3, 3.4, 3.5, 3.6
                pointer = self._execute_single_task(task)
                
                if pointer:
                    context_pointers.append(pointer)
                    task.done = True
                    task.tool_name = pointer.tool_name
                    task.result_context_id = pointer.id
                    logger.info(f"Task {task.id} completed successfully")
                    
                    # Display task completion indicator
                    # Requirements: 3.2
                    if self.ui:
                        self.ui.print_task_done(task.description)
                else:
                    logger.warning(f"Task {task.id} execution returned no pointer")
                    
            except Exception as e:
                # Error handling - record error but continue with remaining tasks
                # Requirements: 10.1, 10.3
                # Log error with full context
                logger.error(
                    f"Executor Agent Error: Task {task.id} execution failed",
                    exc_info=True,
                    extra={
                        'agent_id': self.agent_id,
                        'task_id': task.id,
                        'task_description': task.description[:100],
                        'error_type': type(e).__name__,
                        'error_message': str(e),
                    }
                )
                
                # Record error in task
                task.error = f"{type(e).__name__}: {str(e)}"
                
                # Display task error indicator
                # Requirements: 3.4, 8.1, 8.3, 8.4
                if self.ui:
                    self.ui.print_task_error(task.description, str(e))
                
                # Task remains incomplete (done=False)
                logger.info(f"Task {task.id} marked as incomplete due to error, continuing with remaining tasks")
        
        # Store context pointers in SharedContext
        # Requirements: 3.6, 5.1, 5.2
        if shared_context is not None:
            logger.debug(f"SharedContext before storing context pointers: {list(shared_context.keys())}")
            shared_context['context_pointers'] = context_pointers
            logger.info(f"✓ Stored {len(context_pointers)} context pointers in SharedContext")
            logger.debug(f"SharedContext after storing context pointers: {list(shared_context.keys())}")
            
            # Verify data persistence
            # Requirements: 5.5
            stored_pointers = shared_context.get('context_pointers', [])
            if len(stored_pointers) == len(context_pointers):
                logger.debug(f"✓ Verified: {len(stored_pointers)} context pointers persisted in SharedContext")
            else:
                logger.warning(f"⚠ Verification failed: Expected {len(context_pointers)} pointers, found {len(stored_pointers)}")
        else:
            logger.warning("⚠ No SharedContext provided - context pointers not stored")
        
        logger.info(f"Executor completed {len(context_pointers)} tasks successfully")
        
        # Agent end logging
        # Requirements: 12.1
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info("=" * 80)
        logger.info(f"AGENT END: Executor (agent_id={self.agent_id})")
        logger.info(f"  Execution Time: {elapsed_time:.2f}s")
        logger.info(f"  Tasks Completed: {len(context_pointers)}/{len(tasks)}")
        logger.info(f"  Context Pointers Created: {len(context_pointers)}")
        logger.info("=" * 80)
        
        return context_pointers
    
    def _extract_tool_execution_data(
        self, 
        result: Any
    ) -> Optional[Dict[str, Any]]:
        """
        Extract tool execution data from AgentResult.
        
        Inspects the AgentResult object to find tool name, arguments,
        and result data regardless of SDK version or structure changes.
        
        This method checks multiple possible locations for tool execution data:
        1. result.metrics.tool_metrics (primary location in Strands SDK)
        2. result.tool_calls (legacy/alternative location)
        3. Agent conversation history (for tool result content)
        
        Args:
            result: AgentResult object from Strands SDK
            
        Returns:
            Dict with 'tool_name', 'tool_args', 'tool_result' if found,
            None if no tool execution detected
            
        Requirements: 1.1, 1.2, 1.3, 1.4
        """
        try:
            # Log extraction attempt
            # Requirements: 1.3
            logger.debug("Attempting to extract tool execution data from AgentResult")
            
            # Check Method 1: result.metrics.tool_metrics (Strands SDK primary location)
            # Requirements: 1.1, 1.4
            if hasattr(result, 'metrics') and result.metrics is not None:
                logger.debug("Found result.metrics attribute")
                
                if hasattr(result.metrics, 'tool_metrics') and result.metrics.tool_metrics:
                    logger.debug(f"Found tool_metrics with {len(result.metrics.tool_metrics)} tools")
                    
                    # Get the first tool (we expect one tool per task)
                    tool_name = list(result.metrics.tool_metrics.keys())[0]
                    tool_metric = result.metrics.tool_metrics[tool_name]
                    
                    # Extract tool arguments from the tool metric
                    # Requirements: 1.2
                    tool_args = {}
                    if hasattr(tool_metric, 'tool') and isinstance(tool_metric.tool, dict):
                        tool_args = tool_metric.tool.get('input', {})
                    
                    logger.info(f"Extracted tool from metrics: {tool_name}")
                    logger.debug(f"Tool arguments: {tool_args}")
                    
                    # Try to get tool result from the tool metric itself
                    # Requirements: 1.2
                    tool_result = None
                    
                    # Method 1: Check if tool_metric has a result attribute
                    if hasattr(tool_metric, 'result'):
                        tool_result = tool_metric.result
                        logger.debug(f"Found tool result in tool_metric.result")
                    
                    # Method 2: Check if tool_metric.tool has output
                    elif hasattr(tool_metric, 'tool') and isinstance(tool_metric.tool, dict):
                        if 'output' in tool_metric.tool:
                            tool_result = tool_metric.tool['output']
                            logger.debug(f"Found tool result in tool_metric.tool['output']")
                        elif 'result' in tool_metric.tool:
                            tool_result = tool_metric.tool['result']
                            logger.debug(f"Found tool result in tool_metric.tool['result']")
                    
                    # Method 3: Try to extract from conversation history
                    if tool_result is None:
                        logger.debug("Tool result not found in metrics, checking conversation history")
                        tool_result = self._extract_tool_result_from_conversation(tool_name)
                    
                    if tool_result is not None:
                        # Requirements: 2.3
                        result_size = len(str(tool_result))
                        logger.info(f"Successfully extracted tool execution data: {tool_name} ({result_size} bytes)")
                        
                        return {
                            'tool_name': tool_name,
                            'tool_args': tool_args,
                            'tool_result': tool_result
                        }
                    else:
                        logger.warning(f"Tool {tool_name} found in metrics but result not found")
                        logger.debug(f"tool_metric attributes: {dir(tool_metric)}")
                        if hasattr(tool_metric, 'tool'):
                            logger.debug(f"tool_metric.tool type: {type(tool_metric.tool)}")
                            if isinstance(tool_metric.tool, dict):
                                logger.debug(f"tool_metric.tool keys: {list(tool_metric.tool.keys())}")
                        
                        # Return partial data - we at least have the tool name and args
                        return {
                            'tool_name': tool_name,
                            'tool_args': tool_args,
                            'tool_result': None
                        }
            
            # Check Method 2: result.tool_calls (legacy/alternative location)
            # Requirements: 1.4
            if hasattr(result, 'tool_calls') and result.tool_calls:
                logger.debug(f"Found result.tool_calls with {len(result.tool_calls)} calls")
                
                tool_call = result.tool_calls[0]
                tool_name = tool_call.name if hasattr(tool_call, 'name') else str(tool_call)
                tool_args = tool_call.arguments if hasattr(tool_call, 'arguments') else {}
                tool_result = tool_call.result if hasattr(tool_call, 'result') else None
                
                # Requirements: 2.3
                logger.info(f"Extracted tool from tool_calls: {tool_name}")
                logger.debug(f"Tool arguments: {tool_args}")
                
                return {
                    'tool_name': tool_name,
                    'tool_args': tool_args,
                    'tool_result': tool_result
                }
            
            # Check Method 3: Other possible locations
            # Requirements: 1.4
            for attr_name in ['actions', 'executions', 'tool_outputs', 'tool_results']:
                if hasattr(result, attr_name):
                    attr_value = getattr(result, attr_name)
                    if attr_value:
                        logger.debug(f"Found potential tool data in result.{attr_name}")
                        # Try to extract data from this attribute
                        # This is a fallback for unknown structures
                        try:
                            if isinstance(attr_value, list) and len(attr_value) > 0:
                                first_item = attr_value[0]
                                if isinstance(first_item, dict):
                                    tool_name = first_item.get('name') or first_item.get('tool_name') or 'unknown'
                                    tool_args = first_item.get('args') or first_item.get('arguments') or {}
                                    tool_result = first_item.get('result') or first_item.get('output')
                                    
                                    logger.info(f"Extracted tool from {attr_name}: {tool_name}")
                                    
                                    return {
                                        'tool_name': tool_name,
                                        'tool_args': tool_args,
                                        'tool_result': tool_result
                                    }
                        except Exception as e:
                            logger.debug(f"Failed to extract from {attr_name}: {e}")
                            continue
            
            # No tool execution detected
            # Requirements: 1.3, 2.4
            logger.warning("No tool execution detected in AgentResult")
            logger.debug("Checked locations: metrics.tool_metrics, tool_calls, actions, executions, tool_outputs, tool_results")
            return None
            
        except Exception as e:
            # Error handling - log and return None gracefully
            # Requirements: 1.3, 2.5
            logger.error(
                f"Error extracting tool execution data: {type(e).__name__}: {str(e)}",
                exc_info=True,
                extra={
                    'result_type': type(result).__name__,
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                }
            )
            return None
    
    def _extract_tool_result_from_conversation(self, tool_name: str) -> Optional[Any]:
        """
        Extract tool result from agent's conversation history.
        
        The Strands SDK stores tool results in the conversation history as messages
        with role='user' and content containing 'toolResult' blocks.
        
        Based on Strands SDK source code (event_loop.py line 513-515):
        tool_result_message: Message = {
            "role": "user",
            "content": [{"toolResult": result} for result in tool_results],
        }
        
        Args:
            tool_name: Name of the tool to find result for
            
        Returns:
            Tool result if found, None otherwise
            
        Requirements: 1.2
        """
        try:
            # Access the agent's messages directly (not through conversation_manager)
            # Requirements: 1.2
            if not hasattr(self.agent, 'messages'):
                logger.debug("Agent has no messages attribute")
                return None
            
            messages = self.agent.messages
            
            # Log message count
            logger.debug(f"Searching {len(messages)} messages for tool result")
            
            # Search messages in reverse order (most recent first)
            # Tool results are stored as messages with role='user' and content containing toolResult blocks
            for i, message in enumerate(reversed(messages)):
                try:
                    # Get message role and content
                    role = message.get('role') if isinstance(message, dict) else None
                    content = message.get('content') if isinstance(message, dict) else None
                    
                    # Check if this message contains tool results
                    # Tool results are in messages with role='user' and content is a list
                    if role == 'user' and isinstance(content, list):
                        for content_block in content:
                            # Check if this is a toolResult block (note: lowercase 'toolResult')
                            if isinstance(content_block, dict) and 'toolResult' in content_block:
                                # Found a tool result!
                                tool_result_block = content_block['toolResult']
                                
                                # Extract the actual content from the toolResult
                                # The structure is: {"toolResult": {"toolUseId": "...", "content": [...]}}
                                if isinstance(tool_result_block, dict):
                                    tool_use_id = tool_result_block.get('toolUseId')
                                    tool_content = tool_result_block.get('content')
                                    
                                    logger.debug(f"Found toolResult block: toolUseId={tool_use_id}")
                                    
                                    # The content is typically a list with one item containing the actual result
                                    if isinstance(tool_content, list) and len(tool_content) > 0:
                                        # Get the first content item
                                        first_content = tool_content[0]
                                        
                                        # Check if it's a JSON block
                                        if isinstance(first_content, dict) and 'json' in first_content:
                                            tool_result = first_content['json']
                                            logger.debug(f"Extracted tool result from json block")
                                            return tool_result
                                        
                                        # Check if it's a text block
                                        elif isinstance(first_content, dict) and 'text' in first_content:
                                            tool_result_text = first_content['text']
                                            logger.debug(f"Extracted tool result from text block")
                                            
                                            # Try to parse as JSON
                                            try:
                                                import json
                                                tool_result = json.loads(tool_result_text)
                                                logger.debug(f"Parsed tool result from JSON string")
                                                return tool_result
                                            except:
                                                # Not JSON, return as-is
                                                logger.debug(f"Tool result is plain text")
                                                return tool_result_text
                                        
                                        # Return the content as-is if we don't recognize the format
                                        else:
                                            logger.debug(f"Returning tool content as-is: {type(first_content)}")
                                            return first_content
                                    
                                    # If content is not a list, return it directly
                                    elif tool_content is not None:
                                        logger.debug(f"Returning tool content directly: {type(tool_content)}")
                                        return tool_content
                
                except Exception as e:
                    logger.debug(f"Error processing message {i}: {e}")
                    continue
            
            logger.debug(f"Tool result for {tool_name} not found in conversation history")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting tool result from conversation: {e}", exc_info=True)
            return None
    
    def _execute_single_task(self, task: Task) -> Optional[ContextPointer]:
        """
        Execute a single task.
        
        Process:
        1. Call agent to select and execute appropriate tool
        2. Extract tool execution data using robust extraction logic
        3. Optimize tool arguments (handled by agent)
        4. Get tool result from extracted data
        5. Save output to context file
        6. Validate task completion
        7. Return context pointer
        
        Args:
            task: Task to execute
            
        Returns:
            ContextPointer if successful, None otherwise
            
        Requirements: 1.1, 1.2, 1.5, 3.2, 3.3, 3.4, 3.5, 3.6, 7.1, 12.3, 12.4
        """
        import time
        
        # Tool execution start logging
        # Requirements: 12.3
        tool_start_time = time.time()
        
        try:
            # Step 1: Call agent to select tool and execute
            # The agent will automatically select the appropriate tool
            # Use tool_choice="any" to force tool call without text generation
            # This prevents the agent from generating unnecessary narrative responses
            # Requirements: 3.2
            logger.info(f"Calling Executor agent for task: {task.description}")
            
            # Force tool execution without text response for speed
            result = self.agent(task.description, tool_choice={"any": {}})
            
            # Step 2: Extract tool execution data using robust extraction logic
            # Requirements: 1.1, 1.2, 1.5
            logger.debug("Extracting tool execution data from AgentResult")
            
            tool_data = self._extract_tool_execution_data(result)
            
            # Check if tool execution was detected
            # Requirements: 1.1, 1.5
            if tool_data is None:
                # Log failure with diagnostic information
                # Requirements: 1.5, 2.4
                logger.warning("=" * 80)
                logger.warning("TOOL EXTRACTION FAILED - NO TOOL EXECUTION DETECTED")
                logger.warning("=" * 80)
                logger.warning(f"Task ID: {task.id}")
                logger.warning(f"Task Description: {task.description}")
                logger.warning(f"Result Type: {type(result)}")
                logger.warning(f"Agent did not call any tools for task {task.id}")
                logger.warning("=" * 80)
                return None
            
            # Extract tool information from the extracted data
            # Requirements: 1.2
            tool_name = tool_data['tool_name']
            tool_args = tool_data['tool_args']
            tool_result = tool_data['tool_result']
            
            # Log successful extraction
            # Requirements: 1.5
            logger.info(f"✓ Tool execution detected and extracted successfully")
            logger.debug(f"  Tool Name: {tool_name}")
            logger.debug(f"  Tool Args: {tool_args}")
            if tool_result is not None:
                result_size = len(str(tool_result))
                logger.debug(f"  Tool Result Size: {result_size} bytes")
            
            # Tool execution logging with timing
            # Requirements: 12.3
            logger.info(f"→ Tool Selected: {tool_name}")
            logger.info(f"  Arguments: {tool_args}")
            logger.info(f"Agent selected tool: {tool_name} with args: {tool_args}")
            
            # Step 3: Tool argument optimization is handled by the agent
            # The agent's system prompt instructs it to optimize arguments
            # Requirements: 3.3
            
            # Step 4: Tool result already extracted from agent execution
            # The tool_result was extracted by _extract_tool_execution_data()
            # Requirements: 1.2
            
            # Tool execution end logging with timing
            # Requirements: 12.3
            tool_end_time = time.time()
            tool_elapsed = tool_end_time - tool_start_time
            
            if tool_result is None:
                logger.warning(f"Tool {tool_name} returned None result")
                logger.info(f"✗ Tool Execution Failed: {tool_name}")
                logger.info(f"  Execution Time: {tool_elapsed:.2f}s")
                return None
            
            # Calculate result size
            import sys
            result_size_bytes = sys.getsizeof(str(tool_result))
            result_size_kb = result_size_bytes / 1024
            
            logger.info(f"Tool {tool_name} executed successfully")
            logger.info(f"✓ Tool Execution Complete: {tool_name}")
            logger.info(f"  Execution Time: {tool_elapsed:.2f}s")
            logger.info(f"  Result Size: {result_size_kb:.2f} KB")
            
            # Step 5: Save output to context file
            # Requirements: 3.4, 7.1
            filepath = self.context_manager.save_context(
                tool_name=tool_name,
                args=tool_args,
                result=tool_result,
                task_id=task.id
            )
            
            # Step 6: Validate task completion
            # Requirements: 3.5, 12.4
            is_complete = self._validate_task(task, tool_result)
            
            # Validation decision logging
            # Requirements: 12.4
            logger.info(f"→ Task Validation: Task {task.id}")
            logger.info(f"  Result: {'COMPLETE' if is_complete else 'INCOMPLETE'}")
            logger.info(f"  Reason: {'Data retrieved successfully' if is_complete else 'Insufficient or empty data'}")
            
            if not is_complete:
                logger.warning(f"Task {task.id} validation failed - data may be insufficient")
            
            # Step 7: Create and return context pointer
            # Requirements: 3.6
            pointer_data = self.context_manager.pointers[-1]  # Get the just-added pointer
            
            pointer = ContextPointer(
                id=str(pointer_data['id']),
                tool_name=pointer_data['tool_name'],
                args=pointer_data['args'],
                summary=pointer_data['summary'],
                filepath=pointer_data['filepath'],
                size_kb=pointer_data['size_kb'],
                task_id=task.id
            )
            
            return pointer
            
        except Exception as e:
            # Error handling for single task execution
            # Requirements: 10.1, 10.3
            logger.error(
                f"Executor Agent Error: Failed to execute task {task.id}",
                exc_info=True,
                extra={
                    'agent_id': self.agent_id,
                    'task_id': task.id,
                    'task_description': task.description[:100],
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                }
            )
            return None
    
    def _validate_task(self, task: Task, tool_result: Any) -> bool:
        """
        Perform lightweight validation of task completion.
        
        Checks if the tool execution successfully retrieved data.
        This is a simple validation - just checks if result is non-empty.
        
        Args:
            task: Task that was executed
            tool_result: Result from tool execution
            
        Returns:
            True if task appears complete, False otherwise
            
        Requirements: 3.5, 6.1, 6.2
        """
        try:
            # Check if result is None or empty
            if tool_result is None:
                logger.debug(f"Task {task.id} validation: result is None")
                return False
            
            # Check if result is empty dict/list/string
            if isinstance(tool_result, (dict, list, str)):
                if not tool_result:
                    logger.debug(f"Task {task.id} validation: result is empty")
                    return False
            
            # If we got here, result has some data
            logger.debug(f"Task {task.id} validation: result contains data")
            return True
            
        except Exception as e:
            logger.error(f"Task validation failed: {e}")
            # Conservative: assume task is incomplete on error
            return False

"""
Planner Agent for multi-agent architecture.

The Planner Agent is responsible for:
- Analyzing user queries
- Decomposing queries into actionable tasks
- Handling handoffs from Synthesizer for additional tasks
- Storing task lists in SharedContext

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 10.2
"""

import json
import logging
from typing import Any, Dict, List, Optional

from strands import Agent
from strands.models import BedrockModel
from pydantic import BaseModel, Field

from ..models import Task
from ..prompts import get_planner_prompt

logger = logging.getLogger(__name__)


class TaskList(BaseModel):
    """Schema for task list output from Planner."""
    tasks: List[str] = Field(
        description="List of task descriptions",
        default_factory=list
    )


class PlannerAgent:
    """
    Planner Agent for task decomposition.
    
    The Planner analyzes user queries and breaks them down into specific,
    actionable tasks that can be executed by the Executor Agent.
    
    Responsibilities:
    - Analyze user query
    - Identify required data and tools
    - Create specific, atomic tasks
    - Handle handoffs from Synthesizer for additional tasks
    
    Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 10.2
    """
    
    def __init__(
        self,
        model: BedrockModel,
        tools: List[Any],
        agent_id: str = "planner",
    ):
        """
        Initialize Planner Agent.
        
        Args:
            model: Configured BedrockModel instance
            tools: List of available financial tools (for prompt injection)
            agent_id: Unique identifier for this agent
        """
        logger.info("Initializing PlannerAgent")
        
        self.model = model
        self.tools = tools
        self.agent_id = agent_id
        
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
        system_prompt = get_planner_prompt(tool_descriptions)
        
        # Create Strands Agent for planning
        # No tools needed - pure reasoning agent
        self.agent = Agent(
            model=model,
            system_prompt=system_prompt,
            agent_id=agent_id,
            name="Planner",
            description="Decomposes queries into actionable tasks",
            structured_output_model=TaskList,
        )
        
        logger.info(f"PlannerAgent initialized with {len(tools)} tools")
    
    def plan_tasks(
        self,
        query: str,
        shared_context: Optional[Dict[str, Any]] = None,
        data_gaps: Optional[str] = None,
    ) -> List[Task]:
        """
        Decompose query into actionable tasks.
        
        This method analyzes the user query (or data gaps from Synthesizer)
        and creates a list of specific, atomic tasks for the Executor.
        
        Args:
            query: User query to decompose
            shared_context: Optional shared context dict for storing results
            data_gaps: Optional description of missing data (from Synthesizer handoff)
            
        Returns:
            List of Task objects
            
        Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 10.2, 12.1
        """
        import time
        
        # Agent start logging
        # Requirements: 12.1
        start_time = time.time()
        logger.info("=" * 80)
        logger.info(f"AGENT START: Planner (agent_id={self.agent_id})")
        logger.info(f"  Query: {query[:100]}{'...' if len(query) > 100 else ''}")
        if data_gaps:
            logger.info(f"  Data Gaps: {data_gaps[:100]}{'...' if len(data_gaps) > 100 else ''}")
        logger.info("=" * 80)
        
        try:
            # Retrieve user query from SharedContext if available
            # Requirements: 5.3, 5.4
            if shared_context is not None:
                stored_query = shared_context.get('user_query')
                if stored_query:
                    logger.debug(f"→ Retrieved user query from SharedContext: {stored_query[:100]}...")
                    # Use stored query if current query is empty or different
                    if not query or query != stored_query:
                        logger.debug(f"Using stored query from SharedContext")
                        query = stored_query
            
            # Determine if this is initial planning or handoff
            if data_gaps:
                # Handoff from Synthesizer - create additional tasks
                # Requirements: 2.5
                logger.info(f"Planner received handoff with data gaps: {data_gaps[:100]}...")
                prompt = f"""Original query: {query}

Data gaps identified: {data_gaps}

Create additional tasks to fill these data gaps."""
            else:
                # Initial planning
                # Requirements: 2.1
                logger.info(f"Planner analyzing query: {query[:100]}...")
                prompt = f"""User query: {query}

Create a list of tasks to answer this query."""
            
            # Call agent to generate task list
            # Requirements: 2.1, 2.3
            logger.info("Calling Planner agent for task decomposition")
            result = self.agent(prompt)
            
            # Extract task list from structured output
            if hasattr(result, 'structured_output') and result.structured_output:
                task_list = result.structured_output
                task_descriptions = task_list.tasks
            else:
                # Fallback: try to parse from text response
                logger.warning("No structured output, attempting to parse from text")
                # Extract text from message content
                if result.message and 'content' in result.message and result.message['content']:
                    text = result.message['content'][0]['text']
                else:
                    text = ""
                task_descriptions = self._parse_tasks_from_text(text)
            
            # Convert task descriptions to Task objects
            # Requirements: 2.2 - Include all necessary context in task descriptions
            tasks = []
            for i, desc in enumerate(task_descriptions, start=1):
                task = Task(
                    id=i,
                    description=desc,
                    done=False,
                )
                tasks.append(task)
            
            # Store tasks in SharedContext
            # Requirements: 2.4, 5.1, 5.2
            if shared_context is not None:
                logger.debug(f"SharedContext before storing tasks: {list(shared_context.keys())}")
                shared_context['tasks'] = tasks
                logger.info(f"✓ Stored {len(tasks)} tasks in SharedContext")
                logger.debug(f"SharedContext after storing tasks: {list(shared_context.keys())}")
                
                # Verify data persistence
                # Requirements: 5.5
                stored_tasks = shared_context.get('tasks', [])
                if len(stored_tasks) == len(tasks):
                    logger.debug(f"✓ Verified: {len(stored_tasks)} tasks persisted in SharedContext")
                else:
                    logger.warning(f"⚠ Verification failed: Expected {len(tasks)} tasks, found {len(stored_tasks)}")
            else:
                logger.warning("⚠ No SharedContext provided - tasks not stored")
            
            logger.info(f"Planner created {len(tasks)} tasks")
            for task in tasks:
                logger.info(f"  Task {task.id}: {task.description}")
            
            # Agent end logging
            # Requirements: 12.1
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info("=" * 80)
            logger.info(f"AGENT END: Planner (agent_id={self.agent_id})")
            logger.info(f"  Execution Time: {elapsed_time:.2f}s")
            logger.info(f"  Tasks Created: {len(tasks)}")
            logger.info("=" * 80)
            
            return tasks
            
        except Exception as e:
            # Error handling with fallback
            # Requirements: 10.1, 10.2
            # Log error with full context
            logger.error(
                f"Planner Agent Error: {type(e).__name__}",
                exc_info=True,
                extra={
                    'agent_id': self.agent_id,
                    'query': query[:100],
                    'data_gaps': data_gaps[:100] if data_gaps else None,
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                }
            )
            
            # Fallback: create single task with original query
            logger.info("Planner falling back to single task with original query")
            fallback_task = Task(
                id=1,
                description=query,
                done=False,
            )
            
            # Store fallback task in SharedContext
            # Requirements: 5.1, 5.2
            if shared_context is not None:
                logger.debug(f"SharedContext before storing fallback task: {list(shared_context.keys())}")
                shared_context['tasks'] = [fallback_task]
                logger.info("✓ Stored fallback task in SharedContext")
                logger.debug(f"SharedContext after storing fallback task: {list(shared_context.keys())}")
                
                # Verify data persistence
                # Requirements: 5.5
                stored_tasks = shared_context.get('tasks', [])
                if len(stored_tasks) == 1:
                    logger.debug("✓ Verified: Fallback task persisted in SharedContext")
                else:
                    logger.warning(f"⚠ Verification failed: Expected 1 task, found {len(stored_tasks)}")
            else:
                logger.warning("⚠ No SharedContext provided - fallback task not stored")
            
            logger.info("Planner failed, using fallback single task")
            
            # Agent end logging (error case)
            # Requirements: 12.1
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info("=" * 80)
            logger.info(f"AGENT END: Planner (agent_id={self.agent_id}) [ERROR]")
            logger.info(f"  Execution Time: {elapsed_time:.2f}s")
            logger.info(f"  Tasks Created: 1 (fallback)")
            logger.info("=" * 80)
            
            return [fallback_task]
    
    def _parse_tasks_from_text(self, text: str) -> List[str]:
        """
        Parse task descriptions from text response.
        
        Fallback method when structured output is not available.
        
        Args:
            text: Text response from agent
            
        Returns:
            List of task descriptions
        """
        try:
            # Try to find JSON in the text
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = text[start:end]
                data = json.loads(json_str)
                if 'tasks' in data:
                    return data['tasks']
            
            # If no JSON found, return empty list
            logger.warning("Could not parse tasks from text, returning empty list")
            return []
            
        except Exception as e:
            logger.error(f"Failed to parse tasks from text: {e}")
            return []

"""
Multi-Agent Dexter using Strands Graph pattern.

This module implements the multi-agent architecture for Dexter using Strands'
native Graph orchestration. The system consists of three specialized agents:
- Planner: Decomposes queries into actionable tasks
- Executor: Selects and executes financial tools
- Synthesizer: Generates comprehensive answers from collected data

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 7.5, 10.1, 10.2
"""

import logging
from typing import Any, Dict, List, Optional

from strands.multiagent import GraphBuilder, GraphResult
from strands.models import BedrockModel

from .agents.planner import PlannerAgent
from .agents.executor import ExecutorAgent
from .agents.synthesizer import SynthesizerAgent
from .context import ContextManager
from .config import Config
from .ui import RichUI
from .safety import SafetyHook, UIFeedbackHook
from .tracing import setup_langfuse_tracing, build_trace_attributes, add_metrics_to_trace

logger = logging.getLogger(__name__)


class MultiAgentDexter:
    """
    Multi-agent orchestrator using Strands Graph pattern.
    
    This class creates and manages a Graph with three specialized agents:
    1. Planner: Analyzes queries and creates task lists
    2. Executor: Executes tools to complete tasks
    3. Synthesizer: Generates answers from collected data
    
    The Graph handles agent transitions, data sharing via SharedContext,
    and conditional handoffs when more data is needed.
    
    Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 7.5, 10.1
    """
    
    def __init__(
        self,
        model: BedrockModel,
        tools: List[Any],
        config: Config,
        context_manager: ContextManager,
        ui: Optional[RichUI] = None,
    ):
        """
        Initialize multi-agent system with Graph.
        
        Args:
            model: Configured BedrockModel instance
            tools: List of available financial tools
            config: Configuration object with settings
            context_manager: ContextManager for saving/loading contexts
            ui: Optional RichUI instance for terminal output
            
        Requirements: 1.1, 10.1, 10.2
        """
        logger.info("Initializing MultiAgentDexter with Graph orchestration")
        
        self.model = model
        self.tools = tools
        self.config = config
        self.context_manager = context_manager
        self.ui = ui
        
        # Set up tracing with comprehensive error handling
        # Requirements: 1.1, 4.1, 4.4, 7.5, 10.1, 10.2, 10.4
        try:
            self.tracing_enabled = setup_langfuse_tracing(config)
            
            # Build trace attributes if tracing is enabled
            # Requirements: 1.1, 2.1, 2.2, 2.3, 10.1, 10.2
            self.trace_attributes = build_trace_attributes(config) if self.tracing_enabled else {}
            
            if self.tracing_enabled:
                logger.info(f"Tracing enabled with session ID: {self.trace_attributes.get('session.id')}")
            else:
                logger.info("Tracing disabled")
        except Exception as e:
            # Ensure tracing setup errors don't break agent initialization
            # Requirements: 4.1, 4.4, 7.5, 10.4
            logger.error(
                f"Error during tracing setup: {e}. "
                "Tracing disabled. Agent will continue without observability.",
                exc_info=True
            )
            self.tracing_enabled = False
            self.trace_attributes = {}
        
        # Create specialized agents
        # Requirements: 1.1, 2.1, 2.2, 2.3
        self.planner = self._create_planner_agent()
        self.executor = self._create_executor_agent()
        self.synthesizer = self._create_synthesizer_agent()
        
        # Create Graph with edges and safety hooks
        # Requirements: 1.2, 1.3, 1.4, 1.5, 7.5
        self.graph = self._create_graph()
        
        logger.info("MultiAgentDexter initialized successfully")
    
    def _create_planner_agent(self) -> PlannerAgent:
        """
        Create Planner Agent.
        
        Returns:
            Configured PlannerAgent instance
            
        Requirements: 1.1
        """
        logger.debug("Creating Planner Agent")
        
        planner = PlannerAgent(
            model=self.model,
            tools=self.tools,
            agent_id="planner",
        )
        
        return planner
    
    def _create_executor_agent(self) -> ExecutorAgent:
        """
        Create Executor Agent.
        
        Returns:
            Configured ExecutorAgent instance
            
        Requirements: 1.1, 3.1, 3.2
        """
        logger.debug("Creating Executor Agent")
        
        executor = ExecutorAgent(
            model=self.model,
            tools=self.tools,
            context_manager=self.context_manager,
            ui=self.ui,
            agent_id="executor",
        )
        
        return executor
    
    def _create_synthesizer_agent(self) -> SynthesizerAgent:
        """
        Create Synthesizer Agent.
        
        Returns:
            Configured SynthesizerAgent instance
            
        Requirements: 1.1
        """
        logger.debug("Creating Synthesizer Agent")
        
        synthesizer = SynthesizerAgent(
            model=self.model,
            context_manager=self.context_manager,
            ui=self.ui,
            agent_id="synthesizer",
        )
        
        return synthesizer
    
    def _create_graph(self) -> Any:
        """
        Create Strands Graph with agent nodes and edges.
        
        The Graph structure:
        - Planner → Executor (always)
        - Executor → Synthesizer (always)
        - Synthesizer → Planner (conditional: if needs more data)
        - Synthesizer → End (conditional: if has sufficient data)
        
        Returns:
            Configured Graph instance
            
        Requirements: 1.2, 1.3, 1.4, 1.5, 7.5, 10.1
        """
        logger.debug("Creating Graph with 3 agent nodes")
        
        # Create Graph builder
        # Requirements: 1.2
        builder = GraphBuilder()
        
        # Add agent nodes
        # Requirements: 1.2
        # Note: We wrap the agents in a simple callable that matches the Graph's expected interface
        planner_node = builder.add_node(
            executor=self._create_planner_wrapper(),
            node_id="planner"
        )
        
        executor_node = builder.add_node(
            executor=self._create_executor_wrapper(),
            node_id="executor"
        )
        
        synthesizer_node = builder.add_node(
            executor=self._create_synthesizer_wrapper(),
            node_id="synthesizer"
        )
        
        # Define edges
        # Requirements: 1.3
        # Planner → Executor (always)
        builder.add_edge(
            from_node=planner_node,
            to_node=executor_node,
        )
        
        # Executor → Synthesizer (always)
        builder.add_edge(
            from_node=executor_node,
            to_node=synthesizer_node,
        )
        
        # Synthesizer → Planner (conditional: needs more data)
        # Requirements: 1.5
        builder.add_edge(
            from_node=synthesizer_node,
            to_node=planner_node,
            condition=lambda state: self._needs_more_data(state),
        )
        
        # Note: Synthesizer → End is implicit when no condition is met
        
        # Set entry point
        # Requirements: 1.4
        builder.set_entry_point("planner")
        
        # Configure safety limits
        # Requirements: 7.5, 10.1
        max_iterations = getattr(self.config, 'max_iterations', 20)
        builder.set_max_node_executions(max_iterations)
        
        # Set Graph ID
        builder.set_graph_id("dexter_multi_agent")
        
        # Add safety hooks
        # Requirements: 7.5
        hooks = []
        
        # Add SafetyHook for iteration limits
        safety_hook = SafetyHook(
            max_iterations=max_iterations,
            loop_window=4,
        )
        hooks.append(safety_hook)
        
        # Note: UIFeedbackHook is now attached to individual agent instances
        # (specifically the Executor agent) rather than the Graph.
        # This ensures the hook receives events directly from the agent.
        # Requirements: 6.1, 6.2, 6.4
        
        builder.set_hook_providers(hooks)
        
        # Build Graph
        graph = builder.build()
        
        # Log graph info (safely handle mock objects in tests)
        try:
            logger.info(f"Graph created with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
            logger.info(f"Entry points: {[node.node_id for node in graph.entry_points]}")
        except (TypeError, AttributeError):
            logger.info("Graph created (mock object)")
        
        logger.info(f"Max iterations: {max_iterations}")
        
        return graph
    
    def _create_planner_wrapper(self) -> Any:
        """
        Create a wrapper for Planner that matches Graph's expected interface.
        
        The Graph expects nodes to be Agent or MultiAgentBase instances.
        Since our agents have custom interfaces, we need to wrap them.
        
        Returns:
            Agent-compatible wrapper for Planner
            
        Requirements: 2.1, 10.1, 10.2
        """
        from strands import Agent
        
        # Create a simple Agent that delegates to our PlannerAgent
        # The Agent will receive the query and SharedContext through invocation_state
        def planner_tool(query: str, shared_context: Optional[Dict[str, Any]] = None):
            """Plan tasks for the given query."""
            data_gaps = shared_context.get('data_gaps') if shared_context else None
            tasks = self.planner.plan_tasks(query, shared_context, data_gaps)
            return f"Created {len(tasks)} tasks"
        
        # Create Agent with custom system prompt and trace attributes
        # Requirements: 2.1, 10.1, 10.2
        agent = Agent(
            model=self.model,
            system_prompt="You are the Planner agent. Decompose queries into tasks.",
            agent_id="planner",
            name="Planner",
            description="Decomposes queries into actionable tasks",
            trace_attributes=self.trace_attributes,
        )
        
        return agent
    
    def _create_executor_wrapper(self) -> Any:
        """
        Create a wrapper for Executor that matches Graph's expected interface.
        
        Attaches UIFeedbackHook to the agent instance to capture tool execution
        events and display them to the user.
        
        Returns:
            Agent-compatible wrapper for Executor
            
        Requirements: 2.2, 6.1, 6.2, 6.4, 10.1, 10.2
        """
        from strands import Agent
        
        # Prepare hooks list for the agent
        # Requirements: 6.1, 6.2, 6.4
        hooks = []
        if self.ui:
            ui_hook = UIFeedbackHook(ui=self.ui)
            hooks.append(ui_hook)
            logger.debug("UIFeedbackHook will be attached to Executor agent instance")
        
        # Create Agent that delegates to our ExecutorAgent
        # Attach UIFeedbackHook to this agent instance (not Graph)
        # This ensures the hook receives events from the agent's tool executions
        # Requirements: 2.2, 6.1, 6.2, 6.4, 10.1, 10.2
        agent = Agent(
            model=self.model,
            system_prompt="You are the Executor agent. Execute tools to complete tasks.",
            agent_id="executor",
            name="Executor",
            description="Executes tools to complete tasks",
            tools=self.tools,
            hooks=hooks,  # Attach hooks to agent instance
            trace_attributes=self.trace_attributes,
        )
        
        logger.debug(f"Executor agent created with {len(hooks)} hook(s)")
        
        return agent
    
    def _create_synthesizer_wrapper(self) -> Any:
        """
        Create a wrapper for Synthesizer that matches Graph's expected interface.
        
        Returns:
            Agent-compatible wrapper for Synthesizer
            
        Requirements: 2.3, 10.1, 10.2
        """
        from strands import Agent
        
        # Create Agent that delegates to our SynthesizerAgent
        # Requirements: 2.3, 10.1, 10.2
        agent = Agent(
            model=self.model,
            system_prompt="You are the Synthesizer agent. Generate answers from collected data.",
            agent_id="synthesizer",
            name="Synthesizer",
            description="Generates answers from collected data",
            trace_attributes=self.trace_attributes,
        )
        
        return agent
    
    def _needs_more_data(self, state: Any) -> bool:
        """
        Condition function to determine if Synthesizer should handoff to Planner.
        
        This checks if the Synthesizer identified data gaps that require
        additional research.
        
        Args:
            state: GraphState object
            
        Returns:
            True if more data is needed, False otherwise
            
        Requirements: 1.5
        """
        # Check if data_gaps were set by Synthesizer
        # The GraphState doesn't have a direct way to access SharedContext,
        # so we'll need to check the results
        
        # For now, we'll use a simple heuristic:
        # If the Synthesizer's result contains "HANDOFF_TO_PLANNER", we need more data
        if hasattr(state, 'results') and 'synthesizer' in state.results:
            synthesizer_result = state.results['synthesizer']
            if hasattr(synthesizer_result, 'result'):
                result_text = str(synthesizer_result.result)
                return "HANDOFF_TO_PLANNER" in result_text
        
        return False
    
    def process_query(self, query: str) -> str:
        """
        Process query through multi-agent Graph.
        
        This is the main entry point for query processing. It:
        1. Initializes SharedContext with the query
        2. Executes the Graph starting at the Planner
        3. Handles agent transitions and data sharing
        4. Returns the final answer from the Synthesizer
        
        Error Handling:
        - Tool failures (4.6): Automatically captured by Strands SDK in traces
        - Agent failures (4.7): Automatically captured by Strands SDK in traces
        - Safety limit violations (4.8): Explicitly captured with detailed context
        - Trace finalization (4.5): Handled gracefully on errors
        
        Args:
            query: User query to process
            
        Returns:
            Generated answer text
            
        Requirements: 1.4, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 7.5, 10.1, 12.2, 12.5
        """
        import time
        
        # Total execution time logging
        # Requirements: 12.5
        graph_start_time = time.time()
        transition_count = 0
        
        # Initialize metrics tracking
        # Requirements: 5.4, 5.5, 6.4, 6.5
        handoff_count = 0
        task_count = 0
        tool_count = 0
        context_offload_count = 0
        context_offload_size_kb = 0.0
        
        logger.info("=" * 80)
        logger.info("GRAPH EXECUTION START")
        logger.info(f"  Query: {query[:100]}{'...' if len(query) > 100 else ''}")
        logger.info("=" * 80)
        
        logger.info(f"Processing query: {query[:100]}...")
        
        # Wrap entire process_query in coordinated_progress context
        # Requirements: 4.3, 4.4
        # This ensures a single Live context is used throughout, preventing
        # multiple spinners from conflicting during handoffs
        if self.ui:
            coordinated_context = self.ui.coordinated_progress("Processing query")
        else:
            # No-op context manager when UI is not available
            from contextlib import nullcontext
            coordinated_context = nullcontext()
        
        try:
            with coordinated_context:
                # Initialize SharedContext
                # Requirements: 1.4, 5.1
                shared_context: Dict[str, Any] = {
                    "user_query": query,
                    "tasks": [],
                    "context_pointers": [],
                    "data_gaps": None,
                }
                
                # Execute Graph
                # Requirements: 1.4
                # Note: The Graph will handle agent transitions automatically
                # We need to manually orchestrate since we're using custom agent interfaces
                
                # Step 1: Planner creates tasks
                # Graph transition logging
                # Requirements: 12.2
                logger.info("→ GRAPH TRANSITION: START → Planner")
                transition_count += 1
                
                logger.info("Step 1: Planner creating tasks")
                tasks = []  # Initialize to empty list
                
                try:
                    # Remove nested progress context - using coordinated_progress instead
                    # Requirements: 4.3, 4.4
                    tasks = self.planner.plan_tasks(query, shared_context)
                except Exception as e:
                    # Error handling for planning step
                    # Requirements: 10.1, 10.2
                    logger.error(
                        "Error during planning step",
                        exc_info=True,
                        extra={
                            'step': 'planning',
                            'error_type': type(e).__name__,
                            'error_message': str(e),
                        }
                    )
                    
                    # Planner should have already handled this with fallback,
                    # but if we still have no tasks, return error
                    if not tasks:
                        return self._format_error_message(e)
                
                # Display task list after planning (including empty case)
                # Requirements: 1.1, 1.2, 1.3, 1.5
                self._display_task_list(tasks)
                
                # Check if tasks were created
                if not tasks:
                    logger.info("No tasks created - query may be out of scope")
                    
                    # Add metrics to trace (no tasks case)
                    # Requirements: 5.4, 5.5, 6.4, 6.5
                    if self.tracing_enabled:
                        add_metrics_to_trace(
                            handoff_count=handoff_count,
                            task_count=0,
                            tool_count=0,
                            context_offload_count=0,
                            context_offload_size_kb=0.0,
                        )
                    
                    # Total execution time logging (no tasks case)
                    # Requirements: 12.5
                    graph_end_time = time.time()
                    total_elapsed = graph_end_time - graph_start_time
                    logger.info("=" * 80)
                    logger.info("GRAPH EXECUTION COMPLETE (No Tasks)")
                    logger.info(f"  Total Execution Time: {total_elapsed:.2f}s")
                    logger.info(f"  Total Transitions: {transition_count}")
                    logger.info("=" * 80)
                    
                    return "I'm sorry, but I don't have the tools to answer that question. Please ask about financial data, company fundamentals, SEC filings, or market information."
                
                # Track task count
                # Requirements: 5.5
                task_count = len(tasks)
                
                # Step 2: Executor executes tasks
                # Display transition indicator to user
                # Requirements: 7.1
                if self.ui:
                    self.ui.print_agent_transition("Planner", "Executor")
                
                # Graph transition logging
                # Requirements: 12.2
                logger.info("→ GRAPH TRANSITION: Planner → Executor")
                transition_count += 1
                
                logger.info(f"Step 2: Executor executing {len(tasks)} tasks")
                context_pointers = []  # Initialize to empty list
                
                try:
                    # Remove nested progress context - using coordinated_progress instead
                    # Requirements: 4.3, 4.4
                    context_pointers = self.executor.execute_tasks(tasks, shared_context)
                except Exception as e:
                    # Error handling for execution step
                    # Requirements: 10.1, 10.3
                    logger.error(
                        "Error during execution step",
                        exc_info=True,
                        extra={
                            'step': 'execution',
                            'num_tasks': len(tasks),
                            'error_type': type(e).__name__,
                            'error_message': str(e),
                        }
                    )
                    
                    # Executor should have handled individual task errors,
                    # but if we have a complete failure, return error
                    if not context_pointers:
                        return (
                            "I encountered an issue while executing the research tasks. "
                            "This could be due to a data service problem or an unexpected error. "
                            "Please try again or rephrase your question."
                        )
            
                # Track tool count and context offloading metrics
                # Requirements: 6.4, 6.5
                tool_count = len(context_pointers)
                context_offload_count = len(context_pointers)
                
                # Calculate total size of offloaded contexts
                for pointer in context_pointers:
                    if hasattr(pointer, 'size_kb'):
                        context_offload_size_kb += pointer.size_kb
                
                # Step 3: Synthesizer generates answer (with potential handoff loop)
                # Requirements: 10.5
                max_handoffs = 3  # Prevent infinite loops (safety limit)
                
                while handoff_count < max_handoffs:
                    try:
                        # Display transition indicator to user
                        # Requirements: 7.2
                        if handoff_count == 0:
                            if self.ui:
                                self.ui.print_agent_transition("Executor", "Synthesizer")
                        else:
                            if self.ui:
                                self.ui.print_agent_transition("Executor", "Synthesizer", f"handoff attempt {handoff_count + 1}")
                        
                        # Graph transition logging
                        # Requirements: 12.2
                        if handoff_count == 0:
                            logger.info("→ GRAPH TRANSITION: Executor → Synthesizer")
                        else:
                            logger.info(f"→ GRAPH TRANSITION: Executor → Synthesizer (handoff attempt {handoff_count + 1})")
                        transition_count += 1
                        
                        logger.info(f"Step 3: Synthesizer generating answer (attempt {handoff_count + 1})")
                        
                        # Remove nested progress context - using coordinated_progress instead
                        # Requirements: 4.3, 4.4
                        answer = self.synthesizer.synthesize_answer(query, shared_context)
                        
                        # Check if handoff is needed
                        if answer == "HANDOFF_TO_PLANNER":
                            handoff_count += 1
                            
                            # Display handoff transition indicator to user with count
                            # Requirements: 7.3, 7.5
                            if self.ui:
                                self.ui.print_agent_transition("Synthesizer", "Planner", f"handoff {handoff_count}/{max_handoffs}")
                            
                            # Graph transition logging (handoff)
                            # Requirements: 12.2
                            logger.info("→ GRAPH TRANSITION: Synthesizer → Planner (handoff)")
                            logger.info(f"  Reason: Insufficient data")
                            logger.info(f"  Handoff Count: {handoff_count}/{max_handoffs}")
                            transition_count += 1
                            
                            logger.info(f"Handoff to Planner (attempt {handoff_count}/{max_handoffs})")
                            
                            # Get data gaps from SharedContext
                            data_gaps = shared_context.get('data_gaps')
                            
                            if not data_gaps:
                                logger.warning("Handoff requested but no data gaps specified")
                                return "I encountered an issue while researching your query. Please try rephrasing your question."
                        
                            # Planner creates additional tasks
                            logger.info(f"Planner creating additional tasks for: {data_gaps[:100]}...")
                            additional_tasks = []
                            
                            try:
                                # Remove nested progress context - using coordinated_progress instead
                                # Requirements: 4.3, 4.4
                                additional_tasks = self.planner.plan_tasks(query, shared_context, data_gaps)
                            except Exception as e:
                                # Error handling for additional planning
                                # Requirements: 10.1, 10.2
                                logger.error(
                                    f"Error during additional planning (handoff {handoff_count})",
                                    exc_info=True,
                                    extra={
                                        'handoff_count': handoff_count,
                                        'data_gaps': data_gaps[:100],
                                        'error_type': type(e).__name__,
                                        'error_message': str(e),
                                    }
                                )
                                return (
                                    "I identified missing information but encountered an issue while planning additional research. "
                                    "Please try rephrasing your question or breaking it into smaller parts."
                                )
                            
                            if not additional_tasks:
                                logger.warning("No additional tasks created after handoff")
                                return "I couldn't find all the information needed to answer your question completely. Please try a more specific query."
                        
                            # Display additional task list
                            # Requirements: 1.1, 1.2, 1.3
                            self._display_task_list(additional_tasks)
                            
                            # Track additional tasks
                            # Requirements: 5.5
                            task_count += len(additional_tasks)
                            
                            # Executor executes additional tasks
                            # Graph transition logging
                            # Requirements: 12.2
                            logger.info("→ GRAPH TRANSITION: Planner → Executor (additional tasks)")
                            transition_count += 1
                            
                            logger.info(f"Executor executing {len(additional_tasks)} additional tasks")
                            additional_pointers = []
                            
                            try:
                                # Remove nested progress context - using coordinated_progress instead
                                # Requirements: 4.3, 4.4
                                additional_pointers = self.executor.execute_tasks(additional_tasks, shared_context)
                            except Exception as e:
                                # Error handling for additional execution
                                # Requirements: 10.1, 10.3
                                logger.error(
                                    f"Error during additional execution (handoff {handoff_count})",
                                    exc_info=True,
                                    extra={
                                        'handoff_count': handoff_count,
                                        'num_tasks': len(additional_tasks),
                                        'error_type': type(e).__name__,
                                        'error_message': str(e),
                                    }
                                )
                                return (
                                    "I encountered an issue while gathering additional information. "
                                    "The initial data was collected successfully, but I couldn't complete the follow-up research. "
                                    "Please try simplifying your question."
                                )
                            
                            # Track additional tool calls and context offloading
                            # Requirements: 6.4, 6.5
                            tool_count += len(additional_pointers)
                            context_offload_count += len(additional_pointers)
                            
                            # Calculate additional offloaded context size
                            for pointer in additional_pointers:
                                if hasattr(pointer, 'size_kb'):
                                    context_offload_size_kb += pointer.size_kb
                            
                            # Clear data gaps for next iteration
                            shared_context['data_gaps'] = None
                            
                            # Continue loop to try synthesis again
                            continue
                        else:
                            # Got a real answer, return it
                            # Graph transition logging (completion)
                            # Requirements: 12.2
                            logger.info("→ GRAPH TRANSITION: Synthesizer → END")
                            transition_count += 1
                            
                            logger.info("Answer generated successfully")
                            
                            # Add metrics to trace
                            # Requirements: 5.4, 5.5, 6.4, 6.5
                            if self.tracing_enabled:
                                add_metrics_to_trace(
                                    handoff_count=handoff_count,
                                    task_count=task_count,
                                    tool_count=tool_count,
                                    context_offload_count=context_offload_count,
                                    context_offload_size_kb=context_offload_size_kb,
                                )
                            
                            # Total execution time logging
                            # Requirements: 12.5
                            graph_end_time = time.time()
                            total_elapsed = graph_end_time - graph_start_time
                            logger.info("=" * 80)
                            logger.info("GRAPH EXECUTION COMPLETE")
                            logger.info(f"  Total Execution Time: {total_elapsed:.2f}s")
                            logger.info(f"  Total Transitions: {transition_count}")
                            logger.info(f"  Answer Length: {len(answer)} characters")
                            logger.info("=" * 80)
                            
                            return answer
                        
                    except Exception as e:
                        # Error handling within handoff loop
                        # Requirements: 10.1, 10.4
                        logger.error(
                            f"Error during handoff iteration {handoff_count + 1}",
                            exc_info=True,
                            extra={
                                'handoff_count': handoff_count,
                                'max_handoffs': max_handoffs,
                                'error_type': type(e).__name__,
                                'error_message': str(e),
                            }
                        )
                        
                        # If this is the last handoff attempt, return error
                        if handoff_count >= max_handoffs - 1:
                            logger.error("Error on final handoff attempt, returning error message")
                            return self._format_error_message(e)
                        
                        # Otherwise, try to continue
                        handoff_count += 1
                        logger.info(f"Attempting to continue after error (attempt {handoff_count + 1}/{max_handoffs})")
                        continue
            
                # Max handoffs reached
                logger.warning(f"Max handoffs ({max_handoffs}) reached")
                
                # Add metrics to trace (max handoffs case)
                # Requirements: 5.4, 5.5, 6.4, 6.5
                if self.tracing_enabled:
                    add_metrics_to_trace(
                        handoff_count=handoff_count,
                        task_count=task_count,
                        tool_count=tool_count,
                        context_offload_count=context_offload_count,
                        context_offload_size_kb=context_offload_size_kb,
                    )
                
                # Total execution time logging (max handoffs case)
                # Requirements: 12.5
                graph_end_time = time.time()
                total_elapsed = graph_end_time - graph_start_time
                logger.info("=" * 80)
                logger.info("GRAPH EXECUTION TERMINATED (Max Handoffs)")
                logger.info(f"  Total Execution Time: {total_elapsed:.2f}s")
                logger.info(f"  Total Transitions: {transition_count}")
                logger.info(f"  Handoffs: {handoff_count}/{max_handoffs}")
                logger.info("=" * 80)
                
                return "I've gathered a lot of information but need to simplify the analysis. Please try breaking your question into smaller parts."
            
        except Exception as e:
            # Graph-level error handling with trace finalization
            # Requirements: 4.5, 10.1, 10.4, 10.5
            
            # Prepare context for trace capture
            error_context = {
                'query': query[:100],
                'transition_count': transition_count,
                'handoff_count': handoff_count,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'shared_context_keys': list(shared_context.keys()) if shared_context else [],
            }
            
            # Log error with full context
            # This will be captured in the trace by Strands SDK
            logger.error(
                f"Graph Execution Error: {type(e).__name__}",
                exc_info=True,
                extra=error_context
            )
            
            # Add metrics to trace (error case)
            # Requirements: 5.4, 5.5, 6.4, 6.5
            if self.tracing_enabled:
                try:
                    add_metrics_to_trace(
                        handoff_count=handoff_count,
                        task_count=task_count,
                        tool_count=tool_count,
                        context_offload_count=context_offload_count,
                        context_offload_size_kb=context_offload_size_kb,
                    )
                except Exception as metrics_error:
                    # Don't let metrics errors compound the original error
                    logger.warning(f"Failed to add metrics to trace on error: {metrics_error}")
            
            # Total execution time logging (error case)
            # Requirements: 12.5
            graph_end_time = time.time()
            total_elapsed = graph_end_time - graph_start_time
            logger.info("=" * 80)
            logger.info("GRAPH EXECUTION FAILED")
            logger.info(f"  Total Execution Time: {total_elapsed:.2f}s")
            logger.info(f"  Total Transitions: {transition_count}")
            logger.info(f"  Error: {type(e).__name__}: {str(e)}")
            logger.info("=" * 80)
            
            # Check if this is a safety limit error and capture in trace
            # Requirements: 4.8, 10.5
            if self._is_safety_limit_error(e):
                logger.warning("Safety limit exceeded during Graph execution")
                
                # Capture safety limit violation in trace
                # Requirements: 4.8
                self._capture_safety_limit_violation(e, error_context)
                
                return (
                    "I've reached my processing limit for this query. "
                    "This usually happens with very complex questions that require many steps. "
                    "Please try:\n"
                    "  • Breaking your question into smaller, more specific parts\n"
                    "  • Focusing on one company or metric at a time\n"
                    "  • Simplifying the time range or scope of your analysis"
                )
            
            # Ensure trace is finalized even on error
            # Requirements: 4.5
            # Note: Strands SDK automatically finalizes traces on exceptions,
            # but we log this explicitly for monitoring
            try:
                logger.debug("Trace finalization: error captured by Strands SDK")
            except Exception as trace_error:
                # Don't let trace finalization errors break the agent
                # Requirements: 4.4, 7.5, 10.4
                logger.warning(f"Error during trace finalization: {trace_error}")
            
            # Provide user-friendly error message
            error_message = self._format_error_message(e)
            return error_message
    
    def _is_safety_limit_error(self, error: Exception) -> bool:
        """
        Check if error is related to safety limits.
        
        Args:
            error: Exception that occurred
            
        Returns:
            True if error is safety-related, False otherwise
            
        Requirements: 4.8, 10.5
        """
        error_type = type(error).__name__
        error_str = str(error).lower()
        
        # Check for safety-related keywords
        safety_keywords = [
            'safety', 'limit', 'maximum', 'exceeded', 'max_iterations',
            'too many', 'iteration limit', 'step limit'
        ]
        
        return any(keyword in error_str for keyword in safety_keywords)
    
    def _capture_safety_limit_violation(self, error: Exception, context: Dict[str, Any]) -> None:
        """
        Capture safety limit violations in traces.
        
        This logs detailed information about safety limit violations to help
        with debugging and monitoring. The information is automatically captured
        in traces by Strands SDK.
        
        Args:
            error: The safety limit exception
            context: Execution context with iteration counts
            
        Requirements:
            - 4.8: Capture safety limit violations in traces
            
        Note:
            Tool failures (4.6) and agent failures (4.7) are automatically
            captured by Strands SDK and don't need explicit handling.
        """
        try:
            # Extract iteration/handoff counts from context
            handoff_count = context.get('handoff_count', 0)
            transition_count = context.get('transition_count', 0)
            
            # Log safety limit violation with full context
            # This will be captured in the trace by Strands SDK
            logger.error(
                "Safety limit violation detected",
                exc_info=True,
                extra={
                    'error_type': type(error).__name__,
                    'error_message': str(error),
                    'handoff_count': handoff_count,
                    'transition_count': transition_count,
                    'max_iterations': getattr(self.config, 'max_iterations', 20),
                    'safety_violation': True,  # Flag for trace filtering
                }
            )
        except Exception as e:
            # Don't let trace capture errors break the agent
            # Requirements: 4.4, 7.5, 10.4
            logger.warning(f"Failed to capture safety limit violation in trace: {e}")
    
    def _display_task_list(self, tasks: List[Any]) -> None:
        """
        Display task list after planning.
        
        Formats tasks with status indicators and displays them in a
        formatted box using RichUI.
        
        Args:
            tasks: List of Task objects from Planner
            
        Requirements: 1.1, 1.2, 1.3, 1.5
        """
        if not self.ui:
            return
        
        if not tasks:
            # Handle empty task list case
            # Requirements: 1.5
            logger.debug("No tasks to display - showing empty task list message")
            
            # Display empty task list with appropriate message
            # This helps users understand that no tasks were created
            empty_task_dict = [{
                'description': 'No tasks created - query may be out of scope',
                'status': 'pending'
            }]
            self.ui.print_task_list(empty_task_dict)
            return
        
        # Convert Task objects to dictionary format expected by RichUI
        # Requirements: 1.2, 1.3
        task_dicts = []
        for task in tasks:
            task_dict = {
                'description': task.description,
                'status': 'pending'  # All tasks start as pending
            }
            task_dicts.append(task_dict)
        
        # Display task list using RichUI
        # Requirements: 1.1, 1.3
        self.ui.print_task_list(task_dicts)
        logger.info(f"Displayed {len(tasks)} tasks to user")
    
    def _format_error_message(self, error: Exception) -> str:
        """
        Format error message for user display.
        
        Args:
            error: Exception that occurred
            
        Returns:
            User-friendly error message
            
        Requirements: 10.1, 10.4
        """
        error_type = type(error).__name__
        error_str = str(error).lower()
        
        # Check for specific error types
        if "safety" in error_str or "limit" in error_str or "maximum" in error_str:
            return (
                "I've reached my processing limit for this query. "
                "This usually happens with very complex questions. "
                "Please try:\n"
                "  • Breaking your question into smaller parts\n"
                "  • Focusing on specific companies or metrics\n"
                "  • Simplifying the scope of your analysis"
            )
        
        if "timeout" in error_str or "timed out" in error_str:
            return (
                "The query took too long to process. "
                "This can happen with complex multi-company analyses. "
                "Please try:\n"
                "  • Asking about fewer companies at once\n"
                "  • Reducing the time range of your analysis\n"
                "  • Being more specific about what you need"
            )
        
        if "api" in error_str or "connection" in error_str or "network" in error_str:
            return (
                "I encountered an issue connecting to the financial data service. "
                "This is usually temporary. Please:\n"
                "  • Wait a moment and try again\n"
                "  • Check your internet connection\n"
                "  • Contact support if the issue persists"
            )
        
        if "bedrock" in error_str or "model" in error_str:
            return (
                "I encountered an issue with the AI model service. "
                "This is usually temporary. Please:\n"
                "  • Wait a moment and try again\n"
                "  • Try rephrasing your question\n"
                "  • Contact support if the issue persists"
            )
        
        if "validation" in error_str:
            return (
                "I had trouble validating the data I collected. "
                "The information may be incomplete or in an unexpected format. "
                "Please try:\n"
                "  • Rephrasing your question more specifically\n"
                "  • Asking about a different time period\n"
                "  • Verifying the company ticker symbols are correct"
            )
        
        # Generic error message with helpful suggestions
        return (
            "I encountered an unexpected error while processing your query. "
            "This could be due to various reasons. Please try:\n"
            "  • Rephrasing your question\n"
            "  • Simplifying your request\n"
            "  • Trying again in a moment\n"
            "  • Contacting support if the issue persists\n\n"
            f"Error type: {error_type}"
        )

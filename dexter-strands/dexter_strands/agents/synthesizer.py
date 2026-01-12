"""
Synthesizer Agent for multi-agent architecture.

The Synthesizer Agent is responsible for:
- Retrieving context pointers from SharedContext
- Using LLM-based context selection
- Validating if query can be fully answered
- Identifying data gaps and handing off to Planner
- Generating comprehensive streaming answers
- Integrating with RichUI for streaming

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 7.2, 7.3
"""

import json
import logging
from typing import Any, Dict, List, Optional, Iterator

from strands import Agent
from strands.models import BedrockModel
from pydantic import BaseModel, Field

from ..models import ContextPointer, ValidationResult
from ..context import ContextManager
from ..prompts import get_synthesizer_prompt
from ..ui import RichUI

logger = logging.getLogger(__name__)


class GoalValidation(BaseModel):
    """Schema for goal validation output."""
    is_complete: bool = Field(
        description="Whether the query can be fully answered with available data"
    )
    reason: str = Field(
        description="Explanation of the validation decision"
    )
    missing_data: Optional[str] = Field(
        default=None,
        description="Description of missing data if validation fails"
    )


class SynthesizerAgent:
    """
    Synthesizer Agent for answer generation and validation.
    
    The Synthesizer retrieves collected data, validates if the query can be
    answered, and either generates a comprehensive answer or hands off back
    to the Planner for additional data collection.
    
    Responsibilities:
    - Retrieve context pointers from SharedContext
    - Use LLM to select relevant contexts for the query
    - Validate if query can be fully answered with available data
    - If insufficient data, identify gaps and handoff to Planner
    - If sufficient data, generate streaming answer
    - Integrate with RichUI for streaming display
    
    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 7.2, 7.3
    """
    
    def __init__(
        self,
        model: BedrockModel,
        context_manager: ContextManager,
        ui: Optional[RichUI] = None,
        agent_id: str = "synthesizer",
    ):
        """
        Initialize Synthesizer Agent.
        
        Args:
            model: Configured BedrockModel instance
            context_manager: ContextManager for loading contexts
            ui: Optional RichUI instance for streaming display
            agent_id: Unique identifier for this agent
        """
        logger.info("Initializing SynthesizerAgent")
        
        self.model = model
        self.context_manager = context_manager
        self.ui = ui
        self.agent_id = agent_id
        
        # Create system prompt
        # Requirements: 8.3
        system_prompt = get_synthesizer_prompt()
        
        # Create Strands Agent for synthesis
        # No tools needed - uses context loading internally
        self.agent = Agent(
            model=model,
            system_prompt=system_prompt,
            agent_id=agent_id,
            name="Synthesizer",
            description="Generates answers from collected data",
        )
        
        logger.info("SynthesizerAgent initialized")
    
    def synthesize_answer(
        self,
        query: str,
        shared_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate answer from collected data or handoff to Planner.
        
        This method orchestrates the synthesis process:
        1. Retrieve context pointers from SharedContext
        2. Select relevant contexts using LLM
        3. Validate if query can be answered
        4. If insufficient, identify gaps and handoff to Planner
        5. If sufficient, generate streaming answer
        
        Args:
            query: User query to answer
            shared_context: Optional shared context dict
            
        Returns:
            Generated answer text (or handoff indicator)
            
        Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 7.2, 7.3, 12.1, 12.4
        """
        import time
        
        # Agent start logging
        # Requirements: 12.1
        start_time = time.time()
        logger.info("=" * 80)
        logger.info(f"AGENT START: Synthesizer (agent_id={self.agent_id})")
        logger.info(f"  Query: {query[:100]}{'...' if len(query) > 100 else ''}")
        logger.info("=" * 80)
        
        logger.info(f"Synthesizer processing query: {query[:100]}...")
        
        # Step 1: Retrieve context pointers from SharedContext
        # Requirements: 4.1
        context_pointers = self._retrieve_context_pointers(shared_context)
        
        if not context_pointers:
            # No data collected - answer directly
            logger.info("No context pointers available, answering without data")
            answer = self._generate_answer_without_data(query)
            
            # Agent end logging (no data case)
            # Requirements: 12.1
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info("=" * 80)
            logger.info(f"AGENT END: Synthesizer (agent_id={self.agent_id}) [NO DATA]")
            logger.info(f"  Execution Time: {elapsed_time:.2f}s")
            logger.info(f"  Answer Length: {len(answer)} characters")
            logger.info("=" * 80)
            
            return answer
        
        # Step 2: Select relevant contexts using ContextManager
        # Requirements: 4.2
        selected_contexts = self._select_relevant_contexts(query, context_pointers)
        
        # Step 3: Validate if query can be fully answered
        # Requirements: 4.3, 12.4
        validation = self._validate_goal(query, selected_contexts)
        
        # Validation decision logging
        # Requirements: 12.4
        logger.info(f"→ Goal Validation:")
        logger.info(f"  Result: {'COMPLETE' if validation.is_complete else 'INCOMPLETE'}")
        logger.info(f"  Reason: {validation.reason}")
        if validation.missing_data:
            logger.info(f"  Missing Data: {validation.missing_data}")
        
        # Step 4: Handle insufficient data - handoff to Planner
        # Requirements: 4.4, 5.1, 5.2
        if not validation.is_complete:
            logger.info(f"Goal validation failed: {validation.reason}")
            logger.info(f"Missing data: {validation.missing_data}")
            
            # Store data gaps in SharedContext for Planner
            # Requirements: 5.1, 5.2, 5.4
            if shared_context is not None:
                logger.debug(f"SharedContext before storing data gaps: {list(shared_context.keys())}")
                shared_context['data_gaps'] = validation.missing_data
                logger.info(f"✓ Stored data gaps in SharedContext for handoff")
                logger.debug(f"SharedContext after storing data gaps: {list(shared_context.keys())}")
                
                # Verify data persistence
                # Requirements: 5.5
                stored_gaps = shared_context.get('data_gaps')
                if stored_gaps == validation.missing_data:
                    logger.debug("✓ Verified: Data gaps persisted in SharedContext")
                else:
                    logger.warning("⚠ Verification failed: Data gaps mismatch in SharedContext")
            else:
                logger.warning("⚠ No SharedContext provided - data gaps not stored")
            
            # Agent end logging (handoff case)
            # Requirements: 12.1
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info("=" * 80)
            logger.info(f"AGENT END: Synthesizer (agent_id={self.agent_id}) [HANDOFF]")
            logger.info(f"  Execution Time: {elapsed_time:.2f}s")
            logger.info(f"  Action: Handoff to Planner")
            logger.info("=" * 80)
            
            # Return handoff indicator
            return "HANDOFF_TO_PLANNER"
        
        # Step 5: Generate streaming answer
        # Requirements: 4.5, 7.2, 7.3
        logger.info("Goal validation passed, generating answer")
        answer = self._generate_streaming_answer(query, selected_contexts)
        
        # Agent end logging
        # Requirements: 12.1
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info("=" * 80)
        logger.info(f"AGENT END: Synthesizer (agent_id={self.agent_id})")
        logger.info(f"  Execution Time: {elapsed_time:.2f}s")
        logger.info(f"  Answer Length: {len(answer)} characters")
        logger.info("=" * 80)
        
        return answer
    
    def _retrieve_context_pointers(
        self,
        shared_context: Optional[Dict[str, Any]]
    ) -> List[ContextPointer]:
        """
        Retrieve context pointers from SharedContext.
        
        Args:
            shared_context: Shared context dictionary
            
        Returns:
            List of ContextPointer objects
            
        Requirements: 4.1, 5.1, 5.3, 5.4
        """
        # Log SharedContext state at entry
        # Requirements: 5.4
        if shared_context is None:
            logger.warning("⚠ No shared context provided")
            return []
        
        logger.debug(f"SharedContext at Synthesizer entry: {list(shared_context.keys())}")
        
        # Get context pointers from SharedContext
        # Requirements: 5.3
        pointers_data = shared_context.get('context_pointers', [])
        
        if not pointers_data:
            logger.debug("No context pointers in SharedContext")
            return []
        
        logger.info(f"→ Retrieving {len(pointers_data)} context pointers from SharedContext")
        
        # Convert to ContextPointer objects if needed
        pointers = []
        for i, ptr_data in enumerate(pointers_data):
            if isinstance(ptr_data, ContextPointer):
                pointers.append(ptr_data)
                logger.debug(f"  Pointer {i+1}: {ptr_data.tool_name} (task {ptr_data.task_id})")
            elif isinstance(ptr_data, dict):
                # Convert dict to ContextPointer
                pointer = ContextPointer(**ptr_data)
                pointers.append(pointer)
                logger.debug(f"  Pointer {i+1}: {pointer.tool_name} (task {pointer.task_id})")
            else:
                logger.warning(f"⚠ Unknown pointer type at index {i}: {type(ptr_data)}")
        
        logger.info(f"✓ Retrieved {len(pointers)} context pointers from SharedContext")
        
        # Verify data persistence
        # Requirements: 5.5
        if len(pointers) == len(pointers_data):
            logger.debug(f"✓ Verified: All {len(pointers)} pointers successfully retrieved")
        else:
            logger.warning(f"⚠ Verification warning: Retrieved {len(pointers)} of {len(pointers_data)} pointers")
        
        return pointers
    
    def _select_relevant_contexts(
        self,
        query: str,
        context_pointers: List[ContextPointer]
    ) -> List[Dict[str, Any]]:
        """
        Select relevant contexts using LLM-based selection.
        
        Args:
            query: User query
            context_pointers: Available context pointers
            
        Returns:
            List of loaded context data dictionaries
            
        Requirements: 4.2, 7.1
        """
        logger.info(f"Selecting relevant contexts for query: {query[:100]}...")
        
        # Convert ContextPointer objects to dict format for ContextManager
        pointers_dict = []
        for ptr in context_pointers:
            ptr_dict = {
                'id': int(ptr.id) if isinstance(ptr.id, str) else ptr.id,
                'filepath': ptr.filepath,
                'tool_name': ptr.tool_name,
                'args': ptr.args,
                'summary': ptr.summary,
                'size_kb': ptr.size_kb,
                'task_id': ptr.task_id,
            }
            pointers_dict.append(ptr_dict)
        
        # Use ContextManager for LLM-based selection
        # Requirements: 7.1
        selected_filepaths = self.context_manager.select_relevant_contexts(
            query=query,
            available_pointers=pointers_dict
        )
        
        # Load selected contexts from files
        selected_contexts = self.context_manager.load_contexts(selected_filepaths)
        
        logger.info(f"Selected and loaded {len(selected_contexts)} contexts")
        return selected_contexts
    
    def _validate_goal(
        self,
        query: str,
        selected_contexts: List[Dict[str, Any]]
    ) -> ValidationResult:
        """
        Validate if query can be fully answered with available data.
        
        Uses LLM to assess data sufficiency and identify gaps.
        
        Args:
            query: User query
            selected_contexts: Selected context data
            
        Returns:
            ValidationResult with completion status and missing data
            
        Requirements: 4.3, 6.3, 6.4, 6.5
        """
        logger.info("Validating if goal can be achieved with available data")
        
        try:
            # Format contexts for validation prompt
            formatted_contexts = []
            for ctx in selected_contexts:
                tool_name = ctx.get('tool_name', 'unknown')
                args = ctx.get('args', {})
                summary = ctx.get('summary', 'No summary available')
                formatted_contexts.append(
                    f"- {tool_name} (args: {args}): {summary}"
                )
            
            contexts_text = "\n".join(formatted_contexts) if formatted_contexts else "No contexts available"
            
            # Create validation prompt
            prompt = f"""User Query: {query}

Available Data:
{contexts_text}

Assess whether the available data is sufficient to provide a USEFUL answer to the user's query.

CRITICAL: Be PRACTICAL, not perfectionist. The goal is to provide a useful answer, not a perfect one.
- If you have the CORE data needed, mark as COMPLETE
- Do NOT request additional data just to make the answer "more comprehensive"
- Do NOT request redundant data that's already been collected
- Remember: More data ≠ Better answer

Consider:
- Does the data cover the CORE aspects of the query? (e.g., for "Tesla valuation", do we have Tesla's P/E, EV/EBITDA?)
- Are there CRITICAL companies or metrics COMPLETELY MISSING from the data?
- Can I provide a USEFUL, ACTIONABLE answer with this data?
- Have we already collected similar/redundant data in previous iterations?

RED FLAGS (mark incomplete ONLY if):
- Primary company data is completely missing
- Core metric explicitly requested is absent
- Query asks for comparison but we only have one company's data

GREEN FLAGS (mark complete if):
- We have the primary company's key metrics
- We have enough peer data for basic comparison (even if not all peers)
- We have historical data showing trends (even if not the full requested period)
- Similar data has been collected multiple times already

Guidelines:
- **Comparison queries**: If we have data for the PRIMARY company + at least 2-3 peers, mark as COMPLETE (don't need all possible peers)
- **Trend queries**: If we have data for MOST of the requested period, mark as COMPLETE (don't need every single data point)
- **Valuation queries**: If we have key ratios (P/E, EV/EBITDA) for the company, mark as COMPLETE (don't need every possible ratio)
- **Current price queries**: Most recent available price is sufficient
- **Historical context**: If we have 6-12 months of data, that's sufficient for "historical levels"
- **Redundancy check**: If similar data was collected in previous handoffs, mark as COMPLETE

ONLY mark as incomplete if:
- Primary company's core data is COMPLETELY MISSING
- Query explicitly asks for specific metric that's absent
- Comparison query but we only have the primary company (need at least 2-3 peers)

Temporal Query Handling:
- "Current stock price" = most recent available price (acceptable even if markets are closed)
- "Latest" or "most recent" data = the newest available data point is sufficient
- Weekend/after-hours queries: Previous closing price is acceptable for "current" price requests

If data is insufficient, identify specific gaps (e.g., "Missing Microsoft data entirely", "Need 2023 data but only have 2022").

Respond with:
- is_complete: true if data is sufficient for a useful answer, false only if critical data is missing
- reason: Brief explanation of your assessment
- missing_data: If incomplete, describe what specific CRITICAL data is missing
"""
            
            # Call LLM for validation
            # Use structured output for reliable parsing
            validation_agent = Agent(
                model=self.model,
                system_prompt="You are a validation agent that assesses data sufficiency for answering queries.",
                structured_output_model=GoalValidation,
            )
            
            result = validation_agent(prompt)
            
            # Extract validation from structured output
            if hasattr(result, 'structured_output') and result.structured_output:
                validation_data = result.structured_output
                
                return ValidationResult(
                    is_complete=validation_data.is_complete,
                    reason=validation_data.reason,
                    missing_data=validation_data.missing_data
                )
            else:
                # Fallback: parse from text
                logger.warning("No structured output, attempting to parse from text")
                # Extract text from message content
                if result.message and 'content' in result.message and result.message['content']:
                    text = result.message['content'][0]['text']
                else:
                    text = ""
                return self._parse_validation_from_text(text)
            
        except Exception as e:
            # Conservative: assume data is insufficient on error
            # Requirements: 10.1, 10.4
            # Log error with full context
            logger.error(
                f"Synthesizer Agent Error: Goal validation failed",
                exc_info=True,
                extra={
                    'agent_id': self.agent_id,
                    'query': query[:100],
                    'num_contexts': len(selected_contexts),
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                }
            )
            
            # Return helpful message to user
            return ValidationResult(
                is_complete=False,
                reason=f"Validation error: {type(e).__name__}",
                missing_data="I encountered an issue while validating the data. Please try rephrasing your question or simplifying your request."
            )
    
    def _parse_validation_from_text(self, text: str) -> ValidationResult:
        """
        Parse validation result from text response.
        
        Fallback method when structured output is not available.
        
        Args:
            text: Text response from agent
            
        Returns:
            ValidationResult
        """
        try:
            # Try to find JSON in the text
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = text[start:end]
                data = json.loads(json_str)
                
                return ValidationResult(
                    is_complete=data.get('is_complete', False),
                    reason=data.get('reason', 'Parsed from text'),
                    missing_data=data.get('missing_data')
                )
            
            # If no JSON found, assume incomplete
            logger.warning("Could not parse validation from text, assuming incomplete")
            return ValidationResult(
                is_complete=False,
                reason="Could not parse validation",
                missing_data="Unable to determine data gaps"
            )
            
        except Exception as e:
            logger.error(f"Failed to parse validation from text: {e}")
            return ValidationResult(
                is_complete=False,
                reason="Parse error",
                missing_data="Unable to determine data gaps"
            )
    
    def _generate_streaming_answer(
        self,
        query: str,
        selected_contexts: List[Dict[str, Any]]
    ) -> str:
        """
        Generate streaming answer from selected contexts.
        
        Args:
            query: User query
            selected_contexts: Selected context data
            
        Returns:
            Generated answer text
            
        Requirements: 4.5, 7.2, 7.3, 10.4
        """
        logger.info("Generating streaming answer")
        
        try:
            # Format contexts for answer generation
            formatted_results = []
            for ctx in selected_contexts:
                tool_name = ctx.get('tool_name', 'unknown')
                args = ctx.get('args', {})
                result = ctx.get('result', {})
                formatted_results.append(
                    f"Output of {tool_name} with args {args}:\n{result}"
                )
            
            all_results = "\n\n".join(formatted_results) if formatted_results else "No data available"
            
            # Create answer generation prompt
            prompt = f"""Original user query: "{query}"

Data and results collected from tools:
{all_results}

Based on the data above, provide a comprehensive answer to the user's query.
Include specific numbers, calculations, and insights.
"""
            
            # Generate answer using agent
            result = self.agent(prompt)
            
            # Extract text from message content
            # AgentResult.message is a dict with 'content' key containing list of content blocks
            if result.message and 'content' in result.message and result.message['content']:
                answer_text = result.message['content'][0]['text']
            else:
                answer_text = "I apologize, but I was unable to generate a response."
            
            # Stream answer through UI if available
            # Requirements: 7.2, 7.3
            if self.ui:
                # Convert answer to character iterator for streaming
                char_iterator = iter(answer_text)
                accumulated_answer = self.ui.stream_answer(char_iterator)
                return accumulated_answer
            else:
                # No UI - return answer directly
                logger.debug("No UI available, returning answer without streaming")
                return answer_text
                
        except Exception as e:
            # Error handling for answer generation
            # Requirements: 10.1, 10.4
            logger.error(
                f"Synthesizer Agent Error: Answer generation failed",
                exc_info=True,
                extra={
                    'agent_id': self.agent_id,
                    'query': query[:100],
                    'num_contexts': len(selected_contexts),
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                }
            )
            
            # Return helpful error message to user
            return (
                "I encountered an issue while generating the answer. "
                "The data was collected successfully, but I had trouble synthesizing it. "
                "Please try asking your question in a different way, or contact support if the issue persists."
            )
    
    def _generate_answer_without_data(self, query: str) -> str:
        """
        Generate answer when no data was collected.
        
        Args:
            query: User query
            
        Returns:
            Generated answer text
            
        Requirements: 10.4
        """
        logger.info("Generating answer without collected data")
        
        try:
            prompt = f"""Original user query: "{query}"

No data was collected from tools.

Provide a helpful response explaining that the query may be outside the scope of available financial research tools, or answer using general knowledge if appropriate.
"""
            
            # Generate answer using agent
            result = self.agent(prompt)
            
            # Extract text from message content
            # AgentResult.message is a dict with 'content' key containing list of content blocks
            if result.message and 'content' in result.message and result.message['content']:
                answer_text = result.message['content'][0]['text']
            else:
                answer_text = "I apologize, but I was unable to generate a response."
            
            # Stream answer through UI if available
            if self.ui:
                char_iterator = iter(answer_text)
                accumulated_answer = self.ui.stream_answer(char_iterator)
                return accumulated_answer
            else:
                return answer_text
                
        except Exception as e:
            # Error handling for answer generation without data
            # Requirements: 10.1, 10.4
            logger.error(
                f"Synthesizer Agent Error: Answer generation without data failed",
                exc_info=True,
                extra={
                    'agent_id': self.agent_id,
                    'query': query[:100],
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                }
            )
            
            # Return helpful error message to user
            return (
                "I'm sorry, but I encountered an issue while processing your query. "
                "Your question may be outside the scope of my financial research capabilities, "
                "or there may be a temporary service issue. Please try again or rephrase your question."
            )

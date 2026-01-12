"""Context management for Dexter Strands agent.

This module provides intelligent context management for large tool outputs:
- Offloading large tool results to files
- LLM-based summarization of tool outputs
- Intelligent context selection for answer generation
- Context loading from files

The context management system optimizes token usage while maintaining high
answer quality by storing large tool outputs externally and loading only
relevant data when needed.

## Architecture

Context offloading is implemented at the conversation manager level using
`OffloadingConversationManager` from the `conversation` module. This approach
offloads contexts after messages are properly formatted by the Strands SDK,
ensuring compatibility with the SDK's internal message structure.

The `ContextManager` class in this module provides the core functionality for:
- Saving tool outputs to files with LLM-generated summaries
- Selecting relevant contexts using LLM-based semantic analysis
- Loading selected contexts from disk

See `conversation.py` for the `OffloadingConversationManager` implementation.
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import logging
from pydantic import BaseModel, Field

# Use standard logging - will be configured by setup_logger() in main application
logger = logging.getLogger(__name__)


class SelectedContexts(BaseModel):
    """Represents selected context file IDs for answer generation."""
    context_ids: List[int] = Field(
        ..., 
        description="List of context pointer IDs (0-indexed) that are relevant for answering the query."
    )


class ContextManager:
    """Manages context offloading, selection, and loading.
    
    Responsibilities:
    - Save large tool outputs to files
    - Generate LLM-based summaries
    - Track pointer metadata
    - Select relevant contexts using LLM
    - Load selected contexts from files
    """

    def __init__(self, context_dir: str = ".dexter-strands/context"):
        """Initialize context manager.
        
        Args:
            context_dir: Directory for storing context files
        """
        self.context_dir = Path(context_dir)
        self.context_dir.mkdir(parents=True, exist_ok=True)
        self.pointers: List[Dict[str, Any]] = []
        
        logger.info(f"ContextManager initialized with directory: {self.context_dir}")

    def save_context(
        self,
        tool_name: str,
        args: dict,
        result: Any,
        task_id: Optional[int] = None
    ) -> str:
        """Save tool output to file and create pointer.
        
        Args:
            tool_name: Name of the tool that generated the output
            args: Arguments passed to the tool
            result: Tool output data
            task_id: Optional task ID for tracking
            
        Returns:
            Filepath where context was saved
            
        Process:
        1. Generate filename hash from tool name + args
        2. Generate LLM summary of the output
        3. Save complete context to JSON file
        4. Create pointer metadata
        5. Add pointer to in-memory list
        """
        # Step 1: Generate filename from tool name and args hash
        args_hash = self._hash_args(args)
        filename = f"{tool_name}_{args_hash}.json"
        filepath = self.context_dir / filename
        
        logger.debug(f"Saving context for {tool_name} to {filepath}")
        
        # Step 2: Generate LLM summary (will be implemented in task 3.2)
        summary = self._generate_summary(tool_name, args, result)
        
        # Step 3: Prepare context data structure
        timestamp = datetime.now()
        context_data = {
            "tool_name": tool_name,
            "args": args,
            "summary": summary,
            "timestamp": timestamp.isoformat(),
            "task_id": task_id,
            "result": result
        }
        
        # Step 4: Save to file (will be implemented in task 3.4)
        self._save_to_file(filepath, context_data)
        
        # Step 5: Create pointer metadata (will be implemented in task 3.6)
        result_size_kb = len(json.dumps(result, default=self._json_serializer)) / 1024
        pointer = {
            "id": len(self.pointers),
            "filepath": str(filepath),
            "filename": filename,
            "tool_name": tool_name,
            "args": args,
            "summary": summary,
            "timestamp": timestamp,
            "size_kb": result_size_kb,
            "task_id": task_id
        }
        
        # Step 6: Add pointer to in-memory list
        self.pointers.append(pointer)
        
        logger.info(
            f"Context saved: {tool_name} ({result_size_kb:.1f}KB) -> {filename}"
        )
        
        return str(filepath)

    def _generate_summary(
        self,
        tool_name: str,
        args: dict,
        result: Any
    ) -> str:
        """Generate LLM-based summary of tool output.
        
        Args:
            tool_name: Name of the tool
            args: Tool arguments
            result: Tool output
            
        Returns:
            One-sentence summary describing the data
            
        Example:
            "Apple's annual income statements for 2021-2023, 
             showing revenue of $365B, $383B, $394B"
        """
        try:
            import boto3
            
            # Create boto3 client for Bedrock Runtime
            bedrock = boto3.client(
                service_name='bedrock-runtime',
                region_name=os.getenv('AWS_REGION', 'us-east-1')
            )
            
            # Format the result for preview (truncate if too large)
            result_str = json.dumps(result, default=self._json_serializer, indent=2)
            if len(result_str) > 2000:
                result_str = result_str[:2000] + "\n... (truncated)"
            
            # Format arguments for display
            args_str = json.dumps(args, indent=2)
            
            # Create prompt for summary generation
            prompt = f"""Generate a concise one-sentence summary of this tool output.

Tool: {tool_name}
Arguments: {args_str}

Output Preview:
{result_str}

Provide a brief, informative summary that captures the key information in the output. Focus on:
- What data is included (e.g., company name, time period, data type)
- Key metrics or values if applicable
- Any notable patterns or highlights

Summary (one sentence):"""
            
            # Call Bedrock API directly for summary
            # Using Claude Haiku for fast, cheap summaries
            model_id = os.getenv('BEDROCK_MODEL_ID', 'us.anthropic.claude-haiku-4-5-20251001-v1:0')
            
            response = bedrock.invoke_model(
                modelId=model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 150,
                    "temperature": 0.0,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                })
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            summary = response_body['content'][0]['text'].strip()
            
            # Ensure it's not too long (max 200 chars)
            if len(summary) > 200:
                summary = summary[:197] + "..."
            
            logger.debug(f"Generated summary for {tool_name}: {summary}")
            return summary
            
        except Exception as e:
            # Fallback: create a simple summary without LLM
            logger.warning(f"LLM summary generation failed: {e}, using fallback")
            
            # Create basic summary from tool name and args
            args_summary = ", ".join(f"{k}={v}" for k, v in args.items())
            fallback_summary = f"{tool_name} output with args: {args_summary}"
            
            # Truncate if too long
            if len(fallback_summary) > 200:
                fallback_summary = fallback_summary[:197] + "..."
            
            return fallback_summary

    def _hash_args(self, args: dict) -> str:
        """Generate hash of arguments for filename.
        
        Args:
            args: Tool arguments dictionary
            
        Returns:
            12-character hash string
        """
        # Sort keys for deterministic hashing
        sorted_args = json.dumps(args, sort_keys=True, default=self._json_serializer)
        
        # Generate MD5 hash
        hash_obj = hashlib.md5(sorted_args.encode('utf-8'))
        
        # Return first 12 characters of hex digest
        return hash_obj.hexdigest()[:12]

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for complex types.
        
        Args:
            obj: Object to serialize
            
        Returns:
            JSON-serializable representation
        """
        # Handle Pydantic models
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        elif hasattr(obj, 'dict'):
            return obj.dict()
        
        # Handle datetime objects
        if isinstance(obj, datetime):
            return obj.isoformat()
        
        # Handle Path objects
        if isinstance(obj, Path):
            return str(obj)
        
        # For other types, convert to string
        return str(obj)

    def _save_to_file(self, filepath: Path, context_data: Dict[str, Any]) -> None:
        """Save context data to file atomically.
        
        Args:
            filepath: Path where to save the file
            context_data: Context data to save
            
        Process:
        1. Write to temporary file first
        2. Use os.replace() for atomic operation
        3. Include metadata and result in JSON
        """
        # Step 1: Write to temporary file first
        temp_filepath = filepath.with_suffix('.tmp')
        
        try:
            # Write JSON to temporary file
            with open(temp_filepath, 'w', encoding='utf-8') as f:
                json.dump(context_data, f, indent=2, default=self._json_serializer)
            
            # Step 2: Use os.replace() for atomic operation
            # This ensures the file is either fully written or not written at all
            os.replace(temp_filepath, filepath)
            
            logger.debug(f"Context data written atomically to {filepath}")
            
        except Exception as e:
            # Clean up temporary file if it exists
            if temp_filepath.exists():
                try:
                    temp_filepath.unlink()
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temp file {temp_filepath}: {cleanup_error}")
            
            # Re-raise the original error
            logger.error(f"Failed to save context to {filepath}: {e}")
            raise

    def select_relevant_contexts(
        self,
        query: str,
        available_pointers: List[Dict[str, Any]]
    ) -> List[str]:
        """Select relevant contexts using LLM with timeout protection.
        
        Args:
            query: User's query
            available_pointers: List of pointer metadata
            
        Returns:
            List of filepaths for selected contexts
            
        Process:
        1. Check total size and use fast selection for large contexts
        2. Format pointers for LLM prompt
        3. Call LLM with query + pointer summaries (with timeout)
        4. Parse LLM response for selected IDs
        5. Validate IDs
        6. Map IDs to filepaths
        7. Fallback to all contexts if anything fails
        """
        # Handle empty pointers list
        if not available_pointers:
            logger.debug("No contexts available for selection")
            return []
        
        logger.info(f"Selecting relevant contexts for query: {query[:100]}...")
        logger.debug(f"Available contexts: {len(available_pointers)}")
        
        # Calculate total size for optimization
        total_size_kb = sum(ptr.get('size_kb', 0) for ptr in available_pointers)
        
        # If contexts are very large, use fast keyword-based selection
        if total_size_kb > 200:  # 200KB threshold
            logger.info(f"Large contexts detected ({total_size_kb:.1f}KB), using fast keyword-based selection")
            return self._fast_context_selection(query, available_pointers)
        
        try:
            # Step 1: Format pointers for LLM prompt
            pointer_summaries = []
            for i, pointer in enumerate(available_pointers):
                summary_text = (
                    f"[{i}] {pointer['tool_name']}\n"
                    f"    Args: {json.dumps(pointer['args'])}\n"
                    f"    Summary: {pointer['summary']}\n"
                    f"    Size: {pointer['size_kb']:.1f}KB"
                )
                pointer_summaries.append(summary_text)
            
            pointers_text = "\n\n".join(pointer_summaries)
            
            # Step 2: Create selection prompt
            prompt = f"""User Query: {query}

Available Tool Outputs:
{pointers_text}

Analyze which tool outputs contain data directly relevant to answering the query.
Select only the outputs that are necessary - avoid selecting irrelevant data.
Consider the query's specific requirements (ticker symbols, time periods, metrics, etc.).

Return a JSON object with a "context_ids" field containing a list of IDs (0-indexed) of relevant outputs.

Example format:
{{"context_ids": [0, 2, 5]}}"""
            
            system_prompt = """You are a context selection agent for Dexter, a financial research agent.
Your job is to identify which tool outputs are relevant for answering a user's query.

You will be given:
1. The original user query
2. A list of available tool outputs with summaries

Your task:
- Analyze which tool outputs contain data directly relevant to answering the query
- Select only the outputs that are necessary - avoid selecting irrelevant data
- Consider the query's specific requirements (ticker symbols, time periods, metrics, etc.)
- Return a JSON object with a "context_ids" field containing a list of IDs (0-indexed) of relevant outputs

Example:
If the query asks about "Apple's revenue", select outputs from tools that retrieved Apple's financial data.
If the query asks about "Microsoft's stock price", select outputs from price-related tools for Microsoft."""
            
            # Step 3: Call LLM with structured output (with timeout)
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Context selection timed out")
            
            # Set timeout (30 seconds)
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)
            
            try:
                selected_ids = self._call_llm_for_selection(prompt, system_prompt, len(available_pointers))
            finally:
                signal.alarm(0)  # Cancel alarm
            
            # Step 4: Validate IDs (already done in _call_llm_for_selection)
            
            # Step 5: Map IDs to filepaths
            selected_filepaths = [
                available_pointers[idx]["filepath"] 
                for idx in selected_ids
            ]
            
            logger.info(f"Selected {len(selected_filepaths)} contexts: {selected_ids}")
            return selected_filepaths
            
        except TimeoutError:
            logger.warning("Context selection timed out after 30s, selecting all contexts")
            return [ptr["filepath"] for ptr in available_pointers]
        except Exception as e:
            # Step 6: Fallback to all contexts if anything fails
            logger.warning(f"Context selection failed: {e}, falling back to all contexts")
            return [ptr["filepath"] for ptr in available_pointers]

    def load_contexts(self, filepaths: List[str]) -> List[Dict[str, Any]]:
        """Load context files from disk.
        
        Args:
            filepaths: List of context file paths
            
        Returns:
            List of context data dictionaries
            
        Process:
        1. Read each file
        2. Parse JSON
        3. Extract result data
        4. Handle errors gracefully (skip corrupted files)
        """
        # Step 1: Initialize empty results list (Task 5.1)
        contexts = []
        
        # Handle empty selection case (Task 5.5)
        if not filepaths:
            logger.debug("No context files to load")
            return contexts
        
        logger.info(f"Loading {len(filepaths)} context files")
        
        # Step 2: Read each file from disk (Task 5.2)
        for filepath in filepaths:
            try:
                # Convert to Path object for consistent handling
                file_path = Path(filepath)
                
                # Check if file exists (Task 5.4 - error handling)
                if not file_path.exists():
                    logger.warning(f"Context file not found: {filepath}, skipping")
                    continue
                
                # Read file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                
                # Step 3: Parse JSON content (Task 5.2)
                try:
                    context_data = json.loads(file_content)
                except json.JSONDecodeError as e:
                    # Handle corrupted JSON gracefully (Task 5.4)
                    logger.warning(f"Failed to parse JSON from {filepath}: {e}, skipping")
                    continue
                
                # Step 4: Extract result data (Task 5.2)
                # The context file contains metadata + result
                # We return the full context data (including metadata)
                # so the caller can access both result and metadata if needed
                contexts.append(context_data)
                
                logger.debug(f"Loaded context from {filepath}")
                
            except IOError as e:
                # Handle file I/O errors gracefully (Task 5.4)
                logger.warning(f"Failed to read context file {filepath}: {e}, skipping")
                continue
            except Exception as e:
                # Catch any other unexpected errors (Task 5.4)
                logger.warning(f"Unexpected error loading context from {filepath}: {e}, skipping")
                continue
        
        # Step 5: Return loaded contexts (Task 5.5)
        total_size_kb = sum(
            len(json.dumps(ctx, default=self._json_serializer)) / 1024 
            for ctx in contexts
        )
        
        logger.info(f"Successfully loaded {len(contexts)} contexts ({total_size_kb:.1f}KB total)")
        
        return contexts

    def _call_llm_for_selection(
        self,
        prompt: str,
        system_prompt: str,
        num_contexts: int
    ) -> List[int]:
        """Call LLM for context selection with validation and error handling.
        
        Args:
            prompt: User prompt with query and pointer summaries
            system_prompt: System prompt for context selection
            num_contexts: Total number of available contexts
            
        Returns:
            List of validated context IDs
            
        Raises:
            Exception: If LLM call fails or returns invalid response
        """
        import boto3
        
        try:
            # Create boto3 client for Bedrock Runtime
            bedrock = boto3.client(
                service_name='bedrock-runtime',
                region_name=os.getenv('AWS_REGION', 'us-east-1')
            )
            
            # Use Claude Haiku for fast, cheap context selection
            model_id = os.getenv('BEDROCK_MODEL_ID', 'us.anthropic.claude-haiku-4-5-20251001-v1:0')
            
            # Call Bedrock API
            response = bedrock.invoke_model(
                modelId=model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 500,
                    "temperature": 0.0,
                    "system": system_prompt,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                })
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            response_text = response_body['content'][0]['text'].strip()
            
            logger.debug(f"LLM selection response: {response_text}")
            
            # Parse JSON from response
            # Try to extract JSON if it's wrapped in markdown code blocks
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            
            # Parse JSON
            selection_data = json.loads(response_text)
            
            # Validate structure
            if not isinstance(selection_data, dict) or "context_ids" not in selection_data:
                raise ValueError(f"Invalid response structure: {selection_data}")
            
            selected_ids = selection_data["context_ids"]
            
            # Validate IDs are a list
            if not isinstance(selected_ids, list):
                raise ValueError(f"context_ids must be a list, got: {type(selected_ids)}")
            
            # Validate and filter IDs
            valid_ids = []
            for idx in selected_ids:
                if isinstance(idx, int) and 0 <= idx < num_contexts:
                    valid_ids.append(idx)
                else:
                    logger.warning(f"Invalid context ID: {idx} (must be 0-{num_contexts-1})")
            
            # Handle empty selection - fallback to all contexts
            if not valid_ids:
                logger.warning("No valid context IDs selected, falling back to all contexts")
                valid_ids = list(range(num_contexts))
            
            return valid_ids
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            raise
        except KeyError as e:
            logger.error(f"Missing expected field in LLM response: {e}")
            raise
        except Exception as e:
            logger.error(f"LLM call for context selection failed: {e}")
            raise

    def _fast_context_selection(
        self,
        query: str,
        available_pointers: List[Dict[str, Any]]
    ) -> List[str]:
        """Fast keyword-based context selection for large documents.
        
        This method is used when total context size exceeds 200KB to avoid
        slow LLM-based selection. It uses simple keyword matching instead.
        
        Args:
            query: The user's query
            available_pointers: List of context pointers with metadata
            
        Returns:
            List of selected context file paths
        """
        # Extract keywords from query
        query_lower = query.lower()
        keywords = set(word.strip('.,!?()[]{}"\'') for word in query_lower.split())
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'what', 'how', 'when', 
            'where', 'why', 'who', 'which', 'from', 'about', 'get', 'show', 'tell'
        }
        keywords = keywords - stop_words
        
        # Score each pointer by keyword matches
        scored_pointers = []
        for i, ptr in enumerate(available_pointers):
            # Check keywords in tool name, args, and summary
            text_parts = [
                ptr.get('tool_name', ''),
                json.dumps(ptr.get('args', {})),
                ptr.get('summary', '')
            ]
            text = ' '.join(text_parts).lower()
            
            # Count keyword matches
            matches = sum(1 for kw in keywords if kw in text)
            
            scored_pointers.append((i, matches, ptr))
        
        # Sort by match count (descending)
        scored_pointers.sort(key=lambda x: x[1], reverse=True)
        
        # Select pointers with at least one match, or all if none match
        selected = [
            ptr['filepath'] 
            for i, matches, ptr in scored_pointers 
            if matches > 0
        ]
        
        # If no matches, select all (fallback)
        if not selected:
            logger.warning("No keyword matches found, selecting all contexts")
            selected = [ptr['filepath'] for ptr in available_pointers]
        
        logger.info(f"Fast selection: {len(selected)}/{len(available_pointers)} contexts (keywords: {keywords})")
        return selected

    def get_all_pointers(self) -> List[Dict[str, Any]]:
        """Get all tracked pointer metadata.
        
        Returns:
            Copy of pointers list
        """
        return self.pointers.copy()




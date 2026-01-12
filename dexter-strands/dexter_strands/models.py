"""
Data models for multi-agent architecture.

This module contains Pydantic models used across the multi-agent system:
- Task: Represents a discrete unit of work
- ContextPointer: Metadata for offloaded context
- ValidationResult: Result of validation checks
"""

from typing import Optional
from pydantic import BaseModel, Field


class Task(BaseModel):
    """
    Represents a single research task.
    
    Tasks are created by the Planner Agent and executed by the Executor Agent.
    Each task represents one discrete data retrieval operation.
    """
    
    id: int = Field(..., description="Unique task identifier")
    description: str = Field(..., description="Detailed task description")
    done: bool = Field(default=False, description="Whether task is complete")
    tool_name: Optional[str] = Field(default=None, description="Tool used to complete task")
    result_context_id: Optional[str] = Field(default=None, description="ID of context file with results")
    error: Optional[str] = Field(default=None, description="Error message if task execution failed")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "description": "Get income statement for AAPL for fiscal year 2023",
                "done": False,
                "tool_name": None,
                "result_context_id": None,
                "error": None
            }
        }


class ContextPointer(BaseModel):
    """
    Metadata for offloaded context.
    
    Context pointers track information about tool outputs that have been
    saved to files. They allow the system to reference and load contexts
    without keeping full data in memory.
    """
    
    id: str = Field(..., description="Unique context identifier")
    tool_name: str = Field(..., description="Name of tool that generated this context")
    args: dict = Field(..., description="Arguments passed to the tool")
    summary: str = Field(..., description="LLM-generated summary of the context")
    filepath: str = Field(..., description="Path to the context file")
    size_kb: float = Field(..., description="Size of context file in KB")
    task_id: int = Field(..., description="ID of task that generated this context")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "get_income_statement_abc123",
                "tool_name": "get_income_statement",
                "args": {"ticker": "AAPL", "period": "annual", "limit": 1},
                "summary": "Apple's FY2023 income statement showing revenue of $394B",
                "filepath": ".dexter-strands/context/get_income_statement_abc123.json",
                "size_kb": 12.5,
                "task_id": 1
            }
        }


class ValidationResult(BaseModel):
    """
    Result of validation check.
    
    Used by both task-level validation (Executor) and goal-level validation (Synthesizer).
    Indicates whether validation passed and provides reasoning.
    """
    
    is_complete: bool = Field(..., description="Whether validation passed")
    reason: str = Field(..., description="Explanation of validation result")
    missing_data: Optional[str] = Field(
        default=None, 
        description="Description of missing data (if validation failed)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "is_complete": True,
                "reason": "Successfully retrieved income statement data for AAPL",
                "missing_data": None
            }
        }

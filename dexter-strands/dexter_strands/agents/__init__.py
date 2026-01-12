"""
Multi-agent architecture components.

This module provides specialized agents for the multi-agent system:
- PlannerAgent: Task decomposition and planning
- ExecutorAgent: Tool selection and execution (to be implemented)
- SynthesizerAgent: Answer generation and validation (to be implemented)
"""

from .planner import PlannerAgent

__all__ = ['PlannerAgent']

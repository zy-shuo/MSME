"""
Expert Reasoning Module for Multi-stage Multi-expert Stance Detection Framework.
This module handles reasoning from multiple expert perspectives.
"""

from .knowledge_expert import analyze_knowledge
from .label_expert import analyze_labels
from .pragmatic_expert import analyze_pragmatics
from .meta_judge import aggregate_decisions
from .expert_coordinator import coordinate_experts

__all__ = ['analyze_knowledge', 'analyze_labels', 'analyze_pragmatics', 'aggregate_decisions', 'coordinate_experts']
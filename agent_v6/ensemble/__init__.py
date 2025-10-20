"""
Ensemble reasoning module for agent_v6

Multi-provider ensemble with 4 models + O3 synthesis:
- GPT-5 (high reasoning, high verbosity)
- Claude Opus 4.1 (extended thinking)
- Grok-4 Fast Reasoning (high effort)
- Gemini 2.5 Pro (dynamic thinking with thoughts)
- O3 (synthesis and critic)
"""

__all__ = ["EnsembleTool"]

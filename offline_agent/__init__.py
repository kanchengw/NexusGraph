"""Offline Agent: evaluation, analysis, and optimization pipeline.

Usage:
    python -m offline_agent eval              # Run RAGBench evaluation
    python -m offline_agent analyze           # Generate retrieval analysis report
    python -m offline_agent optimize          # Run LLM optimization suggestions
    python -m offline_agent pipeline          # Full pipeline: eval → analyze → suggest
    python -m offline_agent pipeline --apply  # Full pipeline + apply approved changes
"""

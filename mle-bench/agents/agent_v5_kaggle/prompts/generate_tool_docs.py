"""
Generate tool documentation from registered tools
"""
from typing import Dict, List


def generate_tool_documentation(tool_registry) -> str:
    """
    Generate formatted tool documentation from ToolRegistry

    Args:
        tool_registry: ToolRegistry instance with registered tools

    Returns:
        Formatted markdown string documenting all available tools
    """
    docs = []

    for tool_name, tool in sorted(tool_registry.tools.items()):
        schema = tool.schema
        description = schema.get('description', 'No description available')

        # Format tool entry
        docs.append(f"**{tool_name}**: {description}")

    return "\n\n".join(docs)

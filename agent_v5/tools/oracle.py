"""
Oracle tool - Consult OpenAI o3 for expert guidance
"""
import os
import json
from typing import Dict, Callable, List
from .base import BaseTool


class OracleTool(BaseTool):
    """Consult the Oracle (OpenAI o3) for expert guidance when stuck or confused"""

    def __init__(self, workspace_dir: str, get_conversation_history: Callable[[], List[Dict]]):
        """
        Initialize Oracle tool

        Args:
            workspace_dir: Workspace directory (unused but required by BaseTool)
            get_conversation_history: Callable that returns current conversation history
        """
        super().__init__(workspace_dir)
        self.get_conversation_history = get_conversation_history

    @property
    def name(self) -> str:
        return "Oracle"

    @property
    def schema(self) -> Dict:
        return {
            "name": "Oracle",
            "description": (
                "Consult the wise Oracle when stuck, confused about results, "
                "or need expert strategic guidance. Full conversation history is automatically included. "
                "Use when: CV/leaderboard mismatch detected, stuck after multiple failed iterations, "
                "major strategic decision points, debugging complex issues, or need validation of approach. "
                "The Oracle analyzes patterns across your entire conversation and provides actionable recommendations."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Your specific question for the oracle. Be direct and specific. "
                            "Examples: 'Why is my CV 0.44 but leaderboard score much worse?', "
                            "'Should I pivot to transformers or continue tuning current approach?', "
                            "'What critical bug might I have in my label encoding?', "
                            "'Am I on track for gold medal or do I need a different strategy?'"
                        )
                    }
                },
                "required": ["query"]
            }
        }

    async def execute(self, input: Dict) -> Dict:
        """
        Execute oracle consultation

        Args:
            input: Dict with 'query' key containing the question

        Returns:
            Dict with oracle's analysis and recommendations
        """
        try:
            # Import OpenAI client
            from openai import OpenAI

            query = input["query"]
            
            # Get current conversation history
            conversation_history = self.get_conversation_history()

            # Initialize OpenAI client
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                return {
                    "content": "Error: OPENAI_API_KEY environment variable not set. Cannot consult Oracle.",
                    "is_error": True
                }

            client = OpenAI(api_key=api_key)

            # Build messages array with system instruction, full conversation, and oracle query
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert ML engineer Oracle with deep knowledge of Kaggle competitions, "
                        "model training, debugging, and strategy. You will receive the full conversation history "
                        "of an autonomous agent working on a competition, followed by a specific question.\n\n"
                        "Your task:\n"
                        "1. Analyze the full conversation context thoroughly\n"
                        "2. Identify critical issues, bugs, misalignments, or strategic errors\n"
                        "3. Provide direct, actionable recommendations\n"
                        "4. Be specific about what to fix and how\n"
                        "5. Prioritize high-impact changes\n\n"
                        "Common issues to watch for:\n"
                        "- Label encoding/column order mismatches in predictions\n"
                        "- Train/test contamination or data leakage\n"
                        "- CV/leaderboard score misalignment (often indicates bug)\n"
                        "- Suboptimal model choices for the task\n"
                        "- Missing critical features or preprocessing steps\n"
                        "- Incorrect metric optimization\n"
                        "- Poor hyperparameter choices\n\n"
                        "Be direct. If something is wrong, say exactly what and how to fix it. "
                        "If the approach is fundamentally flawed, recommend a better direction."
                    )
                }
            ]

            # Add full conversation history
            # Convert tool results to readable format for the oracle
            for msg in conversation_history:
                role = msg["role"]
                content = msg["content"]

                if role == "user" and isinstance(content, list):
                    # Tool results - format nicely
                    formatted = []
                    for item in content:
                        if item.get("type") == "tool_result":
                            tool_content = item.get("content", "")
                            formatted.append(f"[Tool Result]: {tool_content}...")  # no truncation for now
                    messages.append({"role": "user", "content": "\n".join(formatted)})
                
                elif role == "assistant" and isinstance(content, list):
                    # Assistant response with text and tool uses
                    text_parts = []
                    tool_uses = []
                    for item in content:
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif item.get("type") == "tool_use":
                            tool_uses.append(f"[Tool: {item.get('name')}({json.dumps(item.get('input', {}))})]")
                    
                    combined = "\n".join(text_parts)
                    if tool_uses:
                        combined += "\n" + "\n".join(tool_uses)
                    
                    if combined.strip():
                        messages.append({"role": "assistant", "content": combined})
                
                elif isinstance(content, str):
                    messages.append({"role": role, "content": content})

            # Add oracle query at the end
            messages.append({
                "role": "user",
                "content": f"[ORACLE QUERY FROM AGENT]: {query}"
            })

            # Call OpenAI o3 model
            response = client.chat.completions.create(
                model="o3",  # MUST be exactly "o3"
                messages=messages,
                max_tokens=8192,
                temperature=1.0  # Default for reasoning models
            )

            oracle_response = response.choices[0].message.content

            return {
                "content": f"🔮 Oracle Analysis:\n\n{oracle_response}",
                "is_error": False,
                "debug_summary": f"Oracle consulted: {query[:100]}..."
            }

        except Exception as e:
            error_msg = str(e)
            
            # Provide helpful error messages
            if "model" in error_msg.lower() and "o3" in error_msg.lower():
                return {
                    "content": (
                        f"Error: OpenAI o3 model not available yet. "
                        f"Original error: {error_msg}"
                    ),
                    "is_error": True
                }
            elif "authentication" in error_msg.lower() or "api" in error_msg.lower():
                return {
                    "content": f"Error: OpenAI API authentication failed. Check OPENAI_API_KEY. Error: {error_msg}",
                    "is_error": True
                }
            else:
                return {
                    "content": f"Error consulting Oracle: {error_msg}",
                    "is_error": True
                }


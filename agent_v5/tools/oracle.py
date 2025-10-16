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
                        "You are an expert ML engineer Oracle providing strategic guidance for Kaggle competitions. "
                        "You will receive the full conversation history of an autonomous agent and a specific question.\n\n"

                        "## Your Approach\n\n"

                        "Analyze the conversation carefully, focusing on:\n"
                        "- What the agent has tried and what results were achieved\n"
                        "- Whether previous Oracle suggestions (if any) helped or hurt performance\n"
                        "- Patterns indicating bugs, data leakage, or strategic misalignment\n"
                        "- Root causes vs symptoms\n\n"

                        "If you previously gave advice that led to worse results:\n"
                        "- Acknowledge the failure explicitly\n"
                        "- Explain why it didn't work\n"
                        "- Suggest a fundamentally different approach\n\n"

                        "## Common Issues to Check\n\n"

                        "**Data integrity:**\n"
                        "- Label encoding/column order mismatches between train and test\n"
                        "- Data leakage (test data used in feature engineering or scaling)\n"
                        "- CV/leaderboard score divergence (often indicates bugs)\n\n"

                        "**Model selection:**\n"
                        "- Task-appropriate model choice (boosting vs neural nets vs linear)\n"
                        "- Correct metric optimization\n"
                        "- Missing critical preprocessing or features\n\n"

                        "**Resource utilization:**\n"
                        "- GPU not used (CPU training is 10-100x slower)\n"
                        "- n_jobs not set to -1 (wasting CPU cores)\n"
                        "- Batch sizes too small (GPU underutilized)\n"
                        "- Single-threaded data loading instead of parallel\n\n"

                        "## Response Guidelines\n\n"

                        "- Be direct and humble - admit uncertainty when appropriate\n"
                        "- Ground recommendations in evidence from logs and results\n"
                        "- If an approach fails after 2-3 iterations, recommend a pivot rather than incremental tuning\n"
                        "- Provide specific, actionable fixes with code examples when relevant\n"
                        "- Think step-by-step and question your own assumptions"
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
                max_completion_tokens=8192,
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


"""
Oracle tool - Consult OpenAI o3 + DeepSeek R1 for expert guidance with multi-model ensemble + critic
"""
import os
import json
import asyncio
from typing import Dict, Callable, List, Tuple
from .base import BaseTool


class OracleTool(BaseTool):
    """
    Consult the Oracle (multi-model ensemble) for expert guidance when stuck or confused

    Architecture:
    1. Query both O3 and DeepSeek-R1 in parallel
    2. O3 Critic compares, synthesizes, and returns unified optimal plan
    """

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
                "Consult the wise Oracle (multi-model ensemble: O3 + DeepSeek-R1 + O3 Critic) when stuck, "
                "confused about results, or need expert strategic guidance. Full conversation history is automatically included. "
                "Use when: CV/leaderboard mismatch detected, stuck after multiple failed iterations, "
                "major strategic decision points, debugging complex issues, or need validation of approach. "
                "The Oracle queries multiple reasoning models in parallel, then synthesizes their insights into "
                "a unified optimal plan. This multi-perspective approach catches blind spots and validates strategies."
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
        Execute oracle consultation with multi-model ensemble + critic

        Architecture:
        1. Query O3 and DeepSeek-R1 in parallel
        2. O3 Critic synthesizes both plans into optimal unified plan

        Args:
            input: Dict with 'query' key containing the question

        Returns:
            Dict with oracle's synthesized analysis and recommendations
        """
        try:
            # Import OpenAI client
            from openai import OpenAI

            query = input["query"]

            # Get current conversation history
            conversation_history = self.get_conversation_history()

            # Initialize OpenAI client for O3
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                return {
                    "content": "Error: OPENAI_API_KEY environment variable not set. Cannot consult Oracle.",
                    "is_error": True
                }

            # Initialize DeepSeek client (OpenAI-compatible)
            deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not deepseek_api_key:
                return {
                    "content": "Error: DEEPSEEK_API_KEY environment variable not set. Cannot consult Oracle.\n"
                              "Get your API key at: https://platform.deepseek.com/api_keys",
                    "is_error": True
                }

            openai_client = OpenAI(api_key=openai_api_key)
            deepseek_client = OpenAI(
                api_key=deepseek_api_key,
                base_url="https://api.deepseek.com"  # DeepSeek's base URL
            )

            # Convert conversation history to messages format
            messages = self._build_messages(conversation_history, query)

            # Step 1: Query both models in parallel
            print("🔮 Oracle: Consulting O3 and DeepSeek-R1 in parallel...")
            o3_plan, deepseek_plan = await asyncio.gather(
                self._query_o3(openai_client, messages),
                self._query_deepseek_r1(deepseek_client, messages)
            )

            # Check for errors in parallel queries
            if o3_plan.startswith("ERROR:") and deepseek_plan.startswith("ERROR:"):
                return {
                    "content": f"Both models failed:\n\nO3: {o3_plan}\n\nDeepSeek-R1: {deepseek_plan}",
                    "is_error": True
                }

            # Step 2: O3 Critic synthesizes both plans
            print("🔮 Oracle: O3 Critic synthesizing optimal plan...")
            final_plan = await self._critic_synthesis(openai_client, messages, o3_plan, deepseek_plan, query)

            # Format final response
            response_content = self._format_response(o3_plan, deepseek_plan, final_plan)

            return {
                "content": response_content,
                "is_error": False,
                "debug_summary": f"Oracle (O3+DeepSeek+Critic): {query[:100]}..."
            }

        except Exception as e:
            error_msg = str(e)

            # Provide helpful error messages
            if "model" in error_msg.lower() and ("o3" in error_msg.lower() or "deepseek" in error_msg.lower()):
                return {
                    "content": (
                        f"Error: Model not available. "
                        f"Original error: {error_msg}"
                    ),
                    "is_error": True
                }
            elif "authentication" in error_msg.lower() or "api" in error_msg.lower():
                return {
                    "content": f"Error: API authentication failed. Check OPENAI_API_KEY. Error: {error_msg}",
                    "is_error": True
                }
            else:
                return {
                    "content": f"Error consulting Oracle: {error_msg}",
                    "is_error": True
                }

    def _build_messages(self, conversation_history: List[Dict], query: str) -> List[Dict]:
        """
        Convert agent conversation history to messages format

        Args:
            conversation_history: Agent's conversation history
            query: User's oracle query

        Returns:
            List of message dicts for LLM API
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert ML engineer Oracle with deep knowledge of Kaggle competitions, "
                    "model training, debugging, and strategy. You will receive the full conversation history "
                    "of an autonomous agent working on a competition, followed by a specific question.\n\n"
                    "CRITICAL SELF-AWARENESS RULES:\n"
                    "• If you previously gave advice in this conversation and results got WORSE, you MUST:\n"
                    "  1. Explicitly acknowledge your prior suggestion failed or caused regression\n"
                    "  2. Explain WHY it didn't work (bug, wrong assumption, bad fit for this task)\n"
                    "  3. Suggest a FUNDAMENTALLY DIFFERENT approach, not a variation of the failed one\n"
                    "• NEVER repeat the same suggestion if metrics regressed after following it\n"
                    "• Be realistic: if an approach isn't working after 2-3 attempts, pivot completely\n"
                    "• If you're uncertain, say so explicitly rather than confidently guessing wrong\n\n"
                    "Your task:\n"
                    "1. ULTRATHINK: Analyze the full conversation context with extreme care\n"
                    "2. Review ALL previous Oracle consultations in this conversation (if any) and their outcomes\n"
                    "3. Identify critical issues, bugs, misalignments, or strategic errors\n"
                    "4. Check if prior Oracle advice was followed and whether it helped or hurt\n"
                    "5. Provide direct, actionable, GROUNDED recommendations based on actual results\n"
                    "6. Prioritize high-impact changes that haven't already failed\n\n"
                    "Common issues to watch for:\n"
                    "- Label encoding/column order mismatches in predictions\n"
                    "- Train/test contamination or data leakage\n"
                    "- CV/leaderboard score misalignment (often indicates bug)\n"
                    "- Suboptimal model choices for the task\n"
                    "- Missing critical features or preprocessing steps\n"
                    "- Incorrect metric optimization\n"
                    "- Poor hyperparameter choices\n"
                    "- CPU-bound code that should use GPU (cuML/RAPIDS/PyTorch)\n"
                    "- Memory issues causing OOM or slow performance\n"
                    "- RESOURCE UNDERUTILIZATION:\n"
                    "  • n_jobs not set to -1 (wasting CPU cores)\n"
                    "  • Batch sizes too small (GPU underutilized, check nvidia-smi)\n"
                    "  • Single-threaded data loading (should use num_workers=-1)\n"
                    "  • Row-by-row processing instead of vectorized operations\n"
                    "  • No mixed precision (wasting GPU compute with float32)\n"
                    "  • Sequential file I/O instead of parallel reading\n\n"
                    "RESPONSE STYLE:\n"
                    "• Be direct but HUMBLE: admit when you don't know or when prior advice failed\n"
                    "• If something is wrong, say exactly what and how to fix it with EVIDENCE from the logs\n"
                    "• If the approach is fundamentally flawed, recommend a DIFFERENT direction, not more tuning\n"
                    "• When results regress: 'My previous suggestion to [X] appears to have caused regression. "
                    "The real issue is [Y]. Instead, try [Z].' \n"
                    "• Think step-by-step, question your own assumptions, consider alternative explanations"
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
                        formatted.append(f"[Tool Result]: {tool_content}")
                if formatted:
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

        return messages

    async def _query_o3(self, client: 'OpenAI', messages: List[Dict]) -> str:
        """
        Query OpenAI O3 model

        Args:
            client: OpenAI client instance
            messages: Conversation messages

        Returns:
            O3's response text or error message
        """
        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="o3",
                messages=messages,
                max_completion_tokens=8192,
                temperature=1.0
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"ERROR: O3 failed - {str(e)}"

    async def _query_deepseek_r1(self, client: 'OpenAI', messages: List[Dict]) -> str:
        """
        Query DeepSeek-R1 model (reasoning mode with Chain of Thought)

        Args:
            client: OpenAI client instance configured for DeepSeek API
                   (base_url="https://api.deepseek.com")
            messages: Conversation messages

        Returns:
            DeepSeek-R1's response text or error message

        Note:
            - Model name is "deepseek-reasoner" (reasoning mode of DeepSeek-V3.2-Exp)
            - Generates Chain of Thought (CoT) before final answer
            - Requires DEEPSEEK_API_KEY environment variable
            - Get API key at: https://platform.deepseek.com/api_keys
        """
        try:
            # DeepSeek-R1 reasoning model
            # API: https://api.deepseek.com
            # Model: deepseek-reasoner (generates CoT reasoning)
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="deepseek-reasoner",  # Reasoning mode (R1)
                messages=messages,
                max_completion_tokens=8192,
                temperature=1.0
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"ERROR: DeepSeek-R1 failed - {str(e)}"

    async def _critic_synthesis(
        self,
        client: 'OpenAI',
        base_messages: List[Dict],
        o3_plan: str,
        deepseek_plan: str,
        original_query: str
    ) -> str:
        """
        Use O3 as critic to synthesize both plans into optimal unified plan

        Args:
            client: OpenAI client instance
            base_messages: Original conversation context
            o3_plan: O3's plan
            deepseek_plan: DeepSeek-R1's plan
            original_query: Original oracle query

        Returns:
            Synthesized optimal plan
        """
        try:
            # Build critic prompt
            critic_messages = base_messages.copy()
            critic_messages.append({
                "role": "user",
                "content": (
                    f"[ORACLE SYNTHESIS TASK]\n\n"
                    f"Original Query: {original_query}\n\n"
                    f"Two expert models provided plans. Your task is to:\n"
                    f"1. Compare both plans critically\n"
                    f"2. Identify strengths and weaknesses of each\n"
                    f"3. Synthesize the best elements into a SINGLE OPTIMAL PLAN\n"
                    f"4. Resolve any contradictions with evidence-based reasoning\n"
                    f"5. Return the unified plan that maximizes chances of success\n\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"PLAN A (OpenAI O3):\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"{o3_plan}\n\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"PLAN B (DeepSeek-R1):\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"{deepseek_plan}\n\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"Provide your SYNTHESIZED OPTIMAL PLAN:\n"
                    f"- Start with a brief comparison (which plan is stronger and why)\n"
                    f"- Then provide the unified optimal plan\n"
                    f"- Be specific, actionable, and grounded in evidence\n"
                    f"- If plans conflict, explain your reasoning for choosing one approach"
                )
            })

            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="o3",
                messages=critic_messages,
                max_completion_tokens=16384,  # More tokens for synthesis
                temperature=1.0
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"ERROR: Critic synthesis failed - {str(e)}"

    def _format_response(self, o3_plan: str, deepseek_plan: str, final_plan: str) -> str:
        """
        Format the final Oracle response with all three plans

        Args:
            o3_plan: O3's plan
            deepseek_plan: DeepSeek-R1's plan
            final_plan: Synthesized plan from O3 critic

        Returns:
            Formatted response string
        """
        return (
            "🔮 **ORACLE CONSULTATION (Multi-Model Ensemble)**\n\n"
            "The Oracle consulted two reasoning models in parallel, then synthesized their insights:\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "📊 **PLAN A: OpenAI O3 Analysis**\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{o3_plan}\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "🧠 **PLAN B: DeepSeek-R1 Analysis**\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{deepseek_plan}\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "✨ **SYNTHESIZED OPTIMAL PLAN (O3 Critic)**\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{final_plan}\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "**Oracle Consultation Complete.** Follow the synthesized optimal plan above."
        )


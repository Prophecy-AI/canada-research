"""
DeepSeek R1 Planner tool - Strategic planning with reasoning model
"""
import os
import json
from typing import Dict, Callable, List
from .base import BaseTool


class DeepSeekPlannerTool(BaseTool):
    """
    Use DeepSeek R1 reasoning model for strategic competition planning

    R1 provides extended reasoning for:
    - Competition strategy formulation
    - Approach selection (features, models, CV)
    - Resource allocation planning
    - Risk identification

    Use BEFORE Oracle (o3) for initial strategy. Oracle handles code review/debugging.
    """

    def __init__(self, workspace_dir: str, get_conversation_history: Callable[[], List[Dict]]):
        """
        Initialize DeepSeek Planner tool

        Args:
            workspace_dir: Workspace directory
            get_conversation_history: Callable that returns current conversation history
        """
        super().__init__(workspace_dir)
        self.get_conversation_history = get_conversation_history

    @property
    def name(self) -> str:
        return "DeepSeekPlanner"

    @property
    def schema(self) -> Dict:
        return {
            "name": "DeepSeekPlanner",
            "description": (
                "Consult DeepSeek R1 reasoning model for deep strategic planning. "
                "Use for: initial competition strategy, approach selection, feature/model brainstorming, "
                "CV strategy design, resource planning. R1 provides extended reasoning (thinking) "
                "to analyze competition patterns and recommend gold-medal approaches. "
                "Use BEFORE Oracle (o3). Oracle is for code review and debugging, not strategic planning."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "planning_query": {
                        "type": "string",
                        "description": (
                            "Your strategic planning question. Be comprehensive and include all context. "
                            "Examples:\n"
                            "- 'Competition: predict house prices. Task: regression, Metric: RMSE. Data: 1500 rows, 80 features (mix of numerical/categorical). What's the optimal gold-medal strategy?'\n"
                            "- 'I have 80GB RAM, A10 GPU 24GB, 8 CPU cores. Data: 500K rows, 200 features. Target: top-1% on time-series forecasting competition. Recommend approach.'\n"
                            "- 'Competition type: NLP sentiment classification, 100K tweets, metric: F1 score. Should I use transformers or traditional ML? What features matter most?'"
                        )
                    }
                },
                "required": ["planning_query"]
            }
        }

    async def execute(self, input: Dict) -> Dict:
        """
        Execute strategic planning with DeepSeek R1

        Args:
            input: Dict with 'planning_query' key

        Returns:
            Dict with R1's strategic analysis and recommendations
        """
        try:
            # Import OpenAI-compatible client (DeepSeek uses OpenAI API format)
            from openai import OpenAI

            planning_query = input["planning_query"]

            # Get current conversation history (for context)
            conversation_history = self.get_conversation_history()

            # Initialize DeepSeek client
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                return {
                    "content": (
                        "Error: DEEPSEEK_API_KEY environment variable not set. "
                        "Cannot consult DeepSeek R1 Planner.\n\n"
                        "Get API key at: https://platform.deepseek.com/\n"
                        "Set with: export DEEPSEEK_API_KEY=your-key-here"
                    ),
                    "is_error": True
                }

            client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"  # DeepSeek API endpoint
            )

            # Build messages array with system instruction and planning query
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert Kaggle grandmaster and ML strategist using DeepSeek R1 reasoning model. "
                        "Your task is to provide gold-medal competitive strategy for Kaggle competitions.\n\n"

                        "You will receive a competition description and must reason deeply about:\n"
                        "1. Competition type analysis (what makes this task unique?)\n"
                        "2. Winning patterns from similar competitions\n"
                        "3. Feature engineering opportunities (high-leverage features)\n"
                        "4. Model selection (what works best for this task/data size?)\n"
                        "5. CV strategy (how to prevent overfitting and ensure leaderboard alignment?)\n"
                        "6. Resource allocation (how to maximize GPU/CPU utilization?)\n"
                        "7. Common pitfalls and how to avoid them\n"
                        "8. Concrete action plan (step-by-step roadmap)\n\n"

                        "Provide a comprehensive strategic plan with:\n"
                        "- **Competition Analysis:** What type of problem is this? What patterns should we look for?\n"
                        "- **Recommended Approach:** Specific models, feature engineering, CV strategy\n"
                        "- **High-Leverage Actions:** What 20% of effort will yield 80% of results?\n"
                        "- **Resource Optimization:** How to use A10 GPU (24GB), CPU cores, RAM efficiently\n"
                        "- **Risk Mitigation:** Common bugs, data leakage risks, overfitting traps\n"
                        "- **Implementation Roadmap:** Ordered list of tasks (baseline → iterations → submission)\n\n"

                        "Be specific and actionable. Think deeply about what separates top-1% solutions from median baselines."
                    )
                }
            ]

            # Add recent conversation context (last 3 turns) for awareness
            recent_context = []
            for msg in conversation_history[-6:]:  # Last 3 turns (user + assistant)
                role = msg["role"]
                content = msg["content"]

                if isinstance(content, str):
                    recent_context.append(f"[{role.upper()}]: {content[:500]}...")  # Truncate long messages
                elif isinstance(content, list):
                    # Extract text from structured content
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text", "")[:300])
                    if text_parts:
                        recent_context.append(f"[{role.upper()}]: {' '.join(text_parts)[:500]}...")

            if recent_context:
                messages.append({
                    "role": "user",
                    "content": f"Recent conversation context:\n\n" + "\n\n".join(recent_context)
                })

            # Add strategic planning query
            messages.append({
                "role": "user",
                "content": f"[STRATEGIC PLANNING QUERY]:\n\n{planning_query}\n\n"
                          f"Please reason deeply and provide a comprehensive gold-medal strategy."
            })

            # Call DeepSeek R1 model with reasoning
            response = client.chat.completions.create(
                model="deepseek-reasoner",  # R1 reasoning model
                messages=messages,
                max_tokens=8192,  # Allow extended reasoning
                temperature=1.0  # Default for reasoning models
            )

            # Extract reasoning and answer
            reasoning_content = response.choices[0].message.reasoning_content or "(no reasoning output)"
            answer_content = response.choices[0].message.content

            # Format output
            output = (
                "🧠 **DeepSeek R1 Strategic Analysis**\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"{answer_content}\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"**Reasoning Trace (R1 Thinking):**\n{reasoning_content[:1000]}...\n"
                f"(Full reasoning: {len(reasoning_content)} chars)"
            )

            return {
                "content": output,
                "is_error": False,
                "debug_summary": f"R1 strategic plan generated ({len(answer_content)} chars)"
            }

        except Exception as e:
            error_msg = str(e)

            # Provide helpful error messages
            if "authentication" in error_msg.lower() or "api" in error_msg.lower() or "401" in error_msg:
                return {
                    "content": (
                        f"Error: DeepSeek API authentication failed.\n\n"
                        f"1. Get API key at: https://platform.deepseek.com/\n"
                        f"2. Set environment variable: export DEEPSEEK_API_KEY=your-key-here\n"
                        f"3. Verify key is valid\n\n"
                        f"Original error: {error_msg}"
                    ),
                    "is_error": True
                }
            elif "model" in error_msg.lower():
                return {
                    "content": (
                        f"Error: DeepSeek model 'deepseek-reasoner' not accessible.\n"
                        f"Verify your API key has access to R1 reasoning model.\n\n"
                        f"Original error: {error_msg}"
                    ),
                    "is_error": True
                }
            else:
                return {
                    "content": f"Error consulting DeepSeek R1 Planner: {error_msg}",
                    "is_error": True
                }

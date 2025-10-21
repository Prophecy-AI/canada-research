"""
Ensemble reasoning tool - Multi-provider consultation with O3 synthesis

Architecture (based on Operand Quant paper):
1. Query 4 models in parallel:
   - GPT-5 (high reasoning, high verbosity)
   - Claude Opus 4.1 (extended thinking)
   - Grok-4 Fast Reasoning (high effort)
   - Gemini 2.5 Pro (dynamic thinking with thought summaries)
2. O3 synthesizes all 4 responses into optimal unified plan

This is the deep-thinking ensemble from the Operand Quant paper.
"""
import os
import json
import asyncio
from typing import Dict, Callable, List
from agent_v5.tools.base import BaseTool


class EnsembleTool(BaseTool):
    """
    Consult the Ensemble (4-model reasoning + O3 synthesis) for expert guidance

    When to use:
    - Stuck on a problem after multiple attempts
    - Need validation of complex strategy
    - Facing critical decision points
    - Results don't match expectations
    - Want diverse perspectives on approach
    """

    def __init__(self, workspace_dir: str, get_conversation_history: Callable[[], List[Dict]]):
        """
        Initialize ensemble tool

        Args:
            workspace_dir: Workspace directory (required by BaseTool)
            get_conversation_history: Callable that returns current conversation history
        """
        super().__init__(workspace_dir)
        self.get_conversation_history = get_conversation_history

    @property
    def name(self) -> str:
        return "ConsultEnsemble"

    @property
    def schema(self) -> Dict:
        return {
            "name": "ConsultEnsemble",
            "description": (
                "Consult the multi-model ensemble (GPT-5 + Claude Opus + Grok-4 + Gemini 2.5 Pro) "
                "synthesized by O3 when you need expert guidance. Use when stuck, confused about results, "
                "or facing major decisions. Full conversation history is automatically included. "
                "The ensemble queries 4 frontier models in parallel, then O3 synthesizes their insights "
                "into a unified optimal plan. This multi-perspective approach catches blind spots."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "problem": {
                        "type": "string",
                        "description": (
                            "Specific problem or question for the ensemble. Be direct and specific. "
                            "Examples: 'Stuck at 0.75 AUC, tried 5 GBM variants', "
                            "'Should I use CNN or transformer for this image task?', "
                            "'Model not converging after 10 epochs, what's wrong?'"
                        )
                    },
                    "context": {
                        "type": "string",
                        "description": (
                            "Additional context the ensemble should know (optional). "
                            "Examples: 'Dataset has 100K samples, 50 features', "
                            "'GPU memory limited to 16GB', "
                            "'Previous attempts: tried ResNet-50 (OOM), ResNet-18 (underfit)'"
                        )
                    }
                },
                "required": ["problem"]
            }
        }

    async def execute(self, input: Dict) -> Dict:
        """
        Execute ensemble consultation

        Flow:
        1. Build messages from conversation history
        2. Query 4 models in parallel (GPT-5, Claude, Grok, Gemini)
        3. O3 synthesizes all 4 responses
        4. Return formatted result

        Args:
            input: Dict with 'problem' and optional 'context'

        Returns:
            Dict with synthesized ensemble advice
        """
        try:
            # Import providers (heavy imports, do only when needed)
            from openai import AsyncOpenAI
            from anthropic import AsyncAnthropic
            from xai_sdk import AsyncClient as XAIAsyncClient
            from google import genai
            from google.genai import types as genai_types

            problem = input["problem"]
            additional_context = input.get("context", "")
            conversation_history = self.get_conversation_history()

            # Validate API keys
            missing_keys = []
            if not os.environ.get("OPENAI_API_KEY"):
                missing_keys.append("OPENAI_API_KEY")
            if not os.environ.get("ANTHROPIC_API_KEY"):
                missing_keys.append("ANTHROPIC_API_KEY")
            if not os.environ.get("XAI_API_KEY"):
                missing_keys.append("XAI_API_KEY")
            if not os.environ.get("GEMINI_API_KEY"):
                missing_keys.append("GEMINI_API_KEY")

            if missing_keys:
                return {
                    "content": f"Error: Missing API keys: {', '.join(missing_keys)}. All 4 providers required for ensemble.",
                    "is_error": True
                }

            # Initialize clients
            openai_client = AsyncOpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                timeout=3600.0  # Extended for reasoning models
            )
            anthropic_client = AsyncAnthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"],
                timeout=3600.0
            )
            xai_client = XAIAsyncClient(
                api_key=os.environ["XAI_API_KEY"],
                timeout=3600  # seconds (not float)
            )
            gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

            # Build messages
            messages = self._build_messages(conversation_history, problem, additional_context)

            print("🔮 Ensemble: Consulting 4 frontier models in parallel...")
            print("   - GPT-5 (high reasoning, high verbosity)")
            print("   - Claude Opus 4.1 (extended thinking)")
            print("   - Grok-4 Fast Reasoning (high effort)")
            print("   - Gemini 2.5 Pro (dynamic thinking)")

            # Query all 4 models in parallel
            gpt5_response, claude_response, grok_response, gemini_response = await asyncio.gather(
                self._query_gpt5(openai_client, messages),
                self._query_claude(anthropic_client, messages),
                self._query_grok(xai_client, messages, problem),
                self._query_gemini(gemini_client, messages, problem, genai_types),
                return_exceptions=True
            )

            # Check for errors
            responses = {
                "GPT-5": gpt5_response,
                "Claude Opus 4.1": claude_response,
                "Grok-4": grok_response,
                "Gemini 2.5 Pro": gemini_response
            }

            valid_responses = {}
            errors = {}

            for name, resp in responses.items():
                if isinstance(resp, Exception):
                    errors[name] = f"ERROR: {str(resp)}"
                elif isinstance(resp, str) and resp.startswith("ERROR:"):
                    errors[name] = resp
                else:
                    valid_responses[name] = resp

            # Check if we have at least one valid response
            if len(valid_responses) == 0:
                error_summary = "\n\n".join([f"{name}: {err}" for name, err in errors.items()])
                return {
                    "content": f"All ensemble models failed:\n\n{error_summary}",
                    "is_error": True
                }

            # Synthesize with O3
            print(f"🔮 Ensemble: O3 synthesizing {len(valid_responses)} expert responses...")
            synthesis = await self._synthesize_with_o3(
                openai_client,
                messages,
                valid_responses,
                errors,
                problem
            )

            # Format response
            response_content = self._format_response(valid_responses, errors, synthesis)

            return {
                "content": response_content,
                "is_error": False,
                "debug_summary": f"Ensemble ({len(valid_responses)}/4 models): {problem}"
            }

        except Exception as e:
            return {
                "content": f"Ensemble consultation failed: {str(e)}",
                "is_error": True
            }

    def _build_messages(
        self,
        conversation_history: List[Dict],
        problem: str,
        additional_context: str
    ) -> List[Dict]:
        """
        Build message history for ensemble models

        Args:
            conversation_history: Agent's full conversation history
            problem: User's specific problem
            additional_context: Optional additional context

        Returns:
            List of messages for LLM APIs
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a world-class AI engineering consultant providing expert guidance. "
                    "You will receive the full conversation history of an autonomous agent and a specific problem. "
                    "Analyze the context deeply and provide actionable, specific recommendations."
                )
            }
        ]

        # Add conversation history (simplified version for ensemble)
        for msg in conversation_history:
            role = msg["role"]
            content = msg["content"]

            if role == "user" and isinstance(content, str):
                messages.append({"role": "user", "content": content})
            elif role == "assistant" and isinstance(content, list):
                # Extract text parts
                text_parts = []
                for item in content:
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                if text_parts:
                    messages.append({"role": "assistant", "content": "\n".join(text_parts)})
            elif role == "tool" and isinstance(content, list):
                # Summarize tool results
                formatted = []
                for item in content:
                    if item.get("type") == "tool_result":
                        tool_name = item.get("tool_name", "tool")
                        tool_output = item.get("content", "")
                        formatted.append(f"[{tool_name}]: {tool_output}")
                if formatted:
                    messages.append({"role": "user", "content": "\n".join(formatted)})

        # Add ensemble query
        query_text = f"[ENSEMBLE CONSULTATION]\n\nProblem: {problem}"
        if additional_context:
            query_text += f"\n\nAdditional Context: {additional_context}"

        messages.append({"role": "user", "content": query_text})

        return messages

    async def _query_gpt5(self, client: 'AsyncOpenAI', messages: List[Dict]) -> str:
        """
        Query GPT-5 with high reasoning and high verbosity

        Per OpenAI docs:
        - reasoning: {effort: "high"} for maximum thinking
        - text: {verbosity: "high"} for detailed output
        - max_output_tokens: 2048 (reduced to control cost/time)
        """
        try:
            response = await client.responses.create(
                model="gpt-5",
                input=[{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"],
                instructions=messages[0]["content"],  # System prompt as instructions
                reasoning={"effort": "high"},
                text={"verbosity": "high"},
                max_output_tokens=2048  # Reduced from 8192
            )
            return response.output_text
        except Exception as e:
            return f"ERROR: GPT-5 failed - {str(e)}"

    async def _query_claude(self, client: 'AsyncAnthropic', messages: List[Dict]) -> str:
        """
        Query Claude Opus 4.1 with extended thinking

        Uses standard Anthropic API with extended_thinking=True parameter
        """
        try:
            # Convert messages format (Anthropic uses different format)
            system_msg = messages[0]["content"]
            conversation_msgs = []

            for msg in messages[1:]:
                role = msg["role"]
                # Anthropic only accepts "user" and "assistant"
                if role == "system":
                    continue
                conversation_msgs.append({
                    "role": "user" if role == "user" else "assistant",
                    "content": msg["content"]
                })

            response = await client.messages.create(
                model="claude-opus-4-1-20250805",
                system=system_msg,
                messages=conversation_msgs,
                max_tokens=8096,  # Reduced from 16384, must be > budget_tokens
                thinking={
                    "type": "enabled",
                    "budget_tokens": 2048  # Reduced from 10000
                }
            )

            # Extract text from response
            text_content = []
            for block in response.content:
                if block.type == "text":
                    text_content.append(block.text)

            return "\n".join(text_content) if text_content else "ERROR: Claude returned no text"

        except Exception as e:
            return f"ERROR: Claude Opus 4.1 failed - {str(e)}"

    async def _query_grok(self, client: 'XAIAsyncClient', messages: List[Dict], problem: str) -> str:
        """
        Query Grok-4 Fast Reasoning with high effort

        Per XAI docs:
        - Model: grok-4-fast-reasoning
        - reasoning_effort not supported (use default)
        - Use chat.create + chat.sample pattern
        """
        try:
            # Build chat with system message
            from xai_sdk.chat import system, user

            chat = client.chat.create(
                model="grok-4-fast-reasoning",
                messages=[system(messages[0]["content"])]
            )

            # Add conversation history
            for msg in messages[1:]:
                if msg["role"] == "user":
                    chat.append(user(msg["content"]))
                # Skip assistant messages for simplicity (Grok SDK doesn't support them in this pattern)

            # Sample response
            response = await chat.sample()
            return response.content

        except Exception as e:
            return f"ERROR: Grok-4 failed - {str(e)}"

    async def _query_gemini(
        self,
        client: 'genai.Client',
        messages: List[Dict],
        problem: str,
        genai_types
    ) -> str:
        """
        Query Gemini 2.5 Pro with dynamic thinking and thought summaries

        Per Google docs:
        - thinking_budget=-1 for dynamic thinking
        - include_thoughts=True for thought summaries
        """
        try:
            # Render conversation as single prompt (Gemini format)
            prompt_parts = [messages[0]["content"]]  # System message

            for msg in messages[1:]:
                role = msg["role"].upper()
                content = msg["content"]
                prompt_parts.append(f"{role}: {content}")

            prompt = "\n\n".join(prompt_parts)

            # Query with thinking config
            response = await asyncio.to_thread(
                client.models.generate_content,
                model="gemini-2.5-pro",
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    thinking_config=genai_types.ThinkingConfig(
                        thinking_budget=-1,  # Dynamic thinking
                        include_thoughts=True  # Get thought summaries
                    )
                )
            )

            # Extract text (includes thought summaries)
            if hasattr(response, "text") and response.text:
                return response.text

            # Fallback: assemble from candidates
            text_parts = []
            for candidate in getattr(response, "candidates", []) or []:
                content = getattr(candidate, "content", None)
                if content:
                    for part in content.parts if hasattr(content, "parts") else content:
                        if hasattr(part, "text") and part.text:
                            text_parts.append(part.text)

            if text_parts:
                return "\n".join(text_parts)

            return "ERROR: Gemini returned no text content"

        except Exception as e:
            return f"ERROR: Gemini 2.5 Pro failed - {str(e)}"

    async def _synthesize_with_o3(
        self,
        client: 'AsyncOpenAI',
        base_messages: List[Dict],
        valid_responses: Dict[str, str],
        errors: Dict[str, str],
        original_problem: str
    ) -> str:
        """
        Use O3 to synthesize all expert responses into unified plan

        Per OpenAI Responses API docs:
        - reasoning: {effort: "high", summary: "auto"}
        - Provides reasoning summary in response
        """
        try:
            # Build synthesis prompt
            synthesis_prompt = f"""[ENSEMBLE SYNTHESIS TASK]

Original Problem: {original_problem}

You received responses from {len(valid_responses)} expert perspectives. Your task:

1. Create a SHORT PARAGRAPH summary for EACH expert perspective (label as "Perspective 1", "Perspective 2", etc - do NOT reveal model names)
2. Then synthesize into ONE unified optimal plan

Format your response EXACTLY like this:

**Expert Perspectives:**

Perspective 1: [2-3 sentence summary of their key recommendation and reasoning]

Perspective 2: [2-3 sentence summary of their key recommendation and reasoning]

Perspective 3: [2-3 sentence summary of their key recommendation and reasoning]

Perspective 4: [2-3 sentence summary of their key recommendation and reasoning]

**Synthesized Optimal Plan:**
[Your unified recommendation - be specific, actionable, resolve contradictions, and provide concrete next steps]

"""

            # Add each expert response
            for i, (model_name, response) in enumerate(valid_responses.items(), 1):
                synthesis_prompt += f"""
{'='*60}
EXPERT PERSPECTIVE {i}:
{'='*60}
{response}

"""

            synthesis_prompt += f"""
{'='*60}

Now provide your response in the exact format specified above.
"""

            # Query O3 with high effort reasoning
            response = await client.responses.create(
                model="gpt-5",
                input=[
                    {"role": "system", "content": "You are an expert synthesizer combining insights from multiple AI models into optimal unified recommendations."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                reasoning={"effort": "medium", "summary": "auto"},
                max_output_tokens=16384
            )

            return response.output_text

        except Exception as e:
            return f"ERROR: O3 synthesis failed - {str(e)}"

    def _format_response(
        self,
        valid_responses: Dict[str, str],
        errors: Dict[str, str],
        synthesis: str
    ) -> str:
        """Format final ensemble response - synthesis only (no full expert responses)"""

        output = ["🔮 **ENSEMBLE CONSULTATION**\n"]
        output.append(f"Consulted {len(valid_responses)} expert perspectives, synthesized by O3.\n")
        output.append("="*70)
        output.append(synthesis)
        output.append("="*70)

        return "\n".join(output)

"""
MemoryCompactor - Hierarchical conversation history compression

Uses O3 to summarize older conversation history while keeping recent messages intact.
Enables handling very long conversations without exceeding context limits.
"""
import os
from typing import List, Dict, Optional
from openai import AsyncOpenAI


class MemoryCompactor:
    """
    Hierarchical memory compaction for long conversations

    Strategy:
    - Keep recent N messages intact (high fidelity)
    - Summarize older messages in tiers (progressive compression)
    - Use O3 for high-quality summarization
    """

    def __init__(
        self,
        keep_recent: int = 20,
        compression_ratio: float = 0.25,
        model: str = "o3"
    ):
        """
        Initialize memory compactor

        Args:
            keep_recent: Number of recent messages to keep intact
            compression_ratio: Target ratio of compressed to original (0.25 = 75% reduction)
            model: Model to use for summarization (default: o3)
        """
        self.keep_recent = keep_recent
        self.compression_ratio = compression_ratio
        self.model = model
        self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    async def compact(
        self,
        conversation_history: List[Dict],
        system_prompt: Optional[str] = None
    ) -> List[Dict]:
        """
        Compact conversation history

        Args:
            conversation_history: Full conversation history
            system_prompt: Optional system prompt to preserve

        Returns:
            Compacted conversation history with summary + recent messages
        """
        # Don't compact if history is short
        if len(conversation_history) <= self.keep_recent:
            return conversation_history

        # Split into old (to compress) and recent (keep intact)
        old_messages = conversation_history[:-self.keep_recent]
        recent_messages = conversation_history[-self.keep_recent:]

        # Summarize old messages
        summary = await self._summarize_messages(old_messages, system_prompt)

        # Build compacted history
        compacted = []

        # Add summary as first message
        compacted.append({
            "role": "user",
            "content": (
                "📝 CONVERSATION SUMMARY (older messages compressed):\n\n"
                f"{summary}\n\n"
                "--- End of summary. Recent messages follow below ---"
            )
        })

        compacted.append({
            "role": "assistant",
            "content": "I've reviewed the conversation summary. Continuing from recent messages..."
        })

        # Add recent messages intact
        compacted.extend(recent_messages)

        return compacted

    async def _summarize_messages(
        self,
        messages: List[Dict],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Summarize a list of messages using O3

        Args:
            messages: Messages to summarize
            system_prompt: Optional context about the conversation

        Returns:
            Summary text
        """
        # Format messages for summarization
        formatted = self._format_messages_for_summary(messages)

        # Create summarization prompt
        prompt = self._build_summarization_prompt(formatted, system_prompt)

        # Call O3 for summarization
        try:
            response = await self.client.responses.create(
                model=self.model,
                input=[{
                    "role": "user",
                    "content": prompt
                }],
                reasoning={
                    "effort": "medium",  # Balance quality and speed
                    "summary": "auto"
                },
                max_output_tokens=8192  # Enough for comprehensive summary
            )

            return response.output_text.strip()

        except Exception as e:
            # Fallback: return basic summary
            return (
                f"Summary of {len(messages)} older messages:\n"
                "The conversation covered multiple topics and tool executions. "
                "Key context has been preserved in recent messages."
            )

    def _format_messages_for_summary(self, messages: List[Dict]) -> str:
        """Format messages into readable text for summarization"""
        formatted_parts = []

        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            # Handle different content types
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                # Extract text from content blocks
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif item.get("type") == "tool_use":
                            text_parts.append(f"[TOOL: {item.get('name', 'unknown')}]")
                        elif item.get("type") == "tool_result":
                            result = item.get("content", "")
                            if isinstance(result, str):
                                # Truncate long tool results
                                if len(result) > 500:
                                    result = result[:500] + "... (truncated)"
                                text_parts.append(f"[RESULT: {result}]")
                text = "\n".join(text_parts)
            else:
                text = str(content)

            # Truncate very long messages
            if len(text) > 1000:
                text = text[:1000] + "... (truncated)"

            formatted_parts.append(f"{role.upper()}: {text}")

        return "\n\n".join(formatted_parts)

    def _build_summarization_prompt(
        self,
        formatted_messages: str,
        system_prompt: Optional[str]
    ) -> str:
        """Build prompt for LLM to summarize conversation"""
        parts = [
            "You are summarizing an older portion of a conversation between an AI agent and a user.",
            "",
            "The agent is an autonomous IDE agent that can:",
            "- Execute code in Jupyter notebooks",
            "- Run scripts in background and monitor them",
            "- Analyze data and create visualizations",
            "- Consult an ensemble of AI models for complex decisions",
            ""
        ]

        if system_prompt:
            parts.extend([
                "Agent's role:",
                system_prompt,
                ""
            ])

        parts.extend([
            "Your task: Create a concise but comprehensive summary of the conversation below.",
            "",
            "Include:",
            "1. Main user goals and requests",
            "2. Key decisions and actions taken",
            "3. Important results or findings",
            "4. Current state of work (files created, processes running, etc.)",
            "5. Any blockers or issues encountered",
            "",
            "Keep the summary factual and preserve important context.",
            f"Target length: ~{int(len(formatted_messages) * self.compression_ratio)} characters",
            "",
            "CONVERSATION TO SUMMARIZE:",
            "=" * 70,
            formatted_messages,
            "=" * 70,
            "",
            "Provide your summary:"
        ])

        return "\n".join(parts)

    def should_compact(self, conversation_history: List[Dict]) -> bool:
        """
        Check if conversation should be compacted

        Args:
            conversation_history: Current conversation history

        Returns:
            True if compaction is recommended
        """
        # Compact if significantly more than keep_recent threshold
        return len(conversation_history) > self.keep_recent * 1.5

    def estimate_token_savings(self, conversation_history: List[Dict]) -> int:
        """
        Estimate token savings from compaction

        Args:
            conversation_history: Current conversation history

        Returns:
            Estimated tokens saved (rough approximation)
        """
        if len(conversation_history) <= self.keep_recent:
            return 0

        old_messages = conversation_history[:-self.keep_recent]

        # Rough estimation: 4 chars per token
        old_chars = sum(
            len(str(msg.get("content", "")))
            for msg in old_messages
        )

        # Assume compression to target ratio
        saved_chars = old_chars * (1 - self.compression_ratio)

        return int(saved_chars / 4)

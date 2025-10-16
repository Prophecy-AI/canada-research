"""
PlanTask tool - Uses reasoning model to create detailed execution plan
"""
import os
from typing import Dict
from anthropic import Anthropic
from agent_v5.tools.base import BaseTool


class PlanTaskTool(BaseTool):
    """
    Creates a detailed execution plan using a reasoning model.

    This tool should be called at the start of complex tasks to:
    - Break down the problem into steps
    - Identify required tools and data
    - Plan the execution strategy
    - Anticipate potential issues
    """

    def __init__(self, workspace_dir: str, get_conversation_history_fn=None):
        """
        Initialize PlanTask tool

        Args:
            workspace_dir: Path to workspace directory
            get_conversation_history_fn: Function that returns current conversation history
        """
        super().__init__(workspace_dir)
        self.get_conversation_history = get_conversation_history_fn
        self.anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        # Use a reasoning-optimized model for planning
        self.planning_model = os.getenv("PLANNING_MODEL", "claude-opus-4-20250514")

    @property
    def name(self) -> str:
        return "PlanTask"

    @property
    def schema(self) -> Dict:
        return {
            "name": "PlanTask",
            "description": (
                "Create a detailed execution plan for a complex task using a reasoning model. "
                "Use this tool at the START of complex multi-step tasks to:\n"
                "- Break down the problem into concrete steps\n"
                "- Identify required tools, data sources, and dependencies\n"
                "- Plan the optimal execution strategy\n"
                "- Anticipate potential issues and edge cases\n"
                "- Establish success criteria\n\n"
                "The plan will guide subsequent tool executions and help avoid mistakes."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "task_description": {
                        "type": "string",
                        "description": (
                            "Clear description of the task to plan. Include:\n"
                            "- What needs to be accomplished\n"
                            "- Any constraints or requirements\n"
                            "- Expected outputs or deliverables"
                        )
                    },
                    "context": {
                        "type": "string",
                        "description": (
                            "Optional additional context about:\n"
                            "- Available data sources or files\n"
                            "- Known constraints or limitations\n"
                            "- Prior attempts or relevant information"
                        )
                    }
                },
                "required": ["task_description"]
            }
        }

    async def execute(self, input: Dict) -> Dict:
        """
        Execute planning using reasoning model

        Args:
            input: Dict with 'task_description' and optional 'context'

        Returns:
            Dict with detailed execution plan
        """
        task_description = input["task_description"]
        additional_context = input.get("context", "")

        # Get conversation history if available (for context)
        conversation_context = ""
        if self.get_conversation_history and callable(self.get_conversation_history):
            history = self.get_conversation_history()
            if history:
                # Include last user message for context
                conversation_context = f"\n\nCurrent conversation context:\n{history[-1].get('content', '')}"

        # Create planning prompt for reasoning model
        planning_prompt = f"""You are a planning expert for an autonomous AI agent. Create a detailed, actionable execution plan for the following task.

**Task:**
{task_description}

{f"**Additional Context:**\n{additional_context}" if additional_context else ""}
{conversation_context}

**Available Tools:**
- Bash: Execute shell commands
- Read: Read file contents
- Write: Create new files
- Edit: Modify existing files
- Glob: Find files by pattern
- Grep: Search file contents
- TodoWrite: Create task lists
- ReadBashOutput: Monitor background processes
- KillShell: Terminate background processes
- ListBashProcesses: List running background processes
- RunSummary: Summarize the current run
- ReadTodoList: Read current todo list
- Oracle: Ask questions about conversation history

**Your Planning Output Should Include:**

1. **Goal Analysis**
   - What is the core objective?
   - What are the success criteria?
   - What are the key challenges?

2. **Step-by-Step Execution Plan**
   - Numbered, concrete steps
   - Which tools to use for each step
   - Dependencies between steps
   - Expected outputs at each stage

3. **Data Requirements**
   - What data/files are needed?
   - Where to find or how to generate them?
   - Any data transformations required?

4. **Risk Assessment**
   - Potential failure points
   - Edge cases to handle
   - Mitigation strategies

5. **Validation Strategy**
   - How to verify each step succeeded?
   - Final validation criteria
   - Testing approach if applicable

**Format your plan clearly with markdown headers and bullet points.**
"""

        try:
            # Call reasoning model for planning
            response = self.anthropic_client.messages.create(
                model=self.planning_model,
                max_tokens=8000,
                temperature=0.0,  # Deterministic planning
                messages=[{
                    "role": "user",
                    "content": planning_prompt
                }]
            )

            plan = response.content[0].text

            # Format output
            output = f"""# Execution Plan (Generated by {self.planning_model})

{plan}

---
*Plan created. Proceed with execution following the steps above.*
"""

            return {
                "content": output,
                "is_error": False,
                "debug_summary": f"Created plan using {self.planning_model}"
            }

        except Exception as e:
            return {
                "content": f"Planning failed: {str(e)}",
                "is_error": True,
                "debug_summary": f"Planning error: {str(e)}"
            }

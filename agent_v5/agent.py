"""
ResearchAgent - Main agentic loop for Agent V5
"""
import os
from typing import List, Dict, AsyncGenerator
from google import genai
from google.genai import types
from agent_v5.tools.registry import ToolRegistry
from debug import log, with_session
from agent_v5.tools.bash import BashTool
from agent_v5.tools.bash_output import ReadBashOutputTool
from agent_v5.tools.kill_shell import KillShellTool
from agent_v5.tools.bash_process_registry import BashProcessRegistry
from agent_v5.tools.read import ReadTool
from agent_v5.tools.write import WriteTool
from agent_v5.tools.edit import EditTool
from agent_v5.tools.glob import GlobTool
from agent_v5.tools.grep import GrepTool
from agent_v5.tools.todo import TodoWriteTool, ReadTodoListTool
from agent_v5.tools.list_bash import ListBashProcessesTool
from agent_v5.tools.run_summary import RunSummaryTool
from agent_v5.tools.cohort import CohortDefinitionTool
from agent_v5.tools.oracle import OracleTool
from agent_v5.tools.elapsed_time import ElapsedTimeTool
import time


class ResearchAgent:
    """Research agent with agentic loop"""

    def __init__(self, session_id: str, workspace_dir: str, system_prompt: str, start_time: float = None):
        self.session_id = session_id
        self.workspace_dir = workspace_dir
        self.system_prompt = system_prompt
        self.conversation_history: List[Dict] = []
        self.tools = ToolRegistry(workspace_dir)
        self.process_registry = BashProcessRegistry()  # Registry for background bash processes
        self.start_time = start_time if start_time is not None else time.time()
        self._register_core_tools()
        self.gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.run = with_session(session_id)(self.run)

    def _register_core_tools(self):
        """Register all core tools"""
        self.tools.register(BashTool(self.workspace_dir, self.process_registry))
        self.tools.register(ReadBashOutputTool(self.workspace_dir, self.process_registry))
        self.tools.register(KillShellTool(self.workspace_dir, self.process_registry))
        self.tools.register(ReadTool(self.workspace_dir))
        self.tools.register(WriteTool(self.workspace_dir))
        self.tools.register(EditTool(self.workspace_dir))
        self.tools.register(GlobTool(self.workspace_dir))
        self.tools.register(GrepTool(self.workspace_dir))
        self.tools.register(TodoWriteTool(self.workspace_dir))
        #self.tools.register(CohortDefinitionTool(self.workspace_dir))
        self.tools.register(RunSummaryTool(self.workspace_dir))
        self.tools.register(ReadTodoListTool(self.workspace_dir))
        self.tools.register(ListBashProcessesTool(self.workspace_dir, self.process_registry))
        self.tools.register(OracleTool(self.workspace_dir, lambda: self.conversation_history))
        self.tools.register(ElapsedTimeTool(self.workspace_dir, self.start_time))

    async def cleanup(self) -> None:
        """
        Cleanup agent resources (kills all background processes)

        IMPORTANT: Call this after agent.run() completes to prevent process leaks.
        Especially critical for long-running agents or when agent is cancelled.
        """
        killed = await self.process_registry.cleanup()
        if killed > 0:
            log(f"✓ Cleaned up {killed} background processes", 1)

    async def run(self, user_message: str) -> AsyncGenerator[Dict, None]:
        """Main agentic loop"""
        log(f"→ Agent.run(session={self.session_id})")

        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "parts": [{"text": user_message}]
        })

        while True:
            log(f"→ API call (turn {len(self.conversation_history)//2})")

            # Convert conversation history to Gemini format
            contents = self._convert_history_to_gemini()

            # Get tool declarations for Gemini
            tool_declarations = self.tools.get_schemas_gemini()

            # Stream response from Gemini
            response_text = ""
            function_calls = []

            try:
                response = self.gemini_client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=self.system_prompt,
                        tools=tool_declarations if tool_declarations else None,
                        temperature=0.0,
                    )
                )

                # Process response parts
                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            # Handle text response
                            if hasattr(part, 'text') and part.text:
                                response_text += part.text
                                yield {"type": "text_delta", "text": part.text}

                            # Handle function calls
                            elif hasattr(part, 'function_call') and part.function_call:
                                fc = part.function_call
                                function_calls.append({
                                    "name": fc.name,
                                    "args": dict(fc.args) if fc.args else {}
                                })

            except Exception as e:
                log(f"✗ Gemini API error: {e}", 2)
                yield {"type": "text_delta", "text": f"\n\n[Error: {str(e)}]"}
                break

            # Add assistant response to history
            assistant_parts = []
            if response_text:
                assistant_parts.append({"text": response_text})
            for fc in function_calls:
                assistant_parts.append({
                    "function_call": {
                        "name": fc["name"],
                        "args": fc["args"]
                    }
                })

            self.conversation_history.append({
                "role": "model",
                "parts": assistant_parts
            })

            # If no function calls, we're done
            if not function_calls:
                log("✓ Agent.run complete", 1)
                break

            # Execute function calls
            log(f"→ Executing {len(function_calls)} tools")

            function_responses = []
            for fc in function_calls:
                result = await self.tools.execute(fc["name"], fc["args"])

                function_responses.append({
                    "function_response": {
                        "name": fc["name"],
                        "response": {"content": result["content"]}
                    }
                })

                yield {
                    "type": "tool_execution",
                    "tool_name": fc["name"],
                    "tool_input": fc["args"],
                    "tool_output": result["content"]
                }

            # Add function responses to history
            self.conversation_history.append({
                "role": "user",
                "parts": function_responses
            })

        yield {"type": "done"}

    def _convert_history_to_gemini(self) -> List[types.Content]:
        """Convert conversation history to Gemini format"""
        contents = []
        for msg in self.conversation_history:
            role = msg["role"]
            parts = msg.get("parts", [])

            # Convert role (user stays user, assistant/model → model)
            gemini_role = "user" if role == "user" else "model"

            # Build content parts
            content_parts = []
            for part in parts:
                if "text" in part:
                    content_parts.append(types.Part(text=part["text"]))
                elif "function_call" in part:
                    fc = part["function_call"]
                    content_parts.append(types.Part(
                        function_call=types.FunctionCall(
                            name=fc["name"],
                            args=fc["args"]
                        )
                    ))
                elif "function_response" in part:
                    fr = part["function_response"]
                    content_parts.append(types.Part(
                        function_response=types.FunctionResponse(
                            name=fr["name"],
                            response=fr["response"]
                        )
                    ))

            if content_parts:
                contents.append(types.Content(role=gemini_role, parts=content_parts))

        return contents

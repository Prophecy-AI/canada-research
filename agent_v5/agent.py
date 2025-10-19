"""
ResearchAgent - Main agentic loop for Agent V5
"""
import json
import os
import time
from typing import List, Dict, AsyncGenerator, Any, Optional

from openai import AsyncOpenAI

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


class ResearchAgent:
    """Research agent with agentic loop"""

    def __init__(self, session_id: str, workspace_dir: str, system_prompt: str, start_time: float = None):
        self.session_id = session_id
        self.workspace_dir = workspace_dir
        self.system_prompt = system_prompt
        self.conversation_history: List[Dict[str, Any]] = []
        self.tools = ToolRegistry(workspace_dir)
        self.process_registry = BashProcessRegistry()  # Registry for background bash processes
        self.start_time = start_time if start_time is not None else time.time()
        self.last_response_id: Optional[str] = None
        self._register_core_tools()
        self.openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
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

    def _build_openai_tools(self) -> List[Dict[str, Any]]:
        """Convert tool schemas to OpenAI Responses-compatible definitions."""
        tool_defs = []
        for schema in self.tools.get_schemas():
            parameters = schema.get("input_schema", {"type": "object", "properties": {}})
            tool_def = {
                "type": "function",
                "name": schema["name"],
                "description": schema.get("description", ""),
                "parameters": parameters,
            }
            if "strict" in schema:
                tool_def["strict"] = schema["strict"]
            tool_defs.append(tool_def)
        return tool_defs

    @staticmethod
    def _parse_tool_arguments(arguments: str) -> Dict[str, Any]:
        """Parse tool call arguments from JSON string."""
        if arguments is None:
            return {}
        try:
            parsed = json.loads(arguments)
            # OpenAI returns arguments as dict; guard against non-dicts
            if isinstance(parsed, dict):
                return parsed
            return {"value": parsed}
        except json.JSONDecodeError:
            return {"raw_arguments": arguments}

    async def run(self, user_message: str) -> AsyncGenerator[Dict, None]:
        """Main agentic loop"""
        log(f"→ Agent.run(session={self.session_id})")

        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        pending_inputs: List[Dict[str, Any]] = [{
            "role": "user",
            "content": user_message
        }]
        previous_response_id = self.last_response_id
        turn_index = 0

        while True:
            turn_index += 1
            log(f"→ API call (turn {turn_index})")

            tools = self._build_openai_tools()

            streamed_text_started = False
            streamed_text_buffer = ""

            request_kwargs: Dict[str, Any] = {
                "model": "gpt-5",
                "input": pending_inputs,
                "instructions": self.system_prompt,
                "tools": tools,
            }
            if previous_response_id:
                request_kwargs["previous_response_id"] = previous_response_id

            async with self.openai_client.responses.stream(**request_kwargs) as stream:
                async for event in stream:
                    if event.type == "response.output_text.delta":
                        if not streamed_text_started:
                            streamed_text_started = True
                            yield {"type": "text_start"}
                        text = event.delta
                        if text:
                            streamed_text_buffer += text
                            yield {"type": "text_delta", "text": text}

                final_response = await stream.get_final_response()

            response_dict = final_response.model_dump(mode="python")
            output_items = response_dict.get("output", [])
            response_id = response_dict.get("id")
            if isinstance(response_id, str):
                self.last_response_id = response_id
                previous_response_id = response_id

            tool_calls = []
            assistant_content: List[Dict[str, Any]] = []
            message_text_found = False

            for item in output_items:
                item_type = item.get("type")
                if item_type == "message":
                    for content_part in item.get("content", []):
                        if content_part.get("type") == "output_text":
                            text = content_part.get("text", "")
                            if text:
                                message_text_found = True
                                assistant_content.append({"type": "text", "text": text})
                elif item_type == "reasoning":
                    assistant_content.append({
                        "type": "reasoning",
                        "content": item.get("content", []),
                        "summary": item.get("summary", [])
                    })
                elif item_type == "function_call":
                    tool_calls.append(item)
                    assistant_content.append({
                        "type": "tool_use",
                        "id": item.get("id"),
                        "name": item.get("name"),
                        "call_id": item.get("call_id"),
                        "input": self._parse_tool_arguments(item.get("arguments", "")),
                        "raw_arguments": item.get("arguments", "")
                    })
                else:
                    assistant_content.append({
                        "type": item_type or "unknown",
                        "raw": item
                    })

            if not message_text_found and streamed_text_buffer:
                assistant_content.append({"type": "text", "text": streamed_text_buffer})

            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_content,
                "raw_output": output_items
            })

            if not tool_calls:
                log("✓ Agent.run complete", 1)
                break

            log(f"→ Executing {len(tool_calls)} tools")

            tool_results = []
            for call in tool_calls:
                name = call.get("name")
                call_id = call.get("call_id")
                arguments = call.get("arguments", "")
                tool_input = self._parse_tool_arguments(arguments)

                result = await self.tools.execute(name, tool_input)
                content_str = result.get("content", "")
                # Ensure the tool output is a string, per API contract
                if not isinstance(content_str, str):
                    content_str = json.dumps(content_str, indent=2)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "tool_name": name,
                    "content": content_str,
                    "is_error": result.get("is_error", False)
                })

                yield {
                    "type": "tool_execution",
                    "tool_name": name,
                    "tool_input": tool_input,
                    "tool_output": content_str
                }

            self.conversation_history.append({
                "role": "tool",
                "content": tool_results
            })

            pending_inputs = [
                {
                    "type": "function_call_output",
                    "call_id": result["tool_use_id"],
                    "output": result["content"]
                }
                for result in tool_results
            ]

        yield {"type": "done"}

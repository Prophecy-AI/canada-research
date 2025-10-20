"""
IDEAgent - Main agent for agent_v6

Extends ResearchAgent from agent_v5 with:
- IDE workspace state tracking
- Jupyter notebook support (first-class)
- Non-blocking script execution
- Memory compaction
- Ensemble reasoning
"""
import os
import time
from typing import List, Dict, AsyncGenerator, Any, Optional

from openai import AsyncOpenAI

from agent_v5.tools.registry import ToolRegistry
from agent_v5.tools.read import ReadTool
from agent_v5.tools.write import WriteTool
from agent_v5.tools.edit import EditTool
from agent_v5.tools.glob import GlobTool
from agent_v5.tools.grep import GrepTool
from agent_v5.tools.elapsed_time import ElapsedTimeTool
from debug import log, with_session

from agent_v6.workspace import IDEWorkspace
from agent_v6.tools.notebook import NotebookTool
from agent_v6.tools.execute_script import ExecuteScriptTool
from agent_v6.tools.check_process import CheckProcessTool
from agent_v6.tools.interrupt_process import InterruptProcessTool
from agent_v6.ensemble.tool import EnsembleTool
from agent_v6.memory import MemoryCompactor


class IDEAgent:
    """
    IDE-centric autonomous agent with multi-provider ensemble reasoning

    Based on agent_v5.ResearchAgent but adds:
    - IDEWorkspace state tracking
    - Jupyter notebook first-class support
    - Non-blocking script execution with monitoring
    - Hierarchical memory compaction
    - Multi-provider ensemble consultation
    """

    def __init__(
        self,
        session_id: str,
        workspace_dir: str,
        system_prompt: str,
        start_time: float = None,
        enable_memory_compaction: bool = True
    ):
        """
        Initialize IDE agent

        Args:
            session_id: Unique session identifier
            workspace_dir: Workspace directory path
            system_prompt: System prompt (agent role/instructions)
            start_time: Session start time (default: now)
            enable_memory_compaction: Enable automatic memory compaction
        """
        self.session_id = session_id
        self.workspace_dir = workspace_dir
        self.system_prompt = system_prompt
        self.conversation_history: List[Dict[str, Any]] = []
        self.start_time = start_time if start_time is not None else time.time()
        self.last_response_id: Optional[str] = None
        self.enable_memory_compaction = enable_memory_compaction

        # Initialize workspace state tracking
        self.workspace = IDEWorkspace(workspace_dir)

        # Initialize tools registry
        self.tools = ToolRegistry(workspace_dir)

        # Initialize memory compactor
        self.memory_compactor = MemoryCompactor(
            keep_recent=20,
            compression_ratio=0.25
        ) if enable_memory_compaction else None

        # Register all tools
        self._register_core_tools()

        # Initialize OpenAI client
        self.openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

        # Wrap run method with session logging
        self.run = with_session(session_id)(self.run)

    def _register_core_tools(self):
        """Register all IDE tools"""
        # File operations (from agent_v5)
        self.tools.register(ReadTool(self.workspace_dir))
        self.tools.register(WriteTool(self.workspace_dir))
        self.tools.register(EditTool(self.workspace_dir))
        self.tools.register(GlobTool(self.workspace_dir))
        self.tools.register(GrepTool(self.workspace_dir))

        # Notebook support (NEW in agent_v6)
        notebook_tool = NotebookTool(self.workspace_dir, self.workspace)
        self.tools.register(notebook_tool)
        self.notebook_tool = notebook_tool  # Keep reference for cleanup

        # Script execution with monitoring (NEW in agent_v6)
        execute_script_tool = ExecuteScriptTool(self.workspace_dir, self.workspace)
        self.tools.register(execute_script_tool)
        self.execute_script_tool = execute_script_tool  # Keep reference

        self.tools.register(CheckProcessTool(self.workspace_dir, execute_script_tool))
        self.tools.register(InterruptProcessTool(self.workspace_dir, execute_script_tool, self.workspace))

        # Ensemble reasoning (NEW in agent_v6)
        self.tools.register(EnsembleTool(
            workspace_dir=self.workspace_dir,
            get_conversation_history=lambda: self.conversation_history
        ))

        # Utilities
        self.tools.register(ElapsedTimeTool(self.workspace_dir, self.start_time))

    async def cleanup(self) -> None:
        """
        Cleanup agent resources

        IMPORTANT: Call this after agent.run() completes to prevent leaks.
        Cleans up:
        - Jupyter kernels
        - Background processes
        """
        # Cleanup notebook kernels
        if hasattr(self, 'notebook_tool'):
            await self.notebook_tool.cleanup()
            log("✓ Cleaned up notebook kernels", 1)

        # Cleanup background processes
        if hasattr(self, 'execute_script_tool'):
            await self.execute_script_tool.cleanup()
            log("✓ Cleaned up background processes", 1)

    def _build_openai_tools(self) -> List[Dict[str, Any]]:
        """Convert tool schemas to OpenAI Responses-compatible definitions"""
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
        """Parse tool call arguments from JSON string"""
        if arguments is None:
            return {}
        try:
            import json
            parsed = json.loads(arguments)
            if isinstance(parsed, dict):
                return parsed
            return {"value": parsed}
        except:
            return {"raw_arguments": arguments}

    async def _maybe_compact_memory(self) -> None:
        """Auto-compact memory if threshold exceeded"""
        if not self.memory_compactor:
            return

        if self.memory_compactor.should_compact(self.conversation_history):
            log(f"→ Compacting memory ({len(self.conversation_history)} messages)")
            savings = self.memory_compactor.estimate_token_savings(self.conversation_history)
            log(f"  Estimated savings: ~{savings} tokens")

            self.conversation_history = await self.memory_compactor.compact(
                self.conversation_history,
                self.system_prompt
            )

            log(f"✓ Memory compacted to {len(self.conversation_history)} messages", 1)

    async def run(self, user_message: str) -> AsyncGenerator[Dict, None]:
        """
        Main agentic loop

        Args:
            user_message: User's message/request

        Yields:
            Stream of events:
            - {"type": "text_start"}: Text streaming begins
            - {"type": "text_delta", "text": str}: Text chunk
            - {"type": "tool_execution", ...}: Tool execution event
        """
        log(f"→ IDEAgent.run(session={self.session_id})")

        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Check if memory compaction needed
        await self._maybe_compact_memory()

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
                log("✓ IDEAgent.run complete", 1)
                break

            log(f"→ Executing {len(tool_calls)} tools")

            # Execute tools
            tool_results = []
            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_id = tool_call.get("id")
                tool_call_id = tool_call.get("call_id")
                tool_input = self._parse_tool_arguments(tool_call.get("arguments", ""))

                log(f"  → {tool_name}({list(tool_input.keys())})")

                result = await self.tools.execute(tool_name, tool_input)

                log(f"  ✓ {tool_name}: {result.get('debug_summary', 'done')}")

                tool_results.append({
                    "type": "function_call_output",
                    "call_id": tool_call_id,
                    "output": result.get("content", "")
                })

                # Yield tool execution event
                yield {
                    "type": "tool_execution",
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "result": result
                }

            # Add tool results to conversation
            self.conversation_history.append({
                "role": "tool",
                "content": tool_results
            })

            # Set pending inputs for next turn
            pending_inputs = tool_results

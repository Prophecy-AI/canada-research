"""
ResearchAgent - Main agentic loop for Agent V5
"""
import asyncio
import os
import time
from typing import List, Dict, AsyncGenerator, Optional
from anthropic import Anthropic
from agent_v5.tools.registry import ToolRegistry
from agent_v5.timeout_manager import TimeoutManager
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
from agent_v5.tools.plan import PlanTaskTool


class ResearchAgent:
    """Research agent with agentic loop and intelligent timeout management"""

    def __init__(
        self,
        session_id: str,
        workspace_dir: str,
        system_prompt: str,
        max_runtime_seconds: Optional[int] = None,  # No limit by default
        max_turns: Optional[int] = None,  # No limit by default
        stall_timeout_seconds: int = 600  # 10 minutes default
    ):
        self.session_id = session_id
        self.workspace_dir = workspace_dir
        self.system_prompt = system_prompt
        self.conversation_history: List[Dict] = []
        self.tools = ToolRegistry(workspace_dir)
        self.process_registry = BashProcessRegistry()  # Registry for background bash processes
        self._register_core_tools()
        self.anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

        # Initialize timeout manager
        self.timeout_manager = TimeoutManager(
            max_runtime_seconds=max_runtime_seconds,
            max_turns=max_turns,
            stall_timeout_seconds=stall_timeout_seconds
        )

        self.run = with_session(session_id)(self.run)

    def _register_core_tools(self):
        """Register all core tools"""
        # Planning tool (uses conversation history for context)
        self.tools.register(PlanTaskTool(self.workspace_dir, lambda: self.conversation_history))

        # Core execution tools
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

    async def cleanup(self) -> None:
        """
        Cleanup agent resources (kills all background processes)

        IMPORTANT: Call this after agent.run() completes to prevent process leaks.
        Especially critical for long-running agents or when agent is cancelled.
        """
        killed = await self.process_registry.cleanup()
        if killed > 0:
            log(f"✓ Cleaned up {killed} background processes", 1)

    def _can_parallelize_tools(self, tool_uses: List[Dict]) -> bool:
        """
        Determine if tools can be executed in parallel.

        Returns False if:
        - Any Write/Edit operations (may affect Read operations)
        - Multiple Bash commands (may have dependencies)
        - ReadBashOutput (needs sequential monitoring)
        - KillShell (affects other processes)

        Returns True for:
        - Multiple Read operations
        - Multiple Glob/Grep operations
        - Mix of read-only MCP tools
        """
        tool_names = [tool["name"] for tool in tool_uses]

        # Sequential execution required for these tools
        sequential_tools = {
            "Write", "Edit", "Bash", "ReadBashOutput", "KillShell", "TodoWrite"
        }

        # Check if any tools require sequential execution
        if any(name in sequential_tools for name in tool_names):
            return False

        # All remaining tools are read-only and can be parallelized
        return True

    def _should_wait_for_process(self, tool_uses: List[Dict], tool_results: List[Dict]) -> bool:
        if len(tool_uses) != 1 or tool_uses[0]["name"] != "ReadBashOutput":
            return False
        content = tool_results[0]["content"]
        return "[RUNNING]" in content and "(no new output since last read)" in content

    async def _wait_for_process(self, shell_id: str) -> str:
        bg_process = self.process_registry.get(shell_id)
        if not bg_process:
            return "Process not found"
        
        while bg_process.process.returncode is None:
            await asyncio.sleep(30)
            
            new_output = bg_process.stdout_data[bg_process.stdout_cursor:].decode('utf-8', errors='replace')
            new_output += bg_process.stderr_data[bg_process.stderr_cursor:].decode('utf-8', errors='replace')
            
            if new_output.strip():
                log(f"→ {shell_id} output: {new_output[:200]}")
                bg_process.stdout_cursor = len(bg_process.stdout_data)
                bg_process.stderr_cursor = len(bg_process.stderr_data)
            else:
                log(f"→ Still waiting for {shell_id} (runtime: {time.time() - bg_process.start_time:.0f}s)")
        
        final_output = bg_process.stdout_data[bg_process.stdout_cursor:].decode('utf-8', errors='replace')
        final_output += bg_process.stderr_data[bg_process.stderr_cursor:].decode('utf-8', errors='replace')
        bg_process.stdout_cursor = len(bg_process.stdout_data)
        bg_process.stderr_cursor = len(bg_process.stderr_data)
        
        runtime = time.time() - bg_process.start_time
        exit_code = bg_process.process.returncode
        
        log(f"✓ {shell_id} completed (exit code: {exit_code}, runtime: {runtime:.0f}s)", 1)
        
        all_output = bg_process.stdout_data.decode('utf-8', errors='replace')
        all_output += bg_process.stderr_data.decode('utf-8', errors='replace')
        
        return (
            f"[COMPLETED] {shell_id} (exit code: {exit_code}, runtime: {runtime:.0f}s)\n"
            f"Command: {bg_process.command}\n\n"
            f"{all_output}"
        )

    async def run(self, user_message: str) -> AsyncGenerator[Dict, None]:
        """Main agentic loop with intelligent timeout management"""
        log(f"→ Agent.run(session={self.session_id})")

        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        while True:
            # Start new turn and check for timeouts
            self.timeout_manager.start_turn()
            timeout_check = self.timeout_manager.check_timeout()

            if timeout_check["timed_out"]:
                log(f"⚠️  {timeout_check['reason']}", 2)
                yield {
                    "type": "timeout",
                    "reason": timeout_check["reason"],
                    "summary": self.timeout_manager.get_summary()
                }
                # Send timeout message to agent for graceful termination
                timeout_msg = (
                    f"\n\n⏱️ **TIMEOUT REACHED**: {timeout_check['reason']}\n\n"
                    f"Please provide a summary of work completed so far and create "
                    f"any necessary output files before terminating."
                )
                self.conversation_history.append({
                    "role": "user",
                    "content": timeout_msg
                })
                # Allow ONE final turn to wrap up
                break

            response_content = []
            tool_uses = []

            log(f"→ API call (turn {self.timeout_manager.turn_count})")

            with self.anthropic_client.messages.stream(
                model="claude-sonnet-4-5-20250929",
                max_tokens=20000,
                system=self.system_prompt,
                messages=self.conversation_history,
                tools=self.tools.get_schemas(),
                temperature=0,
            ) as stream:
                for event in stream:
                    if event.type == "content_block_start":
                        if event.content_block.type == "text":
                            yield {"type": "text_start"}

                    elif event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            text = event.delta.text
                            response_content.append({"type": "text", "text": text})
                            self.timeout_manager.register_activity()  # Track activity
                            yield {"type": "text_delta", "text": text}

                final_message = stream.get_final_message()
                for block in final_message.content:
                    if block.type == "tool_use":
                        tool_uses.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input
                        })

            self.conversation_history.append({
                "role": "assistant",
                "content": response_content + tool_uses
            })

            if not tool_uses:
                log("✓ Agent.run complete", 1)
                break

            log(f"→ Executing {len(tool_uses)} tools")

            # Determine if tools can be parallelized
            can_parallelize = self._can_parallelize_tools(tool_uses)

            if can_parallelize and len(tool_uses) > 1:
                log(f"→ Parallelizing {len(tool_uses)} independent tools")
                # Execute all tools in parallel with asyncio.gather
                results = await asyncio.gather(
                    *[self.tools.execute(tool_use["name"], tool_use["input"])
                      for tool_use in tool_uses],
                    return_exceptions=True
                )

                tool_results = []
                for tool_use, result in zip(tool_uses, results):
                    # Handle exceptions from gather
                    if isinstance(result, Exception):
                        result = {
                            "content": f"Tool execution error: {str(result)}",
                            "is_error": True
                        }

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use["id"],
                        "content": result["content"],
                        "is_error": result.get("is_error", False)
                    })

                    yield {
                        "type": "tool_execution",
                        "tool_name": tool_use["name"],
                        "tool_input": tool_use["input"],
                        "tool_output": result["content"]
                    }
            else:
                # Sequential execution (when tools have dependencies)
                tool_results = []
                for tool_use in tool_uses:
                    result = await self.tools.execute(tool_use["name"], tool_use["input"])
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use["id"],
                        "content": result["content"],
                        "is_error": result.get("is_error", False)
                    })

                    yield {
                        "type": "tool_execution",
                        "tool_name": tool_use["name"],
                        "tool_input": tool_use["input"],
                        "tool_output": result["content"]
                    }

            self.conversation_history.append({
                "role": "user",
                "content": tool_results
            })

        yield {"type": "done"}

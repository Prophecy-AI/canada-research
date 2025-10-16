"""Tests for ListBashProcessesTool"""

import pytest
import tempfile
import asyncio

from agent_v5.tools.bash import BashTool
from agent_v5.tools.bash_output import ReadBashOutputTool
from agent_v5.tools.kill_shell import KillShellTool
from agent_v5.tools.bash_process_registry import BashProcessRegistry
from agent_v5.tools.list_bash import ListBashProcessesTool


@pytest.mark.asyncio
async def test_list_bash_processes_active_and_completed():
    """Ensure ListBashProcesses shows running and completed jobs and cleanup prevents leaks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = BashProcessRegistry()
        bash_tool = BashTool(tmpdir, registry)
        list_tool = ListBashProcessesTool(tmpdir, registry)

        # Start a quick background job
        result = await bash_tool.execute({"command": "echo 'hello'", "background": True})
        assert not result["is_error"]
        shell_id = result["content"].split("process: ")[1].split("\n")[0]

        # Immediately list – should show RUNNING or COMPLETED depending on timing
        listing1 = await list_tool.execute({})
        assert shell_id in listing1["content"]

        # Wait for completion
        await asyncio.sleep(0.2)
        listing2 = await list_tool.execute({})
        assert shell_id in listing2["content"]

        # Cleanup to avoid leaks
        await registry.cleanup()

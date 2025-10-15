"""Ensure BashTool with PTY wrapper flushes output quickly."""

import pytest
import tempfile
import os
import textwrap
import asyncio

from agent_v5.tools.bash import BashTool
from agent_v5.tools.bash_output import ReadBashOutputTool
from agent_v5.tools.bash_process_registry import BashProcessRegistry


@pytest.mark.asyncio
async def test_tty_flush_background():
    """Background Bash process should produce output that can be read incrementally."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a small Python script that prints incrementally
        script_path = os.path.join(tmpdir, "flush_test.py")
        with open(script_path, "w") as f:
            f.write(textwrap.dedent(
                """
                import time, sys
                for i in range(3):
                    print(f"LINE {i}")
                    sys.stdout.flush()
                    time.sleep(0.5)
                """
            ))

        registry = BashProcessRegistry()
        bash_tool = BashTool(tmpdir, registry)
        read_tool = ReadBashOutputTool(tmpdir, registry)

        # Launch background process
        res = await bash_tool.execute({"command": f"python {script_path}", "background": True})
        assert not res["is_error"], res["content"]

        shell_id_line = res["content"].split("process: ")[1]
        shell_id = shell_id_line.split("\n")[0]

        # Wait for the first line to be printed ( > 0.5s )
        await asyncio.sleep(1.0)

        out1 = await read_tool.execute({"shell_id": shell_id})
        assert "LINE 0" in out1["content"], f"Output after 1s was: {out1['content']}"

        # Cleanup background processes
        await registry.cleanup()

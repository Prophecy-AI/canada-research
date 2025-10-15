"""Ensure BashTool with script wrapper flushes output quickly."""

import pytest, tempfile, os, textwrap, asyncio

from agent_v5.tools.bash import BashTool
from agent_v5.tools.bash_output import ReadBashOutputTool
from agent_v5.tools.bash_process_registry import BashProcessRegistry


@pytest.mark.asyncio
async def test_tty_flush_background():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write a script that prints 3 lines with delay
        script_path = os.path.join(tmpdir, "flush_test.py")
        with open(script_path, "w") as f:
            f.write(textwrap.dedent("""
                import time, sys
                for i in range(3):
                    print(f"LINE {i}")
                    sys.stdout.flush()
                    time.sleep(0.5)
            """))

        registry = BashProcessRegistry()
        bash_tool = BashTool(tmpdir, registry)
        read_tool = ReadBashOutputTool(tmpdir, registry)

        res = await bash_tool.execute({"command": f"python {script_path}", "background": True})
        assert not res["is_error"]
        shell_id = res["content"].split("process: ")[1].split("\n")[0]

        # Wait 1s and read
        await asyncio.sleep(1.0)
        out1 = await read_tool.execute({"shell_id": shell_id})
        assert "LINE 0" in out1["content"]

        # Wait for completion
        await registry.cleanup()

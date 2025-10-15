"""
Quick test to verify ReadBashOutput truncation works correctly
"""
import asyncio
import tempfile
from agent_v5.tools.bash import BashTool
from agent_v5.tools.bash_output import ReadBashOutputTool, MAX_OUTPUT_SIZE
from agent_v5.tools.bash_process_registry import BashProcessRegistry


async def test_truncation():
    """Test that large outputs are truncated to MAX_OUTPUT_SIZE"""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = BashProcessRegistry()
        bash_tool = BashTool(tmpdir, registry)
        read_tool = ReadBashOutputTool(tmpdir, registry)
        
        # Generate output larger than MAX_OUTPUT_SIZE (50KB)
        large_size = MAX_OUTPUT_SIZE + 10000  # 60KB
        result = await bash_tool.execute({
            "command": f"python3 -c \"print('x' * {large_size})\"",
            "background": True
        })
        
        assert not result["is_error"], f"Background execution failed: {result}"
        shell_id = result["content"].split("process: ")[1].split("\n")[0]
        
        # Wait for output
        await asyncio.sleep(0.5)
        
        # Read output - should be truncated
        output = await read_tool.execute({"shell_id": shell_id})
        
        print(f"\n=== Test Results ===")
        print(f"Original output size: {large_size:,} chars")
        print(f"MAX_OUTPUT_SIZE: {MAX_OUTPUT_SIZE:,} chars")
        print(f"Returned output size: {len(output['content']):,} chars")
        print(f"Truncation notice present: {'⚠️  Output truncated' in output['content']}")
        print(f"\n=== Actual Output ===")
        print(output["content"])
        print("=" * 80)
        
        # Verify truncation
        assert not output["is_error"], f"Read failed: {output}"
        assert "⚠️  Output truncated" in output["content"], "Missing truncation notice"
        assert f"showing last {MAX_OUTPUT_SIZE:,} chars" in output["content"], "Missing size info"
        
        # The content should be less than MAX_OUTPUT_SIZE + overhead (status line, truncation notice)
        # Allow some overhead for status messages
        assert len(output["content"]) < MAX_OUTPUT_SIZE + 1000, \
            f"Output not truncated: {len(output['content'])} > {MAX_OUTPUT_SIZE + 1000}"
        
        # Cleanup
        await registry.cleanup()
        
        print("✅ Truncation test PASSED!")
        return True


if __name__ == "__main__":
    result = asyncio.run(test_truncation())
    if result:
        print("\n🎉 All truncation tests passed!")


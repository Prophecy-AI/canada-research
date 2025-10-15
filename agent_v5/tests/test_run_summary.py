"""Tests for RunSummaryTool"""

import pytest
import tempfile
import os
import json
from agent_v5.tools.run_summary import RunSummaryTool


@pytest.mark.asyncio
async def test_run_summary_appends_and_latest():
    with tempfile.TemporaryDirectory() as tmpdir:
        tool = RunSummaryTool(tmpdir)

        # First summary
        res1 = await tool.execute({
            "phase": "train",
            "hypothesis": "add feature",
            "action": "Feature engineering",
            "model": "lgbm",
            "metrics": {"cv": 0.8}
        })
        assert not res1["is_error"]
        log_path = os.path.join(tmpdir, ".runs", "run_log.jsonl")
        latest_path = os.path.join(tmpdir, ".runs", "latest.json")
        assert os.path.exists(log_path)
        assert os.path.exists(latest_path)

        # File sizes
        with open(log_path) as f:
            lines = f.readlines()
            assert len(lines) == 1

        with open(latest_path) as f:
            latest = json.load(f)
            assert latest["phase"] == "train"

        # Second summary
        await tool.execute({"phase": "eval", "metrics": {"cv": 0.81}})
        with open(log_path) as f:
            assert len(f.readlines()) == 2

"""
Test NotebookTool with real jupyter_client kernel execution
"""
import os
import json
import tempfile
import shutil
from pathlib import Path

import pytest

from agent_v6.tools.notebook import NotebookTool
from agent_v6.workspace import IDEWorkspace


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for testing"""
    temp_dir = tempfile.mkdtemp(prefix="test_notebook_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_notebook_create_and_execute(temp_workspace):
    """Test creating notebook and executing cells"""
    workspace = IDEWorkspace(temp_workspace)
    tool = NotebookTool(temp_workspace, workspace)

    # Create notebook with code cell
    notebook_path = "test.ipynb"
    abs_path = os.path.join(temp_workspace, notebook_path)

    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": "print('Hello from Jupyter!')",
                "execution_count": None,
                "outputs": []
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(abs_path, 'w') as f:
        json.dump(notebook, f)

    # Execute cell
    result = await tool.execute({
        "notebook_path": notebook_path,
        "cell_index": 0
    })

    # Verify execution
    assert not result["is_error"]
    assert "Hello from Jupyter!" in result["content"]

    # Verify workspace tracking
    nb_state = workspace.get_notebook_state(notebook_path)
    assert nb_state is not None
    assert nb_state.last_executed_cell == 0
    assert nb_state.execution_count == 1
    assert nb_state.kernel_status == "idle"

    # Cleanup
    await tool.cleanup()


@pytest.mark.asyncio
async def test_notebook_persistent_kernel(temp_workspace):
    """Test that variables persist across cell executions"""
    tool = NotebookTool(temp_workspace)

    notebook_path = "persistent.ipynb"
    abs_path = os.path.join(temp_workspace, notebook_path)

    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": "x = 42",
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "source": "print(f'x = {x}')",
                "execution_count": None,
                "outputs": []
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(abs_path, 'w') as f:
        json.dump(notebook, f)

    # Execute first cell (define variable)
    result1 = await tool.execute({
        "notebook_path": notebook_path,
        "cell_index": 0
    })

    assert not result1["is_error"]

    # Execute second cell (use variable)
    result2 = await tool.execute({
        "notebook_path": notebook_path,
        "cell_index": 1
    })

    assert not result2["is_error"]
    assert "x = 42" in result2["content"]

    # Cleanup
    await tool.cleanup()


@pytest.mark.asyncio
async def test_notebook_execute_all(temp_workspace):
    """Test executing all cells"""
    tool = NotebookTool(temp_workspace)

    notebook_path = "multi.ipynb"
    abs_path = os.path.join(temp_workspace, notebook_path)

    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": "a = 1",
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "source": "b = 2",
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "source": "print(a + b)",
                "execution_count": None,
                "outputs": []
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(abs_path, 'w') as f:
        json.dump(notebook, f)

    # Execute all cells
    result = await tool.execute({
        "notebook_path": notebook_path,
        "cell_index": -1
    })

    assert not result["is_error"]
    assert "3" in result["content"]

    # Cleanup
    await tool.cleanup()


@pytest.mark.asyncio
async def test_notebook_error_handling(temp_workspace):
    """Test error handling in cell execution"""
    tool = NotebookTool(temp_workspace)

    notebook_path = "error.ipynb"
    abs_path = os.path.join(temp_workspace, notebook_path)

    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": "raise ValueError('Test error')",
                "execution_count": None,
                "outputs": []
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(abs_path, 'w') as f:
        json.dump(notebook, f)

    # Execute cell with error
    result = await tool.execute({
        "notebook_path": notebook_path,
        "cell_index": 0
    })

    # Should not be is_error (execution succeeded, but cell raised error)
    assert not result["is_error"]
    assert "ERROR:" in result["content"]
    assert "ValueError" in result["content"]

    # Cleanup
    await tool.cleanup()


@pytest.mark.asyncio
async def test_notebook_missing_file(temp_workspace):
    """Test error when notebook doesn't exist"""
    tool = NotebookTool(temp_workspace)

    result = await tool.execute({
        "notebook_path": "nonexistent.ipynb",
        "cell_index": 0
    })

    assert result["is_error"]
    assert "not found" in result["content"]

    # Cleanup
    await tool.cleanup()


@pytest.mark.asyncio
async def test_notebook_create_if_missing(temp_workspace):
    """Test creating notebook if missing"""
    tool = NotebookTool(temp_workspace)

    notebook_path = "new.ipynb"

    result = await tool.execute({
        "notebook_path": notebook_path,
        "cell_index": 0,
        "create_if_missing": True
    })

    # Should fail (notebook created but has no cells)
    assert result["is_error"]
    assert "out of range" in result["content"]

    # But notebook should exist
    assert os.path.exists(os.path.join(temp_workspace, notebook_path))

    # Cleanup
    await tool.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

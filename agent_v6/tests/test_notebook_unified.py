"""
Test JupyterNotebook unified tool with all operations

Tests each operation methodically:
1. list_cells
2. read_cell
3. read_cells
4. insert_cell
5. delete_cell
6. overwrite_cell_source
7. execute_cell
8. insert_execute_code_cell
9. execute_ipython
"""
import os
import json
import tempfile
import shutil
from pathlib import Path

import pytest

from agent_v6.tools.notebook import JupyterNotebook
from agent_v6.workspace import IDEWorkspace


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for testing"""
    temp_dir = tempfile.mkdtemp(prefix="test_notebook_unified_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_notebook(temp_workspace):
    """Create a sample notebook with multiple cells for testing"""
    notebook_path = "sample.ipynb"
    abs_path = os.path.join(temp_workspace, notebook_path)

    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": "# Sample Notebook\n\nThis is a test notebook."
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "source": "x = 42\nprint('x =', x)",
                "outputs": []
            },
            {
                "cell_type": "code",
                "execution_count": 2,
                "metadata": {},
                "source": "y = x * 2\nprint('y =', y)",
                "outputs": []
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": "## Results\n\nThe computation is complete."
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(abs_path, 'w') as f:
        json.dump(notebook, f, indent=2)

    return notebook_path


# ========== TEST: list_cells ==========

@pytest.mark.asyncio
async def test_list_cells_basic(temp_workspace, sample_notebook):
    """Test list_cells operation shows all cells"""
    tool = JupyterNotebook(temp_workspace)

    result = await tool.execute({
        "operation": "list_cells",
        "notebook_path": sample_notebook
    })

    assert not result["is_error"]
    assert "4 cells" in result["content"]
    assert "Index" in result["content"]
    assert "Type" in result["content"]
    assert "markdown" in result["content"]
    assert "code" in result["content"]

    # Cleanup
    await tool.cleanup()


@pytest.mark.asyncio
async def test_list_cells_empty_notebook(temp_workspace):
    """Test list_cells on empty notebook"""
    tool = JupyterNotebook(temp_workspace)

    # Create empty notebook
    notebook_path = "empty.ipynb"
    abs_path = os.path.join(temp_workspace, notebook_path)
    notebook = {"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    with open(abs_path, 'w') as f:
        json.dump(notebook, f)

    result = await tool.execute({
        "operation": "list_cells",
        "notebook_path": notebook_path
    })

    assert not result["is_error"]
    assert "has no cells" in result["content"]

    await tool.cleanup()


@pytest.mark.asyncio
async def test_list_cells_missing_notebook(temp_workspace):
    """Test list_cells on non-existent notebook"""
    tool = JupyterNotebook(temp_workspace)

    result = await tool.execute({
        "operation": "list_cells",
        "notebook_path": "nonexistent.ipynb"
    })

    assert result["is_error"]
    assert "not found" in result["content"]

    await tool.cleanup()


# ========== TEST: read_cell ==========

@pytest.mark.asyncio
async def test_read_cell_basic(temp_workspace, sample_notebook):
    """Test read_cell operation reads specific cell"""
    tool = JupyterNotebook(temp_workspace)

    # Read code cell (index 1)
    result = await tool.execute({
        "operation": "read_cell",
        "notebook_path": sample_notebook,
        "cell_index": 1
    })

    assert not result["is_error"]
    content = json.loads(result["content"])
    assert content["index"] == 1
    assert content["type"] == "code"
    assert "x = 42" in content["source"]
    assert content["execution_count"] == 1

    await tool.cleanup()


@pytest.mark.asyncio
async def test_read_cell_markdown(temp_workspace, sample_notebook):
    """Test read_cell reads markdown cell"""
    tool = JupyterNotebook(temp_workspace)

    # Read markdown cell (index 0)
    result = await tool.execute({
        "operation": "read_cell",
        "notebook_path": sample_notebook,
        "cell_index": 0
    })

    assert not result["is_error"]
    content = json.loads(result["content"])
    assert content["index"] == 0
    assert content["type"] == "markdown"
    assert "Sample Notebook" in content["source"]
    assert "execution_count" not in content  # Markdown cells don't have execution count

    await tool.cleanup()


@pytest.mark.asyncio
async def test_read_cell_out_of_range(temp_workspace, sample_notebook):
    """Test read_cell with invalid index"""
    tool = JupyterNotebook(temp_workspace)

    result = await tool.execute({
        "operation": "read_cell",
        "notebook_path": sample_notebook,
        "cell_index": 999
    })

    assert result["is_error"]
    assert "out of range" in result["content"]

    await tool.cleanup()


@pytest.mark.asyncio
async def test_read_cell_missing_index(temp_workspace, sample_notebook):
    """Test read_cell without cell_index parameter"""
    tool = JupyterNotebook(temp_workspace)

    result = await tool.execute({
        "operation": "read_cell",
        "notebook_path": sample_notebook
    })

    assert result["is_error"]
    assert "cell_index is required" in result["content"]

    await tool.cleanup()


# ========== TEST: read_cells ==========

@pytest.mark.asyncio
async def test_read_cells_basic(temp_workspace, sample_notebook):
    """Test read_cells operation reads all cells"""
    tool = JupyterNotebook(temp_workspace)

    result = await tool.execute({
        "operation": "read_cells",
        "notebook_path": sample_notebook
    })

    assert not result["is_error"]
    cells = json.loads(result["content"])
    assert len(cells) == 4
    assert cells[0]["type"] == "markdown"
    assert cells[1]["type"] == "code"
    assert "x = 42" in cells[1]["source"]

    await tool.cleanup()


# ========== TEST: insert_cell ==========

@pytest.mark.asyncio
async def test_insert_cell_code_at_beginning(temp_workspace, sample_notebook):
    """Test inserting code cell at beginning"""
    tool = JupyterNotebook(temp_workspace)

    result = await tool.execute({
        "operation": "insert_cell",
        "notebook_path": sample_notebook,
        "cell_index": 0,
        "cell_type": "code",
        "cell_source": "# New first cell\nz = 100"
    })

    assert not result["is_error"]
    assert "Inserted code cell at index 0" in result["content"]
    assert "Context:" in result["content"]

    # Verify insertion
    list_result = await tool.execute({
        "operation": "list_cells",
        "notebook_path": sample_notebook
    })
    assert "5 cells" in list_result["content"]

    await tool.cleanup()


@pytest.mark.asyncio
async def test_insert_cell_markdown_at_end(temp_workspace, sample_notebook):
    """Test appending markdown cell"""
    tool = JupyterNotebook(temp_workspace)

    result = await tool.execute({
        "operation": "insert_cell",
        "notebook_path": sample_notebook,
        "cell_index": -1,
        "cell_type": "markdown",
        "cell_source": "## Conclusion\nThe end."
    })

    assert not result["is_error"]
    assert "Inserted markdown cell" in result["content"]

    await tool.cleanup()


@pytest.mark.asyncio
async def test_insert_cell_creates_notebook(temp_workspace):
    """Test insert_cell creates notebook if missing"""
    tool = JupyterNotebook(temp_workspace)

    result = await tool.execute({
        "operation": "insert_cell",
        "notebook_path": "new.ipynb",
        "cell_index": -1,
        "cell_type": "code",
        "cell_source": "print('First cell')"
    })

    assert not result["is_error"]
    assert os.path.exists(os.path.join(temp_workspace, "new.ipynb"))

    await tool.cleanup()


# ========== TEST: delete_cell ==========

@pytest.mark.asyncio
async def test_delete_cell_basic(temp_workspace, sample_notebook):
    """Test deleting a cell"""
    tool = JupyterNotebook(temp_workspace)

    # Delete cell at index 1
    result = await tool.execute({
        "operation": "delete_cell",
        "notebook_path": sample_notebook,
        "cell_index": 1
    })

    assert not result["is_error"]
    assert "Deleted code cell at index 1" in result["content"]
    assert "x = 42" in result["content"]

    # Verify deletion
    list_result = await tool.execute({
        "operation": "list_cells",
        "notebook_path": sample_notebook
    })
    assert "3 cells" in list_result["content"]

    await tool.cleanup()


# ========== TEST: overwrite_cell_source ==========

@pytest.mark.asyncio
async def test_overwrite_cell_source_basic(temp_workspace, sample_notebook):
    """Test overwriting cell source"""
    tool = JupyterNotebook(temp_workspace)

    result = await tool.execute({
        "operation": "overwrite_cell_source",
        "notebook_path": sample_notebook,
        "cell_index": 1,
        "cell_source": "x = 100\nprint('x is now', x)"
    })

    assert not result["is_error"]
    assert "Overwrote cell source" in result["content"]
    assert "OLD" in result["content"]
    assert "NEW" in result["content"]

    # Verify change
    read_result = await tool.execute({
        "operation": "read_cell",
        "notebook_path": sample_notebook,
        "cell_index": 1
    })
    content = json.loads(read_result["content"])
    assert "x = 100" in content["source"]

    await tool.cleanup()


# ========== TEST: execute_cell ==========

@pytest.mark.asyncio
async def test_execute_cell_basic(temp_workspace, sample_notebook):
    """Test executing a code cell"""
    tool = JupyterNotebook(temp_workspace)

    # Execute cell that defines x
    result = await tool.execute({
        "operation": "execute_cell",
        "notebook_path": sample_notebook,
        "cell_index": 1
    })

    assert not result["is_error"]
    assert "x = 42" in result["content"]

    await tool.cleanup()


@pytest.mark.asyncio
async def test_execute_cell_persistent_kernel(temp_workspace):
    """Test that kernel persists across executions"""
    tool = JupyterNotebook(temp_workspace)

    # Create notebook with two cells
    notebook_path = "persist.ipynb"
    abs_path = os.path.join(temp_workspace, notebook_path)
    notebook = {
        "cells": [
            {"cell_type": "code", "source": "a = 123", "execution_count": None, "metadata": {}, "outputs": []},
            {"cell_type": "code", "source": "print('a =', a)", "execution_count": None, "metadata": {}, "outputs": []}
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(abs_path, 'w') as f:
        json.dump(notebook, f)

    # Execute first cell
    await tool.execute({
        "operation": "execute_cell",
        "notebook_path": notebook_path,
        "cell_index": 0
    })

    # Execute second cell (should see variable from first)
    result = await tool.execute({
        "operation": "execute_cell",
        "notebook_path": notebook_path,
        "cell_index": 1
    })

    assert not result["is_error"]
    assert "a = 123" in result["content"]

    await tool.cleanup()


@pytest.mark.asyncio
async def test_execute_cell_not_code(temp_workspace, sample_notebook):
    """Test executing non-code cell fails gracefully"""
    tool = JupyterNotebook(temp_workspace)

    result = await tool.execute({
        "operation": "execute_cell",
        "notebook_path": sample_notebook,
        "cell_index": 0  # markdown cell
    })

    assert result["is_error"]
    assert "not a code cell" in result["content"]

    await tool.cleanup()


# ========== TEST: insert_execute_code_cell ==========

@pytest.mark.asyncio
async def test_insert_execute_code_cell_basic(temp_workspace, sample_notebook):
    """Test inserting and executing a code cell"""
    tool = JupyterNotebook(temp_workspace)

    result = await tool.execute({
        "operation": "insert_execute_code_cell",
        "notebook_path": sample_notebook,
        "cell_index": -1,
        "cell_source": "result = 2 + 2\nprint('Result:', result)"
    })

    assert not result["is_error"]
    assert "Inserted and executed code cell" in result["content"]
    assert "Result: 4" in result["content"]

    await tool.cleanup()


# ========== TEST: execute_ipython ==========

@pytest.mark.asyncio
async def test_execute_ipython_basic(temp_workspace, sample_notebook):
    """Test executing IPython code directly"""
    tool = JupyterNotebook(temp_workspace)

    result = await tool.execute({
        "operation": "execute_ipython",
        "notebook_path": sample_notebook,
        "code": "import sys\nprint(f'Python {sys.version_info.major}.{sys.version_info.minor}')"
    })

    assert not result["is_error"]
    assert "Python 3." in result["content"]

    await tool.cleanup()


@pytest.mark.asyncio
async def test_execute_ipython_magic_command(temp_workspace, sample_notebook):
    """Test IPython magic commands"""
    tool = JupyterNotebook(temp_workspace)

    result = await tool.execute({
        "operation": "execute_ipython",
        "notebook_path": sample_notebook,
        "code": "%pwd"
    })

    assert not result["is_error"]
    # Should return current working directory

    await tool.cleanup()


@pytest.mark.asyncio
async def test_execute_ipython_error_handling(temp_workspace, sample_notebook):
    """Test IPython handles errors gracefully"""
    tool = JupyterNotebook(temp_workspace)

    result = await tool.execute({
        "operation": "execute_ipython",
        "notebook_path": sample_notebook,
        "code": "raise ValueError('Test error')"
    })

    # Should not be is_error (execution succeeded, code raised error)
    assert not result["is_error"]
    assert "ERROR:" in result["content"]
    assert "ValueError" in result["content"]

    await tool.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

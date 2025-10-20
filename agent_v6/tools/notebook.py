"""
NotebookTool - Jupyter notebook execution with jupyter_client

Provides first-class notebook support:
- Execute cells in persistent kernel
- Read cell outputs
- Monitor kernel status
- Handle kernel lifecycle
"""
import os
import json
import asyncio
from pathlib import Path
from typing import Dict, Optional

from agent_v5.tools.base import BaseTool


class NotebookTool(BaseTool):
    """
    Execute code in Jupyter notebook with persistent kernel

    Uses jupyter_client for kernel management.
    Each notebook gets its own kernel that persists across cells.
    """

    def __init__(self, workspace_dir: str, workspace_state=None):
        """
        Initialize notebook tool

        Args:
            workspace_dir: Workspace directory
            workspace_state: Optional IDEWorkspace instance for state tracking
        """
        super().__init__(workspace_dir)
        self.workspace_state = workspace_state
        self.kernels: Dict[str, any] = {}  # notebook_path -> KernelManager

    @property
    def name(self) -> str:
        return "ExecuteNotebookCell"

    @property
    def schema(self) -> Dict:
        return {
            "name": "ExecuteNotebookCell",
            "description": (
                "Execute code cell in Jupyter notebook with persistent kernel. "
                "Each notebook maintains its own kernel session across multiple cell executions. "
                "Use this for iterative data analysis, experimentation, and visualization."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "notebook_path": {
                        "type": "string",
                        "description": "Relative path to notebook file (e.g., 'analysis.ipynb')"
                    },
                    "cell_index": {
                        "type": "integer",
                        "description": "Zero-based index of cell to execute (or -1 to execute all cells)"
                    },
                    "create_if_missing": {
                        "type": "boolean",
                        "description": "Create notebook if it doesn't exist (default: false)"
                    }
                },
                "required": ["notebook_path", "cell_index"]
            }
        }

    async def execute(self, input: Dict) -> Dict:
        """Execute notebook cell"""
        try:
            notebook_path = input["notebook_path"]
            cell_index = input["cell_index"]
            create_if_missing = input.get("create_if_missing", False)

            # Resolve path
            abs_path = os.path.join(self.workspace_dir, notebook_path)

            # Create notebook if requested
            if not os.path.exists(abs_path) and create_if_missing:
                self._create_notebook(abs_path)

            # Validate notebook exists
            if not os.path.exists(abs_path):
                return {
                    "content": f"Notebook not found: {notebook_path}",
                    "is_error": True
                }

            # Get or create kernel for this notebook
            kernel_manager = await self._get_or_create_kernel(notebook_path)

            # Read notebook
            with open(abs_path, 'r') as f:
                notebook = json.load(f)

            cells = notebook.get("cells", [])

            if cell_index == -1:
                # Execute all cells
                results = []
                for i, cell in enumerate(cells):
                    if cell.get("cell_type") == "code":
                        result = await self._execute_cell(
                            kernel_manager,
                            cell,
                            i
                        )
                        results.append(f"Cell {i}:\n{result}")

                        # Track execution
                        if self.workspace_state:
                            self.workspace_state.track_notebook_execution(notebook_path, i)

                output = "\n\n".join(results)
            else:
                # Execute single cell
                if cell_index < 0 or cell_index >= len(cells):
                    return {
                        "content": f"Cell index {cell_index} out of range (0-{len(cells)-1})",
                        "is_error": True
                    }

                cell = cells[cell_index]

                if cell.get("cell_type") != "code":
                    return {
                        "content": f"Cell {cell_index} is not a code cell (type: {cell.get('cell_type')})",
                        "is_error": True
                    }

                output = await self._execute_cell(kernel_manager, cell, cell_index)

                # Track execution
                if self.workspace_state:
                    self.workspace_state.track_notebook_execution(notebook_path, cell_index)

            return {
                "content": output,
                "is_error": False,
                "debug_summary": f"Executed notebook cell(s) in {notebook_path}"
            }

        except Exception as e:
            return {
                "content": f"Notebook execution error: {str(e)}",
                "is_error": True
            }

    async def _get_or_create_kernel(self, notebook_path: str):
        """Get existing kernel or create new one for notebook"""
        if notebook_path in self.kernels:
            return self.kernels[notebook_path]

        # Import jupyter_client here (only when needed)
        from jupyter_client import AsyncKernelManager

        # Create new kernel
        km = AsyncKernelManager()
        await km.start_kernel()

        # Wait for kernel to be ready
        kc = km.client()
        kc.start_channels()
        await kc.wait_for_ready(timeout=30)

        self.kernels[notebook_path] = {"manager": km, "client": kc}

        # Track kernel state
        if self.workspace_state:
            self.workspace_state.track_notebook_kernel(
                notebook_path,
                km.kernel_id,
                "idle"
            )

        return self.kernels[notebook_path]

    async def _execute_cell(self, kernel_info: Dict, cell: Dict, cell_index: int) -> str:
        """Execute a single cell and return output"""
        kc = kernel_info["client"]

        # Get source code
        source = cell.get("source", [])
        if isinstance(source, list):
            code = "".join(source)
        else:
            code = source

        if not code.strip():
            return "(empty cell)"

        # Execute code
        msg_id = kc.execute(code)

        # Collect outputs
        outputs = []
        errors = []

        while True:
            try:
                msg = await asyncio.wait_for(
                    kc.get_iopub_msg(timeout=1),
                    timeout=60.0
                )
            except asyncio.TimeoutError:
                break

            msg_type = msg['header']['msg_type']
            content = msg['content']

            if msg_type == 'stream':
                outputs.append(content['text'])

            elif msg_type == 'execute_result':
                data = content.get('data', {})
                if 'text/plain' in data:
                    outputs.append(data['text/plain'])

            elif msg_type == 'display_data':
                data = content.get('data', {})
                if 'text/plain' in data:
                    outputs.append(data['text/plain'])

            elif msg_type == 'error':
                errors.append(f"{content['ename']}: {content['evalue']}")

            elif msg_type == 'status' and content['execution_state'] == 'idle':
                break

        # Format output
        if errors:
            return f"ERROR:\n{''.join(errors)}"
        elif outputs:
            return ''.join(outputs).strip()
        else:
            return "(no output)"

    def _create_notebook(self, path: str) -> None:
        """Create empty notebook file"""
        notebook = {
            "cells": [],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.11.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 5
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(notebook, f, indent=2)

    async def cleanup(self) -> None:
        """Shutdown all kernels"""
        for notebook_path, kernel_info in self.kernels.items():
            try:
                km = kernel_info["manager"]
                kc = kernel_info["client"]

                kc.stop_channels()
                await km.shutdown_kernel()

                # Track kernel stopped
                if self.workspace_state:
                    self.workspace_state.track_notebook_kernel(
                        notebook_path,
                        km.kernel_id,
                        "stopped"
                    )
            except:
                pass

        self.kernels.clear()

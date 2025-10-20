"""
JupyterNotebook - Unified notebook tool with multiple operations

Provides comprehensive notebook management:
- Cell manipulation (insert, delete, overwrite, read)
- Code execution (execute cells, IPython code)
- Notebook inspection (list cells, read cells)
"""
import os
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any

from agent_v5.tools.base import BaseTool


class JupyterNotebook(BaseTool):
    """
    Unified Jupyter notebook tool with multiple operations

    Supports:
    - list_cells: List basic info of all cells
    - read_cell: Read a specific cell
    - read_cells: Read all cells
    - insert_cell: Insert a cell at specified position
    - delete_cell: Delete a specific cell
    - overwrite_cell_source: Overwrite source of existing cell
    - execute_cell: Execute a cell with timeout and streaming
    - insert_execute_code_cell: Insert and execute a code cell
    - execute_ipython: Execute IPython code directly in kernel
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
        self.kernels: Dict[str, Any] = {}  # notebook_path -> {"manager": KernelManager, "client": KernelClient}

    @property
    def name(self) -> str:
        return "JupyterNotebook"

    @property
    def schema(self) -> Dict:
        return {
            "name": "JupyterNotebook",
            "description": (
                "Unified Jupyter notebook tool with multiple operations for cell manipulation, "
                "code execution, and notebook inspection. Each call specifies an 'operation' parameter "
                "to determine which action to perform."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "list_cells",
                            "read_cell",
                            "read_cells",
                            "insert_cell",
                            "delete_cell",
                            "overwrite_cell_source",
                            "execute_cell",
                            "insert_execute_code_cell",
                            "execute_ipython"
                        ],
                        "description": "Operation to perform on the notebook"
                    },
                    "notebook_path": {
                        "type": "string",
                        "description": "Relative path to notebook file (e.g., 'analysis.ipynb')"
                    },
                    "cell_index": {
                        "type": "integer",
                        "description": "Zero-based index of cell (use -1 for append operations)"
                    },
                    "cell_type": {
                        "type": "string",
                        "enum": ["code", "markdown"],
                        "description": "Type of cell (for insert operations)"
                    },
                    "cell_source": {
                        "type": "string",
                        "description": "Source content for the cell (for insert/overwrite operations)"
                    },
                    "code": {
                        "type": "string",
                        "description": "IPython code to execute (for execute_ipython operation)"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Execution timeout in seconds (default: 60)"
                    },
                    "stream": {
                        "type": "boolean",
                        "description": "Enable streaming progress updates for long-running cells (default: false)"
                    },
                    "progress_interval": {
                        "type": "integer",
                        "description": "Seconds between progress updates when stream=true (default: 5)"
                    }
                },
                "required": ["operation", "notebook_path"]
            }
        }

    async def execute(self, input: Dict) -> Dict:
        """Execute the requested notebook operation"""
        operation = input["operation"]

        # Dispatch to appropriate handler
        handlers = {
            "list_cells": self._list_cells,
            "read_cell": self._read_cell,
            "read_cells": self._read_cells,
            "insert_cell": self._insert_cell,
            "delete_cell": self._delete_cell,
            "overwrite_cell_source": self._overwrite_cell_source,
            "execute_cell": self._execute_cell_op,
            "insert_execute_code_cell": self._insert_execute_code_cell,
            "execute_ipython": self._execute_ipython
        }

        handler = handlers.get(operation)
        if not handler:
            return {
                "content": f"Unknown operation: {operation}",
                "is_error": True
            }

        try:
            return await handler(input)
        except Exception as e:
            return {
                "content": f"Error in {operation}: {str(e)}",
                "is_error": True
            }

    # ========== HELPER METHODS ==========

    def _resolve_notebook_path(self, notebook_path: str) -> str:
        """Resolve relative notebook path to absolute path"""
        return os.path.join(self.workspace_dir, notebook_path)

    def _read_notebook(self, abs_path: str) -> Dict:
        """Read notebook JSON from file"""
        with open(abs_path, 'r') as f:
            return json.load(f)

    def _write_notebook(self, abs_path: str, notebook: Dict) -> None:
        """Write notebook JSON to file"""
        with open(abs_path, 'w') as f:
            json.dump(notebook, f, indent=2)

    def _validate_notebook_exists(self, abs_path: str, notebook_path: str) -> Optional[Dict]:
        """
        Validate notebook exists

        Returns:
            None if valid, error dict if invalid
        """
        if not os.path.exists(abs_path):
            return {
                "content": f"Notebook not found: {notebook_path}",
                "is_error": True
            }
        return None

    def _validate_cell_index(self, cells: List[Dict], cell_index: int, operation: str) -> Optional[Dict]:
        """
        Validate cell index is in range

        Returns:
            None if valid, error dict if invalid
        """
        if cell_index < 0 or cell_index >= len(cells):
            return {
                "content": f"Cell index {cell_index} out of range (0-{len(cells)-1}) for {operation}",
                "is_error": True
            }
        return None

    # ========== OPERATION: list_cells ==========

    async def _list_cells(self, input: Dict) -> Dict:
        """
        List basic information of all cells in the notebook

        Returns a formatted table showing index, type, execution count, and first line
        """
        notebook_path = input["notebook_path"]
        abs_path = self._resolve_notebook_path(notebook_path)

        # Validate notebook exists
        error = self._validate_notebook_exists(abs_path, notebook_path)
        if error:
            return error

        # Read notebook
        notebook = self._read_notebook(abs_path)
        cells = notebook.get("cells", [])

        if not cells:
            return {
                "content": f"Notebook {notebook_path} has no cells",
                "is_error": False
            }

        # Build table
        lines = []
        lines.append("=" * 80)
        lines.append(f"Notebook: {notebook_path} ({len(cells)} cells)")
        lines.append("=" * 80)
        lines.append(f"{'Index':<8} {'Type':<10} {'Count':<8} {'First Line':<50}")
        lines.append("-" * 80)

        for idx, cell in enumerate(cells):
            cell_type = cell.get("cell_type", "unknown")

            # Get execution count for code cells
            exec_count = ""
            if cell_type == "code":
                count = cell.get("execution_count")
                exec_count = str(count) if count is not None else "-"

            # Get first line of source
            source = cell.get("source", "")
            if isinstance(source, list):
                first_line = source[0] if source else ""
            else:
                first_line = source.split('\n')[0] if source else ""

            # Truncate first line if too long
            first_line = first_line.strip()
            if len(first_line) > 50:
                first_line = first_line[:47] + "..."

            lines.append(f"{idx:<8} {cell_type:<10} {exec_count:<8} {first_line:<50}")

        lines.append("=" * 80)

        return {
            "content": "\n".join(lines),
            "is_error": False,
            "debug_summary": f"Listed {len(cells)} cells in {notebook_path}"
        }

    # ========== OPERATION: read_cell ==========

    async def _read_cell(self, input: Dict) -> Dict:
        """Read a specific cell from the notebook"""
        notebook_path = input["notebook_path"]
        cell_index = input.get("cell_index")

        if cell_index is None:
            return {
                "content": "cell_index is required for read_cell operation",
                "is_error": True
            }

        abs_path = self._resolve_notebook_path(notebook_path)

        # Validate notebook exists
        error = self._validate_notebook_exists(abs_path, notebook_path)
        if error:
            return error

        # Read notebook
        notebook = self._read_notebook(abs_path)
        cells = notebook.get("cells", [])

        # Validate cell index
        error = self._validate_cell_index(cells, cell_index, "read_cell")
        if error:
            return error

        # Get cell
        cell = cells[cell_index]

        # Format cell info
        cell_info = {
            "index": cell_index,
            "type": cell.get("cell_type"),
            "source": cell.get("source"),
        }

        # Add outputs for code cells
        if cell.get("cell_type") == "code":
            cell_info["execution_count"] = cell.get("execution_count")
            cell_info["outputs"] = cell.get("outputs", [])

        return {
            "content": json.dumps(cell_info, indent=2),
            "is_error": False,
            "debug_summary": f"Read cell {cell_index} from {notebook_path}"
        }

    # ========== OPERATION: read_cells ==========

    async def _read_cells(self, input: Dict) -> Dict:
        """Read all cells from the notebook"""
        notebook_path = input["notebook_path"]
        abs_path = self._resolve_notebook_path(notebook_path)

        # Validate notebook exists
        error = self._validate_notebook_exists(abs_path, notebook_path)
        if error:
            return error

        # Read notebook
        notebook = self._read_notebook(abs_path)
        cells = notebook.get("cells", [])

        # Format all cells
        cells_info = []
        for idx, cell in enumerate(cells):
            cell_info = {
                "index": idx,
                "type": cell.get("cell_type"),
                "source": cell.get("source"),
            }

            # Add outputs for code cells
            if cell.get("cell_type") == "code":
                cell_info["execution_count"] = cell.get("execution_count")
                cell_info["outputs"] = cell.get("outputs", [])

            cells_info.append(cell_info)

        return {
            "content": json.dumps(cells_info, indent=2),
            "is_error": False,
            "debug_summary": f"Read {len(cells)} cells from {notebook_path}"
        }

    # ========== OPERATION: insert_cell ==========

    async def _insert_cell(self, input: Dict) -> Dict:
        """Insert a cell at specified position"""
        notebook_path = input["notebook_path"]
        cell_index = input.get("cell_index")
        cell_type = input.get("cell_type")
        cell_source = input.get("cell_source", "")

        # Validate required params
        if cell_index is None:
            return {"content": "cell_index is required for insert_cell", "is_error": True}
        if not cell_type:
            return {"content": "cell_type is required for insert_cell", "is_error": True}
        if cell_type not in ["code", "markdown"]:
            return {"content": f"Invalid cell_type: {cell_type} (must be 'code' or 'markdown')", "is_error": True}

        abs_path = self._resolve_notebook_path(notebook_path)

        # Create notebook if doesn't exist
        if not os.path.exists(abs_path):
            self._create_empty_notebook(abs_path)

        # Read notebook
        notebook = self._read_notebook(abs_path)
        cells = notebook.get("cells", [])

        # Handle -1 (append)
        if cell_index == -1:
            cell_index = len(cells)

        # Validate insert position
        if cell_index < 0 or cell_index > len(cells):
            return {
                "content": f"Insert index {cell_index} out of range (0-{len(cells)})",
                "is_error": True
            }

        # Create new cell
        new_cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": cell_source
        }

        if cell_type == "code":
            new_cell["execution_count"] = None
            new_cell["outputs"] = []

        # Insert cell
        cells.insert(cell_index, new_cell)
        notebook["cells"] = cells

        # Write back
        self._write_notebook(abs_path, notebook)

        # Build context (5 cells above and below)
        context_start = max(0, cell_index - 5)
        context_end = min(len(cells), cell_index + 6)
        context_cells = []

        for idx in range(context_start, context_end):
            marker = ">>> " if idx == cell_index else "    "
            cell = cells[idx]
            source_preview = str(cell.get("source", ""))[:50].replace('\n', ' ')
            context_cells.append(f"{marker}[{idx}] {cell.get('cell_type')}: {source_preview}...")

        context = "\n".join(context_cells)

        return {
            "content": f"✓ Inserted {cell_type} cell at index {cell_index}\n\nContext:\n{context}",
            "is_error": False,
            "debug_summary": f"Inserted {cell_type} cell at {cell_index} in {notebook_path}"
        }

    # ========== OPERATION: delete_cell ==========

    async def _delete_cell(self, input: Dict) -> Dict:
        """Delete a specific cell from the notebook"""
        notebook_path = input["notebook_path"]
        cell_index = input.get("cell_index")

        if cell_index is None:
            return {"content": "cell_index is required for delete_cell", "is_error": True}

        abs_path = self._resolve_notebook_path(notebook_path)

        # Validate notebook exists
        error = self._validate_notebook_exists(abs_path, notebook_path)
        if error:
            return error

        # Read notebook
        notebook = self._read_notebook(abs_path)
        cells = notebook.get("cells", [])

        # Validate cell index
        error = self._validate_cell_index(cells, cell_index, "delete_cell")
        if error:
            return error

        # Get cell info before deleting
        deleted_cell = cells[cell_index]
        cell_type = deleted_cell.get("cell_type")
        source_preview = str(deleted_cell.get("source", ""))[:100].replace('\n', ' ')

        # Delete cell
        cells.pop(cell_index)
        notebook["cells"] = cells

        # Write back
        self._write_notebook(abs_path, notebook)

        return {
            "content": f"✓ Deleted {cell_type} cell at index {cell_index}\n\nDeleted content preview:\n{source_preview}...",
            "is_error": False,
            "debug_summary": f"Deleted cell {cell_index} from {notebook_path}"
        }

    # ========== OPERATION: overwrite_cell_source ==========

    async def _overwrite_cell_source(self, input: Dict) -> Dict:
        """Overwrite the source of an existing cell"""
        notebook_path = input["notebook_path"]
        cell_index = input.get("cell_index")
        cell_source = input.get("cell_source")

        if cell_index is None:
            return {"content": "cell_index is required for overwrite_cell_source", "is_error": True}
        if cell_source is None:
            return {"content": "cell_source is required for overwrite_cell_source", "is_error": True}

        abs_path = self._resolve_notebook_path(notebook_path)

        # Validate notebook exists
        error = self._validate_notebook_exists(abs_path, notebook_path)
        if error:
            return error

        # Read notebook
        notebook = self._read_notebook(abs_path)
        cells = notebook.get("cells", [])

        # Validate cell index
        error = self._validate_cell_index(cells, cell_index, "overwrite_cell_source")
        if error:
            return error

        # Get old source for diff
        cell = cells[cell_index]
        old_source = cell.get("source", "")
        if isinstance(old_source, list):
            old_source = "".join(old_source)

        # Overwrite source
        cell["source"] = cell_source

        # Write back
        self._write_notebook(abs_path, notebook)

        # Create diff
        diff_lines = []
        diff_lines.append(f"Cell {cell_index} ({cell.get('cell_type')}):")
        diff_lines.append("--- OLD")
        diff_lines.append(old_source[:200] if len(old_source) > 200 else old_source)
        diff_lines.append("+++ NEW")
        diff_lines.append(cell_source[:200] if len(cell_source) > 200 else cell_source)

        return {
            "content": "✓ Overwrote cell source\n\n" + "\n".join(diff_lines),
            "is_error": False,
            "debug_summary": f"Overwrote cell {cell_index} in {notebook_path}"
        }

    # ========== OPERATION: execute_cell ==========

    async def _execute_cell_op(self, input: Dict) -> Dict:
        """Execute a cell with configurable timeout and optional streaming"""
        notebook_path = input["notebook_path"]
        cell_index = input.get("cell_index")
        timeout_seconds = input.get("timeout", 300)
        stream = input.get("stream", False)
        progress_interval = input.get("progress_interval", 5)

        if cell_index is None:
            return {"content": "cell_index is required for execute_cell", "is_error": True}

        abs_path = self._resolve_notebook_path(notebook_path)

        # Validate notebook exists
        error = self._validate_notebook_exists(abs_path, notebook_path)
        if error:
            return error

        # Read notebook
        notebook = self._read_notebook(abs_path)
        cells = notebook.get("cells", [])

        # Validate cell index
        error = self._validate_cell_index(cells, cell_index, "execute_cell")
        if error:
            return error

        cell = cells[cell_index]

        # Validate it's a code cell
        if cell.get("cell_type") != "code":
            return {
                "content": f"Cell {cell_index} is not a code cell (type: {cell.get('cell_type')})",
                "is_error": True
            }

        # Get or create kernel
        kernel_info = await self._get_or_create_kernel(notebook_path)

        # Execute cell
        output = await self._execute_cell_code(
            kernel_info,
            cell,
            cell_index,
            timeout_seconds,
            stream,
            progress_interval
        )

        # Track execution
        if self.workspace_state:
            self.workspace_state.track_notebook_execution(notebook_path, cell_index)

        return {
            "content": output,
            "is_error": False,
            "debug_summary": f"Executed cell {cell_index} in {notebook_path}"
        }

    # ========== OPERATION: insert_execute_code_cell ==========

    async def _insert_execute_code_cell(self, input: Dict) -> Dict:
        """Insert and execute a code cell"""
        notebook_path = input["notebook_path"]
        cell_index = input.get("cell_index")
        cell_source = input.get("cell_source")

        if cell_index is None:
            return {"content": "cell_index is required for insert_execute_code_cell", "is_error": True}
        if not cell_source:
            return {"content": "cell_source is required for insert_execute_code_cell", "is_error": True}

        # First insert the cell
        insert_result = await self._insert_cell({
            "notebook_path": notebook_path,
            "cell_index": cell_index,
            "cell_type": "code",
            "cell_source": cell_source
        })

        if insert_result["is_error"]:
            return insert_result

        # Determine actual index after insert
        if cell_index == -1:
            # Read notebook to get actual index
            abs_path = self._resolve_notebook_path(notebook_path)
            notebook = self._read_notebook(abs_path)
            actual_index = len(notebook.get("cells", [])) - 1
        else:
            actual_index = cell_index

        # Execute the cell
        exec_result = await self._execute_cell_op({
            "notebook_path": notebook_path,
            "cell_index": actual_index,
            "timeout": input.get("timeout", 300)
        })

        if exec_result["is_error"]:
            return {
                "content": f"Cell inserted but execution failed:\n{exec_result['content']}",
                "is_error": True
            }

        return {
            "content": f"✓ Inserted and executed code cell at index {actual_index}\n\nOutput:\n{exec_result['content']}",
            "is_error": False,
            "debug_summary": f"Inserted and executed cell at {actual_index} in {notebook_path}"
        }

    # ========== OPERATION: execute_ipython ==========

    async def _execute_ipython(self, input: Dict) -> Dict:
        """Execute IPython code directly in the kernel"""
        notebook_path = input["notebook_path"]
        code = input.get("code")
        timeout = input.get("timeout", 60)

        if not code:
            return {"content": "code is required for execute_ipython", "is_error": True}

        # Get or create kernel for this notebook
        kernel_info = await self._get_or_create_kernel(notebook_path)

        # Execute code directly
        kc = kernel_info["client"]

        # Execute code
        msg_id = kc.execute(code)

        # Collect outputs
        outputs = []
        errors = []

        try:
            timeout_at = asyncio.get_event_loop().time() + timeout

            while True:
                remaining = timeout_at - asyncio.get_event_loop().time()
                if remaining <= 0:
                    return {
                        "content": f"Execution timed out after {timeout}s",
                        "is_error": True
                    }

                try:
                    msg = await asyncio.wait_for(
                        kc.get_iopub_msg(timeout=1),
                        timeout=min(remaining, 1)
                    )
                except asyncio.TimeoutError:
                    continue

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

        except Exception as e:
            return {
                "content": f"Execution error: {str(e)}",
                "is_error": True
            }

        # Format output
        if errors:
            return {
                "content": f"ERROR:\n{''.join(errors)}",
                "is_error": False  # Not a tool error, just code error
            }
        elif outputs:
            return {
                "content": ''.join(outputs).strip(),
                "is_error": False
            }
        else:
            return {
                "content": "(no output)",
                "is_error": False
            }

    # ========== KERNEL MANAGEMENT ==========

    async def _get_or_create_kernel(self, notebook_path: str) -> Dict:
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

    async def _execute_cell_code(
        self,
        kernel_info: Dict,
        cell: Dict,
        cell_index: int,
        timeout_seconds: int,
        stream: bool,
        progress_interval: int
    ) -> str:
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

        timeout_at = asyncio.get_event_loop().time() + timeout_seconds

        while True:
            remaining = timeout_at - asyncio.get_event_loop().time()
            if remaining <= 0:
                return f"ERROR: Execution timed out after {timeout_seconds}s"

            try:
                msg = await asyncio.wait_for(
                    kc.get_iopub_msg(timeout=1),
                    timeout=min(remaining, 1)
                )
            except asyncio.TimeoutError:
                # Check if we should send progress update
                if stream:
                    elapsed = timeout_seconds - remaining
                    if int(elapsed) % progress_interval == 0:
                        outputs.append(f"[Still running... {int(elapsed)}s elapsed]\n")
                continue

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

    def _create_empty_notebook(self, path: str) -> None:
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
        self._write_notebook(path, notebook)

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

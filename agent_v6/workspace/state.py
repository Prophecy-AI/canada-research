"""
IDEWorkspace - State tracking for IDE-based autonomous agent

Tracks workspace state across:
- Files: created, modified, deleted
- Notebooks: cells, outputs, kernel state
- Processes: running scripts, resource usage, convergence detection
"""
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field


@dataclass
class FileState:
    """Track state of a single file"""
    path: str
    created_at: float
    modified_at: float
    size_bytes: int
    exists: bool = True


@dataclass
class NotebookState:
    """Track state of a Jupyter notebook"""
    path: str
    kernel_id: Optional[str] = None
    kernel_status: str = "stopped"  # stopped, starting, idle, busy
    num_cells: int = 0
    last_executed_cell: Optional[int] = None
    execution_count: int = 0


@dataclass
class ProcessState:
    """Track state of a running process"""
    pid: int
    command: str
    started_at: float
    status: str = "running"  # running, completed, failed, killed
    exit_code: Optional[int] = None
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    output_lines: int = 0


class IDEWorkspace:
    """
    IDE-centric workspace with state tracking

    Provides:
    - File tracking (creation, modification, deletion)
    - Notebook tracking (kernel state, cell execution)
    - Process tracking (resource usage, convergence)
    """

    def __init__(self, workspace_dir: str):
        """
        Initialize workspace

        Args:
            workspace_dir: Absolute path to workspace directory
        """
        self.workspace_dir = os.path.abspath(workspace_dir)
        Path(self.workspace_dir).mkdir(parents=True, exist_ok=True)

        # State tracking
        self.files: Dict[str, FileState] = {}
        self.notebooks: Dict[str, NotebookState] = {}
        self.processes: Dict[int, ProcessState] = {}

        # Initial scan
        self._scan_workspace()

    def _scan_workspace(self) -> None:
        """Scan workspace directory and populate initial file state"""
        workspace_path = Path(self.workspace_dir)

        for file_path in workspace_path.rglob("*"):
            if file_path.is_file():
                rel_path = str(file_path.relative_to(workspace_path))
                stat = file_path.stat()

                self.files[rel_path] = FileState(
                    path=rel_path,
                    created_at=stat.st_ctime,
                    modified_at=stat.st_mtime,
                    size_bytes=stat.st_size,
                    exists=True
                )

                # Track notebooks separately
                if file_path.suffix == ".ipynb":
                    self.notebooks[rel_path] = NotebookState(
                        path=rel_path,
                        num_cells=self._count_notebook_cells(file_path)
                    )

    def _count_notebook_cells(self, notebook_path: Path) -> int:
        """Count cells in a notebook file"""
        try:
            import json
            with open(notebook_path, 'r') as f:
                nb = json.load(f)
                return len(nb.get("cells", []))
        except:
            return 0

    # ==================== File Operations ====================

    def track_file_created(self, rel_path: str) -> None:
        """Track that a file was created"""
        abs_path = os.path.join(self.workspace_dir, rel_path)
        if os.path.exists(abs_path):
            stat = os.stat(abs_path)
            self.files[rel_path] = FileState(
                path=rel_path,
                created_at=stat.st_ctime,
                modified_at=stat.st_mtime,
                size_bytes=stat.st_size,
                exists=True
            )

            # Track notebooks
            if rel_path.endswith(".ipynb"):
                self.notebooks[rel_path] = NotebookState(
                    path=rel_path,
                    num_cells=self._count_notebook_cells(Path(abs_path))
                )

    def track_file_modified(self, rel_path: str) -> None:
        """Track that a file was modified"""
        abs_path = os.path.join(self.workspace_dir, rel_path)
        if os.path.exists(abs_path):
            stat = os.stat(abs_path)

            if rel_path in self.files:
                self.files[rel_path].modified_at = stat.st_mtime
                self.files[rel_path].size_bytes = stat.st_size
            else:
                self.track_file_created(rel_path)

            # Update notebook cell count
            if rel_path.endswith(".ipynb") and rel_path in self.notebooks:
                self.notebooks[rel_path].num_cells = self._count_notebook_cells(Path(abs_path))

    def track_file_deleted(self, rel_path: str) -> None:
        """Track that a file was deleted"""
        if rel_path in self.files:
            self.files[rel_path].exists = False

        if rel_path in self.notebooks:
            self.notebooks.pop(rel_path, None)

    def get_modified_files(self, since: float) -> List[str]:
        """Get files modified since timestamp"""
        return [
            f.path for f in self.files.values()
            if f.exists and f.modified_at > since
        ]

    def get_created_files(self, since: float) -> List[str]:
        """Get files created since timestamp"""
        return [
            f.path for f in self.files.values()
            if f.exists and f.created_at > since
        ]

    # ==================== Notebook Operations ====================

    def track_notebook_kernel(self, rel_path: str, kernel_id: str, status: str) -> None:
        """Track notebook kernel state"""
        if rel_path not in self.notebooks:
            self.notebooks[rel_path] = NotebookState(path=rel_path)

        self.notebooks[rel_path].kernel_id = kernel_id
        self.notebooks[rel_path].kernel_status = status

    def track_notebook_execution(self, rel_path: str, cell_index: int) -> None:
        """Track notebook cell execution"""
        if rel_path in self.notebooks:
            self.notebooks[rel_path].last_executed_cell = cell_index
            self.notebooks[rel_path].execution_count += 1

    def get_notebook_state(self, rel_path: str) -> Optional[NotebookState]:
        """Get notebook state"""
        return self.notebooks.get(rel_path)

    def get_active_notebooks(self) -> List[str]:
        """Get notebooks with running kernels"""
        return [
            nb.path for nb in self.notebooks.values()
            if nb.kernel_status in ["idle", "busy"]
        ]

    # ==================== Process Operations ====================

    def track_process_started(self, pid: int, command: str) -> None:
        """Track process started"""
        self.processes[pid] = ProcessState(
            pid=pid,
            command=command,
            started_at=time.time(),
            status="running"
        )

    def track_process_stats(self, pid: int, cpu_percent: float, memory_mb: float, output_lines: int) -> None:
        """Update process resource statistics"""
        if pid in self.processes:
            self.processes[pid].cpu_percent = cpu_percent
            self.processes[pid].memory_mb = memory_mb
            self.processes[pid].output_lines = output_lines

    def track_process_completed(self, pid: int, exit_code: int) -> None:
        """Track process completion"""
        if pid in self.processes:
            self.processes[pid].status = "completed" if exit_code == 0 else "failed"
            self.processes[pid].exit_code = exit_code

    def track_process_killed(self, pid: int) -> None:
        """Track process killed"""
        if pid in self.processes:
            self.processes[pid].status = "killed"

    def get_running_processes(self) -> List[ProcessState]:
        """Get all running processes"""
        return [p for p in self.processes.values() if p.status == "running"]

    def get_process_state(self, pid: int) -> Optional[ProcessState]:
        """Get state of specific process"""
        return self.processes.get(pid)

    # ==================== State Summary ====================

    def get_state_summary(self) -> Dict:
        """Get summary of workspace state"""
        return {
            "workspace_dir": self.workspace_dir,
            "files": {
                "total": len(self.files),
                "existing": len([f for f in self.files.values() if f.exists]),
                "deleted": len([f for f in self.files.values() if not f.exists])
            },
            "notebooks": {
                "total": len(self.notebooks),
                "active_kernels": len(self.get_active_notebooks())
            },
            "processes": {
                "total": len(self.processes),
                "running": len(self.get_running_processes()),
                "completed": len([p for p in self.processes.values() if p.status == "completed"]),
                "failed": len([p for p in self.processes.values() if p.status == "failed"])
            }
        }

    def __repr__(self) -> str:
        summary = self.get_state_summary()
        return (
            f"IDEWorkspace(dir={self.workspace_dir}, "
            f"files={summary['files']['existing']}, "
            f"notebooks={summary['notebooks']['total']}, "
            f"processes={summary['processes']['running']} running)"
        )

"""
Test IDEWorkspace state tracking
"""
import os
import json
import time
import tempfile
import shutil
from pathlib import Path

import pytest

from agent_v6.workspace import IDEWorkspace


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for testing"""
    temp_dir = tempfile.mkdtemp(prefix="test_workspace_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_workspace_initialization(temp_workspace):
    """Test workspace initializes correctly"""
    workspace = IDEWorkspace(temp_workspace)

    assert workspace.workspace_dir == temp_workspace
    assert len(workspace.files) == 0
    assert len(workspace.notebooks) == 0
    assert len(workspace.processes) == 0


def test_file_tracking(temp_workspace):
    """Test file creation, modification, deletion tracking"""
    workspace = IDEWorkspace(temp_workspace)

    # Create file
    test_file = "test.txt"
    test_path = os.path.join(temp_workspace, test_file)

    with open(test_path, "w") as f:
        f.write("Hello World")

    workspace.track_file_created(test_file)

    # Verify file tracked
    assert test_file in workspace.files
    assert workspace.files[test_file].exists
    assert workspace.files[test_file].size_bytes > 0

    # Modify file
    time.sleep(0.1)  # Ensure different mtime
    with open(test_path, "a") as f:
        f.write("\nLine 2")

    old_mtime = workspace.files[test_file].modified_at
    workspace.track_file_modified(test_file)

    # Verify modification tracked
    assert workspace.files[test_file].modified_at > old_mtime
    assert workspace.files[test_file].size_bytes > 11

    # Delete file
    os.remove(test_path)
    workspace.track_file_deleted(test_file)

    # Verify deletion tracked
    assert not workspace.files[test_file].exists


def test_notebook_tracking(temp_workspace):
    """Test notebook state tracking"""
    workspace = IDEWorkspace(temp_workspace)

    # Create notebook
    notebook_file = "test.ipynb"
    notebook_path = os.path.join(temp_workspace, notebook_file)

    notebook_content = {
        "cells": [
            {"cell_type": "code", "source": "print('hello')"},
            {"cell_type": "markdown", "source": "# Title"}
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(notebook_path, "w") as f:
        json.dump(notebook_content, f)

    workspace.track_file_created(notebook_file)

    # Verify notebook tracked
    assert notebook_file in workspace.notebooks
    assert workspace.notebooks[notebook_file].num_cells == 2

    # Track kernel
    workspace.track_notebook_kernel(notebook_file, "kernel-123", "idle")
    assert workspace.notebooks[notebook_file].kernel_id == "kernel-123"
    assert workspace.notebooks[notebook_file].kernel_status == "idle"

    # Track execution
    workspace.track_notebook_execution(notebook_file, 0)
    assert workspace.notebooks[notebook_file].last_executed_cell == 0
    assert workspace.notebooks[notebook_file].execution_count == 1

    # Get active notebooks
    active = workspace.get_active_notebooks()
    assert notebook_file in active


def test_process_tracking(temp_workspace):
    """Test process state tracking"""
    workspace = IDEWorkspace(temp_workspace)

    # Track process started
    pid = 12345
    command = "python train.py"

    workspace.track_process_started(pid, command)

    # Verify process tracked
    assert pid in workspace.processes
    assert workspace.processes[pid].status == "running"
    assert workspace.processes[pid].command == command

    # Update stats
    workspace.track_process_stats(pid, cpu_percent=25.5, memory_mb=512.0, output_lines=100)

    assert workspace.processes[pid].cpu_percent == 25.5
    assert workspace.processes[pid].memory_mb == 512.0
    assert workspace.processes[pid].output_lines == 100

    # Mark completed
    workspace.track_process_completed(pid, exit_code=0)

    assert workspace.processes[pid].status == "completed"
    assert workspace.processes[pid].exit_code == 0

    # Verify no longer in running
    running = workspace.get_running_processes()
    assert pid not in [p.pid for p in running]


def test_workspace_scan(temp_workspace):
    """Test workspace scans existing files on init"""
    # Create files before workspace init
    Path(temp_workspace, "existing.txt").write_text("exists")
    Path(temp_workspace, "subdir").mkdir()
    Path(temp_workspace, "subdir", "nested.py").write_text("# python")

    # Initialize workspace
    workspace = IDEWorkspace(temp_workspace)

    # Verify files discovered
    assert "existing.txt" in workspace.files
    assert "subdir/nested.py" in workspace.files


def test_state_summary(temp_workspace):
    """Test workspace state summary"""
    workspace = IDEWorkspace(temp_workspace)

    # Create some state - files must exist
    Path(temp_workspace, "file1.txt").write_text("content1")
    Path(temp_workspace, "file2.txt").write_text("content2")

    workspace.track_file_created("file1.txt")
    workspace.track_file_created("file2.txt")
    workspace.track_file_deleted("file2.txt")

    workspace.track_process_started(100, "cmd1")
    workspace.track_process_started(101, "cmd2")
    workspace.track_process_completed(100, 0)

    summary = workspace.get_state_summary()

    assert summary["files"]["total"] == 2
    assert summary["files"]["existing"] == 1
    assert summary["files"]["deleted"] == 1
    assert summary["processes"]["total"] == 2
    assert summary["processes"]["running"] == 1
    assert summary["processes"]["completed"] == 1


def test_get_modified_files(temp_workspace):
    """Test get files modified since timestamp"""
    workspace = IDEWorkspace(temp_workspace)

    start_time = time.time()

    # Create file before timestamp
    old_file = "old.txt"
    Path(temp_workspace, old_file).write_text("old")
    workspace.track_file_created(old_file)

    # Wait and create file after timestamp
    time.sleep(0.1)
    checkpoint = time.time()
    time.sleep(0.1)

    new_file = "new.txt"
    Path(temp_workspace, new_file).write_text("new")
    workspace.track_file_created(new_file)

    # Get modified files
    modified = workspace.get_modified_files(since=checkpoint)

    assert new_file in modified
    assert old_file not in modified


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

"""
Tests for TaskScheduler - time-aware task scheduling
"""

import pytest
import asyncio
import time
from agent_v5.task_scheduler import (
    TaskScheduler,
    Task,
    TaskPriority,
    TaskComplexity
)


class TestTask:
    """Test Task priority scoring"""

    def test_task_efficiency_score(self):
        """Test efficiency score calculation"""
        task = Task(
            id="test",
            name="Test Task",
            execute_fn=lambda: None,
            duration_min=5,
            duration_typical=10,
            duration_max=20,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.EFFICIENT,
            value_score=0.8
        )

        # Efficiency = value / time = 0.8 / 10 = 0.08
        assert task.efficiency_score == pytest.approx(0.08)

    def test_priority_score_plenty_of_time(self):
        """When there's plenty of time, prefer high-value tasks"""
        quick_task = Task(
            id="quick",
            name="Quick Task",
            execute_fn=lambda: None,
            duration_min=5,
            duration_typical=10,
            duration_max=20,
            priority=TaskPriority.MEDIUM,
            complexity=TaskComplexity.QUICK_WIN,
            value_score=0.5
        )

        valuable_task = Task(
            id="valuable",
            name="Valuable Task",
            execute_fn=lambda: None,
            duration_min=20,
            duration_typical=30,
            duration_max=60,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.EFFICIENT,
            value_score=0.9
        )

        # With 1000s remaining, valuable task should score higher
        time_remaining = 1000
        quick_score = quick_task.get_priority_score(time_remaining)
        valuable_score = valuable_task.get_priority_score(time_remaining)

        assert valuable_score > quick_score

    def test_priority_score_time_crunch(self):
        """When time is tight, prefer quick wins"""
        quick_task = Task(
            id="quick",
            name="Quick Task",
            execute_fn=lambda: None,
            duration_min=5,
            duration_typical=10,
            duration_max=20,
            priority=TaskPriority.MEDIUM,
            complexity=TaskComplexity.QUICK_WIN,
            value_score=0.5
        )

        expensive_task = Task(
            id="expensive",
            name="Expensive Task",
            execute_fn=lambda: None,
            duration_min=50,
            duration_typical=100,
            duration_max=200,
            priority=TaskPriority.MEDIUM,
            complexity=TaskComplexity.EXPENSIVE,
            value_score=0.9
        )

        # With only 30s remaining, quick task should score higher
        time_remaining = 30
        quick_score = quick_task.get_priority_score(time_remaining)
        expensive_score = expensive_task.get_priority_score(time_remaining)

        assert quick_score > expensive_score

    def test_priority_score_critical_always_high(self):
        """Critical tasks should always score high"""
        critical_task = Task(
            id="critical",
            name="Critical Task",
            execute_fn=lambda: None,
            duration_min=50,
            duration_typical=100,
            duration_max=200,
            priority=TaskPriority.CRITICAL,
            complexity=TaskComplexity.EXPENSIVE,
            value_score=0.5
        )

        low_task = Task(
            id="low",
            name="Low Task",
            execute_fn=lambda: None,
            duration_min=5,
            duration_typical=10,
            duration_max=20,
            priority=TaskPriority.LOW,
            complexity=TaskComplexity.QUICK_WIN,
            value_score=0.9
        )

        # Critical should score higher even with less value and more time
        time_remaining = 30
        critical_score = critical_task.get_priority_score(time_remaining)
        low_score = low_task.get_priority_score(time_remaining)

        assert critical_score > low_score


class TestTaskScheduler:
    """Test TaskScheduler"""

    def test_add_task_with_manual_estimates(self):
        """Test adding task with manual duration estimates"""
        scheduler = TaskScheduler(time_budget_seconds=600)

        def dummy_fn():
            return "done"

        scheduler.add_task(
            task_id="test1",
            name="Test Task",
            execute_fn=dummy_fn,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.EFFICIENT,
            value_score=0.8,
            duration_min=10,
            duration_typical=20,
            duration_max=40
        )

        assert "test1" in scheduler.tasks
        task = scheduler.tasks["test1"]
        assert task.duration_typical == 20
        assert task.priority == TaskPriority.HIGH

    def test_time_tracking(self):
        """Test time budget tracking"""
        scheduler = TaskScheduler(time_budget_seconds=600)

        # Before start
        assert scheduler.get_time_elapsed() == 0
        assert scheduler.get_time_remaining() == 600

        # After start
        scheduler.started_at = time.time() - 100  # Started 100s ago
        assert scheduler.get_time_elapsed() == pytest.approx(100, abs=1)
        assert scheduler.get_time_remaining() == pytest.approx(500, abs=1)

    @pytest.mark.asyncio
    async def test_simple_execution(self):
        """Test executing a simple task"""
        scheduler = TaskScheduler(time_budget_seconds=60)

        result_value = None

        async def task_fn():
            nonlocal result_value
            await asyncio.sleep(0.1)  # Simulate work
            result_value = "completed"
            return result_value

        scheduler.add_task(
            task_id="task1",
            name="Simple Task",
            execute_fn=task_fn,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.QUICK_WIN,
            value_score=1.0,
            duration_min=1,
            duration_typical=2,
            duration_max=5
        )

        results = await scheduler.execute_all()

        assert "task1" in results["completed"]
        assert len(results["skipped"]) == 0
        assert len(results["failed"]) == 0
        assert result_value == "completed"

    @pytest.mark.asyncio
    async def test_dependency_resolution(self):
        """Test tasks execute in dependency order"""
        scheduler = TaskScheduler(time_budget_seconds=60)

        execution_order = []

        async def task_a():
            execution_order.append("A")
            await asyncio.sleep(0.01)

        async def task_b():
            execution_order.append("B")
            await asyncio.sleep(0.01)

        async def task_c():
            execution_order.append("C")
            await asyncio.sleep(0.01)

        # C depends on B, B depends on A
        scheduler.add_task(
            task_id="task_c",
            name="Task C",
            execute_fn=task_c,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.QUICK_WIN,
            value_score=1.0,
            depends_on=["task_b"],
            duration_min=1,
            duration_typical=2,
            duration_max=5
        )

        scheduler.add_task(
            task_id="task_b",
            name="Task B",
            execute_fn=task_b,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.QUICK_WIN,
            value_score=1.0,
            depends_on=["task_a"],
            duration_min=1,
            duration_typical=2,
            duration_max=5
        )

        scheduler.add_task(
            task_id="task_a",
            name="Task A",
            execute_fn=task_a,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.QUICK_WIN,
            value_score=1.0,
            duration_min=1,
            duration_typical=2,
            duration_max=5
        )

        await scheduler.execute_all()

        # Should execute in order A -> B -> C
        assert execution_order == ["A", "B", "C"]

    @pytest.mark.asyncio
    async def test_skip_when_out_of_time(self):
        """Test scheduler skips tasks when time runs out"""
        scheduler = TaskScheduler(time_budget_seconds=5)  # Only 5 seconds

        async def quick_task():
            await asyncio.sleep(0.1)
            return "quick"

        async def slow_task():
            await asyncio.sleep(10)  # Takes 10s
            return "slow"

        scheduler.add_task(
            task_id="quick",
            name="Quick Task",
            execute_fn=quick_task,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.QUICK_WIN,
            value_score=1.0,
            duration_min=1,
            duration_typical=2,
            duration_max=3
        )

        scheduler.add_task(
            task_id="slow",
            name="Slow Task",
            execute_fn=slow_task,
            priority=TaskPriority.MEDIUM,  # Not critical
            complexity=TaskComplexity.EXPENSIVE,
            value_score=0.8,
            duration_min=8,
            duration_typical=10,
            duration_max=15
        )

        results = await scheduler.execute_all()

        # Quick task should complete, slow task should be skipped
        assert "quick" in results["completed"]
        assert "slow" in results["skipped"]

    @pytest.mark.asyncio
    async def test_critical_task_runs_despite_time(self):
        """Test critical tasks run even if over time budget"""
        scheduler = TaskScheduler(time_budget_seconds=1)  # Only 1 second

        async def critical_task():
            await asyncio.sleep(0.1)
            return "critical"

        scheduler.add_task(
            task_id="critical",
            name="Critical Task",
            execute_fn=critical_task,
            priority=TaskPriority.CRITICAL,  # CRITICAL priority
            complexity=TaskComplexity.EXPENSIVE,
            value_score=1.0,
            duration_min=5,
            duration_typical=10,  # Needs 10s but only have 1s
            duration_max=20
        )

        results = await scheduler.execute_all()

        # Should complete despite time constraint
        assert "critical" in results["completed"]

    @pytest.mark.asyncio
    async def test_adaptive_reprioritization(self):
        """Test adaptive reprioritization based on remaining time"""
        scheduler = TaskScheduler(time_budget_seconds=10)

        execution_order = []

        async def expensive_task():
            execution_order.append("expensive")
            await asyncio.sleep(0.01)

        async def quick_task():
            execution_order.append("quick")
            await asyncio.sleep(0.01)

        # Add tasks with different priorities and durations
        # With adaptive=True, should prioritize based on remaining time

        scheduler.add_task(
            task_id="expensive",
            name="Expensive Task",
            execute_fn=expensive_task,
            priority=TaskPriority.MEDIUM,
            complexity=TaskComplexity.EXPENSIVE,
            value_score=0.9,
            duration_min=5,
            duration_typical=8,
            duration_max=12
        )

        scheduler.add_task(
            task_id="quick",
            name="Quick Win",
            execute_fn=quick_task,
            priority=TaskPriority.MEDIUM,
            complexity=TaskComplexity.QUICK_WIN,
            value_score=0.7,
            duration_min=1,
            duration_typical=2,
            duration_max=3
        )

        results = await scheduler.execute_all(adaptive=True)

        # With short time budget, quick task should run first
        # (despite expensive having higher value)
        assert execution_order[0] == "quick"

    @pytest.mark.asyncio
    async def test_task_failure_handling(self):
        """Test scheduler handles task failures"""
        scheduler = TaskScheduler(time_budget_seconds=60)

        async def failing_task():
            raise ValueError("Task failed!")

        async def normal_task():
            return "success"

        scheduler.add_task(
            task_id="fail",
            name="Failing Task",
            execute_fn=failing_task,
            priority=TaskPriority.MEDIUM,
            complexity=TaskComplexity.QUICK_WIN,
            value_score=1.0,
            duration_min=1,
            duration_typical=2,
            duration_max=5
        )

        scheduler.add_task(
            task_id="normal",
            name="Normal Task",
            execute_fn=normal_task,
            priority=TaskPriority.MEDIUM,
            complexity=TaskComplexity.QUICK_WIN,
            value_score=1.0,
            duration_min=1,
            duration_typical=2,
            duration_max=5
        )

        results = await scheduler.execute_all()

        assert "fail" in results["failed"]
        assert "normal" in results["completed"]
        assert scheduler.tasks["fail"].error == "Task failed!"

    @pytest.mark.asyncio
    async def test_critical_task_failure_aborts(self):
        """Test critical task failure aborts execution"""
        scheduler = TaskScheduler(time_budget_seconds=60)

        async def critical_fail():
            raise ValueError("Critical failure!")

        async def should_not_run():
            return "should not execute"

        scheduler.add_task(
            task_id="critical",
            name="Critical Task",
            execute_fn=critical_fail,
            priority=TaskPriority.CRITICAL,
            complexity=TaskComplexity.QUICK_WIN,
            value_score=1.0,
            duration_min=1,
            duration_typical=2,
            duration_max=5
        )

        scheduler.add_task(
            task_id="after",
            name="Task After",
            execute_fn=should_not_run,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.QUICK_WIN,
            value_score=1.0,
            depends_on=["critical"],
            duration_min=1,
            duration_typical=2,
            duration_max=5
        )

        results = await scheduler.execute_all()

        # Critical task should fail, subsequent tasks should not run
        assert "critical" in results["failed"]
        assert "after" not in results["completed"]

    def test_summary_output(self):
        """Test summary string generation"""
        scheduler = TaskScheduler(time_budget_seconds=600)

        scheduler.add_task(
            task_id="task1",
            name="Test Task",
            execute_fn=lambda: None,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.EFFICIENT,
            value_score=0.8,
            duration_min=10,
            duration_typical=20,
            duration_max=40
        )

        summary = scheduler.get_summary()

        assert "Task Scheduler Summary" in summary
        assert "Time Budget: 600s" in summary
        assert "Test Task" in summary

"""
Time-Aware Task Scheduler

Dynamically schedules tasks based on:
- Predicted duration (from EstimateTaskDuration)
- Remaining time budget
- Task priority/value
- Task dependencies

Use case: In time-constrained environments (e.g., Kaggle competitions with epoch limits),
this scheduler helps agents make smart decisions about which tasks to run when time is limited.
"""

import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1    # Must run (e.g., load data, make submission)
    HIGH = 2        # Important but can be skipped (e.g., complex model training)
    MEDIUM = 3      # Nice to have (e.g., advanced feature engineering)
    LOW = 4         # Optional (e.g., extensive visualization)


class TaskComplexity(Enum):
    """Task complexity/value trade-off"""
    QUICK_WIN = 1       # Fast, high value (e.g., basic preprocessing)
    EFFICIENT = 2       # Medium time, good value (e.g., simple model)
    EXPENSIVE = 3       # Long time, high value (e.g., complex model)
    EXPLORATORY = 4     # Variable time, uncertain value (e.g., experimentation)


@dataclass
class Task:
    """Represents a schedulable task"""
    id: str
    name: str
    execute_fn: Callable

    # Duration estimates (in seconds)
    duration_min: float
    duration_typical: float
    duration_max: float

    # Scheduling metadata
    priority: TaskPriority
    complexity: TaskComplexity
    value_score: float = 1.0  # Expected value/utility (0-1)

    # Dependencies
    depends_on: List[str] = field(default_factory=list)

    # Execution tracking
    status: str = "pending"  # pending, running, completed, skipped, failed
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None

    def __post_init__(self):
        """Calculate efficiency score"""
        # Efficiency = value / time
        # Use typical duration for calculation
        self.efficiency_score = self.value_score / max(self.duration_typical, 1)

    def get_priority_score(self, time_remaining: float) -> float:
        """
        Calculate priority score for scheduling.
        Higher score = should run sooner.

        Factors:
        - Priority level (critical > high > medium > low)
        - Time remaining vs duration
        - Efficiency (value per second)
        - Complexity (prefer quick wins when time is tight)
        """
        # Base score from priority (0-100)
        priority_scores = {
            TaskPriority.CRITICAL: 100,
            TaskPriority.HIGH: 75,
            TaskPriority.MEDIUM: 50,
            TaskPriority.LOW: 25
        }
        score = priority_scores[self.priority]

        # Boost efficiency (value/time ratio)
        score += self.efficiency_score * 20

        # Time pressure adjustments
        time_ratio = self.duration_typical / max(time_remaining, 1)

        if time_ratio > 1:
            # Task takes longer than remaining time
            # Only run if CRITICAL, otherwise penalize heavily
            if self.priority != TaskPriority.CRITICAL:
                score *= 0.1  # Heavy penalty
        elif time_ratio > 0.5:
            # Task takes > 50% of remaining time
            # Prefer quick wins
            if self.complexity == TaskComplexity.QUICK_WIN:
                score *= 1.5  # Boost quick wins
            elif self.complexity == TaskComplexity.EXPENSIVE:
                score *= 0.7  # Penalize expensive tasks
        else:
            # Plenty of time - prefer high value tasks
            score *= (1 + self.value_score)

        return score


class TaskScheduler:
    """
    Time-aware task scheduler that prioritizes based on duration estimates
    and remaining time budget.
    """

    def __init__(
        self,
        time_budget_seconds: float,
        estimate_tool: Optional[Any] = None
    ):
        """
        Initialize scheduler.

        Args:
            time_budget_seconds: Total time budget for all tasks
            estimate_tool: EstimateTaskDurationTool instance for duration estimates
        """
        self.time_budget = time_budget_seconds
        self.estimate_tool = estimate_tool

        self.tasks: Dict[str, Task] = {}
        self.execution_order: List[str] = []

        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None

    def add_task(
        self,
        task_id: str,
        name: str,
        execute_fn: Callable,
        priority: TaskPriority,
        complexity: TaskComplexity,
        value_score: float = 1.0,
        depends_on: List[str] = None,
        # Optional: let scheduler estimate duration
        task_type: Optional[str] = None,
        data_size_mb: Optional[float] = None,
        # Or provide manual estimates
        duration_min: Optional[float] = None,
        duration_typical: Optional[float] = None,
        duration_max: Optional[float] = None,
    ):
        """
        Add a task to the scheduler.

        Can either:
        1. Provide task_type + data_size_mb for auto-estimation
        2. Provide manual duration estimates
        """
        # Get duration estimates
        if task_type and self.estimate_tool:
            # Use EstimateTaskDuration tool
            estimates = self._estimate_duration(task_type, data_size_mb)
            duration_min = estimates["min"]
            duration_typical = estimates["typical"]
            duration_max = estimates["max"]
        elif None in (duration_min, duration_typical, duration_max):
            raise ValueError(
                "Must provide either (task_type) or (duration_min, duration_typical, duration_max)"
            )

        task = Task(
            id=task_id,
            name=name,
            execute_fn=execute_fn,
            duration_min=duration_min,
            duration_typical=duration_typical,
            duration_max=duration_max,
            priority=priority,
            complexity=complexity,
            value_score=value_score,
            depends_on=depends_on or []
        )

        self.tasks[task_id] = task

    def _estimate_duration(self, task_type: str, data_size_mb: Optional[float]) -> Dict:
        """Call EstimateTaskDuration tool and parse results"""
        import asyncio

        input_dict = {"task_type": task_type}
        if data_size_mb:
            input_dict["data_size_mb"] = data_size_mb

        # Run async tool in sync context
        result = asyncio.run(self.estimate_tool.execute(input_dict))

        if result["is_error"]:
            # Fallback to conservative estimates
            return {"min": 60, "typical": 300, "max": 900}

        # Parse the output (tool returns formatted text)
        # This is a bit hacky - in production, tool should return structured data
        content = result["content"]

        # Simple parsing - look for duration lines
        # In production, you'd want tool to return structured data
        # For now, use conservative defaults
        return {"min": 60, "typical": 300, "max": 900}

    def get_time_remaining(self) -> float:
        """Get remaining time in budget"""
        if not self.started_at:
            return self.time_budget

        elapsed = time.time() - self.started_at
        return max(0, self.time_budget - elapsed)

    def get_time_elapsed(self) -> float:
        """Get elapsed time"""
        if not self.started_at:
            return 0

        end_time = self.completed_at or time.time()
        return end_time - self.started_at

    def _resolve_dependencies(self) -> List[str]:
        """
        Resolve task dependencies and return execution order.
        Uses topological sort.
        """
        # Build dependency graph
        in_degree = {task_id: 0 for task_id in self.tasks}
        adj_list = {task_id: [] for task_id in self.tasks}

        for task_id, task in self.tasks.items():
            for dep_id in task.depends_on:
                if dep_id not in self.tasks:
                    raise ValueError(f"Task {task_id} depends on unknown task {dep_id}")
                adj_list[dep_id].append(task_id)
                in_degree[task_id] += 1

        # Topological sort with priority
        ready_queue = []
        for task_id, degree in in_degree.items():
            if degree == 0:
                ready_queue.append(task_id)

        execution_order = []

        while ready_queue:
            # Sort by priority score (higher = earlier)
            time_remaining = self.get_time_remaining()
            ready_queue.sort(
                key=lambda tid: self.tasks[tid].get_priority_score(time_remaining),
                reverse=True
            )

            # Take highest priority task
            task_id = ready_queue.pop(0)
            execution_order.append(task_id)

            # Update dependencies
            for dependent_id in adj_list[task_id]:
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    ready_queue.append(dependent_id)

        if len(execution_order) != len(self.tasks):
            raise ValueError("Circular dependency detected")

        return execution_order

    async def execute_all(
        self,
        adaptive: bool = True,
        safety_margin: float = 1.2
    ) -> Dict:
        """
        Execute all tasks respecting time budget and priorities.

        Args:
            adaptive: If True, dynamically re-prioritize based on remaining time
            safety_margin: Multiply typical duration by this for conservative estimates

        Returns:
            Dict with execution summary
        """
        self.started_at = time.time()

        # Resolve dependency order
        initial_order = self._resolve_dependencies()

        results = {
            "completed": [],
            "skipped": [],
            "failed": [],
            "time_elapsed": 0,
            "time_budget": self.time_budget
        }

        # Execute tasks
        task_queue = initial_order.copy()

        while task_queue:
            time_remaining = self.get_time_remaining()

            if time_remaining <= 0:
                # Out of time - skip remaining tasks
                for task_id in task_queue:
                    self.tasks[task_id].status = "skipped"
                    results["skipped"].append(task_id)
                break

            # Re-prioritize if adaptive
            if adaptive and len(task_queue) > 1:
                task_queue.sort(
                    key=lambda tid: self.tasks[tid].get_priority_score(time_remaining),
                    reverse=True
                )

            # Get next task
            task_id = task_queue.pop(0)
            task = self.tasks[task_id]

            # Check if we have time for this task
            estimated_duration = task.duration_typical * safety_margin

            if estimated_duration > time_remaining:
                # Not enough time
                if task.priority == TaskPriority.CRITICAL:
                    # Try anyway (critical task)
                    print(f"⚠️  Running CRITICAL task {task.name} despite time constraint")
                else:
                    # Skip task
                    print(f"⏭️  Skipping {task.name} (needs {estimated_duration:.0f}s, have {time_remaining:.0f}s)")
                    task.status = "skipped"
                    results["skipped"].append(task_id)
                    continue

            # Execute task
            print(f"▶️  Running {task.name} (est. {task.duration_typical:.0f}s, {time_remaining:.0f}s remaining)")

            task.status = "running"
            task.started_at = time.time()

            try:
                # Execute (can be sync or async)
                if asyncio.iscoroutinefunction(task.execute_fn):
                    task.result = await task.execute_fn()
                else:
                    task.result = task.execute_fn()

                task.status = "completed"
                task.completed_at = time.time()

                actual_duration = task.completed_at - task.started_at
                print(f"✅ Completed {task.name} in {actual_duration:.1f}s")

                results["completed"].append(task_id)

            except Exception as e:
                task.status = "failed"
                task.completed_at = time.time()
                task.error = str(e)

                print(f"❌ Failed {task.name}: {e}")
                results["failed"].append(task_id)

                # If critical task failed, abort
                if task.priority == TaskPriority.CRITICAL:
                    print(f"🛑 Aborting due to critical task failure")
                    break

        self.completed_at = time.time()
        results["time_elapsed"] = self.get_time_elapsed()

        return results

    def get_summary(self) -> str:
        """Get human-readable execution summary"""
        output = "📊 Task Scheduler Summary\n"
        output += "=" * 60 + "\n\n"

        output += f"Time Budget: {self.time_budget:.0f}s\n"
        output += f"Time Elapsed: {self.get_time_elapsed():.0f}s\n"
        output += f"Time Remaining: {self.get_time_remaining():.0f}s\n\n"

        # Group by status
        by_status = {
            "completed": [],
            "running": [],
            "failed": [],
            "skipped": [],
            "pending": []
        }

        for task_id, task in self.tasks.items():
            by_status[task.status].append(task)

        for status in ["completed", "running", "failed", "skipped", "pending"]:
            tasks = by_status[status]
            if tasks:
                icon = {
                    "completed": "✅",
                    "running": "▶️",
                    "failed": "❌",
                    "skipped": "⏭️",
                    "pending": "⏸️"
                }[status]

                output += f"{icon} {status.upper()} ({len(tasks)}):\n"
                for task in tasks:
                    duration_str = ""
                    if task.started_at and task.completed_at:
                        duration = task.completed_at - task.started_at
                        duration_str = f" ({duration:.1f}s)"

                    output += f"   • {task.name}{duration_str}\n"
                output += "\n"

        return output

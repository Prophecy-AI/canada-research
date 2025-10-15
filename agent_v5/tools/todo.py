"""
Todo tools: TodoWrite (persisting) and ReadTodoList
"""
import os
import json
from typing import Dict, List
from .base import BaseTool


def _todos_path(workspace_dir: str) -> str:
    todos_dir = os.path.join(workspace_dir, ".todos")
    os.makedirs(todos_dir, exist_ok=True)
    return os.path.join(todos_dir, "todos.json")


class TodoWriteTool(BaseTool):
    """Create and update task lists (persisted to .todos/todos.json, enforcing at most one in_progress)."""

    def __init__(self, workspace_dir: str):
        super().__init__(workspace_dir)
        self.todos: List[Dict] = []

    @property
    def name(self) -> str:
        return "TodoWrite"

    @property
    def schema(self) -> Dict:
        return {
            "name": "TodoWrite",
            "description": "Create and update the todo list and persist it to .todos/todos.json. Enforces at most one in_progress task.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "Task description (imperative form)"
                                },
                                "activeForm": {
                                    "type": "string",
                                    "description": "Present continuous form (e.g., 'Running tests')"
                                },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                    "description": "Task status"
                                }
                            },
                            "required": ["content", "activeForm", "status"]
                        }
                    }
                },
                "required": ["todos"]
            }
        }

    async def execute(self, input: Dict) -> Dict:
        try:
            self.todos = input["todos"]
            in_progress_count = sum(1 for t in self.todos if t.get("status") == "in_progress")

            if in_progress_count > 1:
                return {
                    "content": "More than one task marked as in_progress. Only one task may be in_progress.",
                    "is_error": True,
                }

            path = _todos_path(self.workspace_dir)
            with open(path, "w") as f:
                json.dump(self.todos, f, indent=2, ensure_ascii=False)

            return {
                "content": f"Todos saved to {path}",
                "is_error": False,
                "debug_summary": f"{len(self.todos)} tasks",
            }
        except Exception as e:
            return {"content": f"Error writing todos: {str(e)}", "is_error": True}


class ReadTodoListTool(BaseTool):
    """Read the current todo list from .todos/todos.json."""

    @property
    def name(self) -> str:
        return "ReadTodoList"

    @property
    def schema(self) -> Dict:
        return {
            "name": "ReadTodoList",
            "description": "Read the current todo list persisted at .todos/todos.json in the workspace.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        }

    async def execute(self, input: Dict) -> Dict:
        try:
            path = _todos_path(self.workspace_dir)
            if not os.path.exists(path):
                return {"content": "[]", "is_error": False, "debug_summary": "no todos"}
            with open(path, "r") as f:
                data = f.read()
            return {"content": data, "is_error": False, "debug_summary": f"{len(data)} bytes"}
        except Exception as e:
            return {"content": f"Error reading todos: {str(e)}", "is_error": True}

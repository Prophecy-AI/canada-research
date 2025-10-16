import asyncio
import os
from pathlib import Path
from typing import Dict, List
from abc import ABC, abstractmethod


class BaseTool(ABC):
    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def schema(self) -> Dict:
        pass

    @abstractmethod
    async def execute(self, input: Dict) -> str:
        pass


class BashTool(BaseTool):
    @property
    def name(self) -> str:
        return "Bash"

    @property
    def schema(self) -> Dict:
        return {
            "name": "Bash",
            "description": "Execute shell command in workspace directory",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute"
                    }
                },
                "required": ["command"]
            }
        }

    async def execute(self, input: Dict) -> str:
        command = input["command"]
        
        try:
            process = await asyncio.create_subprocess_shell(
                f"cd {self.workspace_dir} && {command}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            output = stdout.decode() + stderr.decode()
            
            return output
        except Exception as e:
            return f"Error: {str(e)}"


class ReadTool(BaseTool):
    @property
    def name(self) -> str:
        return "Read"

    @property
    def schema(self) -> Dict:
        return {
            "name": "Read",
            "description": "Read file contents from workspace",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute or relative path to file"
                    },
                    "offset": {
                        "type": "number",
                        "description": "Line number to start reading from"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Number of lines to read"
                    }
                },
                "required": ["file_path"]
            }
        }

    async def execute(self, input: Dict) -> str:
        file_path = input["file_path"]
        offset = input.get("offset", 0)
        limit = input.get("limit", 2000)
        
        if not file_path.startswith('/'):
            file_path = str(Path(self.workspace_dir) / file_path)
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            selected_lines = lines[offset:offset + limit]
            
            numbered_lines = []
            for i, line in enumerate(selected_lines, start=offset + 1):
                if len(line) > 2000:
                    line = line[:2000] + "... (line truncated)\n"
                numbered_lines.append(f"{i:6d}→{line}")
            
            content = "".join(numbered_lines)
            
            if not content.strip():
                content = "File exists but is empty."
            
            return content
        except Exception as e:
            return f"Error reading file: {str(e)}"


class WriteTool(BaseTool):
    @property
    def name(self) -> str:
        return "Write"

    @property
    def schema(self) -> Dict:
        return {
            "name": "Write",
            "description": "Write content to file in workspace",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to file relative to workspace"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to file"
                    }
                },
                "required": ["file_path", "content"]
            }
        }

    async def execute(self, input: Dict) -> str:
        file_path = input["file_path"]
        content = input["content"]
        full_path = Path(self.workspace_dir) / file_path
        
        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            return f"File written successfully: {file_path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"


class ToolRegistry:
    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir
        self.tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        self.tools[tool.name] = tool

    def get_schemas(self) -> List[Dict]:
        return [tool.schema for tool in self.tools.values()]

    async def execute(self, tool_name: str, input: Dict) -> str:
        if tool_name not in self.tools:
            return f"Unknown tool: {tool_name}"
        
        return await self.tools[tool_name].execute(input)


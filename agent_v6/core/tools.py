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
    async def execute(self, input: Dict) -> Dict:
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

    async def execute(self, input: Dict) -> Dict:
        command = input["command"]
        
        try:
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            
            process = await asyncio.create_subprocess_shell(
                f"cd {self.workspace_dir} && {command}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            
            stdout, stderr = await process.communicate()
            output = stdout.decode() + stderr.decode()
            
            if len(output) > 30000:
                output = output[:30000] + "\n... (output truncated)"
            
            return {
                "content": output,
                "is_error": False
            }
        except Exception as e:
            return {
                "content": f"Error: {str(e)}",
                "is_error": True
            }


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

    async def execute(self, input: Dict) -> Dict:
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
            
            return {
                "content": content,
                "is_error": False
            }
        except Exception as e:
            return {
                "content": f"Error reading file: {str(e)}",
                "is_error": True
            }


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

    async def execute(self, input: Dict) -> Dict:
        file_path = input["file_path"]
        content = input["content"]
        
        try:
            if not file_path.startswith('/'):
                file_path = os.path.join(self.workspace_dir, file_path)
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w') as f:
                f.write(content)
            
            return {
                "content": f"File written successfully: {file_path}",
                "is_error": False
            }
        except Exception as e:
            return {
                "content": f"Error writing file: {str(e)}",
                "is_error": True
            }


class ToolRegistry:
    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir
        self.tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        self.tools[tool.name] = tool

    def get_schemas(self) -> List[Dict]:
        return [tool.schema for tool in self.tools.values()]

    async def execute(self, tool_name: str, input: Dict) -> Dict:
        if tool_name not in self.tools:
            return {
                "content": f"Unknown tool: {tool_name}",
                "is_error": True
            }
        
        return await self.tools[tool_name].execute(input)


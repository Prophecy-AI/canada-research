import os
import asyncio
from typing import List, Dict, AsyncGenerator
from anthropic import Anthropic


class Agent:
    def __init__(self, workspace_dir: str, system_prompt: str, tools: 'ToolRegistry'):
        self.workspace_dir = workspace_dir
        self.system_prompt = system_prompt
        self.tools = tools
        self.conversation_history: List[Dict] = []
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    async def run(self, user_message: str) -> str:
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        full_response = []

        while True:
            response_content = []
            tool_uses = []

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with self.client.messages.stream(
                        model="claude-sonnet-4-5-20250929",
                        max_tokens=20000,
                        system=self.system_prompt,
                        messages=self.conversation_history,
                        tools=self.tools.get_schemas(),
                        temperature=0,
                    ) as stream:
                        for event in stream:
                            if event.type == "content_block_delta":
                                if event.delta.type == "text_delta":
                                    text = event.delta.text
                                    response_content.append({"type": "text", "text": text})
                                    full_response.append(text)
                                    print(text, end="", flush=True)

                        final_message = stream.get_final_message()
                        for block in final_message.content:
                            if block.type == "tool_use":
                                tool_uses.append({
                                    "type": "tool_use",
                                    "id": block.id,
                                    "name": block.name,
                                    "input": block.input
                                })
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"\n⚠️  API error (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}")
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        print(f"\n❌ API error after {max_retries} attempts: {str(e)}")
                        raise

            self.conversation_history.append({
                "role": "assistant",
                "content": response_content + tool_uses
            })

            if not tool_uses:
                break

            print()

            tool_results = []
            for tool_use in tool_uses:
                tool_name = tool_use['name']
                tool_input = tool_use['input']
                
                print(f"\n🔧 Tool: {tool_name}")
                if tool_name == "Bash":
                    cmd = tool_input.get('command', '')
                    print(f"   Command: {cmd[:100]}")
                elif tool_name == "Write":
                    path = tool_input.get('file_path', '')
                    content_len = len(tool_input.get('content', ''))
                    print(f"   Path: {path} ({content_len} bytes)")
                elif tool_name == "Read":
                    path = tool_input.get('file_path', '')
                    print(f"   Path: {path}")
                
                result = await self.tools.execute(tool_name, tool_input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use["id"],
                    "content": result["content"],
                    "is_error": result.get("is_error", False)
                })
                print(f"✓ Completed")

            self.conversation_history.append({
                "role": "user",
                "content": tool_results
            })

        print()
        return "".join(full_response)


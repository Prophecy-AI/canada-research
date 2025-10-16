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
        
        print(f"\n[DEBUG] System prompt length: {len(self.system_prompt)} chars, ~{len(self.system_prompt)//4} tokens")
        print(f"[DEBUG] System prompt first 500 chars:\n{self.system_prompt[:500]}")
        print(f"[DEBUG] System prompt last 500 chars:\n...{self.system_prompt[-500:]}")
        print(f"\n[DEBUG] Conversation history messages: {len(self.conversation_history)}")
        for i, msg in enumerate(self.conversation_history):
            content = msg.get('content', '')
            if isinstance(content, str):
                print(f"[DEBUG]   Message {i}: {msg['role']} - {len(content)} chars")
                if len(content) > 1000:
                    print(f"[DEBUG]     Preview: {content[:200]}...")
            elif isinstance(content, list):
                print(f"[DEBUG]   Message {i}: {msg['role']} - {len(content)} items")

        while True:
            response_content = []
            tool_uses = []

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

            self.conversation_history.append({
                "role": "assistant",
                "content": response_content + tool_uses
            })

            if not tool_uses:
                break

            print()
            
            tool_results = []
            for tool_use in tool_uses:
                print(f"\n🔧 Tool: {tool_use['name']}")
                result = await self.tools.execute(tool_use["name"], tool_use["input"])
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use["id"],
                    "content": result
                })
                print(f"✓ Tool completed")

            self.conversation_history.append({
                "role": "user",
                "content": tool_results
            })

        print()
        return "".join(full_response)


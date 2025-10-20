"""
CLI interface for agent_v6

Simple command-line interface for testing the IDE agent.
"""
import os
import sys
import asyncio
import uuid
from pathlib import Path

from agent_v6.agent import IDEAgent
from agent_v6.prompts.generic_ide import GENERIC_IDE_PROMPT


async def main():
    """Run CLI agent"""
    # Generate session ID
    session_id = str(uuid.uuid4())[:8]

    # Setup workspace
    workspace_dir = os.path.join(os.getcwd(), "workspace", session_id)
    Path(workspace_dir).mkdir(parents=True, exist_ok=True)

    print(f"Agent Session: {session_id}")
    print(f"Workspace: {workspace_dir}")
    print("="*70)
    print()

    # Create agent
    agent = IDEAgent(
        session_id=session_id,
        workspace_dir=workspace_dir,
        system_prompt=GENERIC_IDE_PROMPT,
        enable_memory_compaction=True
    )

    print("IDE Agent ready. Type 'exit' to quit.")
    print("="*70)
    print()

    try:
        while True:
            # Get user input
            try:
                user_input = input("You: ")
            except EOFError:
                break

            if not user_input.strip():
                continue

            if user_input.lower() in ['exit', 'quit']:
                break

            # Run agent
            print("\nAgent: ", end="", flush=True)

            try:
                async for message in agent.run(user_input):
                    if message.get("type") == "text_delta":
                        print(message["text"], end="", flush=True)
                    elif message.get("type") == "tool_execution":
                        # Show tool execution (optional, can comment out)
                        tool_name = message.get("tool_name", "unknown")
                        print(f"\n[🔧 {tool_name}]", end="", flush=True)

                print("\n")  # New line after response

            except KeyboardInterrupt:
                print("\n[Interrupted]")
                break
            except Exception as e:
                print(f"\n[Error: {e}]")

    finally:
        # Cleanup
        print("\nCleaning up...")
        await agent.cleanup()
        print(f"Session {session_id} ended.")


if __name__ == "__main__":
    asyncio.run(main())

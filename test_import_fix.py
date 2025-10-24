#!/usr/bin/env python3
"""
Test script to verify the import fix works
Simulates the Docker container environment
"""
import sys
import os
from pathlib import Path

# Add the agent directory to path (simulating what runner.py does)
AGENT_DIR = Path(__file__).parent / "mle-bench" / "agents" / "agent_v5_kaggle"
sys.path.insert(0, str(AGENT_DIR))

print("=" * 60)
print("Testing Import Fix")
print("=" * 60)
print(f"AGENT_DIR: {AGENT_DIR}")
print(f"sys.path[0]: {sys.path[0]}")
print()

# Test 1: Check files exist
print("1. Checking file structure...")
tools_init = AGENT_DIR / "tools" / "__init__.py"
gpu_validate = AGENT_DIR / "tools" / "gpu_validate.py"
kaggle_agent = AGENT_DIR / "kaggle_agent.py"

assert tools_init.exists(), f"Missing: {tools_init}"
assert gpu_validate.exists(), f"Missing: {gpu_validate}"
assert kaggle_agent.exists(), f"Missing: {kaggle_agent}"
print("   ✓ All required files exist")

# Test 2: Import tools package
print("\n2. Testing tools package import...")
try:
    import tools
    print(f"   ✓ Successfully imported 'tools' package")
    print(f"   Location: {tools.__file__}")
except ImportError as e:
    print(f"   ✗ Failed to import 'tools': {e}")
    sys.exit(1)

# Test 3: Import GPUValidateTool
print("\n3. Testing GPUValidateTool import...")
try:
    from tools.gpu_validate import GPUValidateTool
    print(f"   ✓ Successfully imported 'GPUValidateTool'")
    print(f"   Class: {GPUValidateTool}")
    print(f"   Module: {GPUValidateTool.__module__}")
except ImportError as e:
    print(f"   ✗ Failed to import 'GPUValidateTool': {e}")
    sys.exit(1)

# Test 4: Check GPUValidateTool properties
print("\n4. Testing GPUValidateTool instantiation...")
try:
    tool = GPUValidateTool(workspace_dir="/tmp")
    print(f"   ✓ Successfully created GPUValidateTool instance")
    print(f"   Tool name: {tool.name}")
    print(f"   Schema keys: {list(tool.schema.keys())}")
except Exception as e:
    print(f"   ✗ Failed to instantiate GPUValidateTool: {e}")
    sys.exit(1)

# Test 5: Import memory package (verify it still works)
print("\n5. Testing memory package import...")
try:
    from memory import CompetitionMemory
    print(f"   ✓ Successfully imported 'CompetitionMemory'")
    print(f"   Module: {CompetitionMemory.__module__}")
except ImportError as e:
    print(f"   ✗ Failed to import 'CompetitionMemory': {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED")
print("=" * 60)
print("\nThe import fix is working correctly!")
print("The agent should now start successfully in the Docker container.")

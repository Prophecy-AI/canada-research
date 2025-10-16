#!/bin/bash

echo "======================================"
echo "Agent V6 Local Test"
echo "======================================"

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Error: ANTHROPIC_API_KEY not set"
    echo "Run: export ANTHROPIC_API_KEY='your-key-here'"
    exit 1
fi

cd /Users/kevinson/Desktop/projects/canada-research

echo "Installing dependencies..."
pip install -q anthropic

echo ""
echo "Running agent_v6 local test..."
echo ""

python -m agent_v6.test_local

echo ""
echo "======================================"
echo "Test complete!"
echo "Check test_competition/ for outputs"
echo "======================================"


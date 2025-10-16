#!/bin/bash

echo "======================================"
echo "Agent V6 - mle-bench Integration Test"
echo "======================================"

cd /Users/kevinson/Desktop/projects/canada-research/mle-bench

echo ""
echo "Testing agent_v6 on spaceship-titanic..."
echo ""

python run_agent.py \
  --agent-id agent_v6 \
  --competition-set experiments/splits/spaceship-titanic.txt

echo ""
echo "======================================"
echo "Test complete!"
echo "Check runs/ for results"
echo "======================================"


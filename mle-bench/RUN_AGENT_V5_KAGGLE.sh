#!/bin/bash
# Quick script to build and run agent_v5_kaggle on spaceship-titanic

set -e  # Exit on error

echo "=========================================="
echo "Agent V5 Kaggle - Build & Run"
echo "=========================================="

# Check API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "❌ ERROR: ANTHROPIC_API_KEY not set"
    echo "   Please run: export ANTHROPIC_API_KEY=your-key-here"
    exit 1
fi

echo "✅ ANTHROPIC_API_KEY is set"

# Set build arguments
export SUBMISSION_DIR=/home/submission
export LOGS_DIR=/home/logs
export CODE_DIR=/home/code
export AGENT_DIR=/home/agent

echo ""
echo "=========================================="
echo "Step 1: Build Docker Image"
echo "=========================================="

cd /home/ubuntu/research/canada-research/mle-bench

docker build --platform=linux/amd64 -t agent_v5_kaggle \
  agents/agent_v5_kaggle/ \
  --build-arg SUBMISSION_DIR=$SUBMISSION_DIR \
  --build-arg LOGS_DIR=$LOGS_DIR \
  --build-arg CODE_DIR=$CODE_DIR \
  --build-arg AGENT_DIR=$AGENT_DIR

echo ""
echo "✅ Docker image built successfully"

echo ""
echo "=========================================="
echo "Step 2: Run on a competition"
echo "=========================================="

# Create temporary container config with GPU properly attached
TMP_CONFIG=$(mktemp /tmp/container_config_XXXXXX.json)
cat > "$TMP_CONFIG" << 'EOF'
{
    "mem_limit": "80G",
    "shm_size": "16G",
    "nano_cpus": 8e9,
    "gpus": -1
}
EOF

echo "Using temporary GPU config: $TMP_CONFIG"
cat "$TMP_CONFIG"

for line in $(cat experiments/splits/custom-set.txt); do
	mlebench prepare -c $line
done

git lfs pull

# Run the agent in the background
python run_agent.py \
--agent-id agent_v5_kaggle \
--competition-set experiments/splits/custom-set.txt \
--container-config "$TMP_CONFIG" &

AGENT_PID=$!

# Wait a moment for the container to start
sleep 10

# Get the latest running container
CONTAINER_ID=$(docker ps --latest --format "{{.ID}}")

# Tail the logs (this will follow until the container stops)
docker exec "$CONTAINER_ID" tail -f /home/logs/agent.log

# Wait for the background python process to complete
wait $AGENT_PID

echo "Both processes completed"
# Clean up temporary config
rm -f "$TMP_CONFIG"

echo ""
echo "=========================================="
echo "Step 3: Check Results"
echo "=========================================="

# Find latest run
RUN_GROUP=$(ls -t runs/ | head -1)
echo "Run group: $RUN_GROUP"
echo ""

echo ""
echo "=========================================="
echo "Step 4: Grade Submission"
echo "=========================================="

# Generate submission JSONL
python experiments/make_submission.py \
  --metadata runs/$RUN_GROUP/metadata.json \
  --output runs/$RUN_GROUP/submission.jsonl

# Grade
mlebench grade \
  --submission runs/$RUN_GROUP/submission.jsonl \
  --output-dir runs/$RUN_GROUP

echo ""
echo "=========================================="
echo "COMPLETE!"
echo "=========================================="
echo "Results in: runs/$RUN_GROUP/"
echo ""
echo "View logs:"
echo "  cat runs/$RUN_GROUP/*/logs/*.log"
echo ""
echo "View code:"
echo "  ls runs/$RUN_GROUP/*/code/"
echo ""
echo "View submission:"
echo "  cat runs/$RUN_GROUP/*/submission/submission.csv"
echo ""
echo "View grading results:"
echo "  cat runs/$RUN_GROUP/results.json"

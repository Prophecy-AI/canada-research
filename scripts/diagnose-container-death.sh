#!/bin/bash
# Diagnose why a Docker container died
# Run after a container unexpectedly exits

set -e

echo "=========================================="
echo "Container Death Diagnostics"
echo "=========================================="
echo ""

# Get most recent container (including stopped ones)
CONTAINER_ID=$(docker ps -a --format "{{.ID}}" | head -1)

if [ -z "$CONTAINER_ID" ]; then
    echo "❌ No containers found"
    exit 1
fi

CONTAINER_NAME=$(docker ps -a --filter "id=$CONTAINER_ID" --format "{{.Names}}")
CONTAINER_STATUS=$(docker ps -a --filter "id=$CONTAINER_ID" --format "{{.Status}}")
CONTAINER_IMAGE=$(docker ps -a --filter "id=$CONTAINER_ID" --format "{{.Image}}")

echo "Most recent container:"
echo "  ID: $CONTAINER_ID"
echo "  Name: $CONTAINER_NAME"
echo "  Image: $CONTAINER_IMAGE"
echo "  Status: $CONTAINER_STATUS"
echo ""

# Check exit code
EXIT_CODE=$(docker inspect $CONTAINER_ID --format='{{.State.ExitCode}}')
echo "Exit code: $EXIT_CODE"

case $EXIT_CODE in
    0)
        echo "  → Container exited normally"
        ;;
    137)
        echo "  → Container was KILLED (SIGKILL) - likely OOM or manual kill"
        ;;
    139)
        echo "  → Segmentation fault"
        ;;
    143)
        echo "  → Terminated (SIGTERM)"
        ;;
    *)
        echo "  → Abnormal exit"
        ;;
esac
echo ""

# Check if OOM killed
OOM_KILLED=$(docker inspect $CONTAINER_ID --format='{{.State.OOMKilled}}')
if [ "$OOM_KILLED" = "true" ]; then
    echo "🔴 CAUSE: OUT OF MEMORY (OOM)"
    echo "   The container exceeded its memory limit and was killed by Docker"
    echo ""
fi

# Get container inspect details
echo "=========================================="
echo "Container Resource Limits"
echo "=========================================="
docker inspect $CONTAINER_ID --format='Memory Limit: {{.HostConfig.Memory}} bytes ({{if eq .HostConfig.Memory 0}}unlimited{{else}}limited{{end}})' || echo "Could not read memory limit"
docker inspect $CONTAINER_ID --format='CPU Limit: {{.HostConfig.NanoCpus}}' || echo "Could not read CPU limit"
echo ""

# Show last logs from container
echo "=========================================="
echo "Last 50 Lines of Container Logs"
echo "=========================================="
docker logs --tail 50 $CONTAINER_ID 2>&1
echo ""

# Check system OOM events around the time container died
echo "=========================================="
echo "System OOM Killer Events (last 100)"
echo "=========================================="
dmesg | grep -i "killed process\|out of memory\|oom" | tail -100
echo ""

# Check Docker daemon logs
echo "=========================================="
echo "Docker Daemon Logs (last 20 lines)"
echo "=========================================="
journalctl -u docker.service --no-pager | tail -20
echo ""

# Check system memory at current time
echo "=========================================="
echo "Current System Resources"
echo "=========================================="
echo "Memory:"
free -h
echo ""
echo "Disk:"
df -h /var/lib/docker
echo ""
echo "GPU:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
echo ""

# Recommendations
echo "=========================================="
echo "Diagnostic Summary"
echo "=========================================="
if [ "$OOM_KILLED" = "true" ] || [ "$EXIT_CODE" = "137" ]; then
    echo "🔴 LIKELY CAUSE: Out of Memory (OOM)"
    echo ""
    echo "Solutions:"
    echo "1. Increase Docker container memory limit"
    echo "2. Reduce batch size in training scripts"
    echo "3. Use gradient accumulation instead of large batches"
    echo "4. Enable memory-efficient training (gradient checkpointing)"
    echo "5. Monitor with: scripts/monitor-runner.sh"
elif grep -q "No space left on device" <<< "$(docker logs $CONTAINER_ID 2>&1)"; then
    echo "🔴 LIKELY CAUSE: Disk Full"
    echo ""
    echo "Solutions:"
    echo "1. Clean Docker: docker system prune -af"
    echo "2. Clean old containers: docker container prune -f"
    echo "3. Clean old images: docker image prune -af"
    echo "4. Check disk: df -h"
elif grep -q "CUDA out of memory" <<< "$(docker logs $CONTAINER_ID 2>&1)"; then
    echo "🔴 LIKELY CAUSE: GPU Out of Memory"
    echo ""
    echo "Solutions:"
    echo "1. Reduce batch size"
    echo "2. Use mixed precision training (fp16)"
    echo "3. Enable gradient checkpointing"
    echo "4. Use smaller model"
else
    echo "⚠️  Exit code $EXIT_CODE - check logs above for details"
fi
echo ""
echo "For real-time monitoring during runs:"
echo "  bash scripts/monitor-runner.sh"
echo ""


#!/bin/bash
# Real-time monitoring for GPU runner during MLE-bench runs
# Run this in a separate terminal while competitions are running

set -e

MONITOR_DIR="$HOME/runner-monitoring"
LOG_FILE="$MONITOR_DIR/runner-$(hostname)-$(date +%Y%m%d-%H%M%S).log"

# Create monitoring directory
mkdir -p "$MONITOR_DIR"

echo "=========================================="
echo "GPU Runner Real-Time Monitoring"
echo "=========================================="
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Log file: $LOG_FILE"
echo ""
echo "Monitoring:"
echo "  • CPU/Memory usage (system & containers)"
echo "  • GPU usage (nvidia-smi)"
echo "  • Disk space"
echo "  • Docker container status"
echo "  • OOM killer events"
echo ""
echo "Press Ctrl+C to stop monitoring"
echo "=========================================="
echo ""

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to check OOM killer events
check_oom() {
    local oom_count=$(dmesg | grep -i "killed process" | wc -l)
    echo "$oom_count"
}

# Initial OOM count
INITIAL_OOM=$(check_oom)

# Monitor loop
while true; do
    log "===== Monitoring Snapshot ====="
    
    # System memory
    log "--- System Memory ---"
    free -h | tee -a "$LOG_FILE"
    
    # CPU usage
    log "--- CPU Usage (top 5 processes) ---"
    ps aux --sort=-%cpu | head -6 | tee -a "$LOG_FILE"
    
    # GPU usage
    log "--- GPU Status ---"
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | while read line; do
        log "GPU: $line"
    done
    
    # Docker containers
    log "--- Docker Containers ---"
    docker ps --format "table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Names}}" | tee -a "$LOG_FILE"
    
    # Docker stats (one-shot)
    log "--- Docker Resource Usage ---"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}" 2>/dev/null | tee -a "$LOG_FILE"
    
    # Disk space
    log "--- Disk Space ---"
    df -h / /var/lib/docker | tee -a "$LOG_FILE"
    
    # Check for new OOM events
    CURRENT_OOM=$(check_oom)
    if [ "$CURRENT_OOM" -gt "$INITIAL_OOM" ]; then
        log "⚠️  WARNING: OOM KILLER DETECTED! $((CURRENT_OOM - INITIAL_OOM)) new kills"
        log "Recent OOM events:"
        dmesg | grep -i "killed process" | tail -5 | tee -a "$LOG_FILE"
        INITIAL_OOM=$CURRENT_OOM
    fi
    
    # Check for Docker errors in system logs
    log "--- Recent Docker Errors (last 1 min) ---"
    journalctl -u docker.service --since "1 minute ago" --no-pager | grep -i "error\|kill\|oom" | tail -10 | tee -a "$LOG_FILE" || log "No Docker errors found"
    
    log "===== End Snapshot ====="
    echo "" | tee -a "$LOG_FILE"
    
    # Wait 30 seconds before next snapshot
    sleep 30
done


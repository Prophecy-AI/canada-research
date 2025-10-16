#!/bin/bash
# Lightweight dashboard for monitoring runner in real-time
# Best run in tmux/screen

set -e

# Function to draw dashboard
draw_dashboard() {
    clear
    
    # Header
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║          GPU RUNNER DASHBOARD - $(hostname)                  "
    echo "║          $(date '+%Y-%m-%d %H:%M:%S')                        "
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    
    # System Memory
    echo "┌─ SYSTEM MEMORY ────────────────────────────────────────────────┐"
    free -h | awk 'NR==1{printf "%-10s %10s %10s %10s %10s\n", $1, $2, $3, $4, $7} NR==2{printf "%-10s %10s %10s %10s %10s\n", $1, $2, $3, $4, $7}'
    
    # Memory usage bar
    USED=$(free | awk 'NR==2{print $3}')
    TOTAL=$(free | awk 'NR==2{print $2}')
    PERCENT=$((USED * 100 / TOTAL))
    
    # Color based on usage
    if [ $PERCENT -gt 90 ]; then
        COLOR="🔴"
    elif [ $PERCENT -gt 75 ]; then
        COLOR="🟡"
    else
        COLOR="🟢"
    fi
    
    printf "Usage: $COLOR %3d%% [" $PERCENT
    BARS=$((PERCENT / 2))
    for i in $(seq 1 50); do
        if [ $i -le $BARS ]; then
            printf "█"
        else
            printf "░"
        fi
    done
    printf "]\n"
    echo "└────────────────────────────────────────────────────────────────┘"
    echo ""
    
    # GPU Status
    echo "┌─ GPU STATUS ───────────────────────────────────────────────────┐"
    printf "%-5s %-20s %8s %8s %12s %12s %6s\n" "GPU" "Name" "Util" "Mem%" "Mem Used" "Mem Total" "Temp"
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | while IFS=',' read -r idx name util mem_util mem_used mem_total temp; do
        # Trim whitespace
        idx=$(echo $idx | xargs)
        name=$(echo $name | xargs | cut -c1-20)
        util=$(echo $util | xargs)
        mem_util=$(echo $mem_util | xargs)
        mem_used=$(echo $mem_used | xargs)
        mem_total=$(echo $mem_total | xargs)
        temp=$(echo $temp | xargs)
        
        # Color code based on utilization
        if [ "$util" -gt 80 ]; then
            ICON="🔥"
        elif [ "$util" -gt 50 ]; then
            ICON="⚙️ "
        else
            ICON="💤"
        fi
        
        printf "$ICON%-4s %-20s %7s%% %7s%% %9s MB %9s MB %5s°C\n" "$idx" "$name" "$util" "$mem_util" "$mem_used" "$mem_total" "$temp"
    done
    echo "└────────────────────────────────────────────────────────────────┘"
    echo ""
    
    # Docker Containers
    echo "┌─ DOCKER CONTAINERS ────────────────────────────────────────────┐"
    CONTAINER_COUNT=$(docker ps -q | wc -l)
    echo "Active containers: $CONTAINER_COUNT"
    
    if [ $CONTAINER_COUNT -gt 0 ]; then
        printf "%-12s %-30s %-15s %8s %12s\n" "ID" "Image" "Status" "CPU%" "Memory"
        docker ps --format "{{.ID}},{{.Image}},{{.Status}}" | while IFS=',' read -r id image status; do
            # Get stats for this container
            stats=$(docker stats --no-stream --format "{{.CPUPerc}},{{.MemUsage}}" $id 2>/dev/null || echo "N/A,N/A")
            cpu=$(echo $stats | cut -d',' -f1)
            mem=$(echo $stats | cut -d',' -f2)
            
            # Truncate image name
            image_short=$(echo $image | cut -c1-30)
            status_short=$(echo $status | cut -c1-15)
            
            printf "%-12s %-30s %-15s %8s %12s\n" "${id:0:12}" "$image_short" "$status_short" "$cpu" "$mem"
        done
    else
        echo "No running containers"
    fi
    echo "└────────────────────────────────────────────────────────────────┘"
    echo ""
    
    # Disk Space
    echo "┌─ DISK SPACE ───────────────────────────────────────────────────┐"
    df -h / /var/lib/docker | awk 'NR==1{printf "%-20s %8s %8s %8s %5s %s\n", $1, $2, $3, $4, $5, $6} NR>1{printf "%-20s %8s %8s %8s %5s %s\n", $1, $2, $3, $4, $5, $6}'
    echo "└────────────────────────────────────────────────────────────────┘"
    echo ""
    
    # OOM Events
    OOM_COUNT=$(dmesg | grep -c "killed process" 2>/dev/null || echo 0)
    if [ $OOM_COUNT -gt 0 ]; then
        echo "┌─ ⚠️  OOM KILLER EVENTS: $OOM_COUNT ──────────────────────────────────────┐"
        dmesg | grep "killed process" | tail -3
        echo "└────────────────────────────────────────────────────────────────┘"
        echo ""
    fi
    
    # Footer
    echo "Press Ctrl+C to exit | Refreshes every 5 seconds"
    echo "For detailed logs: bash scripts/monitor-runner.sh"
    echo "After container death: bash scripts/diagnose-container-death.sh"
}

# Main loop
while true; do
    draw_dashboard
    sleep 5
done


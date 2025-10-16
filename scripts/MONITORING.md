# GPU Runner Monitoring & Diagnostics

This directory contains tools to diagnose why Docker containers are being killed during MLE-bench runs.

## Problem

Docker containers die mid-competition with no clear error message. Common causes:
- **Out of Memory (OOM)** - System or Docker runs out of RAM
- **GPU OOM** - CUDA out of memory errors
- **Disk Full** - /var/lib/docker fills up
- **Process killed** - System OOM killer terminates processes

## Setup (One-Time)

### 1. Configure Docker Logging

Run once on each GPU runner to enable persistent logs:

```bash
cd ~/canada-research
sudo bash ./scripts/setup-docker-logging.sh
```

This configures Docker to:
- Keep logs after container death (500MB per container)
- Enable log compression
- Enable live-restore (containers survive daemon restarts)

### 2. Install tmux (for persistent monitoring)

```bash
sudo apt-get install -y tmux
```

## Monitoring Tools

### Option 1: Real-Time Dashboard (Recommended)

Lightweight dashboard that refreshes every 5 seconds:

```bash
# Start in background tmux session
tmux new -d -s monitoring 'bash scripts/runner-dashboard.sh'

# Attach to see dashboard
tmux attach -t monitoring

# Detach: Ctrl+B then D
# Kill: Ctrl+C
```

**Shows:**
- System memory usage (with color-coded bars)
- GPU utilization and memory
- Docker container status and resource usage
- Disk space
- OOM killer events

### Option 2: Detailed Monitoring with Logs

Captures detailed snapshots every 30 seconds to a log file:

```bash
bash scripts/monitor-runner.sh
```

**Captures:**
- System memory (every 30s)
- CPU usage (top 5 processes)
- GPU status (nvidia-smi)
- Docker container stats
- Disk space
- OOM killer events
- Docker daemon errors

**Log location:** `~/runner-monitoring/runner-<hostname>-<timestamp>.log`

## Diagnosis After Container Death

When a container dies unexpectedly:

```bash
bash scripts/diagnose-container-death.sh
```

**Shows:**
- Container exit code and interpretation
- Whether OOM killed the container
- Last 50 lines of container logs
- System OOM events around death time
- Docker daemon errors
- Current system resources
- **Recommended solutions** based on cause

## Common Exit Codes

| Code | Meaning | Likely Cause |
|------|---------|--------------|
| 0 | Normal exit | Task completed successfully |
| 137 | SIGKILL | **OOM killer** or manual kill |
| 139 | Segmentation fault | Memory corruption / bad pointer |
| 143 | SIGTERM | Graceful termination requested |

## OOM Killer Detection

Check if system OOM killer has been active:

```bash
# Check OOM events
dmesg | grep -i "killed process"

# Check Docker OOM kills
docker inspect <container_id> | grep OOMKilled

# Monitor memory continuously
watch -n 1 free -h
```

## Solutions by Problem Type

### 🔴 Out of Memory (OOM)

**Symptoms:**
- Exit code 137
- "OOMKilled": true in docker inspect
- "killed process" in dmesg

**Solutions:**
1. **Increase system memory** (if possible)
2. **Reduce batch size** in training scripts
3. **Use gradient accumulation** instead of large batches
4. **Enable gradient checkpointing** (trades compute for memory)
5. **Monitor training**: Add memory tracking to training scripts
6. **Limit container memory** explicitly to fail fast:
   ```bash
   docker run --memory=16g --memory-swap=16g ...
   ```

### 🔴 GPU Out of Memory

**Symptoms:**
- "CUDA out of memory" in logs
- "RuntimeError: CUDA error" in logs

**Solutions:**
1. **Reduce batch size** (most effective)
2. **Use mixed precision** (fp16/bf16)
3. **Enable gradient checkpointing**
4. **Use smaller model** or fewer layers
5. **Clear GPU cache**: `torch.cuda.empty_cache()`
6. **Monitor GPU memory**:
   ```bash
   watch -n 1 nvidia-smi
   ```

### 🔴 Disk Full

**Symptoms:**
- "No space left on device" in logs
- df shows 100% usage

**Solutions:**
1. **Clean Docker**:
   ```bash
   docker system prune -af
   docker container prune -f
   docker image prune -af
   ```
2. **Clean old runs**:
   ```bash
   cd ~/canada-research/mle-bench
   rm -rf runs/*/  # Keep only latest
   ```
3. **Monitor disk**:
   ```bash
   df -h /var/lib/docker
   ```

## Best Practices

### Before Runs

```bash
# Check disk space
df -h /var/lib/docker

# Check memory
free -h

# Check GPU
nvidia-smi

# Start monitoring
tmux new -d -s monitoring 'bash scripts/runner-dashboard.sh'
```

### During Runs

```bash
# Watch dashboard
tmux attach -t monitoring

# Or tail monitoring logs
tail -f ~/runner-monitoring/*.log
```

### After Container Death

```bash
# Diagnose immediately
bash scripts/diagnose-container-death.sh > diagnosis-$(date +%Y%m%d-%H%M%S).txt

# Check what caused it
cat diagnosis-*.txt
```

## Monitoring During CI/CD

The GitHub Actions workflow already captures:
- Container logs (available in artifacts)
- Exit codes
- Resource usage

But **real-time monitoring on the runner** helps catch issues early.

## Advanced Monitoring

### Continuous Resource Logging

Run in background to log all resources every 10 seconds:

```bash
# Create systemd service for monitoring
sudo tee /etc/systemd/system/runner-monitor.service > /dev/null <<EOF
[Unit]
Description=GPU Runner Monitoring Service
After=docker.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/canada-research
ExecStart=/bin/bash /home/ubuntu/canada-research/scripts/monitor-runner.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable runner-monitor.service
sudo systemctl start runner-monitor.service

# View logs
journalctl -u runner-monitor.service -f
```

### GPU Memory Profiling

Add to training scripts:

```python
import torch

def log_gpu_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")

# Call periodically during training
log_gpu_memory()
```

## Quick Reference

```bash
# Setup (once)
sudo bash scripts/setup-docker-logging.sh
sudo apt-get install -y tmux

# Start monitoring
tmux new -d -s monitoring 'bash scripts/runner-dashboard.sh'

# View dashboard
tmux attach -t monitoring

# After container dies
bash scripts/diagnose-container-death.sh

# Check OOM events
dmesg | grep -i "killed process"

# Check Docker logs
journalctl -u docker.service | tail -100

# Clean Docker
docker system prune -af
```

## Files

- `monitor-runner.sh` - Detailed monitoring with logs
- `diagnose-container-death.sh` - Post-mortem diagnostics
- `setup-docker-logging.sh` - Configure Docker logging
- `runner-dashboard.sh` - Real-time dashboard
- `MONITORING.md` - This file

## Troubleshooting

**Q: Container logs are empty after death**
A: Run `setup-docker-logging.sh` to enable persistent logs

**Q: How do I know if OOM killed my container?**
A: Run `diagnose-container-death.sh` - it checks OOMKilled status

**Q: Dashboard shows high memory but container didn't OOM**
A: Could be disk full, GPU OOM, or timeout - check all diagnostics

**Q: nvidia-smi shows GPU OOM but container still running**
A: Process might be stuck - check with `docker stats` and `docker logs`

## Support

For issues or questions, check the container logs first:

```bash
# Get container ID
docker ps -a

# View logs
docker logs <container_id>

# Run full diagnosis
bash scripts/diagnose-container-death.sh
```


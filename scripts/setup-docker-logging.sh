#!/bin/bash
# Configure Docker for better logging and diagnostics
# Run once on each GPU runner

set -e

echo "=========================================="
echo "Docker Logging Configuration"
echo "=========================================="
echo ""

# Backup existing daemon.json if it exists
DAEMON_JSON="/etc/docker/daemon.json"
if [ -f "$DAEMON_JSON" ]; then
    echo "Backing up existing daemon.json..."
    sudo cp "$DAEMON_JSON" "$DAEMON_JSON.backup-$(date +%Y%m%d-%H%M%S)"
    echo "✅ Backup created"
fi

# Create optimized daemon.json with logging
echo "Creating Docker daemon configuration..."
sudo tee "$DAEMON_JSON" > /dev/null <<'EOF'
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m",
    "max-file": "5",
    "compress": "true"
  },
  "default-ulimits": {
    "nofile": {
      "Name": "nofile",
      "Hard": 64000,
      "Soft": 64000
    }
  },
  "storage-driver": "overlay2",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia",
  "live-restore": true,
  "userland-proxy": false
}
EOF

echo "✅ Configuration written"
echo ""
echo "New Docker configuration:"
cat "$DAEMON_JSON"
echo ""

# Validate JSON
echo "Validating configuration..."
if command -v python3 &> /dev/null; then
    python3 -m json.tool "$DAEMON_JSON" > /dev/null && echo "✅ Valid JSON"
else
    echo "⚠️  python3 not found, skipping validation"
fi
echo ""

# Restart Docker daemon
echo "Restarting Docker daemon..."
sudo systemctl restart docker
sleep 3

# Check Docker status
echo "Checking Docker status..."
sudo systemctl status docker --no-pager | head -10
echo ""

# Verify NVIDIA runtime
echo "Verifying NVIDIA runtime..."
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
echo ""

echo "=========================================="
echo "✅ Docker Logging Configured!"
echo "=========================================="
echo ""
echo "Configuration details:"
echo "  • Max log size: 100MB per file"
echo "  • Max log files: 5 (500MB total per container)"
echo "  • Compression: enabled"
echo "  • Live restore: enabled (containers survive daemon restart)"
echo "  • NVIDIA runtime: default"
echo ""
echo "Logs location:"
echo "  /var/lib/docker/containers/<container_id>/<container_id>-json.log"
echo ""
echo "To view logs after container death:"
echo "  bash scripts/diagnose-container-death.sh"
echo ""


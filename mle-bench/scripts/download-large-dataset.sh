#!/bin/bash
# Download large dataset with resume capability using wget
# Usage: ./scripts/download-large-dataset.sh siim-isic-melanoma-classification

set -e

COMPETITION_ID="${1:-siim-isic-melanoma-classification}"
CACHE_DIR="$HOME/.cache/mle-bench/data/$COMPETITION_ID"

echo "=========================================="
echo "Large Dataset Downloader with Resume"
echo "=========================================="
echo "Competition: $COMPETITION_ID"
echo "Cache dir: $CACHE_DIR"
echo ""

# Create cache directory
mkdir -p "$CACHE_DIR"

# Use mlebench prepare with retry logic
echo "Attempting download with automatic retry..."
echo ""

MAX_RETRIES=5
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo "Attempt $((RETRY_COUNT + 1))/$MAX_RETRIES..."

    if mlebench prepare -c "$COMPETITION_ID"; then
        echo ""
        echo "✅ Download completed successfully!"
        exit 0
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            echo ""
            echo "⚠️  Download failed, retrying in 30 seconds..."
            sleep 30
        fi
    fi
done

echo ""
echo "❌ Failed after $MAX_RETRIES attempts"
echo ""
echo "Alternative: Download manually from Kaggle website"
echo "  1. Go to https://www.kaggle.com/competitions/$COMPETITION_ID/data"
echo "  2. Download the zip file"
echo "  3. Place it in: $CACHE_DIR/"
echo "  4. Run: mlebench prepare -c $COMPETITION_ID"
exit 1

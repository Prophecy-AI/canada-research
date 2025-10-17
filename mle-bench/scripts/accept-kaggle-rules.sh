#!/bin/bash
# Script to help accept Kaggle competition rules for all competitions in a set
# Opens each competition's rules page in your browser
#
# Usage:
#   ./scripts/accept-kaggle-rules.sh [competition-set-file]
#
# Example:
#   ./scripts/accept-kaggle-rules.sh experiments/splits/custom-set.txt

set -e

# Default to custom-set.txt if no argument provided
COMPETITION_SET="${1:-experiments/splits/custom-set.txt}"

# Check if file exists
if [ ! -f "$COMPETITION_SET" ]; then
    echo "❌ ERROR: Competition set file not found: $COMPETITION_SET"
    echo ""
    echo "Usage:"
    echo "  ./scripts/accept-kaggle-rules.sh [competition-set-file]"
    echo ""
    echo "Example:"
    echo "  ./scripts/accept-kaggle-rules.sh experiments/splits/custom-set.txt"
    exit 1
fi

# Count competitions
TOTAL=$(grep -v '^#' "$COMPETITION_SET" | grep -v '^$' | wc -l | tr -d ' ')

if [ "$TOTAL" -eq 0 ]; then
    echo "❌ ERROR: No competitions found in $COMPETITION_SET"
    exit 1
fi

echo "=========================================="
echo "Kaggle Competition Rules Accepter"
echo "=========================================="
echo "Competition set: $COMPETITION_SET"
echo "Total competitions: $TOTAL"
echo ""
echo "This script will:"
echo "  1. Open each competition's rules page in your browser"
echo "  2. Wait for you to accept the rules"
echo "  3. Move to the next competition"
echo ""
echo "For each competition:"
echo "  - Click 'I Understand and Accept' at the bottom"
echo "  - Return to terminal and press Enter"
echo ""
read -p "Press Enter to start..."
echo ""

# Counter
COUNT=0

# Read each line from the file
while IFS= read -r competition_id || [ -n "$competition_id" ]; do
    # Skip empty lines and comments
    if [ -z "$competition_id" ] || [[ "$competition_id" =~ ^# ]]; then
        continue
    fi

    COUNT=$((COUNT + 1))

    echo "=========================================="
    echo "[$COUNT/$TOTAL] $competition_id"
    echo "=========================================="

    RULES_URL="https://www.kaggle.com/c/$competition_id/rules"

    echo "Opening: $RULES_URL"
    echo ""

    # Open in browser (works on macOS, Linux, WSL)
    if command -v open &> /dev/null; then
        # macOS
        open "$RULES_URL"
    elif command -v xdg-open &> /dev/null; then
        # Linux
        xdg-open "$RULES_URL"
    elif command -v wslview &> /dev/null; then
        # WSL
        wslview "$RULES_URL"
    else
        echo "⚠️  Could not auto-open browser. Please manually visit:"
        echo "   $RULES_URL"
    fi

    echo "Instructions:"
    echo "  1. Scroll to bottom of the page"
    echo "  2. Click 'I Understand and Accept'"
    echo "  3. Return here and press Enter to continue"
    echo ""

    # If already accepted, user can just press Enter
    echo "(If you've already accepted, just press Enter to skip)"
    read -p "Press Enter after accepting rules..."

    echo "✅ Marked as accepted"
    echo ""

    # Small delay to avoid overwhelming the browser
    sleep 1

done < "$COMPETITION_SET"

echo "=========================================="
echo "✅ COMPLETE!"
echo "=========================================="
echo "Processed $COUNT competitions"
echo ""
echo "You can now run:"
echo "  cd mle-bench"
echo "  ./RUN_AGENT_V5_KAGGLE.sh"
echo ""
echo "Or prepare datasets manually:"
echo "  mlebench prepare --competition-set $COMPETITION_SET"
echo ""

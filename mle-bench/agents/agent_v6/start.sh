#!/bin/bash
set -e
set -x

echo "========================================="
echo "Agent V6 - New Architecture"
echo "========================================="
echo "Competition ID: ${COMPETITION_ID}"
echo "Code dir: ${CODE_DIR}"
echo "Data dir: /home/data"
echo "Submission dir: ${SUBMISSION_DIR}"
echo "Time limit: ${TIME_LIMIT_MINUTES:-240} minutes"
echo "========================================="

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate agent

# Change to code directory
cd ${CODE_DIR}

# Run the agent with new main.py from core folder
python -u ${AGENT_DIR}/agent_v6/core/main.py 2>&1 | tee ${LOGS_DIR}/agent.log

# Check if submission was created
if [ -f ${SUBMISSION_DIR}/submission.csv ]; then
    echo "✅ Submission file created successfully"
    echo "File size: $(wc -c < ${SUBMISSION_DIR}/submission.csv) bytes"
    echo "Line count: $(wc -l < ${SUBMISSION_DIR}/submission.csv) lines"
    echo ""
    echo "First 5 lines of submission:"
    head -5 ${SUBMISSION_DIR}/submission.csv
    echo ""
else
    echo "❌ ERROR: No submission file found at ${SUBMISSION_DIR}/submission.csv"
    echo "Contents of ${SUBMISSION_DIR}:"
    ls -la ${SUBMISSION_DIR}
    
    # Check workspace for submission
    if [ -f ${CODE_DIR}/submission.csv ]; then
        echo "Found submission in workspace, copying..."
        cp ${CODE_DIR}/submission.csv ${SUBMISSION_DIR}/submission.csv
    fi
    
    exit 1
fi

echo "========================================="
echo "Agent V6 - Complete"
echo "=========================================

"
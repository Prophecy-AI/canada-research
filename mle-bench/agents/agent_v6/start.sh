#!/bin/bash
set -e
set -x

echo "========================================="
echo "Agent V6 - Start"
echo "========================================="
echo "Competition ID: ${COMPETITION_ID}"
echo "Code dir: ${CODE_DIR}"
echo "Data dir: /home/data"
echo "Submission dir: ${SUBMISSION_DIR}"
echo "========================================="

eval "$(conda shell.bash hook)"
conda activate agent

cd ${CODE_DIR}

python -u ${AGENT_DIR}/agent_v6/main.py 2>&1 | tee ${LOGS_DIR}/agent.log

if [ -f ${SUBMISSION_DIR}/submission.csv ]; then
    echo "✅ Submission file created"
    echo "File size: $(wc -c < ${SUBMISSION_DIR}/submission.csv) bytes"
    echo "Line count: $(wc -l < ${SUBMISSION_DIR}/submission.csv) lines"
    echo ""
    echo "First 5 lines:"
    head -5 ${SUBMISSION_DIR}/submission.csv
    echo ""
else
    echo "❌ ERROR: No submission file found at ${SUBMISSION_DIR}/submission.csv"
    echo "Contents of ${SUBMISSION_DIR}:"
    ls -la ${SUBMISSION_DIR}
    exit 1
fi

echo "========================================="
echo "Agent V6 - Complete"
echo "========================================="


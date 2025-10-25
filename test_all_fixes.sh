#!/bin/bash
# Test script to verify all fixes are working
# Run this locally before deploying to GitHub Actions

set -e

echo "============================================================"
echo "Testing All Fixes - MLE-Bench Agent"
echo "============================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Function to print test result
test_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ PASS${NC}: $2"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}❌ FAIL${NC}: $2"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

echo "Test 1: Check import fix - tools/__init__.py exists"
echo "------------------------------------------------------------"
if [ -f "mle-bench/agents/agent_v5_kaggle/tools/__init__.py" ]; then
    test_result 0 "tools/__init__.py exists"
else
    test_result 1 "tools/__init__.py missing"
fi
echo ""

echo "Test 2: Check import fix - kaggle_agent.py uses absolute import"
echo "------------------------------------------------------------"
if grep -q "from tools.gpu_validate import GPUValidateTool" mle-bench/agents/agent_v5_kaggle/kaggle_agent.py; then
    test_result 0 "kaggle_agent.py uses absolute import"
else
    test_result 1 "kaggle_agent.py still uses relative import"
fi
echo ""

echo "Test 3: Check config.yaml has GEMINI_API_KEY"
echo "------------------------------------------------------------"
if grep -q "GEMINI_API_KEY:" mle-bench/agents/agent_v5_kaggle/config.yaml; then
    test_result 0 "config.yaml declares GEMINI_API_KEY"
else
    test_result 1 "config.yaml missing GEMINI_API_KEY"
fi
echo ""

echo "Test 4: Check GitHub workflow has GEMINI_API_KEY"
echo "------------------------------------------------------------"
if grep -q "GEMINI_API_KEY:" .github/workflows/run-mle-bench.yml; then
    test_result 0 "GitHub workflow exports GEMINI_API_KEY"
else
    test_result 1 "GitHub workflow missing GEMINI_API_KEY"
fi
echo ""

echo "Test 5: Check agent.py uses correct Gemini model name"
echo "------------------------------------------------------------"
if grep -q 'model="gemini-2.5-pro"' agent_v5/agent.py; then
    test_result 0 "agent.py uses gemini-2.5-pro (correct)"
elif grep -q 'model="gemini-2.5-pro-002"' agent_v5/agent.py; then
    test_result 1 "agent.py still uses gemini-2.5-pro-002 (incorrect)"
else
    test_result 1 "agent.py model name not found or unexpected"
fi
echo ""

echo "Test 6: Check Python import simulation"
echo "------------------------------------------------------------"
cd mle-bench/agents/agent_v5_kaggle
if python3 -c "import sys; sys.path.insert(0, '.'); from tools.gpu_validate import GPUValidateTool; print('Import successful')" 2>&1 | grep -q "Import successful"; then
    test_result 0 "Python import works"
else
    test_result 1 "Python import fails"
fi
cd ../../..
echo ""

echo "Test 7: Check GEMINI_API_KEY environment variable"
echo "------------------------------------------------------------"
if [ -n "$GEMINI_API_KEY" ]; then
    test_result 0 "GEMINI_API_KEY is set (${#GEMINI_API_KEY} characters)"

    echo ""
    echo "Test 8: Test Gemini API with correct model name"
    echo "------------------------------------------------------------"
    if [ -f "test_gemini_model.py" ]; then
        if python3 test_gemini_model.py > /dev/null 2>&1; then
            test_result 0 "Gemini API test passed"
        else
            echo -e "${YELLOW}⚠️  WARN${NC}: Gemini API test failed (check API key or network)"
            TESTS_FAILED=$((TESTS_FAILED + 1))
        fi
    else
        test_result 1 "test_gemini_model.py not found"
    fi
else
    echo -e "${YELLOW}⚠️  SKIP${NC}: GEMINI_API_KEY not set (cannot test API)"
    echo "         Set it with: export GEMINI_API_KEY='your-key-here'"
    echo "         Then re-run: bash test_all_fixes.sh"
fi
echo ""

echo "============================================================"
echo "Test Summary"
echo "============================================================"
echo -e "Tests Passed: ${GREEN}${TESTS_PASSED}${NC}"
echo -e "Tests Failed: ${RED}${TESTS_FAILED}${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ ALL TESTS PASSED!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Ensure GEMINI_API_KEY secret is set in GitHub repository"
    echo "   (Settings → Secrets → Actions → GEMINI_API_KEY)"
    echo ""
    echo "2. Commit and push changes:"
    echo "   git add ."
    echo "   git commit -m \"Fix import, env var, and Gemini model name issues\""
    echo "   git push"
    echo ""
    echo "3. Trigger GitHub Actions workflow"
    echo "   (Actions → Run MLE-Bench Agent → Run workflow)"
    echo ""
    exit 0
else
    echo -e "${RED}❌ SOME TESTS FAILED${NC}"
    echo ""
    echo "Please fix the failing tests before deploying."
    echo "See the test output above for details."
    echo ""
    exit 1
fi

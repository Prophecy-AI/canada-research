#!/usr/bin/env python3
"""
Real API integration tests for summarize_logs.py

These tests make actual calls to the OpenAI API to validate:
- Model response handling
- Chunking behavior with real data
- End-to-end summarization workflow
- Response parsing robustness

Requirements:
- OPENAI_API_KEY environment variable must be set
- Tests are skipped if API key is not available
- Costs real money (minimal - uses small test cases)

Following agent_v5 pattern: Real API tests (not mocked)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from openai import OpenAI

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).parent))
import summarize_logs as sl


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_workspace():
    """Create temporary workspace for isolated testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def openai_client():
    """Real OpenAI client - requires OPENAI_API_KEY"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set - skipping real API tests")
    return OpenAI(api_key=api_key)


@pytest.fixture
def sample_agent_log():
    """Realistic agent log with issues and successes"""
    return """[2025-01-15 10:00:00] INFO: Agent initialized for competition: titanic
[2025-01-15 10:00:01] INFO: Loading training data from train.csv
[2025-01-15 10:00:02] INFO: Training data shape: (891, 12)
[2025-01-15 10:00:03] INFO: Starting exploratory data analysis
[2025-01-15 10:00:05] ERROR: Missing dependency: scikit-learn
Traceback (most recent call last):
  File "train.py", line 10, in <module>
    from sklearn.ensemble import RandomForestClassifier
ModuleNotFoundError: No module named 'sklearn'
[2025-01-15 10:00:10] INFO: Installing scikit-learn via pip
[2025-01-15 10:00:45] INFO: Successfully installed scikit-learn==1.4.0
[2025-01-15 10:00:50] INFO: Feature engineering - creating Age_Group feature
[2025-01-15 10:00:51] INFO: Feature engineering - extracting Title from Name
[2025-01-15 10:00:55] INFO: Handling missing values in Age column
[2025-01-15 10:01:00] INFO: Training Random Forest model
[2025-01-15 10:01:30] INFO: Model training complete
[2025-01-15 10:01:31] INFO: Cross-validation score: 0.8234
[2025-01-15 10:01:35] INFO: Generating predictions on test set
[2025-01-15 10:01:40] INFO: Creating submission file: submission.csv
[2025-01-15 10:01:41] INFO: Submission complete - 418 predictions
[2025-01-15 10:01:42] WARNING: Model accuracy is below 0.85, consider feature engineering
[2025-01-15 10:01:45] INFO: Agent run complete"""


@pytest.fixture
def sample_agent_log_with_errors():
    """Agent log with multiple critical errors"""
    return """[2025-01-15 10:00:00] INFO: Agent initialized
[2025-01-15 10:00:05] ERROR: CUDA out of memory
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
[2025-01-15 10:00:10] INFO: Falling back to CPU
[2025-01-15 10:05:00] ERROR: File not found: test_data.csv
FileNotFoundError: [Errno 2] No such file or directory: 'test_data.csv'
[2025-01-15 10:05:05] INFO: Attempting to download test_data.csv
[2025-01-15 10:05:30] ERROR: Network timeout after 30s
requests.exceptions.Timeout: HTTPSConnectionPool(host='kaggle.com', port=443)
[2025-01-15 10:05:35] INFO: Agent terminated with errors"""


@pytest.fixture
def create_test_run_structure(temp_workspace):
    """Factory to create test run directory structure"""
    def _create(run_name: str, log_content: str):
        run_dir = temp_workspace / run_name
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Create agent.log
        log_file = logs_dir / "agent.log"
        log_file.write_text(log_content)

        return run_dir

    return _create


# ============================================================================
# Real API Tests - Model Interaction
# ============================================================================

class TestCallModelRealAPI:
    """Test actual OpenAI API calls"""

    def test_successful_log_analysis(self, openai_client, sample_agent_log):
        """Should successfully analyze a real agent log"""
        result = sl.call_model(
            openai_client,
            model=sl.DEFAULT_MODEL,
            system_prompt=sl.SYSTEM_PROMPT,
            user_prompt=f"Competition: titanic\n\nFull log:\n{sample_agent_log}\n\nGenerate the required summary.",
            reasoning_effort="low",  # Use low effort for faster/cheaper tests
            max_output_tokens=1000
        )

        # Verify response structure
        assert isinstance(result, str)
        assert len(result) > 0

        # Verify key sections are present (case-insensitive check)
        result_lower = result.lower()
        assert "issues" in result_lower or "issue" in result_lower
        assert "good decisions" in result_lower or "decision" in result_lower

        # Should mention the actual issue from the log
        assert "scikit-learn" in result_lower or "sklearn" in result_lower or "dependency" in result_lower

        print("\n=== Model Response ===")
        print(result)
        print("=== End Response ===\n")

    def test_handles_error_heavy_log(self, openai_client, sample_agent_log_with_errors):
        """Should identify and categorize multiple errors"""
        result = sl.call_model(
            openai_client,
            model=sl.DEFAULT_MODEL,
            system_prompt=sl.SYSTEM_PROMPT,
            user_prompt=f"Competition: test\n\nFull log:\n{sample_agent_log_with_errors}\n\nGenerate the required summary.",
            reasoning_effort="low",
            max_output_tokens=1000
        )

        assert isinstance(result, str)
        assert len(result) > 0

        # Should identify environment errors
        result_lower = result.lower()
        assert "cuda" in result_lower or "memory" in result_lower or "error" in result_lower
        assert "file not found" in result_lower or "network" in result_lower or "timeout" in result_lower

        print("\n=== Error Analysis ===")
        print(result)
        print("=== End Analysis ===\n")

    def test_response_format_consistency(self, openai_client):
        """Should return consistent format across multiple calls"""
        simple_log = "[INFO] Model trained\n[INFO] Accuracy: 0.95\n[INFO] Complete"

        results = []
        for _ in range(2):  # Make 2 calls to check consistency
            result = sl.call_model(
                openai_client,
                model=sl.DEFAULT_MODEL,
                system_prompt=sl.SYSTEM_PROMPT,
                user_prompt=f"Competition: test\n\nFull log:\n{simple_log}\n\nGenerate the required summary.",
                reasoning_effort="low",
                max_output_tokens=500
            )
            results.append(result)

        # Both should be non-empty strings
        assert all(isinstance(r, str) and len(r) > 0 for r in results)

        # Both should have the required headings (though content may vary slightly)
        for result in results:
            result_lower = result.lower()
            # At least some of the expected sections should be present
            has_sections = any(
                keyword in result_lower
                for keyword in ["issues", "decisions", "observations", "errors"]
            )
            assert has_sections


# ============================================================================
# Real API Tests - Summarization Logic
# ============================================================================

class TestSummarizeLogRealAPI:
    """Test log summarization with real API calls"""

    def test_short_log_single_call(self, openai_client, sample_agent_log):
        """Short log should use single model call"""
        result = sl.summarize_log(
            openai_client,
            model=sl.DEFAULT_MODEL,
            competition_name="titanic",
            log_text=sample_agent_log,
            chunk_size=50000,  # Much larger than log
            reasoning_effort="low",
            max_output_tokens=1000
        )

        assert isinstance(result, str)
        assert len(result) > 0

        # Should mention competition name
        assert "titanic" in result.lower() or "competition" in result.lower()

        # Should identify key insights
        result_lower = result.lower()
        assert "scikit-learn" in result_lower or "sklearn" in result_lower or "dependency" in result_lower

        print("\n=== Short Log Summary ===")
        print(result)
        print("=== End Summary ===\n")

    def test_long_log_chunking(self, openai_client):
        """Long log should be chunked and summarized"""
        # Create a long log by repeating content
        base_log = """[INFO] Training iteration 1
[ERROR] Memory warning
[INFO] Model checkpoint saved
[INFO] Validation accuracy: 0.80
"""
        # Repeat to make it ~1500 chars
        long_log = base_log * 15

        result = sl.summarize_log(
            openai_client,
            model=sl.DEFAULT_MODEL,
            competition_name="test-comp",
            log_text=long_log,
            chunk_size=500,  # Will create ~3 chunks
            reasoning_effort="low",
            max_output_tokens=1000
        )

        assert isinstance(result, str)
        assert len(result) > 0

        # Should still identify patterns despite chunking
        result_lower = result.lower()
        assert "memory" in result_lower or "error" in result_lower or "warning" in result_lower

        print("\n=== Chunked Log Summary ===")
        print(f"Original log length: {len(long_log)} chars")
        print(f"Summary length: {len(result)} chars")
        print(result)
        print("=== End Summary ===\n")


# ============================================================================
# Real API Tests - End-to-End Workflow
# ============================================================================

class TestSummarizeRunsRealAPI:
    """Test full workflow with real API calls"""

    def test_single_run_end_to_end(
        self,
        openai_client,
        temp_workspace,
        create_test_run_structure,
        sample_agent_log
    ):
        """Complete workflow: log file → API call → saved summary"""
        # Create run structure
        run_dir = create_test_run_structure("titanic_abc123", sample_agent_log)
        output_dir = temp_workspace / "output"
        output_dir.mkdir()

        # Execute
        summaries = sl.summarize_runs(
            temp_workspace,
            output_dir,
            model=sl.DEFAULT_MODEL,
            chunk_size=50000,
            reasoning_effort="low",
            max_output_tokens=1000
        )

        # Verify results
        assert len(summaries) == 1
        assert summaries[0].run_name == "titanic_abc123"
        assert isinstance(summaries[0].summary, str)
        assert len(summaries[0].summary) > 0

        # Verify output files
        run_output = output_dir / "titanic_abc123"
        assert run_output.exists()
        assert (run_output / "full_log").exists()
        assert (run_output / "gpt-summary").exists()

        # Verify file contents
        full_log = (run_output / "full_log").read_text()
        assert full_log == sample_agent_log

        gpt_summary = (run_output / "gpt-summary").read_text()
        assert len(gpt_summary) > 0
        assert "scikit-learn" in gpt_summary.lower() or "sklearn" in gpt_summary.lower()

        print("\n=== End-to-End Test Results ===")
        print(f"Run: {summaries[0].run_name}")
        print(f"Summary length: {len(summaries[0].summary)} chars")
        print(f"Summary preview:\n{summaries[0].summary[:300]}...")
        print("=== End Results ===\n")

    def test_multiple_runs_end_to_end(
        self,
        openai_client,
        temp_workspace,
        create_test_run_structure,
        sample_agent_log,
        sample_agent_log_with_errors
    ):
        """Should process multiple runs independently"""
        # Create multiple runs
        create_test_run_structure("comp-a_123", sample_agent_log)
        create_test_run_structure("comp-b_456", sample_agent_log_with_errors)

        output_dir = temp_workspace / "output"
        output_dir.mkdir()

        # Execute
        summaries = sl.summarize_runs(
            temp_workspace,
            output_dir,
            model=sl.DEFAULT_MODEL,
            chunk_size=50000,
            reasoning_effort="low",
            max_output_tokens=1000
        )

        # Verify both processed
        assert len(summaries) == 2
        run_names = [s.run_name for s in summaries]
        assert "comp-a_123" in run_names
        assert "comp-b_456" in run_names

        # Verify different summaries for different content
        summary_a = next(s.summary for s in summaries if s.run_name == "comp-a_123")
        summary_b = next(s.summary for s in summaries if s.run_name == "comp-b_456")

        # They should be different
        assert summary_a != summary_b

        # Summary A should mention sklearn
        assert "scikit-learn" in summary_a.lower() or "sklearn" in summary_a.lower()

        # Summary B should mention errors
        assert "cuda" in summary_b.lower() or "memory" in summary_b.lower() or "error" in summary_b.lower()

        print("\n=== Multiple Runs Test ===")
        print(f"Processed {len(summaries)} runs")
        for summary in summaries:
            print(f"\n{summary.run_name}:")
            print(f"  Length: {len(summary.summary)} chars")
        print("=== End Test ===\n")

    def test_zip_creation_end_to_end(
        self,
        openai_client,
        temp_workspace,
        create_test_run_structure,
        sample_agent_log
    ):
        """Should create valid ZIP archive after summarization"""
        # Create run
        create_test_run_structure("test_run_123", sample_agent_log)

        output_dir = temp_workspace / "output"
        output_dir.mkdir()

        # Summarize
        sl.summarize_runs(
            temp_workspace,
            output_dir,
            model=sl.DEFAULT_MODEL,
            chunk_size=50000,
            reasoning_effort="low",
            max_output_tokens=1000
        )

        # Create ZIP
        zip_path = temp_workspace / "summaries.zip"
        result = sl.zip_output_directory(output_dir, zip_path)

        # Verify ZIP
        assert result == zip_path
        assert zip_path.exists()
        assert zip_path.stat().st_size > 0

        # Verify it's a valid ZIP (can be opened)
        import zipfile
        assert zipfile.is_zipfile(zip_path)

        with zipfile.ZipFile(zip_path, 'r') as zf:
            files = zf.namelist()
            # Should contain both full_log and gpt-summary
            assert any("full_log" in f for f in files)
            assert any("gpt-summary" in f for f in files)

        print("\n=== ZIP Creation Test ===")
        print(f"ZIP size: {zip_path.stat().st_size} bytes")
        print(f"Files in ZIP: {len(files)}")
        print("=== End Test ===\n")


# ============================================================================
# Real API Tests - Edge Cases
# ============================================================================

class TestEdgeCasesRealAPI:
    """Test edge cases with real API"""

    def test_very_short_log(self, openai_client):
        """Should handle minimal log content"""
        short_log = "[INFO] Agent started\n[INFO] Agent finished"

        result = sl.summarize_log(
            openai_client,
            model=sl.DEFAULT_MODEL,
            competition_name="minimal",
            log_text=short_log,
            chunk_size=50000,
            reasoning_effort="low",
            max_output_tokens=500
        )

        assert isinstance(result, str)
        assert len(result) > 0

        print("\n=== Very Short Log Summary ===")
        print(result)
        print("=== End Summary ===\n")

    def test_log_with_unicode(self, openai_client):
        """Should handle logs with unicode characters"""
        unicode_log = """[INFO] Processing data 🚀
[INFO] Model trained successfully ✅
[ERROR] Failed to save: 文件不存在
[WARNING] Performance degraded: Δ = -0.05"""

        result = sl.summarize_log(
            openai_client,
            model=sl.DEFAULT_MODEL,
            competition_name="unicode-test",
            log_text=unicode_log,
            chunk_size=50000,
            reasoning_effort="low",
            max_output_tokens=500
        )

        assert isinstance(result, str)
        assert len(result) > 0

        print("\n=== Unicode Log Summary ===")
        print(result)
        print("=== End Summary ===\n")


# ============================================================================
# Test Summary
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])  # -s to see print output

#!/usr/bin/env python3
"""
Summarize agent logs with Claude Sonnet.

This script collects `agent.log` files extracted from competition containers,
optionally chunks them, feeds them to the configured Anthropic Claude model,
and stores both the raw log and the generated summary in a directory structure
suitable for artifact upload:

<output_dir>/
  <run_id>/
    full_log.txt
    claude-summary.txt

Finally, the directory is zipped so GitHub Actions can upload it as a single
artifact.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from anthropic import Anthropic


# Defaults can be overridden through CLI flags or environment.
DEFAULT_MODEL = os.getenv("CLAUDE_SUMMARY_MODEL", "claude-sonnet-4-5-20250929")
DEFAULT_OUTPUT_DIR = "log_reviews"
DEFAULT_ZIP_NAME = "log_reviews.zip"
MAX_CHARS_PER_CALL = int(os.getenv("LOG_SUMMARY_MAX_CHARS_PER_CALL", "12000"))
MAX_OUTPUT_TOKENS = int(os.getenv("LOG_SUMMARY_MAX_OUTPUT_TOKENS", "8000"))


SYSTEM_PROMPT = """You are an expert reviewer of autonomous Kaggle agent runs with a specific mission:
**Extract structured insights to improve future agent performance.**

Your analysis will be used to:
1. Identify what worked and what failed
2. Update the agent's strategy playbook (kaggle_competition_strategy.txt)
3. Improve system prompts and error handling
4. Build competition memory for future similar tasks

You will receive:
- Full agent execution logs (code, tool calls, outputs, errors)
- Final grading results (score, medal status, thresholds)

You must produce a **structured analysis** following the exact format specified in the final summary instructions."""

CHUNK_SUMMARY_INSTRUCTIONS = """Summarize this log fragment concisely. Focus on:
- Agent's actions and decisions
- Code generated and executed
- Tool usage patterns
- Errors and issues encountered (with timestamps)
- Results and outcomes
- GPU/CPU utilization metrics if present
- Oracle consultations if any
Keep it brief. This is one part of a larger log."""

FINAL_SUMMARY_INSTRUCTIONS = """Now synthesize all chunks into a cohesive run report following this EXACT structure:

## COMPETITION: [competition_name]
**Medal Achieved:** [gold/silver/bronze/none - extract from grading results]
**Score:** [X.XX] (Gold: X.XX | Silver: X.XX | Bronze: X.XX | Median: X.XX)
**Time Budget:** [estimate target from logs] → [calculate actual from timestamps] ([±Z min])

---

### ✅ WHAT WORKED
- Strategy choice: [e.g., "EfficientNet-B2 aligned with playbook ✓"]
- Resource usage: [e.g., "GPU 85%, all 36 cores used ✓"]
- CV strategy: [e.g., "3-fold StratifiedKFold appropriate ✓"]
- Code quality: [e.g., "Clean implementation, minimal bugs ✓"]
- [List 3-5 specific things that went well with evidence from logs]

### ❌ WHAT FAILED
- Technical issues: [e.g., "OpenCV libGL missing - wasted 8 min debugging"]
- Strategy misalignments: [e.g., "Used 5-fold CV (should be 3 for speed)"]
- Bugs: [e.g., "Label encoding bug - predictions in wrong order"]
- Oracle failures: [e.g., "Oracle suggested X but made score worse"]
- [List all failures with timestamps and impact]

### ⚡ EFFICIENCY ANALYSIS
- **GPU utilization:** [X%] (target 70-90%) - extract from nvidia-smi output if present
- **CPU utilization:** [n_jobs setting, num_workers] - check if all 36 cores used
- **Time breakdown:** [Train: X min | Debug: Y min | Inference: Z min] - calculate from timestamps
- **Bottlenecks:** [e.g., "num_workers=4 caused CPU bottleneck, should be 30-36"]
- **Resource underutilization:** [List any inefficiencies like small batch sizes]

### 🎯 SCORE ANALYSIS
- **CV/LB alignment:** [CV X.XX vs LB Y.YY] → [Good/Mismatch/Unknown]
- **Medal gap:** [Calculate: score - threshold] from gold → [Realistic/Unrealistic to close]
- **Improvement potential:** [High/Medium/Low]
- **Reasoning:** [Why agent scored this way - what was the fundamental issue or success factor?]
- **Score context:** [e.g., "Above/below median by X.XX"]

### 🔮 ORACLE CONSULTATIONS
- **Count:** [X times - count from logs]
- **Quality:** [Helpful/Neutral/Harmful - analyze before/after metrics]
- **Examples:**
  - [Turn X: Oracle suggested [Y] → outcome [Z]]
  - [Turn Y: Oracle advice on [topic] → [result]]
- **Self-awareness:** [Did Oracle acknowledge prior mistakes? Did it repeat bad advice?]
- **Net impact:** [Did Oracle consultations help or hurt overall?]

### 💡 LESSONS LEARNED (Add to playbook)
1. [e.g., "Install libgl1-mesa-glx in Dockerfile to avoid OpenCV libGL.so.1 errors"]
2. [e.g., "For 30-min budget, parallel training (3 small models) > 1 large model"]
3. [e.g., "batch_size=32 on A10 24GB = severe underutilization, use 128-256"]
4. [e.g., "albumentations requires cv2 which needs OpenGL libraries in Docker"]
[List 3-7 concrete, actionable lessons with specific technical details]

### 🔄 WHAT TO TRY NEXT TIME
- [e.g., "Try parallel LightGBM + ResNet ensemble for diversity bonus"]
- [e.g., "Use StratifiedGroupKFold instead of GroupKFold (better for this data structure)"]
- [e.g., "Add MixUp augmentation (playbook says +1-2% for image tasks)"]
- [e.g., "Reduce to 2 folds to save time if single model estimated >25 min"]
[List 3-5 specific alternative approaches that could have worked better]

---

**CRITICAL GUIDELINES:**
- Be brutally honest about failures - this is for learning, not vanity
- Use specific evidence from logs (exact timestamps, error messages, code snippets)
- Calculate time estimates from log timestamps (e.g., "Training started 12:34:56, ended 12:47:23 = 12.5 min")
- Extract GPU/CPU metrics from any nvidia-smi, top, htop, or monitoring output
- Analyze grading results mathematically (score vs each threshold, gap calculations)
- Identify alignment/misalignment with kaggle_competition_strategy.txt patterns
- Call out when Oracle advice helped vs when it caused regression
- Focus on actionable insights that can update system prompts or playbook
- If information is missing, say "Unknown" rather than guessing"""


@dataclass
class LogSummary:
    run_name: str
    log_path: Path
    summary: str


def chunk_text(text: str, chunk_size: int) -> Iterable[str]:
    for idx in range(0, len(text), chunk_size):
        yield text[idx : idx + chunk_size]


def call_model(
    client: Anthropic,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=max_output_tokens,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0,  # Deterministic responses for consistency
    )

    # Extract text from response content blocks
    text_parts = []
    for block in response.content:
        if hasattr(block, "text"):
            text_parts.append(block.text)

    if text_parts:
        return "\n".join(text_parts)

    raise RuntimeError("Model response did not contain any text content.")


def summarize_log(
    client: Anthropic,
    *,
    model: str,
    competition_name: str,
    log_text: str,
    chunk_size: int,
    max_output_tokens: int,
) -> str:
    if len(log_text) <= chunk_size:
        user_prompt = (
            f"Competition: {competition_name}\n\n"
            f"Full log:\n{log_text}\n\n"
            "Generate the required summary."
        )
        return call_model(
            client,
            model=model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens,
        )

    fragment_summaries: list[str] = []
    for idx, fragment in enumerate(chunk_text(log_text, chunk_size), start=1):
        user_prompt = (
            f"Competition: {competition_name}\n"
            f"Log chunk {idx} (of unknown total):\n{fragment}\n\n"
            f"{CHUNK_SUMMARY_INSTRUCTIONS}"
        )
        fragment_summary = call_model(
            client,
            model=model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens,
        )
        fragment_summaries.append(f"Chunk {idx} summary:\n{fragment_summary}")

    combined_prompt = (
        f"Competition: {competition_name}\n\n"
        "You are given per-chunk notes. Create a unified run report.\n\n"
        + "\n\n".join(fragment_summaries)
        + "\n\n"
        + FINAL_SUMMARY_INSTRUCTIONS
    )

    return call_model(
        client,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=combined_prompt,
        max_output_tokens=max_output_tokens,
    )


def infer_competition_name(run_dir: Path) -> str:
    # The run directory is typically "<competition_id>_<uuid>".
    # Use the first component as a readable competition identifier.
    name_parts = run_dir.name.split("_")
    return name_parts[0] if name_parts else run_dir.name


def collect_run_dirs(run_group_dir: Path) -> Sequence[Path]:
    if not run_group_dir.exists():
        raise FileNotFoundError(f"Run group directory not found: {run_group_dir}")
    return sorted([p for p in run_group_dir.iterdir() if p.is_dir()])


def summarize_runs(
    run_group_dir: Path,
    output_dir: Path,
    *,
    model: str,
    chunk_size: int,
    max_output_tokens: int,
) -> list[LogSummary]:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is required for log summarization.")

    client = Anthropic(api_key=api_key)
    summaries: list[LogSummary] = []

    # Load grading report from run group level (shared across all competitions)
    grading_data = None
    grading_report_files = list(run_group_dir.glob("*_grading_report.json"))
    if not grading_report_files:
        # Fallback to results.json
        results_path = run_group_dir / "results.json"
        if results_path.exists():
            grading_report_files = [results_path]

    if grading_report_files:
        try:
            import json
            grading_data = json.loads(grading_report_files[0].read_text(errors="replace"))
            print(f"[info] Loaded grading report: {grading_report_files[0].name}", file=sys.stderr)
        except Exception as e:
            print(f"[warn] Failed to parse grading report: {e}", file=sys.stderr)

    for run_dir in collect_run_dirs(run_group_dir):
        log_path = run_dir / "logs" / "agent.log"
        if not log_path.exists():
            print(f"[warn] No agent.log found for {run_dir.name}, skipping.", file=sys.stderr)
            continue

        competition_name = infer_competition_name(run_dir)
        target_dir = output_dir / run_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)

        # Copy raw log so users get full context.
        shutil.copyfile(log_path, target_dir / "full_log.txt")

        log_text = log_path.read_text(errors="replace")

        # Append grading results for this specific competition
        grading_text = ""
        if grading_data and "competition_reports" in grading_data:
            # Find the grading report for this specific competition
            for comp_report in grading_data["competition_reports"]:
                if comp_report.get("competition_id") == competition_name:
                    import json
                    grading_json = json.dumps(comp_report, indent=2)
                    grading_text = f"\n\n{'='*80}\nGRADING RESULTS FOR {competition_name}:\n{'='*80}\n{grading_json}\n{'='*80}\n"
                    break

        # Combine log text with grading results
        combined_log_text = log_text + grading_text

        try:
            summary_text = summarize_log(
                client,
                model=model,
                competition_name=competition_name,
                log_text=combined_log_text,
                chunk_size=chunk_size,
                max_output_tokens=max_output_tokens,
            )
        except Exception as err:  # noqa: BLE001
            summary_text = (
                "Failed to generate summary.\n"
                f"Competition: {competition_name}\n"
                f"Run: {run_dir.name}\n"
                f"Error: {err}"
            )
            print(f"[error] Summary failed for {run_dir.name}: {err}", file=sys.stderr)

        (target_dir / "claude-summary.txt").write_text(summary_text)
        summaries.append(LogSummary(run_dir.name, log_path, summary_text))

    return summaries


def zip_output_directory(output_dir: Path, output_zip: Path) -> Path:
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    if output_zip.exists():
        output_zip.unlink()

    base_name = output_zip.with_suffix("")
    shutil.make_archive(str(base_name), "zip", root_dir=output_dir)
    return output_zip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize agent logs with Claude.")
    parser.add_argument(
        "--run-group",
        required=True,
        help="Name of the run group under runs/ to process.",
    )
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Root directory that contains run groups (default: runs).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Destination directory for summaries (default: <run_group>/<DEFAULT_OUTPUT_DIR>).",
    )
    parser.add_argument(
        "--output-zip",
        default=None,
        help="Path to the ZIP archive to generate (default: <run_group>/<DEFAULT_ZIP_NAME>).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Anthropic model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=MAX_CHARS_PER_CALL,
        help=f"Max characters per model call before chunking (default: {MAX_CHARS_PER_CALL}).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=MAX_OUTPUT_TOKENS,
        help=f"Max output tokens per call (default: {MAX_OUTPUT_TOKENS}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    runs_root = Path(args.runs_root).resolve()
    run_group_dir = runs_root / args.run_group

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else run_group_dir / DEFAULT_OUTPUT_DIR
    )
    output_zip = (
        Path(args.output_zip).resolve()
        if args.output_zip
        else run_group_dir / DEFAULT_ZIP_NAME
    )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summarize_runs(
        run_group_dir,
        output_dir,
        model=args.model,
        chunk_size=args.chunk_size,
        max_output_tokens=args.max_output_tokens,
    )

    zip_output_directory(output_dir, output_zip)
    print(f"Log summaries written to {output_dir} and archived at {output_zip}")


if __name__ == "__main__":
    main()

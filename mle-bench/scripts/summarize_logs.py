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


SYSTEM_PROMPT = """You are an expert reviewer of autonomous Kaggle agent runs.
- You inspect raw execution logs and extract actionable intelligence.
- Always be concrete and reference log snippets if helpful.
- Organize the final response with the following top-level headings:
  Issues, Good Decisions, Bad Decisions, Environment Errors, Missing Dependencies,
  Other Observations, and Recommended Next Actions.
- If a section has nothing to report, write `None`.
- Treat missing or truncated logs as a critical issue."""

CHUNK_SUMMARY_INSTRUCTIONS = """Summarize this log fragment. Focus on:
- Failures, stack traces, or suspicious warnings.
- Successful actions and their impact on score or workflow.
- Configuration or environment notes (missing dependencies, CUDA/S3 issues, etc.).
- Anything that should propagate into the final run summary."""

FINAL_SUMMARY_INSTRUCTIONS = """You are composing the final report for the entire run.
Synthesize the provided fragment summaries into a single concise report using the
required headings:
- Issues
- Good Decisions
- Bad Decisions
- Environment Errors
- Missing Dependencies
- Other Observations
- Recommended Next Actions

Highlight severity and confidence. If information is inconclusive, say so."""


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
        try:
            summary_text = summarize_log(
                client,
                model=model,
                competition_name=competition_name,
                log_text=log_text,
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

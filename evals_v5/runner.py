"""Async eval runner - fire and forget quality checks"""
import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Literal
from openai import AsyncOpenAI
from debug import log

EvalType = Literal["hallucination", "retrieval", "sql", "answer", "code", "objective"]


class EvalRunner:
    """Orchestrate async evals using OpenAI o3"""

    def __init__(self, session_id: str, workspace_dir: str):
        self.session_id = session_id
        self.workspace_dir = workspace_dir
        self.enabled = os.getenv("EVALS_ENABLED") == "1"
        self.client = AsyncOpenAI() if self.enabled else None
        self.evals_dir = Path(workspace_dir) / ".evals"

        if self.enabled:
            self.evals_dir.mkdir(exist_ok=True)

    def submit(self, eval_type: EvalType, data: Dict):
        """Submit eval asynchronously (fire and forget)"""
        if not self.enabled:
            return

        asyncio.create_task(self._run_eval(eval_type, data))

    def submit_batch(self, eval_requests: list[tuple[EvalType, Dict]]):
        """
        Submit multiple evals to run in parallel with asyncio.gather.

        Args:
            eval_requests: List of (eval_type, data) tuples

        Example:
            runner.submit_batch([
                ("hallucination", {"answer": "...", "data": "..."}),
                ("sql", {"sql": "...", "context": "..."}),
                ("answer", {"question": "...", "answer": "..."})
            ])
        """
        if not self.enabled or not eval_requests:
            return

        asyncio.create_task(self._run_batch_evals(eval_requests))

    async def _run_batch_evals(self, eval_requests: list[tuple[EvalType, Dict]]):
        """Run multiple evaluations in parallel using asyncio.gather"""
        try:
            log(f"→ Running {len(eval_requests)} evals in parallel")

            # Execute all evals concurrently
            results = await asyncio.gather(
                *[self._run_eval(eval_type, data)
                  for eval_type, data in eval_requests],
                return_exceptions=True
            )

            # Log any failures
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    eval_type = eval_requests[i][0]
                    log(f"✗ Batch eval {eval_type} failed: {result}", 2)

        except Exception as e:
            log(f"✗ Batch eval execution failed: {e}", 2)

    async def _run_eval(self, eval_type: EvalType, data: Dict):
        """Run specific eval type"""
        try:
            evaluator = self._get_evaluator(eval_type)
            result = await evaluator.evaluate(data, self.client)

            # Store result
            timestamp = int(asyncio.get_event_loop().time())
            result_file = self.evals_dir / f"{eval_type}_{timestamp}.json"
            result_file.write_text(json.dumps({
                "eval_type": eval_type,
                "data": data,
                "result": result,
                "timestamp": timestamp
            }, indent=2))

            log(f"Eval {eval_type}: {result.get('score', 'N/A')}", 1)

        except Exception as e:
            log(f"Eval {eval_type} failed: {e}", 2)

    def _get_evaluator(self, eval_type: EvalType):
        """Get evaluator instance for type"""
        from . import hallucination, retrieval, sql, answer, code, objective

        evaluators = {
            "hallucination": hallucination.HallucinationEvaluator(),
            "retrieval": retrieval.RetrievalEvaluator(),
            "sql": sql.SQLEvaluator(),
            "answer": answer.AnswerEvaluator(),
            "code": code.CodeEvaluator(),
            "objective": objective.ObjectiveEvaluator(),
        }

        return evaluators[eval_type]

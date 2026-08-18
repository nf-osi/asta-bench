import argparse
import asyncio
import json
from pathlib import Path

from inspect_ai.log import EvalLog, read_eval_log
from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    ResponseSchema,
    get_model,
)
from inspect_ai.util import json_schema
from pydantic import BaseModel

from astabench.evals.nf_rag_pubs.task import _extract_response


VERDICT_TO_SCORE = {
    "correct": 1.0,
    "partially_correct": 0.5,
    "incorrect": 0.0,
}


class SemanticJudgement(BaseModel):
    verdict: str
    reason: str


SEMANTIC_JUDGEMENT_SCHEMA = ResponseSchema(
    name="semantic_judgement",
    json_schema=json_schema(SemanticJudgement),
    strict=True,
)


SYSTEM_PROMPT = """You are grading short answers for a scientific literature question answering benchmark.

You will be given:
- a question
- the ideal answer
- the model's predicted answer

Judge how well the predicted answer matches the ideal answer semantically.

Use exactly one of these verdicts:
- correct: the predicted answer is semantically equivalent to the ideal answer
- partially_correct: the predicted answer is meaningfully related and partly right, but incomplete, overly broad, imprecise, or missing an essential qualifier
- incorrect: the predicted answer is wrong, unsupported by the ideal answer, too vague to count, or missing

Important rules:
- Focus on answer meaning, not phrasing.
- Minor wording differences, formatting differences, and harmless synonyms should still be correct.
- If the prediction omits an essential constraint, entity, number, comparison direction, or qualifier, mark partially_correct or incorrect.
- If the prediction contains a correct core answer plus extra wrong information, downgrade it.
- Do not invent facts beyond what is in the ideal answer.

Return JSON with exactly:
{
  "verdict": "correct" | "partially_correct" | "incorrect",
  "reason": "<brief explanation>"
}
"""


def _resolve_logs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]

    candidates = sorted(path.rglob("*.eval"))
    if not candidates:
        candidates = sorted(path.rglob("*.json"))
    return candidates


def _is_short_answer_sample(sample) -> bool:
    if sample.metadata.get("answer_format") == "short_answer":
        return True

    try:
        target = json.loads(sample.target)
    except (TypeError, json.JSONDecodeError):
        return False

    return target.get("answer_format") == "short_answer"


def _build_report_path(log_path: Path) -> Path:
    if log_path.suffix:
        return log_path.with_name(f"{log_path.stem}.short_answer_semantic_scores.json")
    return log_path / "short_answer_semantic_scores.json"


def _extract_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= start:
        raise ValueError("Could not find JSON object in judge response")
    return json.loads(text[start:end])


def _extract_question_text(sample_input: str) -> str:
    marker = "Question: "
    if marker in sample_input:
        return sample_input.rsplit(marker, 1)[-1].strip()
    return sample_input.strip()


async def _judge_sample(model, sample, batch: bool) -> dict:
    target = json.loads(sample.target)
    ideal_answer = str(target.get("ideal_answer", "")).strip()
    _, predicted_answer, _ = _extract_response(sample.output.completion)
    question_text = _extract_question_text(str(sample.input))

    result = {
        "sample_id": sample.id,
        "epoch": sample.epoch,
        "question": question_text,
        "ideal_answer": ideal_answer,
        "predicted_answer": predicted_answer,
    }

    if not predicted_answer:
        result["verdict"] = "incorrect"
        result["score"] = 0.0
        result["reason"] = "No short answer could be extracted from the model output."
        return result

    user_prompt = f"""Question:
{question_text}

Ideal answer:
{ideal_answer}

Predicted answer:
{predicted_answer}
"""

    output = await model.generate(
        [
            ChatMessageSystem(content=SYSTEM_PROMPT),
            ChatMessageUser(content=user_prompt),
        ],
        config=GenerateConfig(
            response_schema=SEMANTIC_JUDGEMENT_SCHEMA,
            temperature=0,
            batch=batch,
        ),
    )

    parsed = _extract_json_object(output.completion)
    verdict = parsed["verdict"]
    if verdict not in VERDICT_TO_SCORE:
        raise ValueError(f"Unexpected verdict '{verdict}' for sample {sample.id}")

    result["verdict"] = verdict
    result["score"] = VERDICT_TO_SCORE[verdict]
    result["reason"] = parsed["reason"]
    return result


def _summarize(results: list[dict]) -> dict:
    total = len(results)
    if total == 0:
        return {
            "total_samples": 0,
            "semantic_score": 0.0,
            "semantic_accuracy": 0.0,
            "partial_rate": 0.0,
            "incorrect_rate": 0.0,
        }

    correct = sum(1 for row in results if row["verdict"] == "correct")
    partial = sum(1 for row in results if row["verdict"] == "partially_correct")
    incorrect = sum(1 for row in results if row["verdict"] == "incorrect")
    mean_score = sum(row["score"] for row in results) / total

    return {
        "total_samples": total,
        "semantic_score": mean_score,
        "semantic_accuracy": correct / total,
        "partial_rate": partial / total,
        "incorrect_rate": incorrect / total,
    }


async def _score_log(
    log_path: Path,
    judge_model_name: str,
    output_path: Path | None,
    batch: bool,
) -> Path:
    eval_log: EvalLog = read_eval_log(log_path)
    if not eval_log.samples:
        raise ValueError(f"No samples found in log: {log_path}")

    samples = [sample for sample in eval_log.samples if _is_short_answer_sample(sample)]
    if not samples:
        raise ValueError(f"No short_answer samples found in log: {log_path}")

    async with get_model(judge_model_name) as judge_model:
        results = []
        for sample in samples:
            results.append(await _judge_sample(judge_model, sample, batch=batch))

    report = {
        "log_path": str(log_path),
        "judge_model": judge_model_name,
        "batch": batch,
        "summary": _summarize(results),
        "samples": results,
    }

    destination = output_path or _build_report_path(log_path)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return destination


async def _main_async(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    log_paths = _resolve_logs(input_path)
    if not log_paths:
        raise ValueError(f"No eval logs found under {input_path}")

    if args.output and len(log_paths) > 1:
        raise ValueError("--output can only be used when scoring a single log file")

    for log_path in log_paths:
        destination = await _score_log(
            log_path=log_path,
            judge_model_name=args.judge_model,
            output_path=Path(args.output) if args.output else None,
            batch=args.batch,
        )
        print(f"Wrote semantic short-answer scores to {destination}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Postprocess nf_rag_pubs short-answer eval logs with an LLM judge."
    )
    parser.add_argument(
        "input",
        help="Path to an Inspect eval log file or a directory containing eval logs.",
    )
    parser.add_argument(
        "--judge-model",
        default="openai/gpt-4.1-mini",
        help="Judge model name understood by Inspect.",
    )
    parser.add_argument(
        "--output",
        help="Optional output JSON path. Only valid when scoring a single log.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Request batch execution from the model provider if supported.",
    )
    args = parser.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())

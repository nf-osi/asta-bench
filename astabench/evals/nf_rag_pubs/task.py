"""NF Knowledge Graph Publication RAG eval.

The agent receives a multiple-choice question about NF research and must:
1. Query the SPARQL+Text endpoint to retrieve relevant publication passages
2. Select the correct answer choice
3. Cite the supporting passages with attribution (PMID, passage number)

Scored on two separate metrics:
- accuracy: correct multiple-choice answer
- passage_f1: F1 over (pmid, passage_num) attribution tuples
"""

import json
import logging
import os
import re
from pathlib import Path

import httpx
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import (
    Metric,
    SampleScore,
    Score,
    Scorer,
    Target,
    metric,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState, use_tools
from inspect_ai.tool import Tool, ToolError, tool

from astabench.evals.utils import not_implemented_solver

logger = logging.getLogger(__name__)

GROUND_TRUTH_PATH = Path(__file__).resolve().parent / "eval_data.yaml"

INSTRUCTION_PREFIX = """\
You have access to a SPARQL+Text interface for the NF-OSI knowledge graph \
with indexed publication text from NF research papers. Use the provided tools \
to search for relevant passages, then answer the multiple-choice question and \
cite the passages that support your answer.

IMPORTANT: No clarification will be provided. Interpret the question as best \
you can and select the most appropriate answer choice.

## SPARQL+Text query examples

Word search:
SELECT ?text WHERE { ?text ql:contains-word "neurofibromatosis" } LIMIT 10

Prefix search:
SELECT ?text WHERE { ?text ql:contains-word "schwann*" } LIMIT 10

Entity + word (every ql:contains-entity MUST be paired with ql:contains-word):
SELECT ?text WHERE {
  ?text ql:contains-entity <https://www.ncbi.nlm.nih.gov/gene/4763> .
  ?text ql:contains-word "mutation*"
} LIMIT 10

Entity-only (use "*" wildcard to satisfy the word requirement):
SELECT ?text WHERE {
  ?text ql:contains-entity <https://www.ncbi.nlm.nih.gov/gene/4763> .
  ?text ql:contains-word "*"
} LIMIT 10

## Response format

Return your final answer as a JSON object with TWO fields:
{
  "answer": "<choice letter>",
  "attribution": [
    {"pmid": "<pmid>", "passage": <number>},
    ...
  ]
}

Each passage in the text index includes an attribution tag in the format: \
[PMID{pmid}-{passage_num}-{section_type}]. Extract the pmid and passage \
number from retrieved text to populate the attribution array.

Return ONLY the JSON object as your final answer, with no other text around it.

Question: """

SPARQL_ENDPOINT = os.environ.get("SPARQL_ENDPOINT", "http://localhost:7001")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def sparql_text_query() -> Tool:
    """Execute a SPARQL+Text query against the NF knowledge graph.

    Returns results as tab-separated values (TSV).
    The graph uses prefix nf: <http://nf-osi.github.com/terms#>.
    Text queries use ql:contains-word and ql:contains-entity predicates.
    """

    async def execute(query: str) -> str:
        """Execute a SPARQL+Text query against the NF knowledge graph.

        Returns results as tab-separated values (TSV).
        The graph uses prefix nf: <http://nf-osi.github.com/terms#>.
        Text queries use ql:contains-word and ql:contains-entity predicates.

        Every ql:contains-entity must be paired with ql:contains-word in the same pattern.
        Use ql:contains-word "*" as a wildcard when only entity matching is needed.

        Args:
            query: The SPARQL+Text query to execute.
        """
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(
                    SPARQL_ENDPOINT,
                    params={"query": query, "action": "tsv_export"},
                    timeout=30,
                )
                r.raise_for_status()
                return r.text
        except httpx.HTTPStatusError as e:
            raise ToolError(f"SPARQL error {e.response.status_code}: {e.response.text}")
        except httpx.RequestError as e:
            raise ToolError(f"Request failed: {e}")

    return execute


@tool
def get_schema() -> Tool:
    """Return all classes and properties defined in the NF ontology.

    Use this to discover the graph structure before writing queries.
    """

    async def execute() -> str:
        """Return all classes and properties defined in the NF ontology.

        Use this to discover the graph structure before writing queries.
        """
        q = """\
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?term ?kind ?label ?comment ?domain ?range WHERE {
  {
    ?term a owl:Class .
    BIND("Class" AS ?kind)
  } UNION {
    ?term a owl:ObjectProperty .
    BIND("ObjectProperty" AS ?kind)
  } UNION {
    ?term a owl:DatatypeProperty .
    BIND("DatatypeProperty" AS ?kind)
  }
  OPTIONAL { ?term rdfs:label ?label }
  OPTIONAL { ?term rdfs:comment ?comment }
  OPTIONAL { ?term rdfs:domain ?domain }
  OPTIONAL { ?term rdfs:range ?range }
} ORDER BY ?kind ?term"""
        async with httpx.AsyncClient() as client:
            r = await client.get(
                SPARQL_ENDPOINT,
                params={"query": q, "action": "tsv_export"},
                timeout=30,
            )
            r.raise_for_status()
            return r.text

    return execute


@tool
def get_entity_types() -> Tool:
    """Return counts of text entities by type.

    Shows what biomedical entity types are available in the text index:
    Gene, DiseaseAnnotation, Chemical, Species, CellLine, Variant.
    """

    async def execute() -> str:
        """Return counts of text entities by type.

        Shows what biomedical entity types are available in the text index:
        Gene, DiseaseAnnotation, Chemical, Species, CellLine, Variant.
        """
        q = """\
PREFIX nf: <http://nf-osi.github.com/terms#>
PREFIX obo: <http://purl.obolibrary.org/obo/>

SELECT ?type (COUNT(?e) AS ?count) WHERE {
  VALUES ?type { nf:Gene nf:DiseaseAnnotation nf:Chemical obo:NCBITaxon_species nf:CellLine nf:Variant }
  ?e a ?type
}
GROUP BY ?type
ORDER BY DESC(?count)"""
        async with httpx.AsyncClient() as client:
            r = await client.get(
                SPARQL_ENDPOINT,
                params={"query": q, "action": "tsv_export"},
                timeout=30,
            )
            r.raise_for_status()
            return r.text

    return execute


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ground_truth(
    path: Path = GROUND_TRUTH_PATH,
    question_style: str = "precise",
) -> list[Sample]:
    """Load eval_data.yaml and convert to inspect_ai Samples.

    Args:
        path: Path to eval_data.yaml.
        question_style: Which question field to use.
            "precise" uses the carefully worded ``question`` field.
            "user_query" uses the colloquial ``user_query`` field.
    """
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f)

    field = "user_query" if question_style == "user_query" else "question"

    samples = []
    for qid, entry in data["ground_truth"].items():
        question = entry[field]
        choices = entry["choices"]
        correct_idx = entry["correct_choice_index"]
        pmcid = entry["pmcid"]
        passage_indices = entry["passage_indices"]

        # Build multiple-choice prompt
        choices_text = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
        full_question = f"{question}\n\n{choices_text}"

        # Target encodes both correct choice and expected passages
        target = {
            "correct_choice_index": correct_idx,
            "attribution": [
                {"pmid": pmcid.replace("PMC", ""), "passage": idx}
                for idx in passage_indices
            ],
        }

        samples.append(
            Sample(
                id=qid,
                input=INSTRUCTION_PREFIX + full_question,
                target=json.dumps(target),
                metadata={
                    "category": qid.rsplit("-", 1)[0],
                    "difficulty": entry.get("difficulty", "unknown"),
                    "question_type": entry.get("question_type", "unknown"),
                },
            )
        )

    return samples


# ---------------------------------------------------------------------------
# Response extraction
# ---------------------------------------------------------------------------

ATTRIBUTION_RE = re.compile(r"\[PMID(\d+)-(\d+)-\w+\]")


def _extract_response(text: str) -> tuple[int | None, list[dict]]:
    """Extract answer choice index and passage list from agent output.

    Returns (choice_index_or_None, [{pmid, passage}, ...]).
    """
    choice_idx = None
    passages: list[dict] = []

    # Try parsing as JSON object with "answer" and "attribution" keys
    try:
        obj_start = text.find("{")
        obj_end = text.rfind("}") + 1
        if obj_start != -1 and obj_end > obj_start:
            parsed = json.loads(text[obj_start:obj_end])
            if isinstance(parsed, dict):
                # Extract choice
                answer = str(parsed.get("answer", "")).strip()
                if len(answer) == 1 and answer.upper() in "ABCDEFGH":
                    choice_idx = ord(answer.upper()) - 65
                elif answer.isdigit():
                    choice_idx = int(answer)

                # Extract passages
                raw_passages = parsed.get("attribution", [])
                if isinstance(raw_passages, list):
                    for p in raw_passages:
                        if isinstance(p, dict) and "pmid" in p and "passage" in p:
                            passages.append({
                                "pmid": str(p["pmid"]).strip(),
                                "passage": int(p["passage"]),
                            })
                    if choice_idx is not None or passages:
                        return choice_idx, passages
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: look for choice letter at start of line
    choice_match = re.search(r"^([A-D])\b", text, re.MULTILINE)
    if choice_match:
        choice_idx = ord(choice_match.group(1).upper()) - 65

    # Fallback: extract attribution tags from text
    for m in ATTRIBUTION_RE.finditer(text):
        passages.append({"pmid": m.group(1), "passage": int(m.group(2))})

    return choice_idx, passages


# ---------------------------------------------------------------------------
# Metrics and scorers
# ---------------------------------------------------------------------------

@metric
def accuracy() -> Metric:
    """Accuracy: fraction of questions with correct answer choice."""

    def metric_fn(scores: list[SampleScore]) -> float:
        correct = sum(1 for s in scores if s.metadata.get("answer_correct", False))
        return correct / max(1, len(scores))

    return metric_fn


@metric
def passage_f1() -> Metric:
    """Mean F1 over passage attribution tuples."""

    def metric_fn(scores: list[SampleScore]) -> float:
        total = sum(s.metadata.get("f1", 0.0) for s in scores)
        return total / max(1, len(scores))

    return metric_fn


@scorer(metrics=[accuracy(), passage_f1(), stderr()])
def score_answer() -> Scorer:
    """Score correct answer choice."""

    async def score(state: TaskState, target: Target) -> Score:
        expected = json.loads(target.text)
        expected_idx = expected["correct_choice_index"]

        choice_idx, _ = _extract_response(state.output.completion)
        correct = choice_idx == expected_idx

        return Score(
            value=1.0 if correct else 0.0,
            answer=str(choice_idx),
            explanation=f"Predicted {choice_idx}, expected {expected_idx}",
            metadata={"answer_correct": correct, "f1": 0.0},
        )

    return score


@scorer(metrics=[accuracy(), passage_f1(), stderr()])
def score_attribution() -> Scorer:
    """Score passage attribution by F1."""

    async def score(state: TaskState, target: Target) -> Score:
        expected = json.loads(target.text)
        expected_passages = expected["attribution"]

        _, predicted_passages = _extract_response(state.output.completion)

        def to_key(p: dict) -> tuple:
            return (str(p.get("pmid", "")).strip(), int(p.get("passage", 0)))

        expected_set = {to_key(p) for p in expected_passages}
        predicted_set = {to_key(p) for p in predicted_passages}

        tp = len(expected_set & predicted_set)
        precision = tp / len(predicted_set) if predicted_set else 0.0
        recall = tp / len(expected_set) if expected_set else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return Score(
            value=f1,
            answer=json.dumps(predicted_passages),
            explanation=f"P={precision:.2f} R={recall:.2f} F1={f1:.2f} ({tp} hits, {len(expected_set)} expected, {len(predicted_set)} predicted)",
            metadata={"answer_correct": False, "f1": f1},
        )

    return score


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

@task
def nf_rag_pubs(
    task_filter: str | None = None,
    task_category: str | None = None,
    question_style: str = "precise",
):
    """NF Knowledge Graph Publication RAG eval.

    The agent queries an NF knowledge graph with SPARQL+Text,
    selects the correct answer, and cites supporting passages.

    Scored on two separate metrics:
    - accuracy: correct multiple-choice answer
    - passage_f1: F1 over (pmid, passage) attribution tuples

    Example rendered question::

        You have access to a SPARQL+Text interface for the NF-OSI knowledge
        graph with indexed publication text from NF research papers. [...]

        Question: What is the only currently FDA-approved pharmacotherapy for
        symptomatic and inoperable plexiform neurofibromas in pediatric NF1
        patients?

        A. Everolimus
        B. Selumetinib
        C. Tofacitinib
        D. Not in knowledgebase

    Expected response::

        {
          "answer": "B",
          "attribution": [
            {"pmid": "9221468", "passage": 1},
            {"pmid": "9221468", "passage": 3}
          ]
        }

    Args:
        task_filter: Comma-separated list of question IDs to include
                     (e.g. "PMC9221468-01,PMC9221468-02").
        task_category: Comma-separated list of category prefixes to include
                       (e.g. "PMC9221468,PMC3484870").
        question_style: "precise" (default) uses carefully worded questions;
                        "user_query" uses colloquial, realistic phrasings.
    """
    samples = load_ground_truth(question_style=question_style)

    if task_filter:
        ids = {t.strip() for t in task_filter.split(",")}
        samples = [s for s in samples if s.id in ids]

    if task_category:
        cats = {c.strip() for c in task_category.split(",")}
        samples = [s for s in samples if s.metadata["category"] in cats]

    if not samples:
        raise ValueError("No samples matched the filter criteria")

    tool_setups = [use_tools(sparql_text_query(), get_schema(), get_entity_types())]

    return Task(
        dataset=MemoryDataset(samples),
        solver=not_implemented_solver(),
        scorer=[score_answer(), score_attribution()],
        setup=tool_setups,
        config=GenerateConfig(max_tool_output=128 * 1024),
    )

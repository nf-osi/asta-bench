"""NF Knowledge Graph RAG eval.

The agent receives a natural-language question about NF research resources
and must query a SPARQL endpoint to retrieve the correct set of results.
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

DEFAULT_PREFIXES = """\
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX nf: <http://nf-osi.github.com/terms#>
PREFIX efo: <http://www.ebi.ac.uk/efo/>
PREFIX obo: <http://purl.obolibrary.org/obo/>
PREFIX prov: <http://www.w3.org/ns/prov#>
"""

INSTRUCTION_PREFIX = """\
You have access to a SPARQL interface for the NF-OSI knowledge graph of data resources and research tools. 
Use the provided tools to answer the following question.

Return your final answer as a JSON array of results, e.g.:
["uuid-1", "uuid-2", "uuid-3"] or [5] or ["name1", "name2"]

Most questions ask for uuid retrieval.
For these, each resource in the graph has nf:resourceId as a string property containing the canonical uuid.
Prefer using nf:resourceId over type-specific IDs (e.g. cellLineId, animalModelId) whenever possible. 
These prefixes are already available in every SPARQL query: nf, rdf, rdfs, owl, xsd, efo, obo, prov.
Use them directly in queries, for example `?tool a nf:CellLine ; nf:resourceId ?rid .`
Do not spend query budget re-declaring these common prefixes -- only declare other prefixes that you need.
For comparison questions, include all entities that tie for the best value unless the question explicitly asks for only one.
Prefer explicit attributes first and add searching text description if this makes sense and remaining resources permit.
Balance search depth and breadth with responsiveness to meet UX and resource requirements; submit an answer before the 45 tool use quotas are reached.

Useful schema sketch of the most important/common classes and properties (use get_schema tool for full ontology):

Study
  -> File
      -> hasModelSystem -> Tool
      -> hasNF1Genotype / hasNF2Genotype -> Genotype
      -> dataType -> Data

Tool
  -> hasMutation -> Mutation
  -> fromDonor -> Donor
  <- aboutResource - Observation - referencesPublication -> Publication
  <- hasResource - Biobank

Tool
  <- resource/development context -> Development
      -> hasPublication -> Publication
      -> hasInvestigator -> Investigator
      -> hasFunder -> Funder

Return ONLY the JSON array as your final answer, with no other text around it.

Question: """

SPARQL_ENDPOINT = os.environ.get("SPARQL_ENDPOINT", "http://localhost:7001")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def sparql_query() -> Tool:
    """Execute a SPARQL query against the NF knowledge graph.

    Returns results as tab-separated values (TSV).
    Common ontology prefixes are prepended automatically.
    """

    async def execute(query: str) -> str:
        """Execute a SPARQL query against the NF knowledge graph.

        Returns results as tab-separated values (TSV).
        Common ontology prefixes are prepended automatically.

        Args:
            query: The SPARQL query to execute.
        """
        try:
            async with httpx.AsyncClient() as client:
                query_with_prefixes = DEFAULT_PREFIXES + "\n" + query
                r = await client.get(
                    SPARQL_ENDPOINT,
                    params={"query": query_with_prefixes, "action": "tsv_export"},
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
def count_by_type() -> Tool:
    """Return instance counts grouped by rdf:type.

    Quick overview of what data is in the graph.
    """

    async def execute() -> str:
        """Return instance counts grouped by rdf:type.

        Quick overview of what data is in the graph.
        """
        q = """\
SELECT ?type (COUNT(?s) AS ?count) WHERE {
  ?s a ?type
} GROUP BY ?type ORDER BY DESC(?count)"""
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

def load_ground_truth(path: Path = GROUND_TRUTH_PATH) -> list[Sample]:
    """Load eval_data.yaml and convert to inspect_ai Samples."""
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f)

    samples = []
    for qid, entry in data["ground_truth"].items():
        question = entry["question"]
        results = entry["results"]

        # Normalise scalars (e.g. PI-002's count) into a single-element list
        # so scoring is uniform.
        if not isinstance(results, list):
            results = [results]

        samples.append(
            Sample(
                id=qid,
                input=INSTRUCTION_PREFIX + question,
                target=json.dumps(results),
                metadata={"category": qid.rsplit("-", 1)[0]},
            )
        )

    return samples


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


def _extract_results(text: str) -> list[str]:
    """Extract result values from agent output.

    Tries JSON array first, then falls back to UUID regex extraction.
    """
    # Try parsing as JSON array
    try:
        arr_start = text.find("[")
        arr_end = text.rfind("]") + 1
        if arr_start != -1 and arr_end > arr_start:
            parsed = json.loads(text[arr_start:arr_end])
            if isinstance(parsed, list):
                return [str(x).lower() for x in parsed]
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: extract anything that looks like a UUID
    return [m.lower() for m in UUID_RE.findall(text)]


@metric
def recall() -> Metric:
    """Recall: fraction of expected UUIDs found by the agent."""

    def metric_fn(scores: list[SampleScore]) -> float:
        return sum(s.score.value for s in scores) / max(1, len(scores))

    return metric_fn


@metric
def f1() -> Metric:
    """F1: harmonic mean of precision and recall over predicted UUIDs."""

    def metric_fn(scores: list[SampleScore]) -> float:
        values = []
        for sample_score in scores:
            metadata = sample_score.score.metadata or {}
            value = metadata.get("f1")
            if value is not None:
                values.append(float(value))
        return sum(values) / max(1, len(values))

    return metric_fn


@scorer(metrics=[recall(), f1(), stderr()])
def score_nf_retrieval() -> Scorer:
    """Score by recall over expected UUID sets."""

    async def score(state: TaskState, target: Target) -> Score:
        expected = set(str(v).lower() for v in json.loads(target.text))
        predicted = set(_extract_results(state.output.completion))

        if not expected:
            return Score(
                value=1.0,
                answer=str(predicted),
                explanation="No expected results",
                metadata={"precision": 1.0, "recall": 1.0, "f1": 1.0},
            )

        hits = expected & predicted
        r = len(hits) / len(expected)
        p = len(hits) / len(predicted) if predicted else 0.0
        f1_value = 2 * p * r / (p + r) if (p + r) else 0.0

        return Score(
            value=r,
            answer=json.dumps(sorted(predicted)),
            explanation=(
                f"Recall {len(hits)}/{len(expected)} = {r:.2f}; "
                f"Precision {len(hits)}/{len(predicted)} = {p:.2f}; "
                f"F1 = {f1_value:.2f}"
            ),
            metadata={"precision": p, "recall": r, "f1": f1_value},
        )

    return score


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

@task
def nf_rag(
    task_filter: str | None = None,
    task_category: str | None = None,
):
    """NF Knowledge Graph RAG eval.

    The agent queries an NF ontology knowledge graph via SPARQL
    and returns results matching the question.

    Args:
        task_filter: Comma-separated list of question IDs to include
                     (e.g. "CL-001,CL-002").
        task_category: Comma-separated list of category prefixes to include
                       (e.g. "CL,MUT").
    """
    samples = load_ground_truth()

    if task_filter:
        ids = {t.strip() for t in task_filter.split(",")}
        samples = [s for s in samples if s.id in ids]

    if task_category:
        cats = {c.strip() for c in task_category.split(",")}
        samples = [s for s in samples if s.metadata["category"] in cats]

    if not samples:
        raise ValueError("No samples matched the filter criteria")

    tool_setups = [use_tools(sparql_query(), get_schema(), count_by_type())]

    return Task(
        dataset=MemoryDataset(samples),
        solver=not_implemented_solver(),
        scorer=score_nf_retrieval(),
        setup=tool_setups,
        config=GenerateConfig(max_tool_output=128 * 1024),
    )

"""NF Knowledge Graph eval via KG MCP.

The agent receives a natural-language question about NF research resources
(cell lines, animal models, antibodies, mutations, etc.) and must query a
SPARQL endpoint exposed through an MCP server to retrieve the correct set of
resource UUIDs.
"""

import asyncio
import concurrent.futures
import json
import logging
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
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
from inspect_ai.tool import Tool, ToolSource
from inspect_ai.tool._mcp._mcp import MCPServerImpl
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client

from astabench.evals.utils import not_implemented_solver

logger = logging.getLogger(__name__)

GROUND_TRUTH_PATH = Path(__file__).resolve().parent / "eval_tools_ground.yaml"

SYSTEM_MESSAGE = """\
You have access to a SPARQL interface for the NF-OSI knowledge graph.
Use the provided tools to answer the question.

Return your final answer as a JSON array of resource UUIDs, e.g.:
["uuid-1", "uuid-2", "uuid-3"]

Return ONLY the JSON array as your final answer, with no other text around it."""


# ---------------------------------------------------------------------------
# MCP tool source (stdio)
# ---------------------------------------------------------------------------

MCP_SERVER_SCRIPT = Path(__file__).resolve().parent / "kg_mcp.py"


def make_kg_mcp_toolsource() -> ToolSource:
    """Create an MCP ToolSource that launches kg_mcp.py via stdio."""

    @asynccontextmanager
    async def _connect():
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(MCP_SERVER_SCRIPT)],
        )
        async with stdio_client(server_params) as (read, write):
            yield read, write

    return MCPServerImpl(_connect)


async def _async_make_kg_mcp_tools() -> list[Tool]:
    """Get the list of Tools from the KG MCP server (async)."""
    source = make_kg_mcp_toolsource()
    return list(await source.tools())


def make_kg_mcp_tools() -> list[Tool]:
    """Get the list of Tools from the KG MCP server (sync wrapper)."""
    coro = _async_make_kg_mcp_tools()
    fut = concurrent.futures.ThreadPoolExecutor().submit(asyncio.run, coro)
    return fut.result()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ground_truth(path: Path = GROUND_TRUTH_PATH) -> list[Sample]:
    """Load eval_tools_ground.yaml and convert to inspect_ai Samples."""
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
                input=question,
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


@scorer(metrics=[recall(), stderr()])
def score_nf_retrieval() -> Scorer:
    """Score by recall over expected UUID sets."""

    async def score(state: TaskState, target: Target) -> Score:
        expected = set(str(v).lower() for v in json.loads(target.text))
        predicted = set(_extract_results(state.output.completion))

        if not expected:
            return Score(value=1.0, answer=str(predicted), explanation="No expected results")

        hits = expected & predicted
        r = len(hits) / len(expected)

        return Score(
            value=r,
            answer=json.dumps(sorted(predicted)),
            explanation=f"Recall {len(hits)}/{len(expected)} = {r:.2f}",
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
    """NF Knowledge Graph retrieval eval via KG MCP.

    The agent queries an NF ontology knowledge graph through a KG MCP
    server and returns resource UUIDs matching the question.

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

    tool_setups = [use_tools(make_kg_mcp_tools())]

    return Task(
        dataset=MemoryDataset(samples, system_message=SYSTEM_MESSAGE),
        solver=not_implemented_solver(),
        scorer=score_nf_retrieval(),
        setup=tool_setups,
    )

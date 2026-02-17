# NF RAG Task

Retrieval evaluation for NF (Neurofibromatosis) research resources. The agent receives a natural-language question about NF research tools -- cell lines, animal models, antibodies, genetic vectors, mutations, etc. -- and must query a knowledge graph through an MCP server to retrieve the correct set of resource UUIDs.

## Setup

From the repo root:

```sh
pip install -e "."
```

The task requires a running SPARQL endpoint for the NF-OSI knowledge graph. By default it expects `http://localhost:7001`; override with:

```sh
export SPARQL_ENDPOINT="http://your-endpoint:port"
```

## Task Description

- 33 questions across 7 categories:
  | Prefix | Category | Count |
  |--------|----------|-------|
  | AB     | Antibodies | 3 |
  | AM     | Animal Models | 5 |
  | CL     | Cell Lines | 9 |
  | CR     | Cross-Reference | 2 |
  | GR     | Genetic Resources / Vectors | 5 |
  | MUT    | Mutations | 6 |
  | PI     | Principal Investigators | 2 |
- Ground truth is defined in `eval_tools_ground.yaml`
- Most answers are lists of resource UUIDs; scalar answers (e.g. counts) are list-wrapped for uniform scoring

## Tools

The agent is given three MCP tools from `kg_mcp.py`, launched as a stdio subprocess:

| Tool | Description |
|------|-------------|
| `sparql_query` | Execute an arbitrary SPARQL query; returns TSV |
| `get_schema` | Discover classes and properties in the NF ontology |
| `count_by_type` | Get instance counts grouped by `rdf:type` |

The graph uses the namespace `nf: <http://nf-osi.github.com/terms#>`.

## Answer Format

The agent must return a JSON array as its final output:

```json
["cdde78f3-a1c3-43b4-a2d9-4f1a7cc88564", "01a755e0-90b9-44d4-ae81-3605fc5a7539"]
```

For scalar answers (e.g. counts), the value should still be in an array: `[4]`.

The scorer first attempts to parse a JSON array from the output. If that fails, it falls back to extracting all UUID-shaped strings via regex.

## Scoring

The primary metric is **recall** -- the fraction of expected values that appear in the agent's output, averaged across questions.

## Usage

```bash
# Run all questions
inspect eval astabench/nf_rag --solver react --model openai/gpt-4o

# Run only cell line questions
inspect eval astabench/nf_rag --solver react --model openai/gpt-4o \
  -T task_category=CL

# Run only mutation and antibody questions
inspect eval astabench/nf_rag --solver react --model openai/gpt-4o \
  -T task_category="MUT,AB"

# Run specific questions by ID
inspect eval astabench/nf_rag --solver react --model openai/gpt-4o \
  -T task_filter="AB-001,CL-007"
```

## Files

| File | Description |
|------|-------------|
| `task.py` | Task definition, MCP wiring, data loading, and scorer |
| `kg_mcp.py` | FastMCP server wrapping the SPARQL endpoint |
| `eval_tools_ground.yaml` | Ground truth: question text and expected UUID sets |

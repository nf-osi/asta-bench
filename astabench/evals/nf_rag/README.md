# NF RAG Task

Retrieval evaluation for NF (Neurofibromatosis) research resources. 
The agent receives a natural-language question about NF research tools**, 
such as cell lines, animal models, antibodies, genetic vectors, mutations, etc., 
and must query a knowledge graph through an MCP server to retrieve the correct set of results.

**Later, retrieval task will also add publications.

## Setup

From the repo root:

```sh
pip install -e "."
```

The task requires a running SPARQL endpoint for the NF-OSI knowledge graph. 
By default it expects `http://localhost:7001`; override with:

```sh
export SPARQL_ENDPOINT="http://your-endpoint:port"
```

## Task Description

Questions across 7 categories:
| Prefix | Category |
|--------|----------|
| AB     | Antibodies |
| AM     | Animal Models |
| CL     | Cell Lines |
| CR     | Cross-Reference |
| GR     | Genetic Reagents |
| MUT    | Mutations |
| PI     | Program / Principal Investigators |


## Tools

The agent is given three tools that query the SPARQL endpoint directly:

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

For single answers (e.g. counts), the value should still be in an array: `[4]`.

The scorer first attempts to parse a JSON array from the output. 
If that fails, it falls back to extracting all UUID-shaped strings via regex.

## Scoring

The primary metric is **recall** -- the fraction of expected values that appear in the agent's output, averaged across questions.

## Usage

Depending on the `--model` specified below, set up your API key:

```
export OPENAI_API_KEY=<your-openai-key>
export ANTHROPIC_API_KEY=<your-anthropic-key>
export GOOGLE_API_KEY=<your-google-key>
```

The framework uses litellm, so see https://models.litellm.ai/ for how to specify model.

**Note:** Use `basic_agent` (not `react`) as the solver. 
The `react` agent does not pick up tools from the task setup; `basic_agent` does.

```bash
# Run all questions (single pass)
inspect eval astabench/nf_rag --solver basic_agent --model anthropic/claude-sonnet-4-5

# Run each question 3 times to measure variance
inspect eval astabench/nf_rag --solver basic_agent --model anthropic/claude-sonnet-4-5 \
  --epochs 3

# Run only cell line questions
inspect eval astabench/nf_rag --solver basic_agent --model anthropic/claude-sonnet-4-5 \
  -T task_category=CL

# Run only mutation and antibody questions
inspect eval astabench/nf_rag --solver basic_agent --model anthropic/claude-sonnet-4-5 \
  -T task_category="MUT,AB"

# Run specific questions by ID
inspect eval astabench/nf_rag --solver basic_agent --model anthropic/claude-sonnet-4-5 \
  -T task_filter="AB-001,CL-007"
```

By default each question is run once. Use `--epochs N` to run each question N times
(useful for measuring consistency across runs). Scores are averaged across epochs.

The `basic_agent` solver has a default message limit of 50. Some questions may require
more tool calls (e.g. schema discovery, query refinement). Increase with `-S message_limit=N`:

```bash
inspect eval astabench/nf_rag --solver basic_agent --model anthropic/claude-sonnet-4-5 \
  -S message_limit=100
```

Output is written to `logs/*` at the root repo. Use `inspect view` to visualize logs.

## Cost accounting

Run `astabench score <log_dir>` (point it at either the directory printed at the end of
`inspect eval` or the top-level `logs/` folder) to aggregate recall metrics and compute
model-usage costs for an NF_RAG run. The command automatically extracts `.eval` archives,
creates `eval_config.json` if it is missing, and uses the override table in
`astabench/config/litellm_cost_overrides.json` as the sole source of pricing data. The scorer
relies on InspectAI's usage logging, so calls that go through Inspect models or `AsyncOpenAI`
with the Inspect bridge are handled automatically. If a custom solver makes model calls outside
of Inspect's wrappers, record them manually so the scorer can report the correct spend:

```python
from astabench.util.model import record_model_usage_with_inspect
from inspect_ai.model import ModelUsage

record_model_usage_with_inspect(
    "openai/gpt-4.1",
    ModelUsage(input_tokens=100, output_tokens=10, total_tokens=110),
)
```

## Files

| File | Description |
|------|-------------|
| `task.py` | Task definition, tool definitions, data loading, and scorer |
| `eval_data.yaml` | Ground truth: questions and expected result sets |

# NF Publication RAG Task

Evaluation for publication-grounded NF question answering over the NF-OSI knowledge graph.
The agent receives a question about NF research, queries a SPARQL+Text endpoint to find
supporting publication passages, answers in the configured format, and cites the
supporting passages.

## Setup

From the repo root:

```sh
pip install -e "."
```

The task expects a running SPARQL endpoint for the NF-OSI knowledge graph.
By default it uses `http://localhost:7001`; override with:

```sh
export SPARQL_ENDPOINT="http://your-endpoint:port"
```

## Task Description

Ground truth is loaded from `eval_data.yaml`. Each sample contains:

- A question
- Four answer choices
- One correct choice index
- An ideal short answer
- A PMID
- One or more supporting passage indices

`test_data.yaml` already uses this same ground-truth structure, so it serves as a
direct single-item example of the data shape expected by the task.

The task can render questions in two styles:

- `precise` (default): uses the curated `question` field
- `user_query`: uses the more colloquial `user_query` field

The task also supports two answer formats:

- `mcq` (default): multiple-choice answering with online answer scoring
- `short_answer`: free-text answering with online citation scoring only

Question IDs are publication-scoped, for example `PMC9221468-01`. The task stores the
publication prefix (for example `PMC9221468`) as the sample category, which is also what
`task_category` filters on.

## Tools

The agent is given three tools:

| Tool | Description |
|------|-------------|
| `sparql_text_query` | Execute a SPARQL+Text query against the publication text index; returns TSV |
| `get_schema` | Return ontology classes and properties to inspect the graph structure |
| `get_entity_types` | Return counts for indexed text entity types |

The publication text index uses attribution tags in the retrieved text of the form
`[PMID{pmid}-{passage_num}-{section_type}]`. The agent is expected to extract `pmid`
and `passage` values from those tags for citation output.

## Answer Format

The final answer must be a JSON object with two fields.

For `answer_format=mcq`:

```json
{
  "answer": "B",
  "attribution": [
    {"pmid": "35741605", "passage": 1},
    {"pmid": "35741605", "passage": 3}
  ]
}
```

- `answer` may be a choice letter (`A`-`H`) or a numeric choice index
- `attribution` must be a list of objects containing `pmid` and `passage`

For `answer_format=short_answer`:

```json
{
  "answer": "Selumetinib",
  "attribution": [
    {"pmid": "35741605", "passage": 1},
    {"pmid": "35741605", "passage": 3}
  ]
}
```

- `answer` is free text
- `attribution` uses the same citation structure as the multiple-choice mode

The scorer first tries to parse the JSON object from the model output. If parsing fails,
it falls back to:

- Extracting a choice letter at the start of a line
- Extracting passage citations from `[PMID...-...-...]` tags found in the output

## Scoring

For `answer_format=mcq`, the task reports two metrics:

- `accuracy`: whether the selected multiple-choice answer matches the ground truth
- `citation_f1`: F1 over the set of `(pmid, passage)` attribution tuples

For `answer_format=short_answer`, the task reports one online metric:

- `citation_f1`: F1 over the set of `(pmid, passage)` attribution tuples

Short-answer semantic answer scoring is intentionally deferred to postprocessing from
logged outputs rather than being computed during the main eval run. The postprocessing
judge uses a 3-way rubric:

- `correct` -> `1.0`
- `partially_correct` -> `0.5`
- `incorrect` -> `0.0`

Attribution scoring is set-based:

- Duplicate citations do not help
- Extra citations reduce precision
- Missing citations reduce recall

## Usage

This task does not provide a usable default solver in `task.py`, so you must pass an
explicit solver on the command line.

Depending on the model, set the relevant API key:

```sh
export OPENAI_API_KEY=<your-openai-key>
export ANTHROPIC_API_KEY=<your-anthropic-key>
export GOOGLE_API_KEY=<your-google-key>
```

Example runs:

```bash
# Run the full publication QA eval
inspect eval astabench/nf_rag_pubs --solver basic_agent --model anthropic/claude-sonnet-4-5

# Run one specific question
inspect eval astabench/nf_rag_pubs --solver basic_agent --model anthropic/claude-sonnet-4-5 \
  -T task_filter=PMC9221468-01

# Run all questions from selected publication prefixes
inspect eval astabench/nf_rag_pubs --solver basic_agent --model anthropic/claude-sonnet-4-5 \
  -T task_category="PMC9221468,PMC3484870"

# Use the colloquial question wording instead of the curated wording
inspect eval astabench/nf_rag_pubs --solver basic_agent --model anthropic/claude-sonnet-4-5 \
  -T question_style=user_query

# Run the short-answer variant and log outputs for later semantic scoring
inspect eval astabench/nf_rag_pubs --solver basic_agent --model anthropic/claude-sonnet-4-5 \
  -T answer_format=short_answer

# Postprocess one short-answer eval log with an LLM judge
python -m astabench.evals.nf_rag_pubs.semantic_score logs/<run>.eval \
  --judge-model openai/gpt-4.1-mini

# Request provider-side batch execution for the judge if supported
python -m astabench.evals.nf_rag_pubs.semantic_score logs/<run>.eval \
  --judge-model openai/gpt-4.1-mini --batch
```

If the agent needs more tool turns, increase the solver message limit:

```bash
inspect eval astabench/nf_rag_pubs --solver basic_agent --model anthropic/claude-sonnet-4-5 \
  -S message_limit=100
```

By default each question runs once. Use `--epochs N` to repeat questions and average the
resulting scores across runs.

Output is written under `logs/` at the repo root. Use `inspect view` to inspect runs.

## Files

| File | Description |
|------|-------------|
| `task.py` | Task definition, prompt, tools, response extraction, and scorers |
| `semantic_score.py` | Postprocesses short-answer eval logs with an LLM semantic judge |
| `eval_data.yaml` | Full publication QA ground truth |
| `test_data.yaml` | Single-item debugging fixture in the same schema expected by the task |

# Customer Support Agent — Assignment 1

LangGraph implementation of the six-module Customer Support Agent described
in `Proposed_Methodology.docx`, evaluated against a single-graph baseline
that mirrors the reference LangGraph "Build a Customer Support Bot" tutorial.

## Structure

```
customer_support_agent/
├── state.py              # shared SupportState schema
├── llm.py                # pluggable LLM interface (offline rule-based stub + optional OpenAI backend)
├── tools/backend.py       # mock flights/hotels/cars/policy backend
├── graph_proposed.py      # proposed 6-module multi-specialist LangGraph agent
├── graph_baseline.py      # baseline single-graph agent (reference-tutorial pattern)
├── tests/test_conversations.py  # 25 labeled synthetic test conversations
├── evaluate.py             # runs both agents over the test set, writes results/
└── results/                # CSV + JSON outputs from the last evaluation run
```

## Setup

```bash
pip install -r requirements.txt
```

By default the agent runs with a deterministic, offline **RuleBasedLLM**
(see `llm.py`) so the whole pipeline is reproducible without any API key —
useful for grading/demo environments with no network access. To use a real
GPT-4-class model instead:

```bash
pip install langchain-openai
export OPENAI_API_KEY=sk-...
```

`llm.get_llm()` automatically switches to the OpenAI backend when the key is
present — no other code changes are required.

## Running the agent interactively

```python
from graph_proposed import PROPOSED_GRAPH

result = PROPOSED_GRAPH.invoke({"user_message": "Cancel my flight BLR-DEL-01"})
print(result["response"])          # asks for confirmation (human-in-the-loop)

result["user_approved"] = True
result = PROPOSED_GRAPH.invoke(result)
print(result["response"])          # booking cancelled
```

## Running the evaluation / comparative analysis

```bash
python evaluate.py
```

This runs both `PROPOSED_GRAPH` and `BASELINE_GRAPH` over the 25 labeled
conversations in `tests/test_conversations.py` and writes:

- `results/proposed_results.csv`
- `results/baseline_results.csv`
- `results/summary.json` — aggregate metrics (see Result Analysis document)

## Notes on the offline LLM stub

`RuleBasedLLM` is a deterministic keyword classifier so every node, edge,
and confirm-before-cancel step can be exercised without an API key. It
separates policy questions from cancellation requests, infers the domain
from a booking id when the message does not name one, and books a new
flight, hotel, or car when the customer is asking to reserve rather than
look something up. `python evaluate.py` resets the mock database between
cases and writes the latest metrics to `results/summary.json`.

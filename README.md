# llm eval runner

A small command line tool that runs a set of questions against any LLM endpoint and reports **accuracy, latency and cost**. Built to measure my own agents (Sentinel Copilot for Aspen) instead of guessing whether they work.

It supports OpenAI compatible endpoints (so also LM Studio, Ollama, vLLM and llama.cpp), the Anthropic Messages API, and a deterministic mock provider used by the test suite.

```
$ llm-eval run examples/aspen.jsonl --provider lmstudio --model glm-4-9b-chat --json results/glm4.json
provider      openai
model         glm-4-9b-chat
cases         8  (passed 6, failed 2, errors 0)
accuracy      75.0%
latency       p50 412 ms, p95 1180 ms
tokens        in 611, out 214
cost          $0.0000
wall time     4.9 s

failures:
  p03        expected ['specification', 'reflux', ...]  got: 'Try increasing the number of iterations.'
  p08        expected ['two liquid', ...]  got: 'Flash3 handles three components.'
```

## Why this exists

Every LLM project I showed people had the same hole: no number that said how good it was. Writing "the agent answers questions about Aspen" is a claim. Writing "75% on a 40 question set, p50 latency 412 ms on a local GLM 4 model" is a result. This tool makes the second sentence cheap to produce and cheap to repeat after every change.

## Install

```bash
git clone https://github.com/ZenNight/llm-eval-runner
cd llm-eval-runner
pip install -e ".[dev]"
pytest
```

Python 3.10 or newer. The only runtime dependency is `httpx`.

## Usage

```bash
# pipeline sanity check, no network
llm-eval run examples/arithmetic.jsonl --provider mock

# local model served by LM Studio on the default port
llm-eval run examples/aspen.jsonl --provider lmstudio --model glm-4-9b-chat

# any OpenAI compatible server, with pricing so cost is reported
export OPENAI_API_KEY=sk-...
llm-eval run examples/aspen.jsonl --provider openai --model gpt-4o-mini \
  --price-in 0.15 --price-out 0.60 --json results/gpt4o-mini.json

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...
llm-eval run examples/aspen.jsonl --provider anthropic --model claude-haiku-4-5 \
  --price-in 1 --price-out 5 --json results/haiku.json

# leaderboard from saved reports
llm-eval compare results/*.json
```

Useful flags: `--system` applies one system prompt to every case, `--tag hysys` runs a subset, `--limit 5` runs the first five cases, `--markdown` prints a table you can paste into a README.

## Dataset format

One JSON object per line. Lines starting with `#` are comments.

```json
{"id": "p01", "prompt": "Which property package is the usual default for hydrocarbon systems in HYSYS?",
 "expected": ["Peng-Robinson", "Peng Robinson", "PR"], "match": "regex", "tags": ["hysys", "thermo"]}
```

| field | required | meaning |
|---|---|---|
| `id` | yes | unique string |
| `prompt` | yes | text sent to the model |
| `expected` | yes | one string or a list; any match counts as a pass |
| `match` | no | `exact`, `contains` (default), `regex`, `numeric` |
| `tolerance` | no | absolute tolerance for `numeric` |
| `system` | no | per case system prompt override |
| `tags` | no | list of strings for `--tag` filtering |

## Design decisions

**Deterministic scoring only, no LLM judge.** A judge model adds cost, nondeterminism and a second model to debug when results look wrong. In practice a regex with alternatives or a numeric tolerance covers almost every question I actually want to ask. If a case cannot be scored deterministically, the case is usually badly written.

**One provider for all OpenAI compatible servers.** LM Studio, Ollama, vLLM and llama.cpp all speak the same chat completions API, so `--provider lmstudio` is just `--provider openai` with a different default base URL. Less code, fewer bugs.

**Errors are recorded, not raised.** If the endpoint fails on case 37 of 50, the run keeps going and the report shows 37 as an error. The exit code is 2 when any error occurred so CI can still catch it.

**Temperature defaults to 0.** Evaluation should be repeatable. If you want to measure variance, run with a higher temperature several times and compare the JSON reports.

**Normalization is conservative.** Answers are lowercased, whitespace collapsed and surrounding punctuation stripped. Nothing else. Aggressive normalization hides real failures.

## Tests

```
pytest -q
13 passed
```

The suite covers scoring rules (including comma decimals and case insensitive regex), dataset validation (duplicate ids, unknown match types, comment lines), aggregation (accuracy, tokens, cost arithmetic), error isolation, and the CLI end to end through the mock provider. No test touches the network.

## Known limitations

- `numeric` takes the **first** number in the answer. "15 times 3 equals 45" scores as 15. Ask the model to answer with only the number, or use `regex`.
- `contains` is substring based, so an expected value of "PR" will match "PRocess". Prefer `regex` with `\b` word boundaries for short tokens.
- Token counts come from the endpoint. LM Studio and some local servers report zero, so cost will read as zero for local models even though electricity is not free.
- Requests are sequential. Concurrency would speed up cloud runs but would also make rate limits and local GPU contention harder to reason about, so it was left out of the first version.
- No retries. A transient 429 counts as an error. Rerun the failed ids or add retries if your provider is flaky.

## Roadmap

- `--concurrency N` for cloud providers
- per tag accuracy breakdown in the report
- a `diff` command to show which cases changed between two JSON reports

## License

MIT

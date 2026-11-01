# SigOP

SigOP helps LLMs build better code harnesses, faster. Designed for open-weight models, it uses PLATEAU and OVERFIT feedback to guide evolution, break out of unproductive loops, and improve performance.

## Quick start

```bash
git clone https://github.com/sigop153/SigOP.git
cd SigOP
uv run python optimize.py --proposer Qwen/Qwen3.5-35B-A3B-FP8 --executor openai/gpt-oss-20b --objective s2d
```

This serves the executor on GPU 0 and the proposer on GPU 1 with vLLM. The proposer is driven as a coding agent in a `claude` CLI session, pointed at the local proposer server through a proxy. No Anthropic model is involved, unless you choose one as the proposer (see [Using Claude models](#using-claude-models)).

For more options, use `--help`:
```bash
uv run python optimize.py --help
```

You can use a YAML file via `--config`:
```bash
uv run python optimize.py --config configs/example.yaml
```

## Requirements
Linux, two GPUs, [uv](https://docs.astral.sh/uv/), the `claude` CLI on `PATH`, and
[bubblewrap](https://github.com/containers/bubblewrap) (`apt install bubblewrap`), which confines
the proposer session to its run folder; pass `--sandbox false` to run without it.
`uv run` installs Python 3.12 and the locked dependencies (vLLM, LiteLLM) on first use.

If you wish to run on a single GPU, serve your own vLLMs and provide the URLs (see [Serving other models](#serving-other-models-via-openai-specification-apis)).

## Using Claude models
```bash
uv run python optimize.py --proposer claude-sonnet-5-5 ...
```

## Resume
```bash
uv run python optimize.py --resume <run_name>
```

## Serving other models (via OpenAI-specification APIs)
```bash
uv run python optimize.py --executor_url http://127.0.0.1:8010/v1 --proposer_url http://127.0.0.1:8020/v1 ...
```

Pass several executors as a comma-separated `--executor_url` list for requests to be spread over them.

## Bring your own objective (benchmark)

The quickest way is a CSV with two columns, `input` and `target`:

```csv
input,target
"I have a rash on my arm and it itches at night",allergy
"My joints are stiff in the morning and swell by evening",arthritis
```

```bash
uv run python optimize.py --objective data/symptoms.csv --proposer ... --executor ...
```

The rows are shuffled and split into four equal parts: proposal, train, val and test.
Answers are graded by exact match, case-insensitive, where the executor is asked to wrap its answer in `[ANSWER]...[/ANSWER]`.

### Custom splits

Presplit your csv into `proposal.csv`, `train.csv`, `val.csv`, and `test.csv`, each with `input` and `target` columns, in one folder. Then run:
```bash
uv run python optimize.py --objective data/symptoms --proposer ... --executor ...
```

### Custom grading
When exact match is not the right grading, or you want to use your own prompt, make a folder:

```
objectives/my-task/
  objective.toml
  TASK.md
  proposal.jsonl   train.jsonl   val.jsonl   test.jsonl
```

Each `.jsonl` line is `{"input": "...", "target": "..."}`. `objective.toml` names the scorer
and the prompt the seed harness wraps around every input:

```toml
name = "Symptom2Disease"
scorer = "exact"               # or "set": split the answer on separators, the sets must match
answer_open = "[DIAGNOSIS]"    # the scorer cuts the answer out between these two
answer_close = "[/DIAGNOSIS]"
temperature = 0.0              # executor sampling; samples > 1 averages pass@1
samples = 1
prompt = """You are an expert medical diagnostician. ... Reply in the format
[DIAGNOSIS]diagnosis_name[/DIAGNOSIS]

## Patient Symptoms
{input}"""
```

Optional keys: `answer_fallback_pattern` (a regex tried when the tags are missing), `separators`
(for the `set` scorer) and `labels` (the closed set of targets, shown to the proposer).

`TASK.md` is what the proposer is told about the task: what the input and target are, how
answers are graded, and what makes the task hard. Then:

```bash
uv run python optimize.py --objective objectives/my-task --proposer ... --executor ...
```

`objectives/s2d` and `objectives/lawbench` are complete examples; `lawbench` uses the `set`
scorer with Chinese separators.

## Folder structure

### Code layout

```
optimize.py          the command
models.yaml          vLLM settings per model
sigop/               the code: loop.py the search, signals.py the two rules, evaluate.py training and
                     scoring, harness.py the harness interface and seeds, proposer.py the Claude Code
                     session, serving.py vLLM and the proxy, objective.py and scorers.py the objectives
objectives/          the built-in objectives
serving/             Qwen chat templates that accept Claude Code's system messages, and the proxy hook
scripts/             import_paper_splits.py, maintainers only: rebuilds the built-in objectives from the paper's data
tests/               pytest, no GPU needed
```

### A run

```
runs/<name>/             the proposer's working folder
  harnesses/             the seed and every candidate
  history.jsonl          one row per candidate: iteration, train score, outcome
  incumbent.json         the current h*
  scores/<name>/         train score and per-item trace
  reports/               the proposer's notes and the SIGNAL files
  sessions/              Claude Code session logs
runs/<name>.heldout/     val and test scores, every decision (detector.jsonl), server logs
```

## Development

```bash
uv run pytest
uv run ruff check .
```

## Citation

```bibtex
@inproceedings{anonymous2027sigop,
  title     = {SigOP: Fully Open-Source Self-Evolving Harness},
  author    = {Anonymous},
  booktitle = {Under review},
  year      = {2027},
}
```

## License

MIT. See `LICENSE`. Parts derive from Meta-Harness (MIT).

"""Evolve a harness for an objective with SigOP.

    python optimize.py --proposer Qwen/Qwen3.5-35B-A3B-FP8 --executor openai/gpt-oss-20b --objective s2d

Without --executor_url and --proposer_url, both models are served here with vLLM, the
executor on the first GPU of --gpus and the proposer on the second, and a proxy is started
for Claude Code. With a URL, that server is used as is; the model must be served under the
name optimize.py derives (the last path component, lower-cased, e.g. gpt-oss-20b).

A --proposer that is an Anthropic model id, claude-sonnet-5-5 or any other claude-*, runs
the proposer session against Anthropic with your own `claude` login or ANTHROPIC_API_KEY;
no proposer server or proxy is started.

--objective: a built-in name (--list_objectives true), an objective folder, or a CSV with
input,target columns. See objective.py for the folder layout.

Every run saves its settings to <run>.heldout/config.yaml. --resume <run_name> continues
that run with those settings; only --iterations, the GPUs and the server URLs are taken
from the new command line.
"""
import random
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from jsonargparse import auto_cli

from sigop.executor import ExecutorConfig
from sigop.loop import Backend, LoopConfig, run_search
from sigop.objective import builtin_names, load_objective
from sigop.proposer import ClaudeCode, ClaudeCodeConfig
from sigop.rundir import RunDir
from sigop.sandbox import default_sandbox
from sigop.serving import (
    HttpBackend,
    Proxy,
    Servers,
    executor_server,
    free_port,
    preflight,
    proposer_server,
    served_name,
)
from sigop.summary import summary

PROPOSER = "Qwen/Qwen3.5-35B-A3B-FP8"
EXECUTOR = "openai/gpt-oss-20b"
OBJECTIVE = "s2d"
ITERATIONS = 50
PATIENCE = 5
TOLERANCE = 0.01
CANDIDATES = 3
PROPOSE_TIMEOUT = 2400
EVAL_STALL = 600
EVAL_CAP = 5400
CONCURRENCY = 32
GPUS = "0,1"
SEED = 42


@dataclass(frozen=True)
class Settings:
    """What defines a run. Saved with it and restored on --resume."""
    proposer: str
    executor: str
    objective: str
    patience: int
    tolerance: float
    candidates: int
    propose_timeout: int
    eval_stall: int
    eval_cap: int
    concurrency: int
    seed: int
    sandbox: bool


def main(
    proposer: str = PROPOSER,
    executor: str = EXECUTOR,
    objective: str = OBJECTIVE,
    iterations: int = ITERATIONS,
    patience: int = PATIENCE,
    tolerance: float = TOLERANCE,
    candidates: int = CANDIDATES,
    propose_timeout: int = PROPOSE_TIMEOUT,
    eval_stall: int = EVAL_STALL,
    eval_cap: int = EVAL_CAP,
    concurrency: int = CONCURRENCY,
    gpus: str = GPUS,
    executor_url: str = "",
    proposer_url: str = "",
    runs_dir: Path = Path("runs"),
    run_name: str = "",
    resume: str = "",
    sandbox: bool = True,
    seed: int = SEED,
    list_objectives: bool = False,
) -> None:
    """Evolve a harness for an objective with SigOP.

    Args:
        proposer: the model that writes harnesses; a claude-* id uses Anthropic through the claude CLI
        executor: the model the harness calls
        objective: built-in name, objective folder, or CSV with input,target columns
        iterations: search budget T
        patience: PLATEAU after this many iterations without a train gain
        tolerance: OVERFIT when val drops by at least this much
        candidates: harnesses the proposer is asked for per iteration
        propose_timeout: seconds a proposer session may take before it is killed
        eval_stall: seconds a harness may go without scoring an item before it is killed
        eval_cap: seconds a harness may take in total before it is killed
        concurrency: concurrent executor requests while scoring
        gpus: two CUDA device ids, executor first, proposer second
        executor_url: one or more OpenAI-compatible endpoints already serving the executor, comma-separated
        proposer_url: an OpenAI-compatible endpoint already serving the proposer
        runs_dir: where run folders are created
        run_name: the run folder's name (default: objective and timestamp)
        resume: the name of a run to continue, with its saved settings, from its last complete iteration
        sandbox: confine the proposer session with bubblewrap to its run folder
        seed: the CSV split, the executor's sampling, and Python's random module for harness code
        list_objectives: print the built-in objectives and exit
    """
    if list_objectives:
        print("\n".join(builtin_names()))
        return
    settings = Settings(proposer, executor, objective, patience, tolerance, candidates, propose_timeout,
                        eval_stall, eval_cap, concurrency, seed, sandbox)
    run = RunDir((runs_dir / (resume or run_name or f"{objective.lower()}-{time.strftime('%Y%m%d-%H%M%S')}")).resolve())
    if resume:
        settings = load_settings(run)
        print(f"resuming {run.root} with its saved settings: {settings.proposer} proposing, "
              f"{settings.executor} executing, objective {settings.objective}", flush=True)
    preflight(serve_executor=not executor_url, serve_proposer=not proposer_url, sandbox=settings.sandbox)
    random.seed(settings.seed)        # harness code that uses `random` without its own generator
    loaded = load_objective(settings.objective, settings.seed)
    devices = gpus.split(",")
    anthropic_model = anthropic_model_of(settings.proposer)
    if anthropic_model_of(settings.executor):
        raise SystemExit("the executor is an OpenAI-compatible endpoint; an Anthropic model is for --proposer only")
    # Resolve the servers before anything is created, so an unsupported model stops here. Servers
    # started here listen on free local ports; an address is only ever given as a URL
    # One GPU per role, executor first; anything else (sharding, several executors, another
    # machine) is served by the user and passed as URLs
    proposer_vllm = (None if (proposer_url or anthropic_model)
                     else proposer_server(settings.proposer, free_port(), devices[-1]))
    executor_vllms = [] if executor_url else [executor_server(settings.executor, free_port(), devices[0])]
    if not resume and run.detector.exists():
        raise SystemExit(f"{run.root} already holds a run: continue it with --resume {run.root.name}, "
                         "or pick another --run_name")
    run.create()
    save_settings(run, settings, iterations, gpus, executor_url, proposer_url)
    print(f"run {run.root}  objective {loaded.spec.name} ({loaded.sizes()})", flush=True)

    servers = Servers()
    signal.signal(signal.SIGTERM, stop_on_sigterm)
    try:
        # Both servers load at once, each on its own GPU; their output is echoed with a prefix
        for number, server in enumerate(executor_vllms):
            print(f"starting executor {settings.executor} on GPU {server.gpu}", flush=True)
            servers.launch_vllm(server, run.heldout, "executor" if len(executor_vllms) == 1 else f"executor{number}")
        if executor_vllms:
            executor_url = ",".join(server.url for server in executor_vllms)
        if proposer_vllm:
            print(f"starting proposer {settings.proposer} on GPU {devices[-1]}", flush=True)
            servers.launch_vllm(proposer_vllm, run.heldout, "proposer")
            proposer_url = proposer_vllm.url
        servers.wait_all()
        # The servers the loop checks before each proposal and scoring; a run stops if one stays down
        backends: list[Backend] = [HttpBackend("executor", f"{url.strip()}/models") for url in executor_url.split(",")]
        if anthropic_model:
            session = ClaudeCodeConfig(base_url=None, model=anthropic_model, timeout_seconds=settings.propose_timeout,
                                       candidates=settings.candidates)
        else:
            proxy = Proxy(free_port(), proposer_url, served_name(settings.proposer))
            servers.start_proxy(proxy, run.heldout)
            session = ClaudeCodeConfig(base_url=proxy.url, timeout_seconds=settings.propose_timeout,
                                       candidates=settings.candidates)
            backends += [HttpBackend("proposer", f"{proposer_url}/models"),
                         HttpBackend("proxy", f"{proxy.url}/health/liveliness")]

        executor_config = ExecutorConfig(
            model=served_name(settings.executor), base_url=executor_url, temperature=loaded.spec.sampling.temperature,
            concurrency=settings.concurrency, cache_dir=runs_dir / ".cache", seed=settings.seed)
        confinement = default_sandbox(run.root, anthropic=bool(anthropic_model)) if settings.sandbox else None
        if confinement is None:
            print("WARNING: --sandbox false: the proposer session can read anything your user can, "
                  "including the heldout folder and the objective's val and test splits", flush=True)
        proposer_agent = ClaudeCode(session, confinement)
        incumbent = run_search(loaded, run, executor_config, proposer_agent,
                               LoopConfig(iterations=iterations, patience=settings.patience,
                                          tolerance=settings.tolerance, concurrency=settings.concurrency,
                                          stall_seconds=settings.eval_stall, cap_seconds=settings.eval_cap,
                                          propose_minutes=settings.propose_timeout // 60), resume=bool(resume),
                               backends=backends)
    finally:
        servers.stop()
    table = summary(run)
    (run.heldout / "summary.md").write_text(table, encoding="utf-8")
    print("\n" + table)
    print(f"final h*: {incumbent.name}  train {incumbent.train:.3f}  val {incumbent.val:.3f}  "
          f"test {incumbent.test:.3f}\nharness: {run.harnesses / (incumbent.name + '.py')}\n"
          f"summary: {run.heldout / 'summary.md'}")


def save_settings(run: RunDir, settings: Settings, iterations: int, gpus: str, executor_url: str,
                  proposer_url: str) -> None:
    """The run's settings, plus this launch's budget and serving, for the record and for --resume."""
    record = {**asdict(settings), "iterations": iterations, "gpus": gpus,
              "executor_url": executor_url, "proposer_url": proposer_url}
    (run.heldout / "config.yaml").write_text(yaml.safe_dump(record, sort_keys=False), encoding="utf-8")


def load_settings(run: RunDir) -> Settings:
    path = run.heldout / "config.yaml"
    if not path.exists() or not run.detector.exists():
        raise SystemExit(f"nothing to resume: {run.root} has no saved settings or decision log")
    with path.open(encoding="utf-8") as handle:
        record = yaml.safe_load(handle)
    return Settings(**{name: record[name] for name in Settings.__dataclass_fields__})


def stop_on_sigterm(*_: object) -> None:
    """A SIGTERM must reach the servers' `finally`, or the vLLM processes outlive the run."""
    raise SystemExit(143)


def anthropic_model_of(name: str) -> str:
    """The model to hand `claude --model`, or "" when the name is not an Anthropic model id."""
    return name if name.startswith("claude-") else ""


if __name__ == "__main__":
    auto_cli(main, as_positional=False)

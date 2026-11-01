"""Start the executor, the proposer and the proxy on this machine, the way the paper's runs
did, and wait until they answer.

    executor  vllm, one GPU, with the model family's reasoning parser
    proposer  vllm, one GPU, with the family's reasoning and tool-call parsers and, for
              Qwen, the chat template in serving/chat_templates/ (Claude Code sends several
              system blocks; the stock template rejects more than one)
    proxy     litellm, speaking the Anthropic API that Claude Code expects and forwarding
              to the proposer's vLLM endpoint; the hook in proxy_hooks.py merges the system
              blocks

The per-family flags live in models.yaml, not here. Each server is a child process
with its own log file; stop() kills all of them.
"""
import importlib.util
import os
import re
import shutil
import socket
import subprocess
import sys
import sysconfig
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MODELS_FILE = ROOT / "models.yaml"
SERVING = ROOT / "serving"
CHAT_TEMPLATES = SERVING / "chat_templates"
READY_TIMEOUT_SECONDS = 1800


def preflight(serve_executor: bool, serve_proposer: bool) -> None:
    """Fail before any server starts when a piece is missing, naming the fix."""
    missing = []
    if (serve_executor or serve_proposer) and importlib.util.find_spec("vllm") is None:
        missing.append("vllm is not installed: run through uv run, or pip install -e .")
    if (serve_executor or serve_proposer) and not Path(sysconfig.get_paths()["include"], "Python.h").exists():
        version = sysconfig.get_python_version()
        missing.append(f"Python {version} development headers are missing (vLLM's Triton kernels compile against "
                       f"them): apt install python{version}-dev, or make the venv with a Python that ships them, "
                       "e.g. uv venv --python 3.12")
    if importlib.util.find_spec("litellm") is None:
        missing.append("litellm is not installed: run through uv run, or pip install -e .")
    if shutil.which("litellm") is None and importlib.util.find_spec("litellm") is not None:
        missing.append("the litellm command is not on PATH (run through uv run)")
    if shutil.which("claude") is None:
        missing.append("the claude CLI is not on PATH: https://docs.anthropic.com/claude-code")
    if missing:
        raise SystemExit("cannot start:\n  " + "\n  ".join(missing))


@dataclass(frozen=True)
class ModelSettings:
    """The vLLM settings of one model, from models.yaml: its own, or its family's with any
    keys the model sets itself on top."""
    name: str
    reasoning_parser: str
    tool_call_parser: str
    max_model_len: int
    chat_template: str = ""
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Models:
    gpu_memory_utilization: float
    entries: tuple[ModelSettings, ...]

    def names(self) -> list[str]:
        return [entry.name for entry in self.entries]


def load_models(path: Path = MODELS_FILE) -> Models:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    families = raw.get("families") or {}
    entries = []
    for name, own in (raw.get("models") or {}).items():
        own = dict(own or {})
        family = own.pop("family", None)
        if family is not None and family not in families:
            raise SystemExit(f"{path}: {name} names family {family!r}, which is not under families:")
        merged = {**(families[family] if family else {}), **own}
        try:
            entries.append(ModelSettings(
                name=str(name), reasoning_parser=str(merged["reasoning_parser"]),
                tool_call_parser=str(merged["tool_call_parser"]), max_model_len=int(merged["max_model_len"]),
                chat_template=str(merged.get("chat_template", "")), flags=tuple(str(x) for x in merged.get("flags", []))))
        except KeyError as missing:
            raise SystemExit(f"{path}: {name} lacks {missing.args[0]}") from None
    return Models(gpu_memory_utilization=float(raw["gpu_memory_utilization"]), entries=tuple(entries))


def settings_of(model: str, models: Models) -> ModelSettings:
    """The entry named exactly as the model, else the one whose last path component matches,
    so a local checkpoint folder (/models/Qwen3.8-27B-FP8) takes its Hugging Face entry's settings."""
    for entry in models.entries:
        if entry.name == model:
            return entry
    tail = model.rstrip("/").split("/")[-1]
    for entry in models.entries:
        if entry.name.split("/")[-1] == tail:
            return entry
    known = "\n".join(f"  {name}" for name in models.names())
    raise SystemExit(f"""{model} is not in {MODELS_FILE}. Models optimize.py can serve:
{known}

To add it, append an entry under models: with the name as you pass it, either with its own
settings or naming a family:

  {model}:
    reasoning_parser: <vLLM reasoning parser>
    tool_call_parser: <vLLM tool-call parser>
    max_model_len: <context length>

Or serve it yourself and pass --executor_url / --proposer_url (README, "Serving other models").""")


@dataclass(frozen=True)
class VllmServer:
    model: str                       # HF id or local path
    served_name: str
    port: int
    gpu: str                         # CUDA_VISIBLE_DEVICES value
    max_model_len: int
    gpu_memory_utilization: float
    flags: tuple[str, ...] = ()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def command(self) -> list[str]:
        return [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", self.model,
                "--served-model-name", self.served_name, "--host", "127.0.0.1", "--port", str(self.port),
                "--max-model-len", str(self.max_model_len),
                "--gpu-memory-utilization", str(self.gpu_memory_utilization), *self.flags]


def free_port() -> int:
    """A local port nothing is listening on, chosen by the OS."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def executor_server(model: str, port: int, gpu: str, models: Models | None = None) -> VllmServer:
    models = models or load_models()
    settings = settings_of(model, models)
    return VllmServer(model=model, served_name=served_name(model), port=port, gpu=gpu,
                      max_model_len=settings.max_model_len, gpu_memory_utilization=models.gpu_memory_utilization,
                      flags=("--reasoning-parser", settings.reasoning_parser, *template_flags(settings), *settings.flags))


def proposer_server(model: str, port: int, gpu: str, models: Models | None = None) -> VllmServer:
    """Claude Code drives the proposer through tool calls, so this role adds the tool-call parser."""
    models = models or load_models()
    settings = settings_of(model, models)
    return VllmServer(model=model, served_name=served_name(model), port=port, gpu=gpu,
                      max_model_len=settings.max_model_len, gpu_memory_utilization=models.gpu_memory_utilization,
                      flags=("--reasoning-parser", settings.reasoning_parser, "--enable-auto-tool-choice",
                             "--tool-call-parser", settings.tool_call_parser, *template_flags(settings), *settings.flags))


def template_flags(settings: ModelSettings) -> tuple[str, ...]:
    return ("--chat-template", str(CHAT_TEMPLATES / settings.chat_template)) if settings.chat_template else ()


def served_name(model: str) -> str:
    return model.rstrip("/").split("/")[-1].lower()


@dataclass(frozen=True)
class Proxy:
    port: int
    proposer_url: str                # the vLLM /v1 endpoint
    proposer_name: str
    token: str = "local"

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def config_text(self) -> str:
        return f"""model_list:
  - model_name: "opus"
    litellm_params:
      model: hosted_vllm/{self.proposer_name}
      api_base: {self.proposer_url}
      api_key: dummy
  - model_name: "*"
    litellm_params:
      model: hosted_vllm/{self.proposer_name}
      api_base: {self.proposer_url}
      api_key: dummy
litellm_settings:
  drop_params: true
  request_timeout: 3600
  num_retries: 3
  callbacks: proxy_hooks.proxy_handler_instance
general_settings:
  master_key: {self.token}
"""


class Child:
    """A server process whose output is written to its log and, while `echo` is on, also
    printed with a prefix. Echo is on during start-up, so a slow or failing start is
    visible, and off once the server answers, so the loop's own output stays readable."""

    def __init__(self, command: list[str], log: Path, prefix: str, extra_environment: dict, cwd: Path | None):
        environment = dict(os.environ)
        environment.update(extra_environment)
        self.log = log
        self.prefix = prefix
        self.echo = True
        self.url = ""                    # the readiness URL still to be awaited, if any
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        env=environment, cwd=cwd, text=True, errors="replace", bufsize=1)
        self.thread = threading.Thread(target=self.pump, daemon=True)
        self.thread.start()

    def pump(self) -> None:
        assert self.process.stdout is not None      # stdout=PIPE above
        with self.log.open("w", encoding="utf-8") as handle:
            for line in self.process.stdout:
                handle.write(line)
                handle.flush()
                if self.echo:
                    print(f"[{self.prefix}] {line.rstrip()}", flush=True)


@dataclass
class Servers:
    """The child processes of one run, stopped together."""
    children: list[Child] = field(default_factory=list)

    def launch_vllm(self, server: VllmServer, log_dir: Path, role: str) -> Child:
        """Start the server without waiting, so several can load at once."""
        child = Child(server.command(), log_dir / f"vllm-{role}.log", role, {"CUDA_VISIBLE_DEVICES": server.gpu}, cwd=None)
        child.url = f"{server.url}/models"
        self.children.append(child)
        return child

    def wait_all(self, timeout_seconds: int = READY_TIMEOUT_SECONDS) -> None:
        """Wait until every launched server answers, watching all of them at once so a crash
        in any one of them is reported as soon as it happens."""
        started = time.time()
        while time.time() - started < timeout_seconds:
            pending = [child for child in self.children if child.url]
            if not pending:
                return
            for child in pending:
                if child.process.poll() is not None:
                    raise SystemExit(failure_message(child))
                if answers(child.url):
                    child.echo = False
                    print(f"[{child.prefix}] ready at {child.url} after {int(time.time() - started)} s; "
                          f"output continues in {child.log}", flush=True)
                    child.url = ""
            time.sleep(5)
        raise TimeoutError("not ready within {} s: {}".format(
            timeout_seconds, ", ".join(f"{c.prefix} ({c.log})" for c in self.children if c.url)))

    def start_proxy(self, proxy: Proxy, log_dir: Path) -> None:
        # LiteLLM imports the callback module from the config file's folder, so the hook is
        # copied beside the config and the proxy runs from there, with absolute paths.
        folder = log_dir.resolve()
        config = folder / f"litellm-{proxy.port}.yaml"
        config.write_text(proxy.config_text(), encoding="utf-8")
        shutil.copy(SERVING / "proxy_hooks.py", folder / "proxy_hooks.py")
        command = ["litellm", "--config", str(config), "--host", "127.0.0.1", "--port", str(proxy.port),
                   "--num_workers", "1"]
        child = Child(command, folder / f"litellm-{proxy.port}.log", "proxy", {}, cwd=folder)
        child.url = f"{proxy.url}/health/liveliness"
        self.children.append(child)
        self.wait_all()

    def stop(self) -> None:
        for child in self.children:
            if child.process.poll() is None:
                child.process.terminate()
        for child in self.children:
            try:
                child.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.process.kill()
        self.children.clear()


@dataclass(frozen=True)
class HttpBackend:
    """A server the search loop checks before each proposal and scoring (loop.Backend)."""
    role: str
    health_url: str

    def up(self) -> bool:
        return answers(self.health_url)


def answers(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status < 500
    except (urllib.error.URLError, OSError):
        return False


def failure_message(child: Child) -> str:
    child.thread.join(timeout=5)
    return (f"{child.prefix} exited with code {child.process.returncode} before {child.url} answered.\n"
            f"log: {child.log}\n{errors_in(child.log)}")


# A line that states a cause, not a traceback frame: "fatal error: Python.h", "RuntimeError: ...",
# "No module named vllm", "CUDA out of memory", "EngineCore failed to start".
CAUSE = re.compile(r"fatal error|\b\w*(?:Error|Exception)\b: |No module named|out of memory|failed to start|"
                   r"not found|permission denied", re.IGNORECASE)
PROCESS_TAG = re.compile(r"^\([A-Za-z]+ pid=\d+\)\s*(?:ERROR|INFO|WARNING)?\s*[\d-]*\s*[\d:]*\s*(?:\[[^\]]*\])?\s*")


def errors_in(log: Path, first: int = 6, last: int = 8) -> str:
    """The first lines that look like an error (the cause is usually early, e.g. a failed
    compile) and the last lines (the final traceback), so the reason is on screen without
    opening the log."""
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    stripped = [PROCESS_TAG.sub("", line).strip() for line in lines]
    causes = [line for line in stripped if CAUSE.search(line) and not line.startswith(("File ", "Traceback"))]
    head = "\n".join(f"  {line[:200]}" for line in causes[:first])
    foot = "\n".join(f"  {line[:200]}" for line in stripped[-last:] if line)
    return f"first errors:\n{head}\nend of log:\n{foot}"


def tail(log: Path, lines: int = 15) -> str:
    try:
        return "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""

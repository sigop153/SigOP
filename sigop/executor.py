"""The executor: an OpenAI-compatible chat endpoint the harness calls, one prompt at a time.

Behaviour kept from Meta-Harness's LLM class: a "Reasoning: medium" system prompt and
harmony parsing for gpt-oss, retries on rate limits and timeouts, a prompt length cap, a
thread pool so a harness can be scored concurrently, and a disk cache keyed on the request
so a re-scored harness costs nothing.
"""
import hashlib
import itertools
import json
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

MAX_PROMPT_CHARS = 224_000
TIMEOUT_SECONDS = 600.0
ATTEMPTS = 6
RETRY_MARKERS = ("429", "rate limit", "too many requests", "retry in", "timed out", "timeout")


@dataclass(frozen=True)
class ExecutorConfig:
    model: str                          # the served model name, e.g. gpt-oss-20b
    base_url: str                       # one or more endpoints, comma-separated: http://127.0.0.1:8010/v1,...
    temperature: float = 0.0
    max_tokens: int = 16384
    concurrency: int = 32
    api_key: str = "local"
    cache_dir: Path | None = None       # None disables the cache
    seed: int | None = None             # sampling seed sent with every request (vLLM honours it)


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class Executor:
    def __init__(self, config: ExecutorConfig):
        self.config = config
        self.is_gpt_oss = "gpt-oss" in config.model.lower()
        urls = [url.strip() for url in config.base_url.split(",") if url.strip()]
        self.clients = [OpenAI(base_url=url, api_key=config.api_key, timeout=TIMEOUT_SECONDS, max_retries=0) for url in urls]
        self.next_client = itertools.cycle(self.clients)
        self.pool = ThreadPoolExecutor(max_workers=config.concurrency)
        self.usage = Usage()
        self._lock = threading.Lock()
        if config.cache_dir:
            config.cache_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, prompt: str) -> str:
        prompt = truncate(prompt)
        cached = self.read_cache(prompt)
        if cached is not None:
            return cached
        content = self.complete(prompt)
        self.write_cache(prompt, content)
        return content

    def batch(self, prompts: list[str]) -> list[str]:
        return list(self.pool.map(self, prompts))

    def complete(self, prompt: str) -> str:
        messages: list[ChatCompletionMessageParam] = []
        if self.is_gpt_oss:
            messages.append({"role": "system", "content": "Reasoning: medium"})
        messages.append({"role": "user", "content": prompt})
        response = self.request(messages)
        content = response.choices[0].message.content or ""
        with self._lock:
            self.usage.calls += 1
            if response.usage:
                self.usage.input_tokens += response.usage.prompt_tokens or 0
                self.usage.output_tokens += response.usage.completion_tokens or 0
        return parse_harmony(content) if self.is_gpt_oss else content

    def request(self, messages: list[ChatCompletionMessageParam]) -> ChatCompletion:
        """One chat completion, retried on rate limits and timeouts with backoff."""
        for attempt in range(1, ATTEMPTS):
            try:
                return self.client().chat.completions.create(
                    model=self.config.model, messages=messages,
                    temperature=self.config.temperature, max_tokens=self.config.max_tokens, seed=self.config.seed)
            except Exception as error:
                if not retryable(error):
                    raise
                time.sleep(min(2 ** attempt, 32) + attempt)
        return self.client().chat.completions.create(
            model=self.config.model, messages=messages,
            temperature=self.config.temperature, max_tokens=self.config.max_tokens, seed=self.config.seed)

    def client(self) -> OpenAI:
        """Requests are spread over the servers in turn (several GPUs serving one executor)."""
        with self._lock:
            return next(self.next_client)

    def cache_path(self, prompt: str) -> Path | None:
        if not self.config.cache_dir:
            return None
        payload = json.dumps({"model": self.config.model, "temperature": self.config.temperature,
                              "max_tokens": self.config.max_tokens, "seed": self.config.seed, "prompt": prompt},
                             sort_keys=True)
        return self.config.cache_dir / (hashlib.sha256(payload.encode()).hexdigest() + ".json")

    def read_cache(self, prompt: str) -> str | None:
        path = self.cache_path(prompt)
        if path is None or not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))["content"]
        except (OSError, ValueError, KeyError):
            return None

    def write_cache(self, prompt: str, content: str) -> None:
        """Written to a unique temporary file first, then renamed into place: two concurrent
        requests with the same prompt (LawBench repeats prompts) must not race on one name."""
        path = self.cache_path(prompt)
        if path is None:
            return
        descriptor, temporary = tempfile.mkstemp(prefix=path.stem, suffix=".tmp", dir=path.parent)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"content": content}))
        os.replace(temporary, path)


def truncate(prompt: str) -> str:
    """Middle-truncate an oversized prompt rather than fail the item."""
    if len(prompt) <= MAX_PROMPT_CHARS:
        return prompt
    half = MAX_PROMPT_CHARS // 2
    return prompt[:half] + "\n\n... [TRUNCATED] ...\n\n" + prompt[-half:]


def retryable(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    if status == 429:
        return True
    text = str(error).lower()
    return any(marker in text for marker in RETRY_MARKERS)


def parse_harmony(content: str) -> str:
    """gpt-oss answers in harmony channels; return the final channel. A server run with
    vLLM's openai_gptoss reasoning parser already strips them, in which case this is a
    no-op. openai_harmony is imported lazily: it is only needed for gpt-oss executors."""
    if "<|channel|>" not in content and "<|message|>" not in content:
        return content
    from openai_harmony import HarmonyEncodingName, Role, load_harmony_encoding

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    try:
        tokens = encoding.encode(content, allowed_special="all")
        messages = encoding.parse_messages_from_completion_tokens(tokens, role=Role.ASSISTANT, strict=False)
    except Exception:  # noqa: BLE001  unparseable harmony is returned as is
        return content
    for message in messages:
        if message.channel == "final":
            return text_of(message)
    return text_of(messages[-1]) if messages else content


def text_of(message) -> str:
    return "".join(getattr(part, "text", "") for part in message.content)

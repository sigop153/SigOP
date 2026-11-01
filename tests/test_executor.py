from pathlib import Path

from sigop.executor import Executor, ExecutorConfig, parse_harmony, retryable, truncate


def config(tmp_path: Path, seed: int | None) -> ExecutorConfig:
    return ExecutorConfig(model="m", base_url="http://127.0.0.1:1/v1", cache_dir=tmp_path, seed=seed)


def test_seed_is_part_of_the_cache_key(tmp_path):
    first = Executor(config(tmp_path, seed=42)).cache_path("hello")
    second = Executor(config(tmp_path, seed=43)).cache_path("hello")
    assert first != second and first == Executor(config(tmp_path, seed=42)).cache_path("hello")


def test_seed_goes_into_the_request(tmp_path, monkeypatch):
    seen = {}

    class Choice:
        message = type("M", (), {"content": "answer"})()

    class Response:
        choices = (Choice(),)
        usage = None

    executor = Executor(config(tmp_path, seed=7))
    for client in executor.clients:
        monkeypatch.setattr(client.chat.completions, "create", lambda **request: seen.update(request) or Response())
    assert executor("prompt") == "answer"
    assert seen["seed"] == 7 and seen["temperature"] == 0.0


def test_truncate_keeps_both_ends():
    text = "a" * 300_000
    cut = truncate(text)
    assert len(cut) < len(text) and "[TRUNCATED]" in cut


def test_retryable_on_rate_limit_and_timeout_only():
    assert retryable(Exception("429 Too Many Requests")) and retryable(Exception("request timed out"))
    assert not retryable(Exception("400 bad request"))


def test_harmony_passthrough_without_channel_markers():
    assert parse_harmony("plain text") == "plain text"


def test_concurrent_identical_prompts_do_not_race_on_the_cache(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    class Choice:
        message = type("M", (), {"content": "same"})()

    class Response:
        choices = (Choice(),)
        usage = None

    executor = Executor(config(tmp_path, seed=1))
    for client in executor.clients:
        monkeypatch.setattr(client.chat.completions, "create", lambda **request: Response())
    with ThreadPoolExecutor(16) as pool:
        answers = list(pool.map(executor, ["the same prompt"] * 64))
    assert answers == ["same"] * 64
    assert executor.read_cache("the same prompt") == "same"


def test_requests_rotate_over_several_servers(tmp_path, monkeypatch):
    class Choice:
        message = type("M", (), {"content": "ok"})()

    class Response:
        choices = (Choice(),)
        usage = None

    executor = Executor(ExecutorConfig(model="m", base_url="http://127.0.0.1:1/v1, http://127.0.0.1:2/v1", seed=1))
    hits = []
    for number, client in enumerate(executor.clients):
        monkeypatch.setattr(client.chat.completions, "create", lambda number=number, **request: hits.append(number) or Response())
    for index in range(4):
        executor(f"prompt {index}")
    assert hits == [0, 1, 0, 1]

import pytest

from sigop.serving import (
    MODELS_FILE,
    Proxy,
    executor_server,
    free_port,
    load_models,
    proposer_server,
    served_name,
    settings_of,
)


def test_served_name_is_the_last_path_component_lower_cased():
    assert served_name("openai/gpt-oss-20b") == "gpt-oss-20b"
    assert served_name("Qwen/Qwen3.5-35B-A3B-FP8") == "qwen3.5-35b-a3b-fp8"
    assert served_name("/models/Qwen3.8-27B/") == "qwen3.8-27b"


def test_models_yaml_lists_the_paper_models():
    models = load_models()
    assert models.names() == ["Qwen/Qwen3.5-35B-A3B-FP8", "Qwen/Qwen3.8-27B-FP8", "deepseek-ai/DeepSeek-V4-Flash-0731",
                              "openai/gpt-oss-20b", "openai/gpt-oss-120b"]
    assert models.gpu_memory_utilization == 0.90


def test_family_members_take_the_family_settings():
    models = load_models()
    for name in ("openai/gpt-oss-20b", "openai/gpt-oss-120b"):
        entry = settings_of(name, models)
        assert (entry.reasoning_parser, entry.tool_call_parser, entry.max_model_len) == ("openai_gptoss", "openai", 131072)


def test_model_keys_override_the_family(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text("gpu_memory_utilization: 0.5\nmodels:\n  x/y:\n    family: f\n    max_model_len: 8192\n"
                    "families:\n  f:\n    reasoning_parser: r\n    tool_call_parser: t\n    max_model_len: 4096\n",
                    encoding="utf-8")
    entry = settings_of("x/y", load_models(path))
    assert (entry.reasoning_parser, entry.max_model_len) == ("r", 8192)


def test_unknown_family_and_missing_keys_are_refused(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text("gpu_memory_utilization: 0.5\nmodels:\n  x/y:\n    family: nope\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="names family 'nope'"):
        load_models(path)
    path.write_text("gpu_memory_utilization: 0.5\nmodels:\n  x/y:\n    reasoning_parser: r\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="lacks tool_call_parser"):
        load_models(path)


def test_lookup_is_by_the_exact_name_passed():
    assert settings_of("Qwen/Qwen3.5-35B-A3B-FP8", load_models()).reasoning_parser == "qwen3"
    with pytest.raises(SystemExit):
        settings_of("qwen/qwen3.5-35b-a3b-fp8", load_models())


def test_unknown_model_error_names_the_file_and_shows_the_entry():
    with pytest.raises(SystemExit) as raised:
        settings_of("mistralai/Mistral-Large", load_models())
    message = str(raised.value)
    assert message.startswith(f"mistralai/Mistral-Large is not in {MODELS_FILE}. Models optimize.py can serve:")
    assert "  openai/gpt-oss-20b" in message
    assert "  mistralai/Mistral-Large:\n    reasoning_parser: <vLLM reasoning parser>" in message
    assert "--proposer_url" in message


def test_executor_gets_the_reasoning_parser_and_context_length_only():
    command = executor_server("openai/gpt-oss-20b", 8010, "0").command()
    assert command[command.index("--reasoning-parser") + 1] == "openai_gptoss"
    assert command[command.index("--max-model-len") + 1] == "131072"
    assert "--enable-auto-tool-choice" not in command and "--tool-call-parser" not in command


@pytest.mark.parametrize("model, tool_parser", [
    ("Qwen/Qwen3.5-35B-A3B-FP8", "qwen3_coder"),
    ("Qwen/Qwen3.8-27B-FP8", "qwen3_coder"),
    ("deepseek-ai/DeepSeek-V4-Flash-0731", "deepseek_v3"),
    ("openai/gpt-oss-20b", "openai"),
])
def test_proposer_adds_the_tool_call_parser(model, tool_parser):
    command = proposer_server(model, 8020, "1").command()
    assert "--enable-auto-tool-choice" in command
    assert command[command.index("--tool-call-parser") + 1] == tool_parser


def test_qwen_uses_the_vendored_chat_template_in_both_roles():
    for server in (executor_server("Qwen/Qwen3.5-35B-A3B-FP8", 8010, "0"), proposer_server("Qwen/Qwen3.8-27B-FP8", 8020, "1")):
        command = server.command()
        template = command[command.index("--chat-template") + 1]
        assert template.endswith(("chat_templates/qwen35_chat_template.jinja", "chat_templates/qwen38_chat_template.jinja"))


def test_proxy_config_maps_opus_and_catch_all_to_the_proposer():
    text = Proxy(8030, "http://127.0.0.1:8020/v1", "gpt-oss-20b").config_text()
    assert text.count("hosted_vllm/gpt-oss-20b") == 2 and 'model_name: "opus"' in text and 'model_name: "*"' in text
    assert "callbacks: proxy_hooks.proxy_handler_instance" in text


def test_free_port_can_be_bound():
    import socket
    port = free_port()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", port))


def test_a_local_checkpoint_folder_matches_its_entry_by_last_component():
    entry = settings_of("/home/me/models/Qwen3.8-27B-FP8", load_models())
    assert entry.name == "Qwen/Qwen3.8-27B-FP8"
    command = proposer_server("/home/me/models/Qwen3.8-27B-FP8", 8020, "1").command()
    assert command[command.index("--model") + 1] == "/home/me/models/Qwen3.8-27B-FP8"
    assert command[command.index("--served-model-name") + 1] == "qwen3.8-27b-fp8"


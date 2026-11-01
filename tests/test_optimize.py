from optimize import Settings, anthropic_model_of, load_settings, save_settings
from sigop.rundir import RunDir


def test_anthropic_models_are_recognised_by_their_id_prefix():
    assert anthropic_model_of("claude-sonnet-5-5") == "claude-sonnet-5-5"
    assert anthropic_model_of("claude-opus-5-5") == "claude-opus-5-5"
    assert anthropic_model_of("Qwen/Qwen3.5-35B-A3B-FP8") == ""
    assert anthropic_model_of("openai/gpt-oss-20b") == ""


def test_settings_round_trip_through_the_run_folder(tmp_path):
    run = RunDir(tmp_path / "r1")
    run.create()
    settings = Settings("Qwen/Qwen3.5-35B-A3B-FP8", "openai/gpt-oss-20b", "lawbench", 5, 0.01, 3, 2400, 600, 5400, 32, 42, True)
    save_settings(run, settings, iterations=50, gpus="0,1", executor_url="", proposer_url="")
    run.detector.write_text("{}\n", encoding="utf-8")
    assert load_settings(run) == settings
    assert "iterations: 50" in (run.heldout / "config.yaml").read_text(encoding="utf-8")

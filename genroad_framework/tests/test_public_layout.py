"""Model-free checks for the public GenRoad Framework layout."""

from __future__ import annotations

import ast
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_example_config_is_valid_and_portable() -> None:
    config_path = ROOT / "configs" / "config.example.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["paths"]["data_root"] == "./data"
    assert config["paths"]["output_root"] == "./output"
    assert config["gui"]["server"]["host"] == "127.0.0.1"
    assert "weather_configs" in config["scene_transform"]
    assert not any("/home/" in line for line in config_path.read_text().splitlines())


def test_web_entrypoint_parses_without_importing_models() -> None:
    source = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "main"
        for node in ast.walk(tree)
    )


def test_production_workflow_remains_available() -> None:
    source = (ROOT / "app" / "ui.py").read_text(encoding="utf-8")
    for function_name in (
        "run_inpainting",
        "sam_segment_click",
        "apply_harmonization_with_sam",
        "apply_weather",
        "save_results",
    ):
        assert f"def {function_name}" in source


def test_repository_does_not_track_local_runtime_artifacts() -> None:
    forbidden = (".cursor", ".gradio", "penv", "config.yaml")
    tracked_files = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
    }
    assert not any(
        any(part in forbidden for part in Path(relative).parts)
        for relative in tracked_files
    )


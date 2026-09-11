"""Portable configuration loading and validation for GenRoad Framework."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

LOGGER = logging.getLogger(__name__)
_CONFIG_CACHE: dict[str, Any] | None = None


def get_project_root() -> Path:
    """Return the directory containing the generator configuration."""
    current = Path(__file__).resolve()
    for parent in (current, *current.parents):
        if (parent / "configs" / "config.yaml").exists():
            return parent
        if (parent / "configs" / "config.example.yaml").exists():
            return parent
    return current.parents[2]


def expand_path(path: str | Path, project_root: Path | None = None) -> Path:
    """Expand environment variables and resolve a path relative to the project."""
    root = project_root or get_project_root()
    expanded = os.path.expanduser(os.path.expandvars(str(path)))
    if not expanded:
        return root
    resolved = Path(expanded)
    return (root / resolved).resolve() if not resolved.is_absolute() else resolved.resolve()


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML configuration and attach its resolved project root."""
    path = Path(config_path) if config_path else get_project_root() / "configs" / "config.yaml"
    if not path.exists():
        LOGGER.warning("Configuration file not found: %s", path)
        return {}
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    project_root = expand_path(config.get("paths", {}).get("project_root", ""))
    config.setdefault("paths", {})["_resolved_project_root"] = str(project_root)
    return config


def get_value(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Read a nested configuration value."""
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def get_path(config: dict[str, Any], *keys: str, default: str = "") -> Path:
    """Read and resolve a nested path value."""
    project_root = Path(
        get_value(config, "paths", "_resolved_project_root", default=str(get_project_root()))
    )
    return expand_path(get_value(config, *keys, default=default), project_root)


def get_model_path(config: dict[str, Any], model_type: str) -> str:
    """Return a Hugging Face model ID or resolved local model path."""
    key_map = {
        "inpainting": ("inpainting", "model_id"),
        "scene_transform": ("scene_transform", "model_path"),
        "depth_estimation": ("depth_estimation", "model_id"),
        "segmentation": ("segmentation", "model_id"),
        "clip": ("clip", "model_id"),
    }
    section_key = key_map.get(model_type)
    if section_key is None:
        return ""
    model_id = get_value(config, "models", *section_key, default="")
    if not isinstance(model_id, str):
        return ""
    if model_id.startswith(("./", "../", "/", "~", "$")):
        project_root = Path(
            get_value(config, "paths", "_resolved_project_root", default=str(get_project_root()))
        )
        return str(expand_path(model_id, project_root))
    return model_id


def get_gui_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Return normalized defaults consumed by the web interface."""
    project_root = Path(
        get_value(config, "paths", "_resolved_project_root", default=str(get_project_root()))
    )
    image_folder = get_value(config, "paths", "gui", "default_image_folder", default="")
    output_folder = get_value(config, "paths", "gui", "default_output_folder", default="")
    if not image_folder:
        image_folder = get_value(config, "paths", "data_root", default="./data")
    if not output_folder:
        output_folder = expand_path(
            get_value(config, "paths", "output_root", default="./output"), project_root
        ) / "gui"
    defaults = get_value(config, "gui", "defaults", default={})
    return {
        "image_folder": str(expand_path(image_folder, project_root)),
        "output_folder": str(expand_path(output_folder, project_root)),
        "inpainting_steps": defaults.get("inpainting_steps", 50),
        "inpainting_guidance": defaults.get("inpainting_guidance", 15.0),
        "weather_steps": defaults.get("weather_steps", 30),
        "weather_guidance": defaults.get("weather_guidance", 7.5),
        "weather_img_guidance": defaults.get("weather_img_guidance", 1.5),
        "default_weather_effects": defaults.get(
            "default_weather_effects", ["snow", "rain", "fog", "night", "dawn"]
        ),
    }


def validate_config(config: dict[str, Any]) -> dict[str, list[str]]:
    """Validate required sections and local paths without loading model weights."""
    errors: list[str] = []
    warnings: list[str] = []
    if not config:
        errors.append("Configuration is empty.")
        return {"errors": errors, "warnings": warnings}
    if not get_model_path(config, "inpainting"):
        errors.append("models.inpainting.model_id is required.")
    scene_model = get_model_path(config, "scene_transform")
    if not scene_model:
        errors.append("models.scene_transform.model_path is required.")
    elif Path(scene_model).is_absolute() and not Path(scene_model).exists():
        warnings.append(f"CosXL checkpoint not found: {scene_model}")
    data_root = get_path(config, "paths", "data_root", default="./data")
    if not data_root.exists():
        warnings.append(f"Data directory does not exist yet: {data_root}")
    return {"errors": errors, "warnings": warnings}


def get_config(reload: bool = False) -> dict[str, Any]:
    """Return the cached default configuration."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None or reload:
        _CONFIG_CACHE = load_config()
    return _CONFIG_CACHE


def print_config_summary(config: dict[str, Any]) -> None:
    """Print a concise human-readable configuration summary."""
    print(f"Project root: {get_project_root()}")
    print(f"Data root: {get_path(config, 'paths', 'data_root')}")
    print(f"Output root: {get_path(config, 'paths', 'output_root')}")
    print(f"Inpainting model: {get_model_path(config, 'inpainting')}")
    print(f"Scene transform model: {get_model_path(config, 'scene_transform')}")
    validation = validate_config(config)
    for warning in validation["warnings"]:
        print(f"Warning: {warning}")
    for error in validation["errors"]:
        print(f"Error: {error}")

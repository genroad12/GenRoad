"""Application state and lazy model access for the web interface."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from genroad.utils.config_helper import (
    get_gui_defaults,
    get_model_path,
    get_project_root,
    load_config,
)
from genroad.utils.object_specs import get_all_object_keys

LOGGER = logging.getLogger(__name__)

WEATHER_PROMPTS: Dict[str, str] = {
    "snow": "Change the season to winter with snow, photo-realistic.",
    "rain": "Change the weather to rainy, photo-realistic.",
    "fog": "Change the weather to foggy, photo-realistic.",
    "night": "Change to a night scene with lights on, photo-realistic.",
    "sunny": "Change the weather to a bright sunny day, photo-realistic.",
    "autumn": "Change the season to autumn, photo-realistic.",
    "spring": "Change the season to spring, photo-realistic.",
    "dawn": "Change to a dawn scene, photo-realistic.",
}


class WebAppState:
    """State for one web application session and its lazy-loaded models."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.current_image: Optional[Image.Image] = None
        self.current_image_path: Optional[Path] = None
        self.accepted_image: Optional[Image.Image] = None
        self.inpainted_image: Optional[Image.Image] = None
        self.weather_results: Dict[str, Image.Image] = {}
        self.accepted_weather: Dict[str, Image.Image] = {}
        self.bbox: Optional[Tuple[int, int, int, int]] = None
        self.current_folder: Optional[Path] = None
        self.image_files: List[Path] = []
        self.current_index = 0
        self.annotations: List[Dict] = []
        self.click_state = "idle"
        self.first_click: Optional[Tuple[int, int]] = None
        self.raw_inpainted_image: Optional[Image.Image] = None
        self.harmonized_image: Optional[Image.Image] = None
        self.inpaint_source: Optional[Image.Image] = None
        self.inpaint_mask: Optional[Image.Image] = None
        self.last_inpaint_obj_type: Optional[str] = None
        self.last_inpaint_seed: Optional[int] = None
        self.current_object_accepted = False
        self.pending_inpaintings: List[Dict] = []
        self.combined_mask: Optional[Image.Image] = None
        self.sam_masks: List[Dict] = []
        self.sam_combined_mask: Optional[np.ndarray] = None
        self.sam_click_point: Optional[Tuple[int, int]] = None
        self.sam_current_mask: Optional[np.ndarray] = None
        self.sam_current_score = 0.0
        self._inpainter = None
        self._scene_editor = None
        self._sam_segmenter = None
        self.object_types = get_all_object_keys()

    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration and apply configured weather prompts."""
        config = load_config(config_path)
        for name, item in config.get("scene_transform", {}).get(
            "weather_configs", {}
        ).items():
            if isinstance(item, dict) and item.get("prompt"):
                WEATHER_PROMPTS[name] = item["prompt"]
        return config

    @property
    def gui_defaults(self) -> dict:
        """Return values used to initialize GUI controls."""
        return get_gui_defaults(self.config)

    @property
    def inpainter(self):
        """Lazily load the Stable Diffusion inpainting model."""
        if self._inpainter is None:
            from genroad.models.inpainter import StableDiffusionInpainter

            model_id = get_model_path(self.config, "inpainting")
            self._inpainter = StableDiffusionInpainter(
                model_id=model_id or "stabilityai/stable-diffusion-2-inpainting"
            )
        return self._inpainter

    @property
    def scene_editor(self):
        """Lazily load the CosXL Edit scene transformation model."""
        if self._scene_editor is None:
            from genroad.models.scene_editor import SceneEditor

            model_path = get_model_path(self.config, "scene_transform")
            if not model_path:
                model_path = str(get_project_root() / "models" / "cosxl_edit.safetensors")
            self._scene_editor = SceneEditor(model_path=model_path)
        return self._scene_editor

    @property
    def sam_segmenter(self):
        """Lazily load the SAM segmentation model."""
        if self._sam_segmenter is None:
            from genroad.models.sam_segmenter import SAMSegmenter

            self._sam_segmenter = SAMSegmenter(
                model_variant="vit-base",
                device="cuda" if torch.cuda.is_available() else "cpu",
                dtype="float32",
            )
        return self._sam_segmenter

    def clear_sam_state(self) -> None:
        """Clear accepted and preview SAM masks."""
        self.sam_masks = []
        self.sam_combined_mask = None
        self.sam_click_point = None
        self.sam_current_mask = None
        self.sam_current_score = 0.0

    def clear_gpu_memory(self) -> None:
        """Release Python and CUDA caches after a failed generation."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

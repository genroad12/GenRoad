#!/usr/bin/env python3
"""
Web-based "GenRoad: A Generative Framework for Synthesizing Anomalies in Autonomous Driving" GUI using Gradio.
Lightweight version with click-based bbox selection.
"""

from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
from typing import Optional, Tuple, List, Dict
import logging
import random
import torch
import gc
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from genroad.utils.object_specs import get_all_object_keys, get_prompt, OBJECT_PROMPTS
from genroad.utils.config_helper import (
    load_config, get_path, get_value, get_model_path, 
    get_gui_defaults, get_project_root, expand_path
)
from genroad.models.scene_editor import SceneEditor

WEATHER_PROMPTS = {
    "snow": "change the season to winter with snow, photo-realistic",
    "rain": "change the weather to rainy, photo-realistic",
    "fog": "change the weather to foggy, photo-realistic",
    "night": "change to night scene with lights on, photo-realistic",
    "sunny": "change the weather to bright sunny day, photo-realistic",
    "autumn": "change the season to autumn, photo-realistic",
    "spring": "change the season to spring, photo-realistic",
    "dawn": "change to dawn scene, photo-realistic",
}


class _LegacyWebAppState:
    """Compatibility-only state implementation retained during migration."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.current_image: Optional[Image.Image] = None  # Original loaded image
        self.current_image_path: Optional[Path] = None
        self.accepted_image: Optional[Image.Image] = None  # Last accepted/approved result
        self.inpainted_image: Optional[Image.Image] = None  # Current working result (may be rejected)
        self.weather_results: Dict[str, Image.Image] = {}  # Current working weather results
        self.accepted_weather: Dict[str, Image.Image] = {}  # Accepted weather results
        self.bbox: Optional[Tuple[int, int, int, int]] = None
        self.current_folder: Optional[Path] = None
        self.image_files: List[Path] = []
        self.current_index: int = 0
        
        # Annotations for accepted inpainting results
        self.annotations: List[Dict] = []  # [{bbox, label, seed, timestamp}, ...]
        
        # Click state for bbox selection
        self.click_state = "idle"  # idle, first_click, second_click
        self.first_click: Optional[Tuple[int, int]] = None
        
        # Harmonization workflow state
        self.raw_inpainted_image: Optional[Image.Image] = None  # Raw inpainting without harmonization
        self.harmonized_image: Optional[Image.Image] = None  # Harmonized result
        self.inpaint_source: Optional[Image.Image] = None  # Source image used for inpainting
        self.inpaint_mask: Optional[Image.Image] = None  # Mask used for inpainting
        self.last_inpaint_obj_type: Optional[str] = None  # Object type of last inpainting
        self.last_inpaint_seed: Optional[int] = None  # Seed of last inpainting
        self.current_object_accepted: bool = False  # Flag to prevent duplicate annotations
        
        # Multi-object harmonization support
        self.pending_inpaintings: List[Dict] = []  # [{bbox, mask, obj_type, seed}, ...]
        self.combined_mask: Optional[Image.Image] = None  # Combined mask for all pending objects
        
        # SAM 2 Segmentation for harmonization masks
        self.sam_masks: List[Dict] = []  # [{mask, bbox, obj_type, score}, ...]
        self.sam_combined_mask: Optional[np.ndarray] = None  # Combined SAM masks
        self.sam_click_point: Optional[Tuple[int, int]] = None  # Last click for SAM
        self.sam_current_mask: Optional[np.ndarray] = None  # Current preview mask
        self.sam_current_score: float = 0.0  # Current mask confidence
        
        self._inpainter = None
        self._scene_editor = None
        self._harmonizer = None  # Separate harmonizer instance for the tab
        self._sam_segmenter = None  # SAM 2 for harmonization masks
        self.object_types = get_all_object_keys()
        logger.info("WebAppState initialized")
    
    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration using config_helper."""
        config = load_config(config_path)
        weather_configs = config.get("scene_transform", {}).get("weather_configs", {})
        for weather_name, weather_config in weather_configs.items():
            prompt = weather_config.get("prompt") if isinstance(weather_config, dict) else None
            if prompt:
                WEATHER_PROMPTS[weather_name] = prompt
        return config
    
    @property
    def gui_defaults(self) -> dict:
        """Get GUI default values from config."""
        return get_gui_defaults(self.config)
    
    @property
    def inpainter(self):
        if self._inpainter is None:
            logger.info("Loading StableDiffusionInpainter...")
            from genroad.models.inpainter import StableDiffusionInpainter
            
            # Get model path from config (handles both HF IDs and local paths)
            model_id = get_model_path(self.config, 'inpainting')
            if not model_id:
                # Fallback to HuggingFace model
                model_id = "stabilityai/stable-diffusion-2-inpainting"
                logger.warning(f"No inpainting model configured, using default: {model_id}")
            
            logger.info(f"Loading inpainting model: {model_id}")
            self._inpainter = StableDiffusionInpainter(model_id=model_id)
            logger.info("Inpainter loaded")
        return self._inpainter
    
    @property
    def scene_editor(self):
        if self._scene_editor is None:
            logger.info("Loading SceneEditor...")
            from genroad.models.scene_editor import SceneEditor
            
            # Get model path from config
            model_path = get_model_path(self.config, 'scene_transform')
            if not model_path:
                # Try default location
                model_path = str(get_project_root() / "models" / "cosxl_edit.safetensors")
                logger.warning(f"No scene transform model configured, trying: {model_path}")
            
            logger.info(f"Loading scene editor model: {model_path}")
            self._scene_editor = SceneEditor(model_path=model_path)
            logger.info("SceneEditor loaded")
        return self._scene_editor
    
    @property
    def sam_segmenter(self):
        """Lazy-load SAM segmenter for harmonization masks."""
        if self._sam_segmenter is None:
            logger.info("Loading SAM Segmenter...")
            from genroad.models.sam_segmenter import SAMSegmenter
            
            # Use vit-base for good speed/quality balance
            self._sam_segmenter = SAMSegmenter(
                model_variant="vit-base",
                device="cuda" if torch.cuda.is_available() else "cpu",
                dtype="float32"  # SAM requires float32 for stability
            )
            logger.info("SAM Segmenter loaded")
        return self._sam_segmenter
    
    def clear_sam_state(self):
        """Reset SAM segmentation state."""
        self.sam_masks = []
        self.sam_combined_mask = None
        self.sam_click_point = None
        self.sam_current_mask = None
        self.sam_current_score = 0.0
    
    def clear_gpu_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


from app.state import WEATHER_PROMPTS as _CONFIGURED_WEATHER_PROMPTS
from app.state import WebAppState

WEATHER_PROMPTS = _CONFIGURED_WEATHER_PROMPTS
app_state: Optional[WebAppState] = None


def initialize_app(config_path: Optional[str] = None) -> str:
    global app_state
    app_state = WebAppState(config_path)
    return "✓ Ready"


def load_folder(folder_path: str) -> Tuple[List[str], str, str]:
    global app_state
    if app_state is None:
        initialize_app()
    
    folder = Path(folder_path).expanduser()
    if not folder.exists() or not folder.is_dir():
        return [], f"✗ Invalid folder: {folder_path}", ""
    
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    files = []
    for ext in extensions:
        files.extend(folder.glob(f"*{ext}"))
        files.extend(folder.glob(f"*{ext.upper()}"))
    
    files = sorted(set(files), key=lambda x: x.name)
    if not files:
        return [], f"✗ No images in {folder_path}", ""
    
    app_state.current_folder = folder
    app_state.image_files = files
    app_state.current_index = 0
    
    choices = [f.name for f in files]
    return choices, f"✓ Found {len(files)} images", str(files[0])


def select_image(name: str) -> Tuple[Optional[np.ndarray], str, str]:
    global app_state
    if not app_state or not app_state.image_files:
        return None, "Load folder first", ""
    
    for i, f in enumerate(app_state.image_files):
        if f.name == name:
            app_state.current_index = i
            break
    
    path = app_state.image_files[app_state.current_index]
    try:
        app_state.current_image = Image.open(path)
        app_state.current_image_path = path
        # Reset all working state when loading new image
        app_state.accepted_image = None
        app_state.inpainted_image = None
        app_state.weather_results = {}
        app_state.accepted_weather = {}
        app_state.annotations = []
        app_state.click_state = "idle"
        app_state.first_click = None
        
        return np.array(app_state.current_image), f"✓ {name} ({app_state.current_index+1}/{len(app_state.image_files)})", str(path)
    except Exception as e:
        return None, f"✗ Error: {e}", ""


def navigate(direction: str) -> Tuple[Optional[np.ndarray], str, str, str]:
    global app_state
    if not app_state or not app_state.image_files:
        return None, "Load folder first", "", ""
    
    if direction == "prev":
        app_state.current_index = max(0, app_state.current_index - 1)
    else:
        app_state.current_index = min(len(app_state.image_files) - 1, app_state.current_index + 1)
    
    path = app_state.image_files[app_state.current_index]
    try:
        app_state.current_image = Image.open(path)
        app_state.current_image_path = path
        # Reset all working state when navigating to new image
        app_state.accepted_image = None
        app_state.inpainted_image = None
        app_state.weather_results = {}
        app_state.accepted_weather = {}
        app_state.annotations = []
        app_state.click_state = "idle"
        app_state.first_click = None
        
        return np.array(app_state.current_image), f"✓ {path.name}", str(path), path.name
    except Exception as e:
        return None, f"✗ Error: {e}", "", ""


def handle_image_click(image, evt: gr.SelectData, x1, y1, x2, y2):
    """Handle click on image to set bbox coordinates."""
    global app_state
    if app_state is None or image is None:
        return x1, y1, x2, y2, "Load an image first", "Not selected", image, image
    
    click_x, click_y = evt.index[0], evt.index[1]
    
    # Get original image (without any bbox drawn)
    source = app_state.accepted_image or app_state.current_image
    if source is None:
        source = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    
    if app_state.click_state == "idle" or app_state.click_state == "second_click":
        # First click - set top-left
        app_state.first_click = (click_x, click_y)
        app_state.click_state = "first_click"
        
        # Draw a small marker at first click point
        img_with_marker = draw_bbox_on_image(np.array(source), click_x-5, click_y-5, click_x+5, click_y+5, color='yellow')
        
        return (click_x, click_y, x2, y2, 
                f"🖱️ First corner: ({click_x}, {click_y}) - Click for second corner", 
                "Selecting...",
                img_with_marker, img_with_marker)
    else:
        # Second click - set bottom-right
        fx, fy = app_state.first_click
        new_x1 = min(fx, click_x)
        new_y1 = min(fy, click_y)
        new_x2 = max(fx, click_x)
        new_y2 = max(fy, click_y)
        
        app_state.click_state = "second_click"
        
        # Draw bbox on the source image
        img_with_bbox = draw_bbox_on_image(np.array(source), new_x1, new_y1, new_x2, new_y2)
        bbox_text = f"({new_x1}, {new_y1}) → ({new_x2}, {new_y2})"
        
        return (new_x1, new_y1, new_x2, new_y2,
                f"✓ BBox selected: {bbox_text}",
                bbox_text,
                img_with_bbox, img_with_bbox)


def reset_bbox_selection():
    """Reset bbox click state."""
    global app_state
    if app_state:
        app_state.click_state = "idle"
        app_state.first_click = None
        
        # Return original image without bbox
        source = app_state.accepted_image or app_state.current_image
        if source:
            return np.array(source), "Not selected", "Selection reset - click to start"
    return None, "Not selected", "Selection reset"


def draw_bbox_on_image(image, x1, y1, x2, y2, color='red', width=3):
    """Draw bbox rectangle on image."""
    if image is None:
        return None
    img = Image.fromarray(image.copy()) if isinstance(image, np.ndarray) else image.copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle([int(x1), int(y1), int(x2), int(y2)], outline=color, width=width)
    return np.array(img)


def preview_bbox(image, x1, y1, x2, y2):
    """Preview bbox on image."""
    if image is None:
        return None
    return draw_bbox_on_image(image, x1, y1, x2, y2)


def run_inpainting(x1, y1, x2, y2, obj_type, seed, steps, guidance, progress=gr.Progress()):
    """Run inpainting WITHOUT harmonization - raw result only. Supports multi-object mode."""
    global app_state
    if app_state is None or app_state.current_image is None:
        return None, None, "Load an image first"
    
    if x1 >= x2 or y1 >= y2:
        return None, None, "Invalid bbox coordinates"
    
    try:
        progress(0.1, desc="Preparing...")
        
        # For multi-object mode: use raw_inpainted_image if we have pending objects
        # Otherwise use accepted_image or current_image
        if app_state.raw_inpainted_image is not None and len(app_state.pending_inpaintings) > 0:
            source = app_state.raw_inpainted_image
            logger.info(f"Using RAW image with {len(app_state.pending_inpaintings)} pending objects")
        elif app_state.accepted_image is not None:
            source = app_state.accepted_image
            logger.info(f"Using ACCEPTED image as source ({len(app_state.annotations)} objects)")
        else:
            source = app_state.current_image
            logger.info("Using ORIGINAL image as source (no accepted yet)")
        
        w, h = source.size
        bbox = (max(0, min(int(x1), w)), max(0, min(int(y1), h)), 
                max(0, min(int(x2), w)), max(0, min(int(y2), h)))
        
        # Store first inpainting's source for harmonization reference
        if len(app_state.pending_inpaintings) == 0:
            if app_state.accepted_image is not None:
                app_state.inpaint_source = app_state.accepted_image.copy()
            else:
                app_state.inpaint_source = app_state.current_image.copy()
        
        # Store current bbox
        app_state.bbox = bbox
        
        # Create mask for this object
        mask = Image.new('L', (w, h), 0)
        ImageDraw.Draw(mask).rectangle(bbox, fill=255)
        app_state.inpaint_mask = mask
        
        progress(0.2, desc="Getting prompt...")
        prompt = get_prompt(obj_type) or f"a {obj_type} on the road, photorealistic"
        
        if seed < 0:
            seed = random.randint(0, 2**32 - 1)
        
        progress(0.3, desc="Inpainting (no harmonization)...")
        logger.info(f"Inpainting: {obj_type}, bbox={bbox}, seed={seed}")
        
        # Disable harmonization for raw result
        original_use_harmonizer = app_state.inpainter.use_harmonizer
        app_state.inpainter.use_harmonizer = False
        
        result = app_state.inpainter.inpaint(
            image=source, mask=mask, prompt=prompt, bbox=bbox,
            negative_prompt="blurry, low quality, distorted",
            num_inference_steps=steps, guidance_scale=guidance, seed=seed,
            apply_harmonization=False  # Explicitly disable
        )
        
        # Restore setting
        app_state.inpainter.use_harmonizer = original_use_harmonizer
        
        progress(0.9, desc="Done!")
        app_state.inpainted_image = result
        app_state.raw_inpainted_image = result.copy()  # Store raw result for harmonization
        app_state.harmonized_image = None  # Reset harmonized result
        app_state.last_inpaint_obj_type = obj_type
        app_state.last_inpaint_seed = seed
        
        # Add to pending inpaintings list for multi-object harmonization
        # Store previous state for UNDO/REJECT capability
        # Also store crop info for SAM-based paste (prevents bbox shadow effect)
        crop_info = getattr(app_state.inpainter, '_last_crop_info', None)
        
        pending_obj = {
            "bbox": bbox,
            "mask": np.array(mask),
            "obj_type": obj_type,
            "seed": seed,
            "previous_raw_image": source.copy(),  # Store source BEFORE this inpainting for undo
            # Crop info for SAM-based paste
            "crop_info": crop_info.copy() if crop_info else None,
        }
        
        if crop_info:
            logger.info(f"Stored crop info: box={crop_info['crop_box']}, crop_size={crop_info['inpainted_crop'].size}")
        
        # Find existing entry with similar bbox (within 30px threshold)
        existing_idx = None
        threshold = 30
        for i, p in enumerate(app_state.pending_inpaintings):
            existing_bbox = p["bbox"]
            if (abs(existing_bbox[0] - bbox[0]) < threshold and
                abs(existing_bbox[1] - bbox[1]) < threshold and
                abs(existing_bbox[2] - bbox[2]) < threshold and
                abs(existing_bbox[3] - bbox[3]) < threshold):
                existing_idx = i
                break
        
        if existing_idx is not None:
            # Update existing entry (retry with new seed/object type)
            # Keep the original previous_raw_image from first attempt
            prev_raw = app_state.pending_inpaintings[existing_idx].get("previous_raw_image")
            pending_obj["previous_raw_image"] = prev_raw if prev_raw else source.copy()
            app_state.pending_inpaintings[existing_idx] = pending_obj
            logger.info(f"Updated pending object #{existing_idx+1}: {obj_type} at {bbox}")
        else:
            # Add new entry
            app_state.pending_inpaintings.append(pending_obj)
            logger.info(f"Added new pending object #{len(app_state.pending_inpaintings)}: {obj_type} at {bbox}")
        
        # Update combined mask (union of all pending masks)
        combined = np.zeros((h, w), dtype=np.uint8)
        for p in app_state.pending_inpaintings:
            combined = np.maximum(combined, p["mask"])
        app_state.combined_mask = Image.fromarray(combined)
        
        logger.info(f"Total pending objects: {len(app_state.pending_inpaintings)}")
        app_state.current_object_accepted = False  # Reset acceptance flag for new object
        
        # Comparison
        comp_w = w * 2 + 10
        comp = Image.new('RGB', (comp_w, h), (50, 50, 50))
        src_copy = source.copy()
        ImageDraw.Draw(src_copy).rectangle(bbox, outline='red', width=3)
        comp.paste(src_copy, (0, 0))
        comp.paste(result, (w + 10, 0))
        
        logger.info(f"✓ Inpainting complete (raw, no harmonization): {obj_type}, seed={seed}")
        return np.array(result), np.array(comp), f"✓ Done (raw) | {obj_type} | Seed: {seed} | Ready for harmonization"
    
    except Exception as e:
        logger.error(f"Inpainting error: {e}")
        app_state.clear_gpu_memory()
        return None, None, f"✗ Error: {e}"

# ============================================================================
# HARMONIZATION FUNCTIONS
# ============================================================================

def get_harmonization_preview(mask_dilate, feather_radius, diff_threshold):
    """Generate mask preview for ALL pending objects."""
    global app_state
    if app_state is None or app_state.raw_inpainted_image is None:
        return None, "No inpainting result to harmonize"
    
    if app_state.inpaint_source is None:
        return None, "Missing source image"
    
    # Use combined mask if available, otherwise single mask
    if app_state.combined_mask is not None:
        mask_array = np.array(app_state.combined_mask)
    elif app_state.inpaint_mask is not None:
        mask_array = np.array(app_state.inpaint_mask)
    else:
        return None, "Missing mask data"
    
    try:
        import cv2
        
        # Get images
        original = np.array(app_state.inpaint_source)
        inpainted = np.array(app_state.raw_inpainted_image)
        
        # Apply dilation to mask
        if mask_dilate > 0:
            kernel_size = int(mask_dilate) * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            mask_dilated = cv2.dilate(mask_array, kernel, iterations=1)
        else:
            mask_dilated = mask_array.copy()
        
        # Create difference mask
        diff = cv2.absdiff(original, inpainted)
        diff_gray = np.max(diff, axis=2)
        _, diff_binary = cv2.threshold(diff_gray, int(diff_threshold), 255, cv2.THRESH_BINARY)
        
        # Constrain to dilated mask region
        kernel_expand = np.ones((5, 5), np.uint8)
        expanded_mask = cv2.dilate(mask_dilated, kernel_expand, iterations=3)
        diff_binary = cv2.bitwise_and(diff_binary, expanded_mask)
        
        # Clean up
        kernel_close = np.ones((7, 7), np.uint8)
        diff_binary = cv2.morphologyEx(diff_binary, cv2.MORPH_CLOSE, kernel_close)
        kernel_open = np.ones((3, 3), np.uint8)
        diff_binary = cv2.morphologyEx(diff_binary, cv2.MORPH_OPEN, kernel_open)
        
        # Apply feathering
        if feather_radius > 0:
            blur_size = int(feather_radius) * 2 + 1
            feathered = cv2.GaussianBlur(diff_binary.astype(np.float32), (blur_size, blur_size), 0)
        else:
            feathered = diff_binary.astype(np.float32)
        
        # Create visualization: green overlay on inpainted image
        vis = inpainted.copy()
        mask_colored = np.zeros_like(vis)
        mask_colored[:, :, 1] = (feathered / feathered.max() * 255).astype(np.uint8) if feathered.max() > 0 else 0
        vis = (vis * 0.6 + mask_colored * 0.4).astype(np.uint8)
        
        # Draw ALL pending bboxes (different colors for each)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        for i, pending in enumerate(app_state.pending_inpaintings):
            bbox = pending["bbox"]
            color = colors[i % len(colors)]
            cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            # Add label
            label = f"{i+1}:{pending['obj_type']}"
            cv2.putText(vis, label, (bbox[0], bbox[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Legacy: Draw single bbox if no pending
        if not app_state.pending_inpaintings and app_state.bbox:
            cv2.rectangle(vis, (app_state.bbox[0], app_state.bbox[1]), 
                         (app_state.bbox[2], app_state.bbox[3]), (255, 0, 0), 2)
        
        num_objects = len(app_state.pending_inpaintings) or 1
        obj_types = ", ".join([p["obj_type"] for p in app_state.pending_inpaintings]) if app_state.pending_inpaintings else "single"
        return vis, f"Mask preview | {num_objects} object(s): {obj_types} | Dilate: {mask_dilate}px | Feather: {feather_radius}px"
    
    except Exception as e:
        logger.error(f"Mask preview error: {e}")
        return None, f"✗ Error: {e}"


def apply_harmonization(harm_method, mask_dilate, feather_radius, diff_threshold, progress=gr.Progress()):
    """Apply harmonization to ALL pending objects."""
    global app_state
    if app_state is None or app_state.raw_inpainted_image is None:
        return None, None, None, "No inpainting result to harmonize"
    
    if app_state.inpaint_source is None:
        return None, None, None, "Missing source image"
    
    try:
        num_objects = len(app_state.pending_inpaintings) or 1
        progress(0.2, desc=f"Preparing harmonization for {num_objects} object(s)...")
        
        from genroad.models.harmonizer import ImageHarmonizer, HarmonizationMethod
        
        # Create/configure harmonizer
        harmonizer = ImageHarmonizer(
            method=harm_method,
            blend_strength=0.7,
            mask_feather_radius=int(feather_radius),
            use_difference_mask=True,
            difference_threshold=int(diff_threshold),
        )
        
        progress(0.5, desc=f"Applying {harm_method} to {num_objects} object(s)...")
        
        # Use combined mask for multi-object mode, otherwise single mask
        if app_state.combined_mask is not None:
            mask_array = np.array(app_state.combined_mask)
        else:
            mask_array = np.array(app_state.inpaint_mask)
        
        # Apply morphological dilation to the mask
        if mask_dilate > 0:
            import cv2
            kernel_size = int(mask_dilate) * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            mask_array = cv2.dilate(mask_array, kernel, iterations=1)
        
        # For multi-object, compute combined bbox
        if app_state.pending_inpaintings:
            all_bboxes = [p["bbox"] for p in app_state.pending_inpaintings]
            combined_bbox = (
                min(b[0] for b in all_bboxes),
                min(b[1] for b in all_bboxes),
                max(b[2] for b in all_bboxes),
                max(b[3] for b in all_bboxes)
            )
        else:
            combined_bbox = app_state.bbox
        
        # Apply harmonization
        result = harmonizer.harmonize(
            inpainted_image=app_state.raw_inpainted_image,
            original_image=app_state.inpaint_source,
            mask=mask_array,
            bbox=combined_bbox,
        )
        
        app_state.harmonized_image = result.image
        
        progress(0.9, desc="Creating comparison...")
        
        # Create side-by-side comparison
        raw_arr = np.array(app_state.raw_inpainted_image)
        harm_arr = np.array(result.image)
        
        # Get mask preview
        mask_preview, _ = get_harmonization_preview(mask_dilate, feather_radius, diff_threshold)
        
        progress(1.0)
        
        logger.info(f"✓ Harmonization applied: {harm_method}")
        return (np.array(app_state.raw_inpainted_image), 
                np.array(result.image), 
                mask_preview,
                f"✓ Harmonization applied | Method: {harm_method}")
    
    except Exception as e:
        logger.error(f"Harmonization error: {e}")
        return None, None, None, f"✗ Error: {e}"


def is_duplicate_annotation(bbox, label, threshold=20):
    """Check if an annotation with similar bbox and same label already exists."""
    global app_state
    if not app_state or not app_state.annotations:
        return False
    
    for ann in app_state.annotations:
        if ann.get("label") != label:
            continue
        existing_bbox = ann.get("bbox", [0, 0, 0, 0])
        # Check if bboxes are similar (within threshold pixels)
        if (abs(existing_bbox[0] - bbox[0]) < threshold and
            abs(existing_bbox[1] - bbox[1]) < threshold and
            abs(existing_bbox[2] - bbox[2]) < threshold and
            abs(existing_bbox[3] - bbox[3]) < threshold):
            return True
    return False


def find_matching_annotation(obj_type, inpaint_bbox, threshold=50):
    """
    Find existing annotation that matches the object type and bbox.
    
    Returns annotation index if found, None otherwise.
    """
    global app_state
    if not app_state or not app_state.annotations:
        return None
    
    for i, ann in enumerate(app_state.annotations):
        if ann.get("label") != obj_type:
            continue
        
        # Check bbox proximity
        existing_bbox = ann.get("bbox", [0, 0, 0, 0])
        if (abs(existing_bbox[0] - inpaint_bbox[0]) < threshold and
            abs(existing_bbox[1] - inpaint_bbox[1]) < threshold and
            abs(existing_bbox[2] - inpaint_bbox[2]) < threshold and
            abs(existing_bbox[3] - inpaint_bbox[3]) < threshold):
            return i
    
    return None


def expand_bbox(bbox, expand_percent, img_width, img_height):
    """
    Expand bounding box by a percentage.
    
    Args:
        bbox: (x1, y1, x2, y2) tuple
        expand_percent: Expansion ratio (e.g., 0.05 for 5%)
        img_width, img_height: Image dimensions for boundary checking
    
    Returns:
        Expanded bbox tuple (x1, y1, x2, y2) clipped to image bounds
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    # Expand by percentage (half on each side)
    expand_w = int(width * expand_percent / 2)
    expand_h = int(height * expand_percent / 2)
    
    # Apply expansion with boundary checks
    new_x1 = max(0, x1 - expand_w)
    new_y1 = max(0, y1 - expand_h)
    new_x2 = min(img_width, x2 + expand_w)
    new_y2 = min(img_height, y2 + expand_h)
    
    return (new_x1, new_y1, new_x2, new_y2)


def accept_harmonization():
    """Accept the harmonized result for all pending objects.

    Existing annotations are updated when a matching object is found.
    SAM bounding boxes are preferred and expanded according to configuration.
    """
    global app_state
    if app_state is None or app_state.harmonized_image is None:
        return None, None, None, "No harmonized result to accept", "", "", ""
    
    # Update both inpainted_image and accepted_image
    app_state.inpainted_image = app_state.harmonized_image.copy()
    app_state.accepted_image = app_state.harmonized_image.copy()
    
    # Get image dimensions for bbox expansion
    img_width, img_height = app_state.harmonized_image.size
    
    # Get bbox expansion percentage from config (default 5%)
    try:
        from genroad.utils.config_helper import get_config
        config = get_config()
        bbox_expand = config.get("models", {}).get("sam", {}).get("bbox_expand_percent", 0.05)
    except Exception:
        bbox_expand = 0.05  # Default 5%
    
    logger.info(f"SAM bbox expansion: {bbox_expand * 100:.1f}%")
    
    # Build SAM masks list indexed for matching
    sam_masks_list = list(app_state.sam_masks) if app_state.sam_masks else []
    
    added_count = 0
    updated_count = 0
    
    if app_state.pending_inpaintings:
        for i, pending in enumerate(app_state.pending_inpaintings):
            obj_type = pending["obj_type"]
            seed = pending["seed"]
            inpaint_bbox = pending["bbox"]
            
            # Determine final bbox: prefer SAM, fallback to inpainting
            final_bbox = None
            bbox_source = "inpainting"
            
            # Try to match SAM mask by index (most reliable)
            if i < len(sam_masks_list):
                sam_bbox = sam_masks_list[i].get("bbox")
                if sam_bbox:
                    # Expand SAM bbox by configured percentage
                    final_bbox = expand_bbox(sam_bbox, bbox_expand, img_width, img_height)
                    bbox_source = f"SAM+{int(bbox_expand*100)}%"
            
            # If no SAM bbox, use inpainting bbox (no expansion)
            if final_bbox is None:
                final_bbox = inpaint_bbox
            
            # Check if this object already has an annotation (UPDATE instead of ADD)
            existing_idx = find_matching_annotation(obj_type, inpaint_bbox)
            
            if existing_idx is not None:
                # UPDATE existing annotation with SAM bbox
                app_state.annotations[existing_idx]["bbox"] = list(final_bbox)
                app_state.annotations[existing_idx]["bbox_source"] = bbox_source
                app_state.annotations[existing_idx]["updated"] = datetime.now().isoformat()
                app_state.annotations[existing_idx]["harmonized"] = True
                updated_count += 1
                logger.info(f"✓ Updated annotation #{app_state.annotations[existing_idx]['id']}: "
                           f"{obj_type} bbox -> {final_bbox} (from {bbox_source})")
            else:
                # ADD new annotation (no existing match)
                annotation = {
                    "id": len(app_state.annotations) + 1,
                    "bbox": list(final_bbox),
                    "label": obj_type,
                    "seed": int(seed) if seed >= 0 else None,
                    "timestamp": datetime.now().isoformat(),
                    "harmonized": True,
                    "bbox_source": bbox_source
                }
                app_state.annotations.append(annotation)
                added_count += 1
                logger.info(f"✓ Added annotation: {obj_type} at {final_bbox} (from {bbox_source})")
    
    else:
        # Legacy single-object mode
        if sam_masks_list:
            sam_mask = sam_masks_list[0]
            bbox = sam_mask.get("bbox") or app_state.bbox or (0, 0, 0, 0)
            # Expand SAM bbox
            bbox = expand_bbox(bbox, bbox_expand, img_width, img_height)
            obj_type = sam_mask.get("obj_type") or app_state.last_inpaint_obj_type or "unknown"
            bbox_source = f"SAM+{int(bbox_expand*100)}%"
        else:
            bbox = app_state.bbox or (0, 0, 0, 0)
            obj_type = app_state.last_inpaint_obj_type or "unknown"
            bbox_source = "inpainting"
        
        seed = app_state.last_inpaint_seed or -1
        inpaint_bbox = app_state.bbox or (0, 0, 0, 0)
        
        existing_idx = find_matching_annotation(obj_type, inpaint_bbox)
        
        if existing_idx is not None:
            app_state.annotations[existing_idx]["bbox"] = list(bbox)
            app_state.annotations[existing_idx]["bbox_source"] = bbox_source
            app_state.annotations[existing_idx]["updated"] = datetime.now().isoformat()
            updated_count = 1
        else:
            annotation = {
                "id": len(app_state.annotations) + 1,
                "bbox": list(bbox),
                "label": obj_type,
                "seed": int(seed) if seed >= 0 else None,
                "timestamp": datetime.now().isoformat(),
                "harmonized": True,
                "bbox_source": bbox_source
            }
            app_state.annotations.append(annotation)
            added_count = 1
    
    # Clear pending list and SAM state after acceptance
    app_state.pending_inpaintings = []
    app_state.combined_mask = None
    app_state.raw_inpainted_image = None
    
    # Clear SAM state
    app_state.clear_sam_state()
    
    # Mark as accepted
    app_state.current_object_accepted = True
    
    logger.info(f"✓ Harmonization accepted: {added_count} added, {updated_count} updated")
    
    result_img = np.array(app_state.harmonized_image)
    status_parts = []
    if added_count > 0:
        status_parts.append(f"{added_count} added")
    if updated_count > 0:
        status_parts.append(f"{updated_count} updated")
    status_msg = f"✓ Harmonization accepted | {', '.join(status_parts)} | Total: {len(app_state.annotations)} | Ready for Weather"
    
    return (
        result_img,  # result_image in Tab 2
        result_img,  # source_display in Tab 2
        result_img,  # main_image in Tab 1
        status_msg,
        get_annotations_display(),
        get_accepted_objects_display(),
        get_pending_info()
    )


def reject_harmonization():
    """Reject harmonization and keep raw inpainting. Keeps pending objects for retry."""
    global app_state
    if app_state is None or app_state.raw_inpainted_image is None:
        return None, "No raw result available"
    
    # Reset harmonized image only (keep pending objects for retry with different params)
    app_state.harmonized_image = None
    app_state.inpainted_image = app_state.raw_inpainted_image.copy()
    num_pending = len(app_state.pending_inpaintings)
    logger.info(f"Harmonization rejected, {num_pending} pending object(s) kept for retry")
    return np.array(app_state.raw_inpainted_image), f"Harmonization rejected | {num_pending} object(s) ready for retry"


# =============================================================================
# SAM 2 SEGMENTATION FUNCTIONS
# =============================================================================

def sam_segment_click(evt: gr.SelectData):
    """Handle click on image for SAM segmentation with BBox guidance."""
    global app_state
    if app_state is None or app_state.raw_inpainted_image is None:
        return None, "No inpainting result available for segmentation"
    
    point = (evt.index[0], evt.index[1])
    app_state.sam_click_point = point
    
    try:
        import cv2
        
        # Find the closest pending inpainting bbox to the click point
        # This helps segment the ENTIRE object (not just one part)
        target_bbox = None
        target_obj_type = None
        
        if app_state.pending_inpaintings:
            min_dist = float('inf')
            for pending in app_state.pending_inpaintings:
                p_bbox = pending["bbox"]
                # Check if click is inside bbox
                if (p_bbox[0] <= point[0] <= p_bbox[2] and 
                    p_bbox[1] <= point[1] <= p_bbox[3]):
                    target_bbox = p_bbox
                    target_obj_type = pending["obj_type"]
                    logger.info(f"Click inside bbox: {target_bbox} ({target_obj_type})")
                    break
                # Calculate distance to bbox center
                center_x = (p_bbox[0] + p_bbox[2]) / 2
                center_y = (p_bbox[1] + p_bbox[3]) / 2
                dist = ((point[0] - center_x)**2 + (point[1] - center_y)**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    target_bbox = p_bbox
                    target_obj_type = pending["obj_type"]
        
        # Use BBox-guided segmentation if bbox available
        if target_bbox:
            logger.info(f"SAM BBox-guided segmentation: point={point}, bbox={target_bbox}")
            mask, score = app_state.sam_segmenter.segment_point_with_bbox(
                app_state.raw_inpainted_image,
                point=point,
                bbox=target_bbox,
                use_bbox_if_available=True,
                fallback_to_point=True,
                use_crop=True
            )
            method = f"BBox-guided ({target_obj_type})"
        else:
            # Fallback to point-only segmentation
            logger.info(f"SAM point-only segmentation: point={point}")
            mask, score = app_state.sam_segmenter.segment_point(
                app_state.raw_inpainted_image,
                point=point,
                point_label=1,
                multimask_output=True,
                return_best=True,
                use_crop=True
            )
            method = "Point-only"
        
        # Store current mask
        app_state.sam_current_mask = mask
        app_state.sam_current_score = score
        
        # Visualize mask on image
        vis = app_state.sam_segmenter.visualize_mask(
            app_state.raw_inpainted_image,
            mask,
            color=(0, 255, 0),  # Green
            alpha=0.4,
            show_contour=True
        )
        
        # Draw click point
        cv2.circle(vis, point, 8, (255, 0, 0), -1)  # Red dot
        cv2.circle(vis, point, 10, (255, 255, 255), 2)  # White border
        
        # Draw target bbox if used
        if target_bbox:
            cv2.rectangle(vis, (target_bbox[0], target_bbox[1]), 
                         (target_bbox[2], target_bbox[3]), (0, 255, 255), 2)  # Cyan for guide bbox
        
        # Get mask bbox for display
        mask_bbox = app_state.sam_segmenter.get_mask_bbox(mask, padding=5)
        if mask_bbox:
            cv2.rectangle(vis, (mask_bbox[0], mask_bbox[1]), 
                         (mask_bbox[2], mask_bbox[3]), (255, 255, 0), 2)  # Yellow for result
        
        status = f"✓ {method} | Score: {score:.3f} | Click 'Accept' or click another point"
        return vis, status
        
    except Exception as e:
        logger.error(f"SAM segmentation error: {e}")
        import traceback
        traceback.print_exc()
        return None, f"✗ Error: {e}"


def sam_add_to_current():
    """
    Add current click to the preview mask (multi-click union).
    
    For multi-part objects, click multiple components and merge them.
    """
    global app_state
    if app_state is None or app_state.sam_current_mask is None:
        return None, "No mask to add to"
    
    if app_state.raw_inpainted_image is None:
        return None, "No image available"
    
    # Current mask already contains the latest click
    # Just visualize it with "Add more clicks" message
    
    import cv2
    vis = app_state.sam_segmenter.visualize_mask(
        app_state.raw_inpainted_image,
        app_state.sam_current_mask,
        color=(0, 255, 0),
        alpha=0.4,
        show_contour=True
    )
    
    # Draw click point
    if app_state.sam_click_point:
        cv2.circle(vis, app_state.sam_click_point, 8, (255, 0, 0), -1)
        cv2.circle(vis, app_state.sam_click_point, 10, (255, 255, 255), 2)
    
    mask_area = np.sum(app_state.sam_current_mask > 0)
    return vis, f"Mask area: {mask_area} px | Click more points to expand or 'Accept'"


def sam_expand_with_click(evt: gr.SelectData):
    """
    Handle click to EXPAND current mask (union mode).
    
    Add a new click to the current mask using a union operation.
    This is useful for multi-part objects.
    """
    global app_state
    if app_state is None or app_state.raw_inpainted_image is None:
        return None, "No image available"
    
    point = (evt.index[0], evt.index[1])
    
    try:
        import cv2
        
        # Get new mask for this click
        new_mask, new_score = app_state.sam_segmenter.segment_point(
            app_state.raw_inpainted_image,
            point=point,
            point_label=1,
            multimask_output=True,
            return_best=True,
            use_crop=True
        )
        
        # If we have an existing preview mask, merge with it (union)
        if app_state.sam_current_mask is not None:
            merged_mask = np.maximum(app_state.sam_current_mask, new_mask)
            app_state.sam_current_mask = merged_mask
            app_state.sam_current_score = max(app_state.sam_current_score, new_score)
            logger.info(f"Expanded mask with point {point}, combined score={app_state.sam_current_score:.3f}")
        else:
            app_state.sam_current_mask = new_mask
            app_state.sam_current_score = new_score
        
        app_state.sam_click_point = point
        
        # Visualize
        vis = app_state.sam_segmenter.visualize_mask(
            app_state.raw_inpainted_image,
            app_state.sam_current_mask,
            color=(0, 255, 0),
            alpha=0.4,
            show_contour=True
        )
        
        cv2.circle(vis, point, 8, (255, 0, 0), -1)
        cv2.circle(vis, point, 10, (255, 255, 255), 2)
        
        mask_bbox = app_state.sam_segmenter.get_mask_bbox(app_state.sam_current_mask, padding=5)
        if mask_bbox:
            cv2.rectangle(vis, (mask_bbox[0], mask_bbox[1]), 
                         (mask_bbox[2], mask_bbox[3]), (255, 255, 0), 2)
        
        mask_area = np.sum(app_state.sam_current_mask > 0)
        return vis, f"✓ Expanded | Area: {mask_area} px | Click more or 'Accept'"
        
    except Exception as e:
        logger.error(f"Expand mask error: {e}")
        return None, f"✗ Error: {e}"


def sam_accept_mask():
    """Accept current SAM mask and add to list."""
    global app_state
    if app_state is None or app_state.sam_current_mask is None:
        return None, "No mask to accept", "No masks"
    
    try:
        mask = app_state.sam_current_mask
        bbox = app_state.sam_segmenter.get_mask_bbox(mask, padding=5)
        
        if bbox is None:
            return None, "Empty mask - click on an object", get_sam_masks_info()
        
        # Find object type from pending inpaintings (closest bbox match)
        obj_type = "unknown"
        if app_state.pending_inpaintings:
            click_center = app_state.sam_click_point
            if click_center:
                min_dist = float('inf')
                for pending in app_state.pending_inpaintings:
                    p_bbox = pending["bbox"]
                    # Check if click is inside pending bbox
                    if (p_bbox[0] <= click_center[0] <= p_bbox[2] and 
                        p_bbox[1] <= click_center[1] <= p_bbox[3]):
                        obj_type = pending["obj_type"]
                        break
                    # Otherwise find closest bbox center
                    center = ((p_bbox[0] + p_bbox[2]) / 2, (p_bbox[1] + p_bbox[3]) / 2)
                    dist = ((click_center[0] - center[0])**2 + (click_center[1] - center[1])**2)**0.5
                    if dist < min_dist:
                        min_dist = dist
                        obj_type = pending["obj_type"]
        
        # Add to masks list
        app_state.sam_masks.append({
            "mask": mask.copy(),
            "bbox": bbox,
            "obj_type": obj_type,
            "score": app_state.sam_current_score,
            "click_point": app_state.sam_click_point
        })
        
        # Update combined mask
        if app_state.sam_combined_mask is None:
            app_state.sam_combined_mask = mask.copy()
        else:
            app_state.sam_combined_mask = np.maximum(app_state.sam_combined_mask, mask)
        
        # Clear current mask
        app_state.sam_current_mask = None
        app_state.sam_current_score = 0.0
        
        # Create visualization of all masks
        vis = visualize_all_sam_masks()
        
        logger.info(f"Accepted SAM mask: {obj_type} at {bbox}, total: {len(app_state.sam_masks)}")
        return vis, f"✓ Mask accepted: {obj_type} | Total: {len(app_state.sam_masks)} mask(s)", get_sam_masks_info()
        
    except Exception as e:
        logger.error(f"Accept mask error: {e}")
        return None, f"✗ Error: {e}", get_sam_masks_info()


def apply_sam_masked_paste():
    """
    Apply SAM-based paste: Extract ONLY the object from cropped inpainting using SAM mask,
    then paste it onto the original/accepted image. This eliminates the "bbox shadow" effect.
    
    Uses advanced blending techniques to avoid boundary artifacts:
    1. Multi-level feathering for smooth transitions
    2. Color matching at boundaries
    3. Optional Poisson-like blending for seamless results
    
    Returns:
        PIL.Image: Clean image with only the objects pasted (no bbox artifacts)
    """
    import cv2
    
    global app_state
    if app_state is None:
        logger.warning("apply_sam_masked_paste: No app_state")
        return None
    
    if not app_state.sam_masks or not app_state.pending_inpaintings:
        logger.warning("apply_sam_masked_paste: No SAM masks or pending inpaintings")
        return None
    
    # Start with the base image (before any inpainting)
    if app_state.inpaint_source is not None:
        base_image = app_state.inpaint_source.copy()
    elif app_state.accepted_image is not None:
        base_image = app_state.accepted_image.copy()
    else:
        base_image = app_state.current_image.copy()
    
    base_array = np.array(base_image).astype(np.float32)
    
    logger.info(f"SAM-masked paste: {len(app_state.sam_masks)} masks, {len(app_state.pending_inpaintings)} pending objects")
    
    # Process each SAM mask and corresponding pending inpainting
    for i, sam_mask_info in enumerate(app_state.sam_masks):
        sam_mask = sam_mask_info["mask"]  # Full-size SAM mask
        click_point = sam_mask_info.get("click_point", (0, 0))
        
        # Find the corresponding pending inpainting (by click point proximity)
        matched_pending = None
        for pending in app_state.pending_inpaintings:
            p_bbox = pending["bbox"]
            if (p_bbox[0] - 50 <= click_point[0] <= p_bbox[2] + 50 and 
                p_bbox[1] - 50 <= click_point[1] <= p_bbox[3] + 50):
                matched_pending = pending
                break
        
        if matched_pending is None:
            if i < len(app_state.pending_inpaintings):
                matched_pending = app_state.pending_inpaintings[i]
        
        if matched_pending is None or matched_pending.get("crop_info") is None:
            logger.warning(f"SAM mask {i}: No matching crop info found, skipping")
            continue
        
        crop_info = matched_pending["crop_info"]
        crop_box = crop_info["crop_box"]
        inpainted_crop = crop_info["inpainted_crop"]
        
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1
        
        # Extract the SAM mask region corresponding to the crop area
        sam_crop_mask = sam_mask[crop_y1:crop_y2, crop_x1:crop_x2]
        
        if sam_crop_mask.shape != (crop_h, crop_w):
            logger.warning(f"Mask size mismatch: {sam_crop_mask.shape} vs ({crop_h}, {crop_w})")
            continue
        
        if not np.any(sam_crop_mask > 0):
            logger.warning(f"SAM mask {i}: Empty mask in crop region")
            continue
        
        # Get arrays
        inpainted_crop_array = np.array(inpainted_crop).astype(np.float32)
        base_crop = base_array[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        
        # ====================================================================
        # CONTROLLED BLENDING - Preserve object details, smooth edges only
        # ====================================================================
        
        # Step 1: Create binary mask
        binary_mask = (sam_crop_mask > 127).astype(np.uint8) * 255
        
        # Step 2: Create feathered alpha mask
        # - Core: 100% inpainted so object details are preserved.
        # - Edge: gradual blend for a smooth transition.
        
        # Erode slightly to define core region (object fully visible here)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        core_mask = cv2.erode(binary_mask, kernel_small, iterations=1)
        
        # Create transition zone: area between core and slightly expanded mask
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        outer_mask = cv2.dilate(binary_mask, kernel_medium, iterations=1)
        
        # Feather the outer mask for smooth edge transitions
        # Use moderate blur - not too aggressive
        feathered_outer = cv2.GaussianBlur(outer_mask.astype(np.float32), (21, 21), 0)
        feathered_outer = feathered_outer / 255.0
        
        # Core stays at 100%, transition zone is feathered
        core_alpha = core_mask.astype(np.float32) / 255.0
        
        # Final alpha: core is 1.0, edges transition smoothly
        alpha = np.maximum(core_alpha, feathered_outer * 0.95)
        alpha = alpha.clip(0, 1)
        
        # Step 3: Alpha blend - object preserved, edges smooth
        alpha_3ch = np.stack([alpha] * 3, axis=-1)
        blended_crop = inpainted_crop_array * alpha_3ch + base_crop * (1 - alpha_3ch)
        
        logger.info(f"SAM-masked paste {i}: Controlled blending (core preserved, edges feathered)")
        
        # Paste back
        base_array[crop_y1:crop_y2, crop_x1:crop_x2] = blended_crop
        
        logger.info(f"SAM-masked paste {i}: obj={sam_mask_info.get('obj_type', 'unknown')}, crop={crop_box}")
    
    # Convert back to PIL
    result_image = Image.fromarray(base_array.astype(np.uint8))
    
    logger.info("✓ SAM-masked paste complete: Seamless blending applied")
    return result_image


def sam_reject_mask():
    """Reject current SAM mask preview."""
    global app_state
    if app_state is None:
        return None, "No state available"
    
    app_state.sam_current_mask = None
    app_state.sam_current_score = 0.0
    
    # Show existing masks or raw image
    if app_state.sam_masks:
        vis = visualize_all_sam_masks()
    elif app_state.raw_inpainted_image is not None:
        vis = np.array(app_state.raw_inpainted_image)
    else:
        vis = None
    
    return vis, "Mask rejected | Click on another object"


def sam_clear_all_masks():
    """Clear all SAM masks."""
    global app_state
    if app_state is None:
        return None, "No state available", "No masks"
    
    num_cleared = len(app_state.sam_masks)
    app_state.clear_sam_state()
    
    if app_state.raw_inpainted_image is not None:
        vis = np.array(app_state.raw_inpainted_image)
    else:
        vis = None
    
    return vis, f"✓ Cleared {num_cleared} mask(s) | Click on objects to segment", "No masks"


def sam_remove_last_mask():
    """Remove the last added SAM mask."""
    global app_state
    if app_state is None or not app_state.sam_masks:
        return None, "No masks to remove", "No masks"
    
    removed = app_state.sam_masks.pop()
    
    # Recalculate combined mask
    if app_state.sam_masks:
        app_state.sam_combined_mask = app_state.sam_masks[0]["mask"].copy()
        for m in app_state.sam_masks[1:]:
            app_state.sam_combined_mask = np.maximum(app_state.sam_combined_mask, m["mask"])
        vis = visualize_all_sam_masks()
    else:
        app_state.sam_combined_mask = None
        vis = np.array(app_state.raw_inpainted_image) if app_state.raw_inpainted_image else None
    
    return vis, f"✓ Removed mask: {removed['obj_type']}", get_sam_masks_info()


def visualize_all_sam_masks():
    """Create visualization of all accepted SAM masks."""
    global app_state
    if app_state is None or app_state.raw_inpainted_image is None:
        return None
    
    import cv2
    vis = np.array(app_state.raw_inpainted_image)
    
    # Color palette for different masks
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 255, 0),  # Yellow
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
    ]
    
    for i, mask_info in enumerate(app_state.sam_masks):
        mask = mask_info["mask"]
        color = colors[i % len(colors)]
        
        # Overlay mask
        overlay = np.zeros_like(vis)
        mask_bool = mask > 127
        overlay[mask_bool] = color
        vis = (vis * 0.6 + overlay * 0.4).astype(np.uint8)
        
        # Draw contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 2)
        
        # Draw bbox and label
        bbox = mask_info["bbox"]
        cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        label = f"{i+1}:{mask_info['obj_type']}"
        cv2.putText(vis, label, (bbox[0], bbox[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return vis


def get_sam_masks_info():
    """Get info about accepted SAM masks."""
    global app_state
    if app_state is None or not app_state.sam_masks:
        return "No masks accepted"
    
    lines = []
    for i, m in enumerate(app_state.sam_masks):
        bbox = m["bbox"]
        lines.append(f"{i+1}. {m['obj_type']} | Score: {m['score']:.2f} | BBox: ({bbox[0]},{bbox[1]})-({bbox[2]},{bbox[3]})")
    
    return f"{len(app_state.sam_masks)} mask(s):\n" + "\n".join(lines)


def apply_harmonization_with_sam(harm_method, mask_dilate, feather_radius, progress=gr.Progress()):
    """
    Apply harmonization using SAM masks.
    
    NEW: First applies SAM-masked paste to eliminate bbox shadow effect,
    then applies color/light harmonization.
    """
    global app_state
    if app_state is None or app_state.raw_inpainted_image is None:
        return None, None, None, "No inpainting result to harmonize"
    
    if app_state.inpaint_source is None:
        return None, None, None, "Missing source image"
    
    # Check if we have SAM masks
    if not app_state.sam_masks and app_state.sam_combined_mask is None:
        return None, None, None, "No SAM masks! Click on objects to segment them first"
    
    try:
        num_masks = len(app_state.sam_masks)
        progress(0.1, desc=f"Step 1: SAM-masked paste ({num_masks} object(s))...")
        
        # ========================================================================
        # STEP 1: Apply SAM-masked paste (eliminates bbox shadow effect)
        # ========================================================================
        # This extracts ONLY the objects from cropped inpainting using SAM masks
        # and pastes them onto the original image - no bbox artifacts!
        sam_pasted_image = apply_sam_masked_paste()
        
        if sam_pasted_image is None:
            # Fallback to raw inpainting if SAM paste fails
            logger.warning("SAM-masked paste failed, using raw inpainting")
            sam_pasted_image = app_state.raw_inpainted_image.copy()
        else:
            logger.info("✓ SAM-masked paste complete: bbox artifacts removed")
        
        progress(0.3, desc="Step 2: Preparing harmonization...")
        
        from genroad.models.harmonizer import ImageHarmonizer
        import cv2
        
        # Create harmonizer
        harmonizer = ImageHarmonizer(
            method=harm_method,
            blend_strength=0.7,
            mask_feather_radius=int(feather_radius),
            use_difference_mask=False,  # Use SAM mask directly
        )
        
        progress(0.5, desc=f"Step 3: Applying {harm_method}...")
        
        # Use SAM combined mask
        mask_array = app_state.sam_combined_mask.copy()
        
        # Apply dilation
        if mask_dilate > 0:
            kernel_size = int(mask_dilate) * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            mask_array = cv2.dilate(mask_array, kernel, iterations=1)
        
        # Apply feathering
        if feather_radius > 0:
            blur_size = int(feather_radius) * 2 + 1
            mask_array = cv2.GaussianBlur(mask_array.astype(np.float32), (blur_size, blur_size), 0)
            mask_array = (mask_array / mask_array.max() * 255).astype(np.uint8) if mask_array.max() > 0 else mask_array
        
        # Calculate combined bbox
        all_bboxes = [m["bbox"] for m in app_state.sam_masks]
        combined_bbox = (
            min(b[0] for b in all_bboxes),
            min(b[1] for b in all_bboxes),
            max(b[2] for b in all_bboxes),
            max(b[3] for b in all_bboxes)
        )
        
        # ========================================================================
        # STEP 2: Apply color/light harmonization on the clean pasted image
        # ========================================================================
        result = harmonizer.harmonize(
            inpainted_image=sam_pasted_image,  # Use SAM-pasted image instead of raw
            original_image=app_state.inpaint_source,
            mask=mask_array,
            bbox=combined_bbox,
        )
        
        app_state.harmonized_image = result.image
        
        progress(0.9, desc="Creating preview...")
        
        # Create mask visualization (show on SAM-pasted image)
        vis = np.array(sam_pasted_image).copy()
        mask_overlay = np.zeros_like(vis)
        mask_bool = mask_array > 127 if mask_array.dtype == np.uint8 else mask_array > 0.5
        mask_overlay[mask_bool] = (0, 255, 0)
        vis = (vis * 0.6 + mask_overlay * 0.4).astype(np.uint8)
        
        progress(1.0)
        
        logger.info(f"✓ SAM-based harmonization applied: {harm_method}, {num_masks} mask(s), bbox shadow eliminated")
        return (
            np.array(app_state.raw_inpainted_image),
            np.array(result.image),
            vis,
            f"✓ Harmonization applied | Method: {harm_method} | {num_masks} SAM mask(s)"
        )
        
    except Exception as e:
        logger.error(f"SAM harmonization error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, f"✗ Error: {e}"


def get_sam_mask_preview(mask_dilate, feather_radius):
    """Preview the SAM mask with dilation and feathering."""
    global app_state
    if app_state is None or app_state.raw_inpainted_image is None:
        return None, "No inpainting result"
    
    if not app_state.sam_masks and app_state.sam_combined_mask is None:
        # Return raw image with message
        return np.array(app_state.raw_inpainted_image), "No SAM masks - click on objects to segment"
    
    try:
        import cv2
        
        mask_array = app_state.sam_combined_mask.copy()
        
        # Apply dilation
        if mask_dilate > 0:
            kernel_size = int(mask_dilate) * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            mask_array = cv2.dilate(mask_array, kernel, iterations=1)
        
        # Apply feathering
        if feather_radius > 0:
            blur_size = int(feather_radius) * 2 + 1
            mask_array = cv2.GaussianBlur(mask_array.astype(np.float32), (blur_size, blur_size), 0)
            if mask_array.max() > 0:
                mask_array = (mask_array / mask_array.max() * 255).astype(np.uint8)
        
        # Create visualization
        vis = np.array(app_state.raw_inpainted_image).copy()
        mask_overlay = np.zeros_like(vis)
        mask_bool = mask_array > 127 if mask_array.dtype == np.uint8 else mask_array > 0.5
        mask_overlay[mask_bool] = (0, 255, 0)
        vis = (vis * 0.6 + mask_overlay * 0.4).astype(np.uint8)
        
        # Draw contours
        mask_binary = (mask_array > 127).astype(np.uint8) * 255 if mask_array.dtype == np.uint8 else (mask_array > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
        
        num_masks = len(app_state.sam_masks)
        return vis, f"Mask preview | {num_masks} mask(s) | Dilate: {mask_dilate}px | Feather: {feather_radius}px"
        
    except Exception as e:
        logger.error(f"Mask preview error: {e}")
        return None, f"✗ Error: {e}"


def get_pending_info():
    """Get info about pending objects for harmonization."""
    global app_state
    if app_state is None or not app_state.pending_inpaintings:
        return "No pending objects"
    
    lines = []
    for i, p in enumerate(app_state.pending_inpaintings):
        bbox = p["bbox"]
        lines.append(f"{i+1}. {p['obj_type']} at ({bbox[0]},{bbox[1]})-({bbox[2]},{bbox[3]})")
    
    return f"{len(app_state.pending_inpaintings)} object(s):\n" + "\n".join(lines)


def clear_pending_objects():
    """Clear all pending inpaintings and start fresh."""
    global app_state
    if app_state is None:
        return None, "No state available"
    
    num_cleared = len(app_state.pending_inpaintings)
    app_state.pending_inpaintings = []
    app_state.combined_mask = None
    app_state.raw_inpainted_image = None
    app_state.harmonized_image = None
    app_state.inpaint_mask = None
    
    # Reset to accepted_image or current_image
    if app_state.accepted_image is not None:
        img = np.array(app_state.accepted_image)
    elif app_state.current_image is not None:
        img = np.array(app_state.current_image)
    else:
        img = None
    
    logger.info(f"Cleared {num_cleared} pending object(s)")
    return img, f"✓ Cleared {num_cleared} pending object(s) | Ready for new inpainting"


def accept_result(x1, y1, x2, y2, obj_type, seed):
    """Accept current inpainting result as the new base image and add annotation."""
    global app_state
    if app_state is None or app_state.inpainted_image is None:
        return None, None, None, "No result to accept", get_annotations_display(), "No objects accepted yet"
    
    bbox = (int(x1), int(y1), int(x2), int(y2))
    
    # Check for duplicate annotation
    if is_duplicate_annotation(bbox, obj_type):
        # Already accepted - just update the image without adding annotation
        app_state.accepted_image = app_state.inpainted_image.copy()
        accepted_img = np.array(app_state.accepted_image)
        logger.info(f"Already accepted: {obj_type} at {bbox}")
        return (
            accepted_img, accepted_img, accepted_img,
            f"⚠️ Already accepted: {obj_type}",
            get_annotations_display(),
            get_accepted_objects_display()
        )
    
    # Check if this was already accepted via harmonization
    if hasattr(app_state, 'current_object_accepted') and app_state.current_object_accepted:
        # Reset flag and skip adding annotation
        app_state.current_object_accepted = False
        app_state.accepted_image = app_state.inpainted_image.copy()
        accepted_img = np.array(app_state.accepted_image)
        logger.info(f"Object already accepted via harmonization: {obj_type}")
        return (
            accepted_img, accepted_img, accepted_img,
            f"⚠️ Already accepted via harmonization: {obj_type}",
            get_annotations_display(),
            get_accepted_objects_display()
        )
    
    # Save current inpainted result as accepted
    app_state.accepted_image = app_state.inpainted_image.copy()
    
    # Add annotation for this inpainting
    annotation = {
        "id": len(app_state.annotations) + 1,
        "bbox": list(bbox),
        "label": obj_type,
        "seed": int(seed) if seed >= 0 else None,
        "timestamp": datetime.now().isoformat()
    }
    app_state.annotations.append(annotation)
    logger.info(f"✓ Result accepted: {obj_type} at {bbox}")
    
    accepted_img = np.array(app_state.accepted_image)
    
    # Return: source_display, main_image, preview_image, status, annotations_tab3, accepted_objects_tab2
    return (
        accepted_img, 
        accepted_img, 
        accepted_img, 
        f"✓ Accepted: {obj_type} (#{len(app_state.annotations)})", 
        get_annotations_display(),
        get_accepted_objects_display()
    )


def get_annotations_display():
    """Get formatted display of all annotations."""
    global app_state
    if app_state is None or not app_state.annotations:
        return "No objects added yet"
    
    lines = []
    for ann in app_state.annotations:
        bbox = ann['bbox']
        lines.append(f"#{ann['id']}: {ann['label']} at ({bbox[0]},{bbox[1]})-({bbox[2]},{bbox[3]})")
    return "\n".join(lines)


def get_annotation_choices():
    """Get annotation choices for dropdown."""
    global app_state
    if app_state is None or not app_state.annotations:
        return []
    return [f"#{ann['id']}: {ann['label']}" for ann in app_state.annotations]


def draw_all_annotations(image):
    """Draw all annotations on image."""
    global app_state
    if app_state is None or image is None or not app_state.annotations:
        return image
    
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        img = image.copy()
    
    draw = ImageDraw.Draw(img)
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FFA500', '#800080']
    
    for i, ann in enumerate(app_state.annotations):
        bbox = ann['bbox']
        color = colors[i % len(colors)]
        # Draw rectangle
        draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=color, width=3)
        # Draw label
        label = f"#{ann['id']}: {ann['label']}"
        draw.text((bbox[0], bbox[1] - 15), label, fill=color)
    
    return np.array(img)


def get_annotation_details(selection):
    """Get details of selected annotation for editing."""
    global app_state
    if not selection or app_state is None or not app_state.annotations:
        return 0, 0, 0, 0, None, "", "Select an object, then click on image to redefine bbox"
    
    # Parse selection: "#1: label"
    try:
        ann_id = int(selection.split(":")[0].replace("#", ""))
        ann = next((a for a in app_state.annotations if a['id'] == ann_id), None)
        if ann:
            bbox = ann['bbox']
            bbox_text = f"({bbox[0]}, {bbox[1]}) → ({bbox[2]}, {bbox[3]})"
            return (bbox[0], bbox[1], bbox[2], bbox[3], ann['label'], bbox_text,
                    f"✓ Editing #{ann_id}: {ann['label']} - Click on image to change bbox")
    except Exception as e:
        logger.error(f"Get annotation error: {e}")
    return 0, 0, 0, 0, None, "", "Invalid selection"


def update_annotation(selection, x1, y1, x2, y2, label):
    """Update annotation with new bbox and label."""
    global app_state
    if not selection or app_state is None:
        return "No selection", get_annotations_display(), None
    
    try:
        ann_id = int(selection.split(":")[0].replace("#", ""))
        for ann in app_state.annotations:
            if ann['id'] == ann_id:
                ann['bbox'] = [int(x1), int(y1), int(x2), int(y2)]
                ann['label'] = label
                ann['updated'] = datetime.now().isoformat()
                logger.info(f"✓ Updated annotation #{ann_id}")
                
                # Draw updated annotations
                img_with_ann = draw_all_annotations(app_state.accepted_image or app_state.current_image)
                return f"✓ Updated #{ann_id}", get_annotations_display(), img_with_ann
    except Exception as e:
        logger.error(f"Update error: {e}")
    
    return "✗ Update failed", get_annotations_display(), None


def delete_annotation(selection):
    """Delete selected annotation."""
    global app_state
    if not selection or app_state is None:
        return "No selection", get_annotations_display(), gr.update(choices=get_annotation_choices(), value=None), None
    
    try:
        ann_id = int(selection.split(":")[0].replace("#", ""))
        app_state.annotations = [a for a in app_state.annotations if a['id'] != ann_id]
        logger.info(f"✓ Deleted annotation #{ann_id}")
        
        img_with_ann = draw_all_annotations(app_state.accepted_image or app_state.current_image)
        choices = get_annotation_choices()
        return f"✓ Deleted #{ann_id}", get_annotations_display(), gr.update(choices=choices, value=None), img_with_ann
    except Exception as e:
        logger.error(f"Delete error: {e}")
    
    return "✗ Delete failed", get_annotations_display(), gr.update(choices=get_annotation_choices(), value=None), None


def refresh_annotation_view():
    """Refresh annotation view with current image and annotations."""
    global app_state
    if app_state is None:
        return None, "No image", gr.update(choices=[], value=None), 0, 0, 0, 0, None, "", "No image"
    
    source = app_state.accepted_image or app_state.current_image
    if source is None:
        return None, "No image", gr.update(choices=[], value=None), 0, 0, 0, 0, None, "", "No image"
    
    img_with_ann = draw_all_annotations(source)
    choices = get_annotation_choices()
    
    # Reset click state for annotation editing
    app_state.ann_click_state = "idle"
    app_state.ann_first_click = None
    
    # Return all fields to reset the edit form
    return (
        img_with_ann, 
        get_annotations_display(), 
        gr.update(choices=choices, value=None),  # Reset dropdown
        0, 0, 0, 0,  # Reset bbox fields (hidden)
        None,  # Reset label dropdown
        "",  # Reset bbox display
        "Select an object, then click on image to edit bbox"
    )


def sam_annotation_click(image, evt: gr.SelectData, selection):
    """
    SAM-based bbox detection for annotation editing.
    
    Single click -> SAM segments object -> Updates bbox
    """
    global app_state
    if app_state is None or image is None:
        return 0, 0, 0, 0, image, "", "Load an image first"
    
    if not selection:
        return 0, 0, 0, 0, image, "", "⚠️ Select an object first, then click to detect its bbox with SAM"
    
    point = (evt.index[0], evt.index[1])
    
    try:
        import cv2
        
        source = app_state.accepted_image or app_state.current_image
        if source is None:
            return 0, 0, 0, 0, image, "", "No source image"
        
        # Run SAM segmentation
        logger.info(f"SAM annotation bbox at point: {point}")
        mask, score = app_state.sam_segmenter.segment_point(
            source,
            point=point,
            point_label=1,
            multimask_output=True,
            return_best=True,
            use_crop=True
        )
        
        # Get bbox from mask
        bbox = app_state.sam_segmenter.get_mask_bbox(mask, padding=3)
        
        if bbox is None:
            return 0, 0, 0, 0, image, "", "⚠️ No object detected - try clicking on the object"
        
        new_x1, new_y1, new_x2, new_y2 = bbox
        
        # Create visualization
        if isinstance(source, Image.Image):
            img = source.copy()
        else:
            img = Image.fromarray(source)
        
        img_arr = np.array(img)
        
        # Draw SAM mask overlay
        vis = app_state.sam_segmenter.visualize_mask(
            img_arr, mask, color=(0, 255, 0), alpha=0.3, show_contour=True
        )
        
        # Draw all existing annotations
        vis_pil = Image.fromarray(vis)
        draw = ImageDraw.Draw(vis_pil)
        colors = ['#FF0000', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FFA500', '#800080']
        for i, ann in enumerate(app_state.annotations):
            ann_bbox = ann['bbox']
            color = colors[i % len(colors)]
            draw.rectangle([ann_bbox[0], ann_bbox[1], ann_bbox[2], ann_bbox[3]], outline=color, width=2)
            draw.text((ann_bbox[0], ann_bbox[1] - 15), f"#{ann['id']}: {ann['label']}", fill=color)
        
        # Draw new SAM bbox in bright green
        draw.rectangle([new_x1, new_y1, new_x2, new_y2], outline='#00FF00', width=4)
        draw.text((new_x1, new_y1 - 20), f"SAM BBOX (score: {score:.2f})", fill='#00FF00')
        
        # Draw click point
        cv2.circle(np.array(vis_pil), point, 8, (255, 0, 0), -1)
        
        preview = np.array(vis_pil)
        bbox_text = f"({new_x1}, {new_y1}) → ({new_x2}, {new_y2})"
        
        logger.info(f"SAM annotation bbox: {bbox}, score={score:.3f}")
        
        return (
            new_x1, new_y1, new_x2, new_y2,
            preview,
            bbox_text,
            f"✓ SAM detected bbox | Score: {score:.2f} | Click 'Save Changes' to update"
        )
        
    except Exception as e:
        logger.error(f"SAM annotation error: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0, 0, 0, image, "", f"✗ Error: {e}"


def handle_annotation_click(image, evt: gr.SelectData, selection, x1, y1, x2, y2):
    """Handle click on annotation image to edit bbox."""
    global app_state
    if app_state is None or image is None:
        return x1, y1, x2, y2, image, "", "Load an image first"
    
    if not selection:
        return x1, y1, x2, y2, image, "", "⚠️ Select an object first, then click to edit its bbox"
    
    click_x, click_y = evt.index[0], evt.index[1]
    
    # Initialize annotation click state if needed
    if not hasattr(app_state, 'ann_click_state'):
        app_state.ann_click_state = "idle"
        app_state.ann_first_click = None
    
    if app_state.ann_click_state == "idle" or app_state.ann_click_state == "second_click":
        # First click - set top-left
        app_state.ann_first_click = (click_x, click_y)
        app_state.ann_click_state = "first_click"
        return x1, y1, x2, y2, image, "Selecting...", f"🖱️ First corner: ({click_x}, {click_y}) - Click for second corner"
    else:
        # Second click - set bottom-right
        fx, fy = app_state.ann_first_click
        new_x1, new_y1 = min(fx, click_x), min(fy, click_y)
        new_x2, new_y2 = max(fx, click_x), max(fy, click_y)
        
        app_state.ann_click_state = "second_click"
        
        # Draw preview with new bbox
        source = app_state.accepted_image or app_state.current_image
        if source:
            img = source.copy() if isinstance(source, Image.Image) else Image.fromarray(source)
            draw = ImageDraw.Draw(img)
            # Draw all existing annotations
            colors = ['#FF0000', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FFA500', '#800080']
            for i, ann in enumerate(app_state.annotations):
                bbox = ann['bbox']
                color = colors[i % len(colors)]
                draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=color, width=2)
                draw.text((bbox[0], bbox[1] - 15), f"#{ann['id']}: {ann['label']}", fill=color)
            # Draw new bbox in bright green (thicker)
            draw.rectangle([new_x1, new_y1, new_x2, new_y2], outline='#00FF00', width=4)
            draw.text((new_x1, new_y1 - 20), "NEW BBOX", fill='#00FF00')
            preview = np.array(img)
        else:
            preview = image
        
        bbox_text = f"({new_x1}, {new_y1}) → ({new_x2}, {new_y2})"
        return new_x1, new_y1, new_x2, new_y2, preview, bbox_text, f"✓ New bbox selected - Click 'Save Changes' to apply"


def reset_annotation_click():
    """Reset annotation click state."""
    global app_state
    if app_state:
        app_state.ann_click_state = "idle"
        app_state.ann_first_click = None
    return "Click state reset - ready for new selection"


def get_accepted_objects_display():
    """Get display text for accepted objects in Tab 2."""
    global app_state
    if app_state is None or not app_state.annotations:
        return "No objects accepted yet"
    
    lines = []
    for ann in app_state.annotations:
        bbox = ann['bbox']
        lines.append(f"✓ #{ann['id']}: {ann['label']} ({bbox[0]},{bbox[1]})-({bbox[2]},{bbox[3]})")
    return "\n".join(lines)


def reject_result():
    """Reject current inpainting result, revert to previous state.
    
    Workflow:
    - If there are pending inpaintings, remove the LAST one and revert to its previous state
    - If no pending inpaintings, revert to accepted_image or current_image
    """
    global app_state
    if app_state is None:
        return None, None, "Not initialized"
    
    # Check if we have pending inpaintings to undo
    if app_state.pending_inpaintings:
        # Remove the last pending inpainting
        removed = app_state.pending_inpaintings.pop()
        removed_obj_type = removed.get("obj_type", "unknown")
        previous_raw = removed.get("previous_raw_image")
        
        logger.info(f"Rejected: {removed_obj_type} at {removed.get('bbox')}")
        
        if previous_raw is not None:
            # Restore to the state BEFORE this inpainting was applied
            app_state.raw_inpainted_image = previous_raw.copy()
            app_state.inpainted_image = previous_raw.copy()
            display = np.array(previous_raw)
            
            # Update combined mask (recalculate from remaining pending)
            if app_state.pending_inpaintings:
                w, h = previous_raw.size
                combined = np.zeros((h, w), dtype=np.uint8)
                for p in app_state.pending_inpaintings:
                    combined = np.maximum(combined, p["mask"])
                app_state.combined_mask = Image.fromarray(combined)
                remaining = len(app_state.pending_inpaintings)
                status = f"✓ Rejected '{removed_obj_type}' | Reverted to previous state | {remaining} object(s) remaining"
            else:
                # No more pending - clear combined mask and raw_inpainted
                app_state.combined_mask = None
                app_state.raw_inpainted_image = None
                app_state.inpaint_source = None
                status = f"✓ Rejected '{removed_obj_type}' | Reverted to last accepted"
        else:
            # No previous_raw stored - fall back to accepted/original
            if app_state.accepted_image is not None:
                display = np.array(app_state.accepted_image)
                app_state.raw_inpainted_image = None
                app_state.inpainted_image = None
                status = f"✓ Rejected '{removed_obj_type}' | Reverted to last accepted"
            else:
                display = np.array(app_state.current_image) if app_state.current_image else None
                app_state.raw_inpainted_image = None
                app_state.inpainted_image = None
                status = f"✓ Rejected '{removed_obj_type}' | Reverted to original"
            
            # Clear combined mask since we're back to base
            if not app_state.pending_inpaintings:
                app_state.combined_mask = None
                app_state.inpaint_source = None
    else:
        # No pending inpaintings - just clear current inpainting
        app_state.inpainted_image = None
        app_state.raw_inpainted_image = None
        app_state.combined_mask = None
        app_state.inpaint_source = None
        
        # Show the accepted image (or original if nothing accepted yet)
        if app_state.accepted_image is not None:
            display = np.array(app_state.accepted_image)
            status = "✓ Reverted to last accepted result"
        elif app_state.current_image is not None:
            display = np.array(app_state.current_image)
            status = "✓ Reverted to original image"
        else:
            display = None
            status = "No image to revert to"
    
    logger.info(status)
    # Return: source_display, result_image, status, pending_info
    return display, None, status, get_pending_info()


def get_current_base_image():
    """Get the current base image for display."""
    global app_state
    if app_state is None:
        return None
    
    if app_state.accepted_image is not None:
        return np.array(app_state.accepted_image)
    elif app_state.current_image is not None:
        return np.array(app_state.current_image)
    return None


def get_weather_source():
    """Get the source image for weather effects.
    Only uses accepted_image or current_image - never uses non-accepted inpainted_image.
    """
    global app_state
    if app_state and app_state.accepted_image is not None:
        return app_state.accepted_image
    elif app_state and app_state.current_image is not None:
        return app_state.current_image
    return None


def apply_single_weather(weather_type: str, seed: int, steps: int, guidance: float, 
                         img_guidance: float = 1.5, progress=gr.Progress()):
    """Apply a single weather effect with specific seed."""
    global app_state
    source = get_weather_source()
    if source is None:
        # Return existing image if available instead of None
        existing = get_weather_image(weather_type)
        return existing, f"✗ No source image available"
    
    try:
        progress(0.3, desc=f"Applying {weather_type}...")
        prompt = WEATHER_PROMPTS.get(weather_type, f"transform to {weather_type}")
        
        # Pass seed to edit_scene which creates proper generator
        actual_seed = seed if seed >= 0 else None
        
        result = app_state.scene_editor.edit_scene(
            image=source, 
            prompt=prompt,
            num_inference_steps=steps, 
            guidance_scale=guidance,
            image_guidance_scale=img_guidance,
            seed=actual_seed  # Pass seed to pipeline's generator
        )
        app_state.weather_results[weather_type] = result
        progress(1.0)
        
        seed_info = f"seed: {seed}" if seed >= 0 else "random seed"
        return np.array(result), f"✓ {weather_type} ({seed_info}) | img_g={img_guidance}"
    except Exception as e:
        logger.error(f"Weather error for {weather_type}: {e}")
        app_state.clear_gpu_memory()
        # Return existing image on error instead of None
        existing = get_weather_image(weather_type)
        return existing, f"✗ Error: {e}"


def accept_weather(weather_type: str):
    """Accept a weather result."""
    global app_state
    if app_state is None:
        return f"✗ Not initialized"
    
    if weather_type in app_state.weather_results:
        app_state.accepted_weather[weather_type] = app_state.weather_results[weather_type].copy()
        return f"✓ {weather_type} accepted"
    return f"✗ No result for {weather_type}"


def reject_weather(weather_type: str):
    """Reject a weather result, revert to accepted if available."""
    global app_state
    if app_state is None:
        return None, f"✗ Not initialized"
    
    # Remove from working results
    if weather_type in app_state.weather_results:
        del app_state.weather_results[weather_type]
    
    # Show accepted version if available
    if weather_type in app_state.accepted_weather:
        return np.array(app_state.accepted_weather[weather_type]), f"✓ Reverted to accepted {weather_type}"
    
    return None, f"✓ {weather_type} rejected"


def get_weather_image(weather_type: str):
    """Get current weather image (working or accepted)."""
    global app_state
    if app_state is None:
        return None
    
    # Prefer working result, fallback to accepted
    if weather_type in app_state.weather_results:
        return np.array(app_state.weather_results[weather_type])
    elif weather_type in app_state.accepted_weather:
        return np.array(app_state.accepted_weather[weather_type])
    return None


def apply_weather(weathers, steps, guidance, progress=gr.Progress()):
    """Apply multiple weather effects at once (initial generation)."""
    global app_state
    source = get_weather_source()
    if source is None:
        return [None] * len(WEATHER_PROMPTS), "No image available"
    if not weathers:
        return [None] * len(WEATHER_PROMPTS), "Select at least one weather"
    
    try:
        for i, w in enumerate(weathers):
            progress((i + 0.5) / len(weathers), desc=f"Applying {w}...")
            prompt = WEATHER_PROMPTS.get(w, f"transform to {w}")
            
            result = app_state.scene_editor.edit_scene(
                image=source, prompt=prompt,
                num_inference_steps=steps, guidance_scale=guidance
            )
            app_state.weather_results[w] = result
            progress((i + 1) / len(weathers))
        
        # Return images in order of weather types
        result_images = []
        for w in WEATHER_PROMPTS.keys():
            if w in app_state.weather_results:
                result_images.append(np.array(app_state.weather_results[w]))
            else:
                result_images.append(None)
        
        return result_images + [f"✓ Applied {len(weathers)} effects"]
    except Exception as e:
        logger.error(f"Weather error: {e}")
        app_state.clear_gpu_memory()
        return [None] * len(WEATHER_PROMPTS) + [f"✗ Error: {e}"]


def save_results(save_type, output_dir, prefix):
    global app_state
    if app_state is None:
        return "Not initialized"
    
    try:
        out = Path(output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        
        if not prefix:
            prefix = app_state.current_image_path.stem if app_state.current_image_path else "result"
        
        saved_paths = []
        
        # Base metadata for JSON
        base_metadata = {
            # Store only a portable identifier; never expose the workstation
            # filesystem path in public metadata.
            "source_image": app_state.current_image_path.name if app_state.current_image_path else None,
            "timestamp": datetime.now().isoformat(),
            "annotations": app_state.annotations.copy() if app_state.annotations else []
        }
        
        # Save accepted image (final approved result with all objects)
        if save_type in ["inpainted", "all"]:
            img_to_save = app_state.accepted_image or app_state.inpainted_image
            if img_to_save:
                p = out / f"{prefix}_inpainted.png"
                img_to_save.save(p)
                saved_paths.append(str(p))
                
                # Save JSON for inpainted image
                json_data = {
                    **base_metadata,
                    "image_path": str(p),
                    "type": "inpainted"
                }
                json_path = out / f"{prefix}_inpainted.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                saved_paths.append(str(json_path))
        
        if save_type in ["weather", "all"]:
            # Save weather images with individual JSON files
            for style_name in WEATHER_PROMPTS.keys():
                img = app_state.accepted_weather.get(style_name) or app_state.weather_results.get(style_name)
                if img:
                    p = out / f"{prefix}_{style_name}.png"
                    img.save(p)
                    saved_paths.append(str(p))
                    
                    # Save JSON for each weather style
                    json_data = {
                        **base_metadata,
                        "image_path": str(p),
                        "type": "weather",
                        "style": style_name,
                        "style_prompt": WEATHER_PROMPTS.get(style_name, "")
                    }
                    json_path = out / f"{prefix}_{style_name}.json"
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, indent=2, ensure_ascii=False)
                    saved_paths.append(str(json_path))
        
        if not saved_paths:
            return "No results to save"
        
        # Format output with full paths
        img_count = len([p for p in saved_paths if p.endswith('.png')])
        json_count = len([p for p in saved_paths if p.endswith('.json')])
        result_text = f"✓ Saved {img_count} image(s) + {json_count} JSON(s):\n"
        for path in saved_paths:
            result_text += f"  • {path}\n"
        return result_text.strip()
    except Exception as e:
        logger.error(f"Save error: {e}")
        return f"✗ Error: {e}"


def create_interface():
    obj_types = get_all_object_keys()
    weather_types = list(WEATHER_PROMPTS.keys())
    
    # Get GUI defaults from config
    global app_state
    if app_state is not None:
        gui_defaults = app_state.gui_defaults
    else:
        # Fallback to direct config load
        gui_defaults = get_gui_defaults(load_config())
    
    default_image_folder = gui_defaults.get('image_folder', '')
    default_output_folder = gui_defaults.get('output_folder', '')
    default_weather_effects = gui_defaults.get('default_weather_effects', ["snow", "rain", "fog", "night", "dawn"])
    
    with gr.Blocks(title="GenRoad Framework", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎯 GenRoad: A Generative Framework for Synthesizing Anomalies in Autonomous Driving")
        gr.Markdown("**Workflow:** Load Image → Click to Select BBox → Inpaint → Harmonize → Annotate → Apply Weather/Scene Effects → Save Results")
        
        status = gr.Textbox(label="Status", value="Ready", interactive=False)
        
        # Note: All image components now have 
        
        with gr.Tabs():
            # Tab 1: Load & BBox
            with gr.TabItem("1️⃣ Load & BBox"):
                gr.Markdown("### 🖱️ Click twice on image: first for top-left corner, second for bottom-right corner")
                
                with gr.Row():
                    with gr.Column(scale=1, min_width=250):
                        folder_input = gr.Textbox(
                            label="Folder Path",
                            value=default_image_folder
                        )
                        load_btn = gr.Button("📂 Load Folder", variant="primary")
                        img_dropdown = gr.Dropdown(label="Select Image", choices=[], interactive=True)
                        
                        with gr.Row():
                            prev_btn = gr.Button("⬅️ Prev")
                            next_btn = gr.Button("Next ➡️")
                        
                        path_display = gr.Textbox(label="Path", interactive=False, max_lines=2)
                        
                        gr.Markdown("---")
                        gr.Markdown("### 📐 Selected BBox")
                        bbox_info_tab1 = gr.Textbox(label="Coordinates", value="Not selected", interactive=False, max_lines=1)
                        reset_bbox_btn = gr.Button("↩️ Reset Selection", variant="secondary", size="sm")
                        
                        # Hidden fields for bbox coordinates (still needed for inpainting)
                        bbox_x1 = gr.Number(value=100, visible=False)
                        bbox_y1 = gr.Number(value=100, visible=False)
                        bbox_x2 = gr.Number(value=300, visible=False)
                        bbox_y2 = gr.Number(value=300, visible=False)
                    
                    with gr.Column(scale=3):
                        main_image = gr.Image(
                            label="Click to Select BBox (2 clicks: top-left → bottom-right)", 
                            type="numpy", 
                            height=650, 
                            interactive=False
                        )
                
                # Hidden preview_image (needed for Tab 2 sync but not displayed)
                preview_image = gr.Image(visible=False, type="numpy")
            
            # Tab 2: Inpainting (Optimized Layout)
            with gr.TabItem("2️⃣ Inpainting"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=280):
                        # Quick Actions at TOP for fast workflow
                        obj_type = gr.Dropdown(choices=obj_types, value=obj_types[0] if obj_types else None, label="Object Type")
                        
                        with gr.Row():
                            inpaint_btn = gr.Button("🎨 Run Inpainting", variant="primary", size="lg")
                            retry_btn = gr.Button("🔄 Retry", size="lg")
                        
                        with gr.Row():
                            accept_btn = gr.Button("✅ Accept", variant="primary")
                            reject_btn = gr.Button("❌ Reject", variant="stop")
                        
                        gr.Markdown("---")
                        
                        # Compact previews
                        with gr.Row():
                            source_display = gr.Image(label="Base", type="numpy", height=120, interactive=False)
                            bbox_preview_tab2 = gr.Image(label="BBox", type="numpy", height=120, interactive=False)
                        
                        # Hidden bbox_info (still needed for logic but not displayed prominently)
                        bbox_info = gr.Textbox(value="Not set", visible=False)
                        
                        with gr.Accordion("⚙️ Settings", open=False):
                            seed_input = gr.Number(label="Seed (-1=random)", value=-1, precision=0)
                            steps = gr.Slider(10, 100, 50, step=5, label="Steps")
                            guidance = gr.Slider(1, 20, 7.5, step=0.5, label="Guidance")
                        
                        with gr.Accordion("📋 Accepted Objects", open=False):
                            accepted_objects_display = gr.Textbox(
                                label="", 
                                value="No objects accepted yet",
                                lines=3, 
                                interactive=False
                            )
                    
                    with gr.Column(scale=2):
                        result_image = gr.Image(label="Inpainting Result", type="numpy", height=380)
                        compare_image = gr.Image(label="Comparison (Base | Result)", type="numpy", height=180)
            
            # Tab 3: Harmonization with SAM (Optimized Layout)
            with gr.TabItem("3️⃣ Harmonization"):
                gr.Markdown("*Click object → Accept mask → Apply harmonization*", elem_classes=["info-text"])
                
                with gr.Row():
                    with gr.Column(scale=1, min_width=260):
                        # Quick Actions at TOP
                        with gr.Row():
                            sam_accept_btn = gr.Button("✅ Accept Mask", variant="primary")
                            sam_reject_btn = gr.Button("❌ Reject", variant="stop")
                        
                        apply_harm_btn = gr.Button("🎨 Apply Harmonization", variant="primary", size="lg")
                        
                        with gr.Row():
                            accept_harm_btn = gr.Button("✅ Accept All", variant="primary")
                            reject_harm_btn = gr.Button("❌ Reject", variant="stop")
                        
                        harm_status = gr.Textbox(label="", value="Click on object to segment", interactive=False, max_lines=1)
                        
                        gr.Markdown("---")
                        
                        # SAM Controls - Compact
                        sam_expand_mode = gr.Checkbox(
                            label="🔄 Expand Mode (multi-part objects)",
                            value=False
                        )
                        
                        with gr.Row():
                            sam_undo_btn = gr.Button("↩️ Undo", size="sm")
                            sam_clear_btn = gr.Button("🗑️ Clear", variant="stop", size="sm")
                        
                        sam_masks_info = gr.Textbox(
                            label="SAM Masks",
                            value="No masks",
                            interactive=False, lines=2, max_lines=3
                        )
                        
                        with gr.Accordion("⚙️ Settings", open=False):
                            harm_method = gr.Dropdown(
                                choices=["none", "color_transfer", "histogram_match", "poisson_blend", 
                                        "lab_adjust", "multi_band_blend", "combined"],
                                value="combined",
                                label="Method"
                            )
                            harm_mask_dilate = gr.Slider(0, 50, 5, step=1, label="Mask Dilation")
                            harm_feather = gr.Slider(0, 100, 30, step=5, label="Feather")
                            preview_mask_btn = gr.Button("👁️ Preview Mask", size="sm")
                        
                        # Hidden
                        harm_diff_threshold = gr.Slider(5, 50, 15, step=1, visible=False)
                        
                        with gr.Accordion("📦 Pending Objects", open=False):
                            pending_info = gr.Textbox(
                                label="", 
                                value="No pending objects", 
                                interactive=False, lines=2
                            )
                            clear_pending_btn = gr.Button("🗑️ Clear Inpaintings", variant="stop", size="sm")
                    
                    with gr.Column(scale=2):
                        sam_segment_image = gr.Image(
                            label="🖱️ Click on object (green = mask)", 
                            type="numpy", 
                            height=420,
                            interactive=False,
                            
                        )
                        
                        with gr.Row():
                            raw_inpaint_display = gr.Image(label="Raw", type="numpy", height=200)
                            harmonized_display = gr.Image(label="Harmonized", type="numpy", height=200)
                        
                        harm_mask_preview = gr.Image(label="Mask Preview", type="numpy", height=150, visible=True)
            
            # Tab 4: Annotations (Auto-refresh on tab select)
            with gr.TabItem("4️⃣ Annotations") as ann_tab:
                with gr.Row():
                    with gr.Column(scale=1, min_width=240):
                        # Objects list VISIBLE at top
                        annotations_display = gr.Textbox(
                            label="📋 Objects", 
                            value="No objects added yet",
                            lines=5, 
                            interactive=False
                        )
                        
                        ann_select = gr.Dropdown(label="Select Object", choices=[], interactive=True)
                        ann_bbox_display = gr.Textbox(label="BBox", value="", interactive=False, max_lines=1)
                        
                        with gr.Row():
                            update_ann_btn = gr.Button("💾 Save", variant="primary")
                            delete_ann_btn = gr.Button("🗑️ Delete", variant="stop")
                        
                        ann_status = gr.Textbox(label="", value="Select object, click to edit", interactive=False, max_lines=1)
                        
                        gr.Markdown("---")
                        
                        with gr.Row():
                            ann_sam_mode = gr.Checkbox(label="🎯 SAM", value=False)
                            reset_click_btn = gr.Button("↩️ Reset", size="sm")
                        
                        with gr.Accordion("✏️ Change Label", open=False):
                            ann_label = gr.Dropdown(choices=obj_types, label="New Label", interactive=True)
                        
                        refresh_ann_btn = gr.Button("🔄 Refresh View", variant="secondary", size="sm")
                        
                        # Hidden fields for bbox coordinates
                        ann_x1 = gr.Number(value=0, visible=False)
                        ann_y1 = gr.Number(value=0, visible=False)
                        ann_x2 = gr.Number(value=0, visible=False)
                        ann_y2 = gr.Number(value=0, visible=False)
                    
                    with gr.Column(scale=3):
                        ann_image = gr.Image(
                            label="2-click: top-left → bottom-right | SAM: single click", 
                            type="numpy", 
                            height=580, 
                            interactive=False,
                            
                        )
            
            # Tab 5: Weather
            with gr.TabItem("5️⃣ Weather"):
                gr.Markdown("### 🌦️ Weather & Style Effects")
                gr.Markdown("*Select effects to show their panels, then Generate. Accept/Reject each individually.*")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Default selection from config
                        weather_select = gr.CheckboxGroup(
                            choices=weather_types, 
                            value=default_weather_effects, 
                            label="Select Effects (panels will appear below)"
                        )
                        
                        with gr.Accordion("⚙️ Generation Settings", open=False):
                            w_steps = gr.Slider(10, 100, 30, step=5, label="Steps")
                            w_guidance = gr.Slider(1, 20, 7.5, step=0.5, label="Text Guidance")
                            w_img_guidance = gr.Slider(0.5, 3.0, 1.5, step=0.1, label="Image Guidance")
                        
                        weather_btn = gr.Button("🌦️ Generate Selected", variant="primary", size="lg")
                
                gr.Markdown("---")
                gr.Markdown("### Results")
                
                # Create individual controls for each weather type with visibility control
                weather_columns = {}
                weather_images = {}
                weather_seeds = {}
                weather_regen_btns = {}
                weather_accept_btns = {}
                weather_reject_btns = {}
                weather_statuses = {}
                
                # All weather types in a single row with dynamic visibility
                # Large preview area for selected image
                gr.Markdown("### 🔍 Click on any result to see full size preview")
                weather_preview = gr.Image(label="Full Size Preview", type="numpy", height=450, interactive=False)
                
                gr.Markdown("---")
                gr.Markdown("### Results")
                gr.Markdown("*Click 🔄 to regenerate with custom parameters*")
                
                # Storage for per-style parameters
                weather_steps = {}
                weather_guidance = {}
                weather_img_guidance = {}
                
                with gr.Row():
                    for w in weather_types:
                        # Show panels for default selected styles
                        is_visible = w in default_weather_effects
                        with gr.Column(min_width=220, visible=is_visible) as col:
                            gr.Markdown(f"**{w.upper()}**")
                            weather_images[w] = gr.Image(label=w, type="numpy", height=160, interactive=False)
                            
                            with gr.Accordion("⚙️ Parameters", open=False):
                                weather_steps[w] = gr.Slider(10, 100, 30, step=5, label="Steps")
                                weather_guidance[w] = gr.Slider(1, 20, 7.5, step=0.5, label="Guidance")
                                weather_img_guidance[w] = gr.Slider(0.5, 3.0, 1.5, step=0.1, label="Img Guidance")
                            
                            weather_seeds[w] = gr.Number(label="Seed", value=-1, precision=0)
                            with gr.Row():
                                weather_regen_btns[w] = gr.Button("🔄 Regen", size="sm")
                                weather_accept_btns[w] = gr.Button("✅", size="sm", variant="primary")
                                weather_reject_btns[w] = gr.Button("❌", size="sm")
                            weather_statuses[w] = gr.Textbox(label="", value="", interactive=False, max_lines=1)
                        weather_columns[w] = col
            
            # Tab 6: Save
            with gr.TabItem("6️⃣ Save"):
                output_dir = gr.Textbox(label="Output Dir", value=default_output_folder)
                prefix = gr.Textbox(label="Prefix (empty=auto)", value="")
                save_type = gr.Radio(["inpainted", "weather", "all"], value="all", label="Save")
                
                with gr.Row():
                    save_btn = gr.Button("💾 Save", variant="primary")
                    save_next_btn = gr.Button("💾 Save & Next")
                
                save_status = gr.Textbox(label="Status", interactive=False)
        
        # Events
        def on_load_folder(folder):
            choices, st, path = load_folder(folder)
            if choices:
                # Also load first image
                img, img_st, img_path = select_image(choices[0])
                # Clear weather statuses and images, set previews to new image
                return [gr.update(choices=choices, value=choices[0]), st, path, img, img, img, img, "Not set", "No objects accepted yet"] + \
                       [""] * len(weather_types) + [None] * len(weather_types)
            return [gr.update(choices=[]), st, path, None, None, None, None, "Not set", "No objects accepted yet"] + \
                   [""] * len(weather_types) + [None] * len(weather_types)
        
        load_folder_outputs = [img_dropdown, status, path_display, main_image, source_display,
                               preview_image, bbox_preview_tab2, bbox_info, accepted_objects_display] + \
                              [weather_statuses[w] for w in weather_types] + \
                              [weather_images[w] for w in weather_types]
        load_btn.click(on_load_folder, [folder_input], load_folder_outputs)
        
        def on_select(name):
            if not name:
                # Return None for image outputs + empty strings for weather statuses + cleared previews
                return [None, "Select an image", "", None, None, None, "Not set", "No objects accepted yet"] + \
                       [""] * len(weather_types) + [None] * len(weather_types)
            img, st, path = select_image(name)
            # Clear weather statuses and images when changing image
            # Also set preview_image and bbox_preview_tab2 to new image
            return [img, st, path, img, img, img, "Not set", "No objects accepted yet"] + \
                   [""] * len(weather_types) + [None] * len(weather_types)
        
        img_dropdown.change(
            on_select, 
            [img_dropdown], 
            [main_image, status, path_display, source_display, preview_image, bbox_preview_tab2, bbox_info, accepted_objects_display] + 
            [weather_statuses[w] for w in weather_types] + 
            [weather_images[w] for w in weather_types]
        )
        
        def on_prev():
            img, st, path, name = navigate("prev")
            # Clear weather statuses and images, reset previews to new image
            return [img, st, path, name, img, img, img, "Not set", "No objects accepted yet"] + \
                   [""] * len(weather_types) + [None] * len(weather_types)
        
        def on_next():
            img, st, path, name = navigate("next")
            # Clear weather statuses and images, reset previews to new image
            return [img, st, path, name, img, img, img, "Not set", "No objects accepted yet"] + \
                   [""] * len(weather_types) + [None] * len(weather_types)
        
        nav_outputs = [main_image, status, path_display, img_dropdown, source_display, 
                       preview_image, bbox_preview_tab2, bbox_info, accepted_objects_display] + \
                      [weather_statuses[w] for w in weather_types] + \
                      [weather_images[w] for w in weather_types]
        
        prev_btn.click(on_prev, outputs=nav_outputs)
        next_btn.click(on_next, outputs=nav_outputs)
        
        # Click on image for bbox - updates main_image with bbox drawn
        main_image.select(
            handle_image_click,
            [main_image, bbox_x1, bbox_y1, bbox_x2, bbox_y2],
            [bbox_x1, bbox_y1, bbox_x2, bbox_y2, status, bbox_info_tab1, main_image, bbox_preview_tab2]
        )
        
        # Reset bbox selection
        reset_bbox_btn.click(
            reset_bbox_selection,
            outputs=[main_image, bbox_info_tab1, status]
        )
        
        def retry():
            return random.randint(0, 2**32-1)
        
        # Accept: Save result as new base image and add annotation
        accept_btn.click(
            accept_result,
            inputs=[bbox_x1, bbox_y1, bbox_x2, bbox_y2, obj_type, seed_input],
            outputs=[source_display, main_image, preview_image, status, annotations_display, accepted_objects_display]
        )
        
        # Reject: Discard current result, revert to previous state
        reject_btn.click(
            reject_result,
            outputs=[source_display, result_image, status, pending_info]
        )
        
        # ================================================================
        # HARMONIZATION TAB HANDLERS (SAM 2 Integration)
        # ================================================================
        
        # Update raw inpainting display when inpainting completes
        def update_harm_displays():
            """Update harmonization tab displays after inpainting."""
            global app_state
            if app_state and app_state.raw_inpainted_image:
                pending = get_pending_info()
                # Also clear SAM state for new segmentation
                app_state.clear_sam_state()
                return (np.array(app_state.raw_inpainted_image),  # sam_segment_image
                        np.array(app_state.raw_inpainted_image),  # raw_inpaint_display
                        None,  # harmonized_display
                        None,  # harm_mask_preview
                        f"Click on objects to segment | {pending}",  # harm_status
                        pending,  # pending_info
                        "No masks - click on objects")  # sam_masks_info
            return None, None, None, None, "No inpainting result available", "No pending objects", "No masks"
        
        # Chain after inpainting to update harmonization tab
        inpaint_btn.click(
            run_inpainting,
            [bbox_x1, bbox_y1, bbox_x2, bbox_y2, obj_type, seed_input, steps, guidance],
            [result_image, compare_image, status]
        ).then(
            update_harm_displays,
            outputs=[sam_segment_image, raw_inpaint_display, harmonized_display, harm_mask_preview, 
                    harm_status, pending_info, sam_masks_info]
        )
        
        # Retry with new seed
        retry_btn.click(retry, outputs=[seed_input]).then(
            run_inpainting,
            [bbox_x1, bbox_y1, bbox_x2, bbox_y2, obj_type, seed_input, steps, guidance],
            [result_image, compare_image, status]
        ).then(
            update_harm_displays,
            outputs=[sam_segment_image, raw_inpaint_display, harmonized_display, harm_mask_preview, 
                    harm_status, pending_info, sam_masks_info]
        )
        
        # SAM Segmentation: Click on image to segment (supports expand mode)
        def handle_sam_click(evt: gr.SelectData, expand_mode: bool):
            """Route click to appropriate handler based on expand mode."""
            if expand_mode and app_state and app_state.sam_current_mask is not None:
                # Expand mode: add to current mask
                return sam_expand_with_click(evt)
            else:
                # Normal mode: new segmentation with BBox guidance
                return sam_segment_click(evt)
        
        sam_segment_image.select(
            handle_sam_click,
            inputs=[sam_expand_mode],
            outputs=[sam_segment_image, harm_status]
        )
        
        # SAM: Accept current mask
        sam_accept_btn.click(
            sam_accept_mask,
            outputs=[sam_segment_image, harm_status, sam_masks_info]
        )
        
        # SAM: Reject current mask
        sam_reject_btn.click(
            sam_reject_mask,
            outputs=[sam_segment_image, harm_status]
        )
        
        # SAM: Undo last mask
        sam_undo_btn.click(
            sam_remove_last_mask,
            outputs=[sam_segment_image, harm_status, sam_masks_info]
        )
        
        # SAM: Clear all masks
        sam_clear_btn.click(
            sam_clear_all_masks,
            outputs=[sam_segment_image, harm_status, sam_masks_info]
        )
        
        # Preview final mask button (SAM-based)
        preview_mask_btn.click(
            get_sam_mask_preview,
            inputs=[harm_mask_dilate, harm_feather],
            outputs=[harm_mask_preview, harm_status]
        )
        
        # Apply harmonization (SAM-based)
        apply_harm_btn.click(
            apply_harmonization_with_sam,
            inputs=[harm_method, harm_mask_dilate, harm_feather],
            outputs=[raw_inpaint_display, harmonized_display, harm_mask_preview, harm_status]
        )
        
        # Accept harmonization
        accept_harm_btn.click(
            accept_harmonization,
            outputs=[result_image, source_display, main_image, harm_status, 
                    annotations_display, accepted_objects_display, pending_info]
        )
        
        # Reject harmonization
        reject_harm_btn.click(
            reject_harmonization,
            outputs=[result_image, harm_status]
        )
        
        # Clear pending objects
        def clear_and_update():
            img, status = clear_pending_objects()
            # Also clear SAM state
            if app_state:
                app_state.clear_sam_state()
            return img, img, status, get_pending_info(), "No masks"
        
        clear_pending_btn.click(
            clear_and_update,
            outputs=[sam_segment_image, raw_inpaint_display, harm_status, pending_info, sam_masks_info]
        )
        
        # Auto-update mask preview when parameters change
        harm_mask_dilate.change(
            get_sam_mask_preview,
            inputs=[harm_mask_dilate, harm_feather],
            outputs=[harm_mask_preview, harm_status]
        )
        harm_feather.change(
            get_sam_mask_preview,
            inputs=[harm_mask_dilate, harm_feather],
            outputs=[harm_mask_preview, harm_status]
        )
        
        # Weather: Update panel visibility when selection changes
        def update_weather_visibility(selected):
            """Show/hide weather panels based on selection."""
            updates = []
            for w in weather_types:
                updates.append(gr.update(visible=(w in selected)))
            return updates
        
        weather_select.change(
            update_weather_visibility,
            [weather_select],
            [weather_columns[w] for w in weather_types]
        )
        
        # Weather: Generate selected effects (skip already accepted ones)
        def on_generate_weather(weathers, steps, guidance, img_guidance, progress=gr.Progress()):
            """Generate only NEW weather effects (skip already accepted ones)."""
            global app_state
            source = get_weather_source()
            if source is None:
                return [get_weather_image(w) for w in weather_types] + ["No image available"]
            if not weathers:
                return [get_weather_image(w) for w in weather_types] + ["Select at least one effect"]
            
            # Filter out already accepted weathers - only generate new ones
            to_generate = [w for w in weathers if w not in app_state.accepted_weather]
            
            if not to_generate:
                return [get_weather_image(w) for w in weather_types] + ["All selected effects already accepted"]
            
            try:
                for i, w in enumerate(to_generate):
                    progress((i + 0.5) / len(to_generate), desc=f"Generating {w}...")
                    prompt = WEATHER_PROMPTS.get(w, f"transform to {w}")
                    
                    result = app_state.scene_editor.edit_scene(
                        image=source, prompt=prompt,
                        num_inference_steps=int(steps), 
                        guidance_scale=float(guidance),
                        image_guidance_scale=float(img_guidance)
                    )
                    app_state.weather_results[w] = result
                    progress((i + 1) / len(to_generate))
                
                # Return all images (existing accepted + newly generated)
                result_images = [get_weather_image(w) for w in weather_types]
                
                skipped = len(weathers) - len(to_generate)
                if skipped > 0:
                    return result_images + [f"✓ Generated {len(to_generate)} new, skipped {skipped} | img_g={img_guidance}"]
                return result_images + [f"✓ Generated {len(to_generate)} effects | img_g={img_guidance}"]
            except Exception as e:
                logger.error(f"Weather error: {e}")
                app_state.clear_gpu_memory()
                return [get_weather_image(w) for w in weather_types] + [f"✗ Error: {e}"]
        
        weather_outputs = [weather_images[w] for w in weather_types] + [status]
        weather_btn.click(on_generate_weather, [weather_select, w_steps, w_guidance, w_img_guidance], weather_outputs)
        
        # Individual weather regenerate/accept/reject handlers
        for w in WEATHER_PROMPTS.keys():
            # Regenerate with new seed
            def make_regen_handler(weather_type):
                def handler(seed, steps, guidance, img_guidance):
                    # ALWAYS generate new random seed for regeneration
                    # This ensures different results each time regenerate is clicked
                    new_seed = random.randint(0, 2**32-1)
                    img, st = apply_single_weather(weather_type, new_seed, int(steps), 
                                                  float(guidance), float(img_guidance))
                    return img, new_seed, st
                return handler
            
            # Use per-style parameters for regeneration
            weather_regen_btns[w].click(
                make_regen_handler(w),
                [weather_seeds[w], weather_steps[w], weather_guidance[w], weather_img_guidance[w]],
                [weather_images[w], weather_seeds[w], weather_statuses[w]]
            )
            
            # Accept
            def make_accept_handler(weather_type):
                def handler():
                    return accept_weather(weather_type)
                return handler
            
            weather_accept_btns[w].click(
                make_accept_handler(w),
                outputs=[weather_statuses[w]]
            )
            
            # Reject
            def make_reject_handler(weather_type):
                def handler():
                    return reject_weather(weather_type)
                return handler
            
            weather_reject_btns[w].click(
                make_reject_handler(w),
                outputs=[weather_images[w], weather_statuses[w]]
            )
            
            # Click on image to show large preview
            def make_preview_handler(weather_type):
                def handler():
                    img = get_weather_image(weather_type)
                    if img is not None:
                        return img
                    return None
                return handler
            
            weather_images[w].select(
                make_preview_handler(w),
                outputs=[weather_preview]
            )
        
        save_btn.click(save_results, [save_type, output_dir, prefix], [save_status])
        
        def save_and_next(st, od, pr):
            save_st = save_results(st, od, pr)
            img, nav_st, path, name = navigate("next")
            # Clear weather statuses and images, reset previews
            return [save_st, img, nav_st, path, name, img, img, img, "Not set", "No objects accepted yet"] + \
                   [""] * len(weather_types) + [None] * len(weather_types)
        
        save_next_outputs = [save_status, main_image, status, path_display, img_dropdown, source_display,
                             preview_image, bbox_preview_tab2, bbox_info, accepted_objects_display] + \
                            [weather_statuses[w] for w in weather_types] + \
                            [weather_images[w] for w in weather_types]
        save_next_btn.click(
            save_and_next,
            [save_type, output_dir, prefix],
            save_next_outputs
        )
        
        # Annotation Tab Events
        refresh_ann_btn.click(
            refresh_annotation_view,
            outputs=[ann_image, annotations_display, ann_select, ann_x1, ann_y1, ann_x2, ann_y2, ann_label, ann_bbox_display, ann_status]
        )
        
        # Auto-refresh when Annotation tab is selected
        ann_tab.select(
            refresh_annotation_view,
            outputs=[ann_image, annotations_display, ann_select, ann_x1, ann_y1, ann_x2, ann_y2, ann_label, ann_bbox_display, ann_status]
        )
        
        ann_select.change(
            get_annotation_details,
            inputs=[ann_select],
            outputs=[ann_x1, ann_y1, ann_x2, ann_y2, ann_label, ann_bbox_display, ann_status]
        )
        
        update_ann_btn.click(
            update_annotation,
            inputs=[ann_select, ann_x1, ann_y1, ann_x2, ann_y2, ann_label],
            outputs=[ann_status, annotations_display, ann_image]
        )
        
        delete_ann_btn.click(
            delete_annotation,
            inputs=[ann_select],
            outputs=[ann_status, annotations_display, ann_select, ann_image]
        )
        
        # Click on annotation image to edit bbox (supports SAM mode)
        def handle_ann_click_with_mode(image, evt: gr.SelectData, selection, x1, y1, x2, y2, sam_mode: bool):
            """Route click to SAM or manual handler based on mode."""
            if sam_mode:
                return sam_annotation_click(image, evt, selection)
            else:
                return handle_annotation_click(image, evt, selection, x1, y1, x2, y2)
        
        ann_image.select(
            handle_ann_click_with_mode,
            inputs=[ann_image, ann_select, ann_x1, ann_y1, ann_x2, ann_y2, ann_sam_mode],
            outputs=[ann_x1, ann_y1, ann_x2, ann_y2, ann_image, ann_bbox_display, ann_status]
        )
        
        # Reset click selection
        reset_click_btn.click(
            reset_annotation_click,
            outputs=[ann_status]
        )
        
        demo.load(initialize_app, outputs=[status])
    
    return demo

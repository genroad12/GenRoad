"""
Stable Diffusion 2 Inpainting module for object insertion with crop-based approach.

Features:
- Crop-based inpainting for better quality
- Enhanced prompts with lighting consistency
- Optional post-processing harmonization
- CLIP-based validation with retry mechanism
- Object-specific mask sizing
"""

import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from typing import Union, Optional, List, Tuple, Dict
from dataclasses import dataclass
from diffusers import StableDiffusionInpaintPipeline

# Import harmonizer and object specs
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genroad.utils.object_specs import get_negative_prompt, ENHANCED_NEGATIVE_PROMPT


@dataclass
class InpaintingResult:
    """Result of inpainting with validation info."""
    image: Image.Image
    is_valid: bool
    confidence: float
    attempts: int
    object_type: Optional[str] = None
    validation_details: Optional[dict] = None


class StableDiffusionInpainter:
    """Wrapper for Stable Diffusion 2 Inpainting model using crop-based inpainting."""
    
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-inpainting",  # HuggingFace ID or local path
        device: str = "cuda",
        dtype: str = "float16",
        num_inference_steps: int = 50,
        guidance_scale: float = 18.0,  # Higher for better prompt adherence
        negative_prompt: Optional[str] = None,
        use_harmonizer: bool = True,
        harmonizer_method: str = "combined",
        harmonizer_strength: float = 0.7,
        harmonizer_mask_feather: int = 50,
        # Mask blurring for softer transitions
        mask_blur_radius: int = 5,
        mask_expand_pixels: int = 2,
        # Validation settings
        use_validator: bool = True,
        validation_threshold: float = 0.25,
        max_retries: int = 3,
        # Validation config maps
        validation_clip_thresholds: Optional[Dict[str, float]] = None,
        validation_diff_settings: Optional[Dict[str, Dict[str, float]]] = None,
        validation_diff_min_change_default: float = 0.15,
        validation_diff_min_mean_diff_default: float = 20.0,
    ):
        """
        Initialize the inpainting model.
        
        Args:
            model_id: Path to SD2-Inpainting model
            device: Computation device (cuda/cpu)
            dtype: Model dtype (float16/float32)
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale for prompt adherence
            negative_prompt: Custom negative prompt (uses enhanced default if None)
            use_harmonizer: Whether to apply post-inpainting harmonization
            harmonizer_method: Harmonization method (color_transfer, poisson_blend, combined, etc.)
            harmonizer_strength: Strength of harmonization (0-1)
            harmonizer_mask_feather: Mask feather radius for smooth harmonization transitions
            mask_blur_radius: Gaussian blur radius for mask feathering (0 = no blur)
            mask_expand_pixels: Pixels to expand mask before blurring (helps cover edges)
            use_validator: Whether to use CLIP validation
            validation_threshold: Confidence threshold for valid detection
            max_retries: Maximum retry attempts for failed validation
        """
        self.model_id = model_id
        self.device = device
        self.dtype = torch.float16 if dtype == "float16" else torch.float32
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        
        # Use enhanced negative prompt by default
        self.negative_prompt = negative_prompt or ENHANCED_NEGATIVE_PROMPT
        
        # Harmonizer settings
        self.use_harmonizer = use_harmonizer
        self.harmonizer_method = harmonizer_method
        self.harmonizer_strength = harmonizer_strength
        self.harmonizer_mask_feather = harmonizer_mask_feather
        self.harmonizer = None
        
        # Mask blurring settings
        self.mask_blur_radius = mask_blur_radius
        self.mask_expand_pixels = mask_expand_pixels
        
        # Validation settings
        self.use_validator = use_validator
        self.validation_threshold = validation_threshold
        self.max_retries = max_retries
        self.validation_clip_thresholds = validation_clip_thresholds or {}
        self.validation_diff_settings = validation_diff_settings or {}
        self.validation_diff_min_change_default = validation_diff_min_change_default
        self.validation_diff_min_mean_diff_default = validation_diff_min_mean_diff_default
        self.validator = None
        
        self.pipeline = None
        self._loaded = False
        
        # Store last crop info for SAM-based paste (prevents bbox shadow)
        self._last_crop_info = None
    
    def load(self) -> None:
        """Load the inpainting pipeline."""
        if self._loaded:
            return
        
        print(f"Loading inpainting model: {self.model_id}")
        
        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)
        
        if self.device == "cuda":
            self.pipeline.enable_attention_slicing()
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        
        # Load harmonizer if enabled
        if self.use_harmonizer:
            from genroad.models.harmonizer import ImageHarmonizer
            self.harmonizer = ImageHarmonizer(
                method=self.harmonizer_method,
                blend_strength=self.harmonizer_strength,
                mask_feather_radius=self.harmonizer_mask_feather,
                use_difference_mask=True,  # Use difference-based mask for natural boundaries
            )
            print(f"  Harmonizer enabled: method={self.harmonizer_method}, strength={self.harmonizer_strength}, feather={self.harmonizer_mask_feather}px, diff_mask=True")
        
        # Load validator if enabled
        if self.use_validator:
            try:
                from genroad.models.inpainting_validator import InpaintingValidator
                self.validator = InpaintingValidator(
                    device=self.device,
                    threshold=self.validation_threshold,
                    clip_thresholds=self.validation_clip_thresholds,
                    diff_settings=self.validation_diff_settings,
                    diff_min_change_default=self.validation_diff_min_change_default,
                    diff_min_mean_diff_default=self.validation_diff_min_mean_diff_default,
                )
                if self.validator.load():
                    print(f"  Validator enabled: threshold={self.validation_threshold}, max_retries={self.max_retries}")
                else:
                    print("  Validator could not load CLIP, will skip validation")
                    self.validator = None
            except Exception as e:
                print(f"  Validator disabled: {e}")
                self.validator = None
        
        self._loaded = True
        print("Inpainting model loaded successfully.")
    
    def unload(self) -> None:
        """Unload the pipeline to free memory."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        self.harmonizer = None
        
        if self.validator is not None:
            self.validator.unload()
            self.validator = None
        
        self._loaded = False
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
    
    def _enhance_prompt(self, prompt: str) -> str:
        """
        Enhance prompt for better visibility and lighting consistency.
        """
        # Lighting and quality enhancements
        enhancements = (
            "highly detailed, realistic natural lighting matching the scene, "
            "proper shadow direction consistent with sun position, "
            "photorealistic, sharp focus, professional photography, 4k quality"
        )
        return f"{prompt}, {enhancements}"
    
    def _feather_mask(
        self,
        mask: np.ndarray,
        blur_radius: Optional[int] = None,
        expand_pixels: Optional[int] = None,
    ) -> np.ndarray:
        """
        Apply feathering/blurring to mask for softer transitions.
        
        This creates a gradual transition at the mask edges, which helps
        the inpainted region blend more naturally with the surroundings.
        
        Args:
            mask: Binary mask (0 or 255)
            blur_radius: Gaussian blur radius (uses instance default if None)
            expand_pixels: Pixels to expand mask before blurring
            
        Returns:
            Feathered mask with gradual edge transitions
        """
        blur_radius = blur_radius if blur_radius is not None else self.mask_blur_radius
        expand_pixels = expand_pixels if expand_pixels is not None else self.mask_expand_pixels
        
        if blur_radius <= 0:
            return mask
        
        feathered = mask.copy().astype(np.float32)
        
        # Step 1: Optionally expand the mask slightly to cover any edge artifacts
        if expand_pixels > 0:
            kernel = np.ones((expand_pixels * 2 + 1, expand_pixels * 2 + 1), np.uint8)
            feathered = cv2.dilate(feathered, kernel, iterations=1)
        
        # Step 2: Apply Gaussian blur for soft edges
        # Ensure kernel size is odd
        kernel_size = blur_radius * 2 + 1
        feathered = cv2.GaussianBlur(feathered, (kernel_size, kernel_size), 0)
        
        # Step 3: Normalize to 0-255 range
        feathered = np.clip(feathered, 0, 255).astype(np.uint8)
        
        return feathered
    
    def _create_crop_blend_mask(
        self,
        crop_shape: Tuple[int, int],
        object_mask: np.ndarray,
        crop_box: Tuple[int, int, int, int],
        full_shape: Tuple[int, int],
        edge_fade_pixels: int = 30,
    ) -> np.ndarray:
        """
        Create a blend mask for the crop region that fades at edges.
        
        This mask ensures:
        1. The object area is fully blended (white)
        2. The crop edges fade to 0 (seamless transition with original)
        
        Args:
            crop_shape: (height, width) of crop
            object_mask: Binary mask of the object within the crop
            crop_box: (x1, y1, x2, y2) of the crop in full image
            full_shape: (height, width) of full image
            edge_fade_pixels: How many pixels to fade at crop edges
            
        Returns:
            Blend mask for the crop region
        """
        crop_h, crop_w = crop_shape
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
        full_h, full_w = full_shape
        
        # Start with the object mask expanded a bit
        blend_mask = object_mask.copy().astype(np.float32)
        
        # Expand object mask to cover more area
        if self.mask_expand_pixels > 0:
            kernel = np.ones((self.mask_expand_pixels * 4 + 1, self.mask_expand_pixels * 4 + 1), np.uint8)
            blend_mask = cv2.dilate(blend_mask, kernel, iterations=2)
        
        # Apply strong blur to create gradient
        blur_size = edge_fade_pixels * 2 + 1
        blend_mask = cv2.GaussianBlur(blend_mask, (blur_size, blur_size), 0)
        
        # Create edge fade gradient for crop boundaries
        edge_mask = np.ones((crop_h, crop_w), dtype=np.float32)
        
        # Fade at edges that are not image boundaries
        for i in range(edge_fade_pixels):
            fade = i / edge_fade_pixels
            # Top edge (if not at image top)
            if crop_y1 > 0 and i < crop_h:
                edge_mask[i, :] = min(edge_mask[i, :].min(), fade)
            # Bottom edge (if not at image bottom)
            if crop_y2 < full_h and crop_h - 1 - i >= 0:
                edge_mask[crop_h - 1 - i, :] = min(edge_mask[crop_h - 1 - i, :].min(), fade)
            # Left edge (if not at image left)
            if crop_x1 > 0 and i < crop_w:
                edge_mask[:, i] = np.minimum(edge_mask[:, i], fade)
            # Right edge (if not at image right)
            if crop_x2 < full_w and crop_w - 1 - i >= 0:
                edge_mask[:, crop_w - 1 - i] = np.minimum(edge_mask[:, crop_w - 1 - i], fade)
        
        # Blur the edge mask for smoother transition
        edge_mask = cv2.GaussianBlur(edge_mask, (21, 21), 0)
        
        # Combine: use object mask but respect edge fading
        # The blend mask is the maximum of object region and edge-faded crop
        combined = np.maximum(blend_mask / 255.0, 0) * edge_mask
        
        # Ensure object area is preserved
        object_area = (object_mask > 127).astype(np.float32)
        object_area_dilated = cv2.dilate(object_area, np.ones((5, 5), np.uint8), iterations=1)
        combined = np.maximum(combined, object_area_dilated * edge_mask)
        
        return (combined * 255).astype(np.uint8)
    
    def _create_soft_composite(
        self,
        original: Image.Image,
        inpainted: Image.Image,
        mask: np.ndarray,
        blur_radius: int = 5,
    ) -> Image.Image:
        """
        Create a soft composite of original and inpainted images using feathered mask.
        
        This blends the inpainted region with the original using the feathered mask
        as an alpha channel, creating seamless transitions.
        
        Args:
            original: Original image
            inpainted: Inpainted image
            mask: Binary mask (will be feathered)
            blur_radius: Blur radius for feathering
            
        Returns:
            Composited image with soft transitions
        """
        # Feather the mask with larger radius for smoother blend
        feathered_mask = self._feather_mask(mask, blur_radius=blur_radius * 2, expand_pixels=blur_radius)
        
        # Convert to float for blending
        orig_array = np.array(original).astype(np.float32)
        inp_array = np.array(inpainted).astype(np.float32)
        
        # Normalize mask to 0-1
        alpha = feathered_mask.astype(np.float32) / 255.0
        
        # Expand alpha to 3 channels
        if len(alpha.shape) == 2:
            alpha = np.stack([alpha] * 3, axis=-1)
        
        # Blend: result = inpainted * alpha + original * (1 - alpha)
        blended = inp_array * alpha + orig_array * (1 - alpha)
        
        return Image.fromarray(blended.astype(np.uint8))
    
    @torch.no_grad()
    def inpaint(
        self,
        image: Union[Image.Image, np.ndarray, str],
        mask: Union[Image.Image, np.ndarray],
        prompt: str,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        enhance_prompt: bool = True,
        crop_padding: int = 100,  # Reduced from 150 for larger mask ratio
        target_size: int = 512,
        apply_harmonization: Optional[bool] = None,
        mask_blur_radius: Optional[int] = None,
        use_soft_composite: bool = True,
        debug_dir: Optional[str] = None,  # Directory to save debug masks
        min_crop_size: int = 256,  # Reduced from 300 for larger mask ratio in SD
        min_mask_ratio: float = 0.12,  # Increased from 0.05 for better object visibility
        use_dynamic_padding: bool = True,  # Use dynamic padding based on object size
        small_object_padding_multiplier: float = 1.5,  # Reduced from 2.0 for better mask ratio
    ) -> Image.Image:
        """
        Perform inpainting using crop-based approach for better quality.
        
        Args:
            image: Input image
            mask: Binary mask (white = area to inpaint)
            prompt: Text prompt describing what to generate
            bbox: Bounding box (x1, y1, x2, y2) for cropping. If None, derived from mask.
            negative_prompt: What to avoid (uses enhanced default if None)
            num_inference_steps: Override default steps
            guidance_scale: Override default guidance
            seed: Random seed
            enhance_prompt: Whether to enhance the prompt
            crop_padding: Padding around bbox for context
            target_size: Target size for SD processing (512 recommended)
            apply_harmonization: Override instance setting for harmonization
            mask_blur_radius: Override default mask blur radius for soft edges
            use_soft_composite: Use feathered mask for final compositing
            debug_dir: Directory to save debug mask visualizations (None = don't save)
            min_crop_size: Minimum crop size in pixels (ensures adequate context)
            min_mask_ratio: Minimum mask area ratio in 512x512 image for good SD output
            
        Returns:
            Inpainted (and optionally harmonized) image at original size
        """
        if not self._loaded:
            self.load()
        
        # Load image if path
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Keep reference to original for harmonization
        original_image = image.copy()
        original_size = image.size
        
        # Convert mask to numpy if PIL
        if isinstance(mask, Image.Image):
            mask_array = np.array(mask)
        else:
            mask_array = mask.copy()
        
        # Ensure binary mask
        mask_array = (mask_array > 127).astype(np.uint8) * 255
        
        # Get bbox from mask if not provided
        if bbox is None:
            rows = np.any(mask_array > 0, axis=1)
            cols = np.any(mask_array > 0, axis=0)
            if not rows.any() or not cols.any():
                print("  [Inpainter] WARNING: Empty mask, returning original image")
                return image
            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]
            bbox = (x1, y1, x2 + 1, y2 + 1)
        
        # Get bbox dimensions and mask size
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]
        bbox_center_x = (bbox[0] + bbox[2]) // 2
        bbox_center_y = (bbox[1] + bbox[3]) // 2
        
        # Dynamic padding: use more padding for smaller objects
        # Small objects need more context for SD to understand the scene
        effective_padding = crop_padding
        if use_dynamic_padding:
            bbox_size = max(bbox_w, bbox_h)
            if bbox_size < 100:
                # Very small object: use maximum padding multiplier
                effective_padding = int(crop_padding * small_object_padding_multiplier)
            elif bbox_size < 150:
                # Small object: use scaled padding
                scale = 1.0 + (150 - bbox_size) / 100 * (small_object_padding_multiplier - 1)
                effective_padding = int(crop_padding * scale)
            print(f"  [Inpainter] Dynamic padding: bbox_size={bbox_size}, padding={effective_padding}")
        
        # Calculate minimum padding needed to achieve min_mask_ratio
        # mask_area / crop_area = mask_area / (crop_w * crop_h) >= min_mask_ratio
        # For square crops resized to 512x512:
        # If bbox is 40x60 = 2400 pixels, and we want 5% of 512x512 = 13107 pixels
        # Scale factor = sqrt(13107 / 2400) = 2.34x
        mask_pixels = np.sum(mask_array > 127)
        target_mask_pixels = int(target_size * target_size * min_mask_ratio)
        
        # Calculate ideal crop size based on mask ratio
        # mask_in_sd / (512*512) = min_mask_ratio
        # mask_in_crop / (crop_size^2) * (512/crop_size)^2 = min_mask_ratio
        # => crop_size = sqrt(mask_in_crop / min_mask_ratio)
        ideal_crop_size = int(np.sqrt(mask_pixels / min_mask_ratio)) if mask_pixels > 0 else min_crop_size
        
        # Use the larger of: bbox with padding, min_crop_size, or ideal size for mask ratio
        effective_crop_size = max(
            max(bbox_w, bbox_h) + 2 * effective_padding,  # Dynamic padding
            min_crop_size,  # Minimum crop size
            ideal_crop_size,  # Ideal size for mask ratio
        )
        
        # Calculate dynamic padding based on effective crop size
        dynamic_padding_w = (effective_crop_size - bbox_w) // 2
        dynamic_padding_h = (effective_crop_size - bbox_h) // 2
        
        # Calculate crop region with dynamic padding
        crop_x1 = max(0, bbox_center_x - effective_crop_size // 2)
        crop_y1 = max(0, bbox_center_y - effective_crop_size // 2)
        crop_x2 = min(original_size[0], crop_x1 + effective_crop_size)
        crop_y2 = min(original_size[1], crop_y1 + effective_crop_size)
        
        # Adjust if we hit image boundaries
        if crop_x2 - crop_x1 < effective_crop_size:
            crop_x1 = max(0, crop_x2 - effective_crop_size)
        if crop_y2 - crop_y1 < effective_crop_size:
            crop_y1 = max(0, crop_y2 - effective_crop_size)
        
        crop_box = (crop_x1, crop_y1, crop_x2, crop_y2)
        crop_width = crop_x2 - crop_x1
        crop_height = crop_y2 - crop_y1
        
        # Estimate mask percentage after resize
        mask_in_crop = mask_pixels  # Assuming mask is fully within crop
        expected_mask_percent = mask_in_crop / (crop_width * crop_height) * 100 if crop_width * crop_height > 0 else 0
        
        print(f"  [Inpainter] Bbox: {bbox} ({bbox_w}x{bbox_h}), Crop: {crop_box} ({crop_width}x{crop_height})")
        print(f"  [Inpainter] Mask pixels: {mask_pixels}, expected in SD: ~{expected_mask_percent:.1f}%")
        
        # Crop image and mask
        cropped_image = image.crop(crop_box)
        cropped_mask = mask_array[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # Resize to target size for SD
        cropped_resized = cropped_image.resize((target_size, target_size), Image.LANCZOS)
        
        # Check if mask is too small after resize - expand it if needed
        # Calculate what the mask percentage will be after resize
        cropped_mask_pixels = np.sum(cropped_mask > 127)
        resized_mask_ratio = cropped_mask_pixels / (crop_width * crop_height) if crop_width * crop_height > 0 else 0
        min_pixels_needed = int(target_size * target_size * min_mask_ratio)
        expected_pixels_after_resize = int(resized_mask_ratio * target_size * target_size)
        
        # If mask is too small, dilate it to meet minimum requirements
        if expected_pixels_after_resize < min_pixels_needed:
            # Calculate dilation needed
            current_mask_size = np.sqrt(cropped_mask_pixels)
            target_mask_size = np.sqrt(min_pixels_needed * (crop_width * crop_height) / (target_size * target_size))
            dilation_needed = int((target_mask_size - current_mask_size) / 2) + 1
            dilation_needed = max(3, min(dilation_needed, 30))  # Limit dilation
            
            print(f"  [Inpainter] Mask too small ({expected_pixels_after_resize}px < {min_pixels_needed}px), dilating by {dilation_needed}px")
            
            # Dilate the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_needed * 2 + 1, dilation_needed * 2 + 1))
            cropped_mask = cv2.dilate(cropped_mask, kernel, iterations=1)
            
            # Update counts
            cropped_mask_pixels = np.sum(cropped_mask > 127)
            expected_pixels_after_resize = int(cropped_mask_pixels / (crop_width * crop_height) * target_size * target_size)
            print(f"  [Inpainter] After dilation: ~{expected_pixels_after_resize}px expected")
        
        # Apply mask feathering for softer transitions
        blur_radius = mask_blur_radius if mask_blur_radius is not None else self.mask_blur_radius
        if blur_radius > 0:
            feathered_cropped_mask = self._feather_mask(cropped_mask, blur_radius=blur_radius)
            print(f"  [Inpainter] Mask blur radius: {blur_radius}px")
        else:
            feathered_cropped_mask = cropped_mask
        
        # Resize mask - use LANCZOS for feathered mask to preserve gradients
        mask_pil = Image.fromarray(feathered_cropped_mask, mode="L")
        if blur_radius > 0:
            mask_resized = mask_pil.resize((target_size, target_size), Image.LANCZOS)
        else:
            mask_resized = mask_pil.resize((target_size, target_size), Image.NEAREST)
        
        # Verify mask
        mask_np = np.array(mask_resized)
        white_pixels = (mask_np > 127).sum()  # Count pixels above threshold for feathered mask
        total_pixels = target_size * target_size
        print(f"  [Inpainter] Mask: {white_pixels} active pixels ({100*white_pixels/total_pixels:.1f}%)")
        
        if white_pixels == 0:
            print("  [Inpainter] WARNING: No white pixels in resized mask!")
            return image
        
        # Save debug masks if debug_dir is provided
        if debug_dir is not None:
            import os
            os.makedirs(debug_dir, exist_ok=True)
            
            # 1. Original binary mask (full size)
            debug_mask_original = Image.fromarray(mask_array, mode="L")
            debug_mask_original.save(os.path.join(debug_dir, "01_mask_original_binary.png"))
            
            # 2. Cropped binary mask
            debug_cropped_mask = Image.fromarray(cropped_mask, mode="L")
            debug_cropped_mask.save(os.path.join(debug_dir, "02_mask_cropped_binary.png"))
            
            # 3. Feathered/blurred mask (cropped)
            debug_feathered = Image.fromarray(feathered_cropped_mask, mode="L")
            debug_feathered.save(os.path.join(debug_dir, "03_mask_feathered.png"))
            
            # 4. Resized mask for SD (512x512)
            mask_resized.save(os.path.join(debug_dir, "04_mask_resized_for_sd.png"))
            
            # 5. Cropped image region
            cropped_image.save(os.path.join(debug_dir, "05_crop_input.png"))
            
            # 6. Resized crop for SD
            cropped_resized.save(os.path.join(debug_dir, "06_crop_resized_for_sd.png"))
            
            # 7. Mask overlay on cropped image
            crop_array = np.array(cropped_image)
            mask_colored = np.zeros_like(crop_array)
            mask_colored[:, :, 0] = feathered_cropped_mask  # Red channel
            overlay = (crop_array * 0.6 + mask_colored * 0.4).astype(np.uint8)
            Image.fromarray(overlay).save(os.path.join(debug_dir, "07_mask_overlay_on_crop.png"))
            
            print(f"  [Inpainter] Debug masks saved to: {debug_dir}")
        
        # Enhance prompt
        if enhance_prompt:
            prompt = self._enhance_prompt(prompt)
        
        print(f"  [Inpainter] Prompt: {prompt[:80]}...")
        
        # Set parameters
        steps = num_inference_steps or self.num_inference_steps
        guidance = guidance_scale or self.guidance_scale
        neg_prompt = negative_prompt or self.negative_prompt
        
        # Set seed
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Run inpainting
        result = self.pipeline(
            prompt=prompt,
            negative_prompt=neg_prompt,
            image=cropped_resized,
            mask_image=mask_resized,
            height=target_size,
            width=target_size,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        ).images[0]
        
        # Resize result back to crop size
        result_resized = result.resize((crop_width, crop_height), Image.LANCZOS)
        
        # Composite result back into original image
        final_image = image.copy()
        
        # Store crop info for SAM-based paste later (prevents bbox shadow effect)
        self._last_crop_info = {
            "crop_box": crop_box,  # (x1, y1, x2, y2)
            "inpainted_crop": result_resized.copy(),  # Cropped inpainting result
            "original_crop": cropped_image.copy(),  # Original cropped region
            "crop_mask": cropped_mask.copy(),  # Mask within crop
        }
        
        # Save inpainting result debug images
        if debug_dir is not None:
            import os
            # 8. Raw SD output (512x512)
            result.save(os.path.join(debug_dir, "08_sd_output_raw.png"))
            # 9. Resized SD output
            result_resized.save(os.path.join(debug_dir, "09_sd_output_resized.png"))
        
        if use_soft_composite and blur_radius > 0:
            # Create a blend mask that covers the crop area with faded edges
            # This prevents the rectangular crop artifact
            crop_blend_mask = self._create_crop_blend_mask(
                crop_shape=(crop_height, crop_width),
                object_mask=cropped_mask,  # Original binary mask for the crop
                crop_box=crop_box,
                full_shape=(original_size[1], original_size[0]),  # (h, w)
                edge_fade_pixels=max(30, blur_radius * 3),
            )
            
            # Save crop blend mask debug
            if debug_dir is not None:
                import os
                # 10. Crop blend mask
                Image.fromarray(crop_blend_mask, mode="L").save(
                    os.path.join(debug_dir, "10_crop_blend_mask.png")
                )
                # 11. Blend mask overlay on result
                result_array = np.array(result_resized)
                blend_colored = np.zeros_like(result_array)
                blend_colored[:, :, 1] = crop_blend_mask  # Green channel
                overlay = (result_array * 0.6 + blend_colored * 0.4).astype(np.uint8)
                Image.fromarray(overlay).save(
                    os.path.join(debug_dir, "11_blend_mask_overlay.png")
                )
            
            # Blend the crop result with original crop using the blend mask
            orig_crop = np.array(cropped_image).astype(np.float32)
            result_crop = np.array(result_resized).astype(np.float32)
            
            # Normalize mask to 0-1
            alpha = crop_blend_mask.astype(np.float32) / 255.0
            alpha = np.stack([alpha] * 3, axis=-1)
            
            # Blend within crop region
            blended_crop = result_crop * alpha + orig_crop * (1 - alpha)
            blended_crop_img = Image.fromarray(blended_crop.astype(np.uint8))
            
            # Save blended crop debug
            if debug_dir is not None:
                import os
                # 12. Blended crop result
                blended_crop_img.save(os.path.join(debug_dir, "12_blended_crop.png"))
            
            # Paste the blended crop back
            final_image.paste(blended_crop_img, (crop_x1, crop_y1))
            print(f"  [Inpainter] Soft composite with crop edge blending applied")
        else:
            # Standard hard paste
            final_image.paste(result_resized, (crop_x1, crop_y1))
        
        # Save final result before harmonization
        if debug_dir is not None:
            import os
            final_image.save(os.path.join(debug_dir, "13_final_before_harmonization.png"))
        
        # Apply harmonization if enabled
        should_harmonize = apply_harmonization if apply_harmonization is not None else self.use_harmonizer
        
        if should_harmonize and self.harmonizer is not None:
            print(f"  [Inpainter] Applying harmonization ({self.harmonizer_method}) with difference mask...")
            harmonized = self.harmonizer.harmonize(
                inpainted_image=final_image,
                original_image=original_image,
                mask=mask_array,
                bbox=bbox,
                debug_dir=debug_dir,  # Pass debug directory for mask visualization
            )
            final_image = harmonized.image
            print(f"  [Inpainter] Harmonization complete (diff_mask={harmonized.adjustments.get('difference_mask', False)})")
        
        return final_image
    
    def inpaint_with_validation(
        self,
        image: Union[Image.Image, np.ndarray, str],
        mask: Union[Image.Image, np.ndarray],
        prompt: str,
        object_type: str,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        seed: Optional[int] = None,
        max_retries: Optional[int] = None,
        mask_scale_on_retry: float = 1.2,
        **kwargs,
    ) -> InpaintingResult:
        """
        Perform inpainting with CLIP-based validation and retry mechanism.
        
        If the initial inpainting fails validation, retries with:
        - Different random seed
        - Scaled mask size (larger by mask_scale_on_retry factor)
        
        Args:
            image: Input image
            mask: Binary mask
            prompt: Text prompt
            object_type: Type of object for validation
            bbox: Bounding box for object
            seed: Initial random seed
            max_retries: Override default max retries
            mask_scale_on_retry: Factor to scale mask on retry (>1 = larger)
            **kwargs: Additional arguments for inpaint()
            
        Returns:
            InpaintingResult with validation info
        """
        if not self._loaded:
            self.load()
        
        max_attempts = max_retries if max_retries is not None else self.max_retries
        
        # Convert image for validation reference
        if isinstance(image, str):
            original_image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            original_image = Image.fromarray(image)
        else:
            original_image = image.copy()
        
        # Convert mask to numpy
        if isinstance(mask, Image.Image):
            mask_array = np.array(mask)
        else:
            mask_array = mask.copy()
        
        # Get bbox from mask if not provided
        if bbox is None:
            rows = np.any(mask_array > 0, axis=1)
            cols = np.any(mask_array > 0, axis=0)
            if rows.any() and cols.any():
                y1, y2 = np.where(rows)[0][[0, -1]]
                x1, x2 = np.where(cols)[0][[0, -1]]
                bbox = (int(x1), int(y1), int(x2 + 1), int(y2 + 1))
        
        current_seed = seed
        current_mask = mask_array.copy()
        best_result = None
        best_confidence = -1.0
        
        for attempt in range(max_attempts):
            print(f"  [Inpainter] Attempt {attempt + 1}/{max_attempts}")
            
            # Run inpainting
            result_image = self.inpaint(
                image=original_image,
                mask=current_mask,
                prompt=prompt,
                bbox=bbox,
                seed=current_seed,
                apply_harmonization=False,  # Validate before harmonization
                **kwargs,
            )
            
            # Validate if validator is available
            if self.validator is not None:
                validation = self.validator.validate_with_difference(
                    original_image=original_image,
                    inpainted_image=result_image,
                    mask=current_mask,
                    bbox=bbox,
                    object_type=object_type,
                )
                
                print(f"    Validation: valid={validation.is_valid}, "
                      f"confidence={validation.confidence:.3f}, "
                      f"object_score={validation.object_score:.3f}")
                
                # Track best result
                if validation.confidence > best_confidence:
                    best_confidence = validation.confidence
                    best_result = (result_image, validation)
                
                if validation.is_valid:
                    # Apply harmonization now
                    if self.use_harmonizer and self.harmonizer is not None:
                        harmonized = self.harmonizer.harmonize(
                            inpainted_image=result_image,
                            original_image=original_image,
                            mask=current_mask,
                            bbox=bbox,
                        )
                        result_image = harmonized.image
                    
                    return InpaintingResult(
                        image=result_image,
                        is_valid=True,
                        confidence=validation.confidence,
                        attempts=attempt + 1,
                        object_type=object_type,
                        validation_details=validation.details,
                    )
            else:
                # No validator - assume valid
                if self.use_harmonizer and self.harmonizer is not None:
                    harmonized = self.harmonizer.harmonize(
                        inpainted_image=result_image,
                        original_image=original_image,
                        mask=current_mask,
                        bbox=bbox,
                    )
                    result_image = harmonized.image
                
                return InpaintingResult(
                    image=result_image,
                    is_valid=True,
                    confidence=1.0,
                    attempts=1,
                    object_type=object_type,
                    validation_details={"skipped": True},
                )
            
            # Prepare for retry with adaptive strategies
            if attempt < max_attempts - 1:
                # Strategy based on attempt number:
                # Attempt 1: Just change seed
                # Attempt 2: Change seed + expand mask + increase guidance
                # Attempt 3: Change seed + more expansion + different guidance
                
                # Change seed
                if current_seed is not None:
                    current_seed = current_seed + 1000 + attempt * 123
                else:
                    current_seed = np.random.randint(0, 1000000)
                
                # Adaptive mask expansion (more aggressive as attempts increase)
                if mask_scale_on_retry > 1.0:
                    # Dilate the mask - more aggressive on later attempts
                    base_kernel = int(5 * (mask_scale_on_retry - 1))
                    kernel_size = base_kernel * (attempt + 1) + 3  # At least 3px dilation
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                    current_mask = cv2.dilate(mask_array.copy(), kernel, iterations=1)
                
                # Adjust guidance scale on retries (try different values)
                # Higher guidance = more prompt adherence, lower = more natural
                retry_guidance = [self.guidance_scale, self.guidance_scale * 1.2, self.guidance_scale * 0.85]
                kwargs['guidance_scale'] = retry_guidance[min(attempt, len(retry_guidance) - 1)]
                
                # Increase crop padding on later retries for more context
                if attempt >= 1:
                    kwargs['crop_padding'] = kwargs.get('crop_padding', 150) + 50 * attempt
                
                print(f"    Retry {attempt + 2}: seed={current_seed}, mask_expand={kernel_size}px, "
                      f"guidance={kwargs.get('guidance_scale', self.guidance_scale):.1f}")
        
        # Return best result if no valid result found
        print(f"  [Inpainter] No valid result after {max_attempts} attempts, using best (confidence={best_confidence:.3f})")
        
        if best_result is not None:
            result_image, validation = best_result
            
            # Apply harmonization to best result
            if self.use_harmonizer and self.harmonizer is not None:
                harmonized = self.harmonizer.harmonize(
                    inpainted_image=result_image,
                    original_image=original_image,
                    mask=mask_array,
                    bbox=bbox,
                )
                result_image = harmonized.image
            
            return InpaintingResult(
                image=result_image,
                is_valid=False,
                confidence=best_confidence,
                attempts=max_attempts,
                object_type=object_type,
                validation_details=validation.details,
            )
        
        # Fallback - return last result
        return InpaintingResult(
            image=result_image,
            is_valid=False,
            confidence=0.0,
            attempts=max_attempts,
            object_type=object_type,
        )
    
    def __call__(
        self,
        image: Union[Image.Image, np.ndarray, str],
        mask: Union[Image.Image, np.ndarray],
        prompt: str,
        **kwargs,
    ) -> Image.Image:
        """Alias for inpaint method."""
        return self.inpaint(image, mask, prompt, **kwargs)


# Note: Object prompts are now centralized in utils/object_specs.py
# Use: from utils.object_specs import OBJECT_PROMPTS, get_prompt, get_all_object_keys

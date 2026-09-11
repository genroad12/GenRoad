"""
Scene editing module using CosXL Edit (SDXL InstructPix2Pix).
Based on cityscape-adverse implementation: https://github.com/naufalso/cityscape-adverse
"""

import os
import torch
import numpy as np
from PIL import Image
from typing import Union, Optional, Dict, Any, List


class SceneEditor:
    """
    Scene editor using CosXL Edit for weather/environment transformations.
    This follows the cityscape-adverse benchmark approach.
    
    Reference: https://arxiv.org/pdf/2411.00425
    """
    
    # Weather condition prompts following cityscape-adverse
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
    
    def __init__(
        self,
        model_path: str = "./models/cosxl_edit.safetensors",  # Local path to CosXL Edit safetensors
        cache_dir: str = "./models/huggingface",
        device: str = "cuda",
        dtype: str = "float16",
        num_inference_steps: int = 20,
        guidance_scale: float = 7.0,
        resolution: int = 2048,
    ):
        """
        Initialize the CosXL Edit scene editor.
        
        Args:
            model_path: Path to CosXL edit safetensors file
            cache_dir: Cache directory for VAE and other models
            device: Computation device
            dtype: Model dtype
            num_inference_steps: Denoising steps (20 recommended for CosXL)
            guidance_scale: Guidance scale (7.0 recommended)
            resolution: Target resolution for processing
        """
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.device = device
        self.dtype = torch.float16 if dtype == "float16" else torch.float32
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.resolution = resolution
        
        self.pipeline = None
        self._loaded = False
    
    def load(self) -> None:
        """Load the CosXL Edit pipeline."""
        if self._loaded:
            return
        
        print(f"Loading CosXL Edit model from: {self.model_path}")
        
        from diffusers import EDMEulerScheduler, StableDiffusionXLInstructPix2PixPipeline, AutoencoderKL
        
        # Use local CosXL edit checkpoint
        edit_file = self.model_path
        
        # Load VAE
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=self.dtype,
            cache_dir=self.cache_dir,
        )
        
        # Load pipeline
        self.pipeline = StableDiffusionXLInstructPix2PixPipeline.from_single_file(
            edit_file,
            num_in_channels=8,
            is_cosxl_edit=True,
            vae=vae,
            torch_dtype=self.dtype,
            cache_dir=self.cache_dir,
        )
        
        # Set scheduler as per cityscape-adverse
        self.pipeline.scheduler = EDMEulerScheduler(
            sigma_min=0.002,
            sigma_max=120.0,
            sigma_data=1.0,
            prediction_type="v_prediction",
            sigma_schedule="exponential",
        )
        
        self.pipeline.to(self.device)
        self.pipeline.set_progress_bar_config(disable=True)
        
        # Enable memory optimizations
        if self.device == "cuda":
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
            except Exception:
                self.pipeline.enable_attention_slicing()
        
        self._loaded = True
        print("CosXL Edit model loaded successfully.")
    
    def unload(self) -> None:
        """Unload the pipeline to free memory."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        self._loaded = False
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
    
    def _resize_image(self, image: Image.Image) -> Image.Image:
        """
        Resize image while maintaining aspect ratio.
        Ensures dimensions are multiples of 8.
        """
        original_width, original_height = image.size
        max_dim = max(original_width, original_height)
        scale_factor = self.resolution / max_dim
        
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        
        # Ensure dimensions are multiples of 8
        new_width = (new_width // 8) * 8
        new_height = (new_height // 8) * 8
        
        # Minimum size
        new_width = max(8, new_width)
        new_height = max(8, new_height)
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    @torch.no_grad()
    def edit_scene(
        self,
        image: Union[Image.Image, np.ndarray, str],
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        image_guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Edit scene with a text instruction.
        
        Args:
            image: Input image
            prompt: Editing instruction
            negative_prompt: What to avoid
            num_inference_steps: Override default steps
            guidance_scale: Override default guidance (text guidance)
            image_guidance_scale: Override default image guidance (1.5 default, higher = more like original)
            seed: Random seed for reproducibility
            
        Returns:
            Edited image
        """
        if not self._loaded:
            self.load()
        
        # Load image if path
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        original_size = image.size
        
        # Resize for processing
        image_resized = self._resize_image(image)
        width, height = image_resized.size
        
        # Set parameters
        steps = num_inference_steps or self.num_inference_steps
        guidance = guidance_scale or self.guidance_scale
        img_guidance = image_guidance_scale if image_guidance_scale is not None else 1.5
        
        # Ensure steps is valid (avoid off-by-one scheduler errors)
        steps = max(1, min(steps, 100))
        
        # Set seed - create generator with proper device
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
        
        # Clear any cached scheduler state
        self.pipeline.scheduler.set_timesteps(steps, device=self.device)
        
        # Run editing
        result = self.pipeline(
            prompt=prompt,
            image=image_resized,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            guidance_scale=guidance,
            image_guidance_scale=img_guidance,
            num_inference_steps=steps,
            generator=generator,
        ).images[0]
        
        # Resize back to original
        if result.size != original_size:
            result = result.resize(original_size, Image.Resampling.LANCZOS)
        
        return result
    
    def apply_weather(
        self,
        image: Union[Image.Image, np.ndarray, str],
        weather: str,
        custom_prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Apply weather transformation to an image.
        
        Args:
            image: Input image
            weather: Weather type ('snow', 'rain', 'fog', 'night', etc.)
            custom_prompt: Override default weather prompt
            seed: Random seed
            
        Returns:
            Weather-transformed image
        """
        # Get prompt
        if custom_prompt:
            prompt = custom_prompt
        elif weather in self.WEATHER_PROMPTS:
            prompt = self.WEATHER_PROMPTS[weather]
        else:
            raise ValueError(f"Unknown weather type: {weather}. Choose from {list(self.WEATHER_PROMPTS.keys())}")
        
        return self.edit_scene(
            image=image,
            prompt=prompt,
            seed=seed,
        )
    
    def apply_all_weather_conditions(
        self,
        image: Union[Image.Image, np.ndarray, str],
        weather_types: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Image.Image]:
        """
        Apply multiple weather conditions to an image.
        
        Args:
            image: Input image
            weather_types: List of weather types (all if None)
            seed: Base random seed
            
        Returns:
            Dictionary mapping weather type to transformed image
        """
        results = {}
        # Use all available weather types if none specified
        weathers = weather_types or list(self.WEATHER_PROMPTS.keys())
        
        for i, weather in enumerate(weathers):
            current_seed = seed + i if seed is not None else None
            results[weather] = self.apply_weather(
                image=image,
                weather=weather,
                seed=current_seed,
            )
        
        return results
    
    def __call__(
        self,
        image: Union[Image.Image, np.ndarray, str],
        weather: str,
        **kwargs,
    ) -> Image.Image:
        """Alias for apply_weather method."""
        return self.apply_weather(image, weather, **kwargs)

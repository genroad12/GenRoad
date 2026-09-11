"""
Image Harmonization Module for Post-Inpainting Processing

This module provides various techniques to harmonize inpainted objects
with the surrounding scene, addressing lighting and color consistency issues.

Techniques included:
1. Color Transfer - Match color statistics of inpainted region to surroundings
2. Histogram Matching - Match luminance/color histograms
3. Poisson Blending - Seamless gradient-domain blending
4. LAB Color Adjustment - Perceptually uniform color space adjustments
5. Multi-Band Blending - Pyramid-based smooth transitions

Key improvement: Uses difference-based mask from original vs inpainted image
for natural object detection and smooth transitions.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


class HarmonizationMethod(Enum):
    """Available harmonization methods."""
    NONE = "none"
    COLOR_TRANSFER = "color_transfer"
    HISTOGRAM_MATCH = "histogram_match"
    POISSON_BLEND = "poisson_blend"
    LAB_ADJUST = "lab_adjust"
    MULTI_BAND_BLEND = "multi_band_blend"
    COMBINED = "combined"  # Uses multiple methods


@dataclass
class HarmonizationResult:
    """Result of harmonization process."""
    image: Image.Image
    method: str
    adjustments: dict  # Details about what was adjusted


class ImageHarmonizer:
    """
    Post-inpainting image harmonizer to blend objects with scene lighting.
    
    This class provides multiple harmonization techniques that can be used
    individually or combined for best results.
    
    All methods use feathered masks to prevent visible bbox artifacts.
    """
    
    def __init__(
        self,
        method: str = "combined",
        blend_strength: float = 0.7,
        color_transfer_strength: float = 0.5,
        preserve_contrast: bool = True,
        context_radius: int = 100,  # Increased for better context sampling
        mask_feather_radius: int = 50,  # Feathering radius for smooth transitions
        use_difference_mask: bool = True,  # Use difference-based mask
        difference_threshold: int = 15,  # Threshold for difference detection
    ):
        """
        Initialize the harmonizer.
        
        Args:
            method: Harmonization method to use
            blend_strength: Overall blending strength (0-1)
            color_transfer_strength: Color transfer intensity (0-1)
            preserve_contrast: Whether to preserve original contrast
            context_radius: Radius around bbox to sample context colors
            mask_feather_radius: Radius for feathering the mask edges
            use_difference_mask: Use difference between original and inpainted
            difference_threshold: Pixel difference threshold for mask creation
        """
        self.method = HarmonizationMethod(method)
        self.blend_strength = blend_strength
        self.color_transfer_strength = color_transfer_strength
        self.preserve_contrast = preserve_contrast
        self.context_radius = context_radius
        self.mask_feather_radius = mask_feather_radius
        self.use_difference_mask = use_difference_mask
        self.difference_threshold = difference_threshold
    
    def _create_feathered_mask(
        self,
        mask: np.ndarray,
        feather_radius: Optional[int] = None,
    ) -> np.ndarray:
        """
        Create a feathered version of the mask for smooth blending.
        
        Args:
            mask: Binary mask (0-255)
            feather_radius: Radius for Gaussian blur (uses default if None)
            
        Returns:
            Feathered mask with values 0.0-1.0
        """
        radius = feather_radius if feather_radius is not None else self.mask_feather_radius
        
        if radius <= 0:
            return (mask > 127).astype(np.float32)
        
        # Convert to float
        mask_float = mask.astype(np.float32) / 255.0
        
        # Expand mask slightly before feathering
        kernel = np.ones((5, 5), np.uint8)
        mask_expanded = cv2.dilate((mask_float * 255).astype(np.uint8), kernel, iterations=2)
        mask_float = mask_expanded.astype(np.float32) / 255.0
        
        # Apply Gaussian blur for feathering
        blur_size = radius * 2 + 1
        feathered = cv2.GaussianBlur(mask_float, (blur_size, blur_size), 0)
        
        return feathered
    
    def _create_difference_mask(
        self,
        original: np.ndarray,
        inpainted: np.ndarray,
        binary_mask: Optional[np.ndarray] = None,
        feather_radius: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a mask based on pixel differences between original and inpainted images.
        
        This method automatically detects which pixels changed during inpainting,
        creating a more natural mask that captures the actual object boundaries.
        
        Args:
            original: Original image before inpainting
            inpainted: Image after inpainting
            binary_mask: Optional binary mask to constrain the region
            feather_radius: Radius for feathering (uses default if None)
            
        Returns:
            Tuple of (raw_difference_mask, feathered_mask) both 0.0-1.0
        """
        radius = feather_radius if feather_radius is not None else self.mask_feather_radius
        
        # Calculate absolute difference
        diff = cv2.absdiff(original, inpainted)
        
        # Convert to grayscale (max across channels for any change)
        diff_gray = np.max(diff, axis=2)
        
        # Apply threshold to create binary mask
        _, diff_binary = cv2.threshold(
            diff_gray, self.difference_threshold, 255, cv2.THRESH_BINARY
        )
        
        # If binary mask provided, constrain to that region
        if binary_mask is not None:
            if len(binary_mask.shape) == 3:
                binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_RGB2GRAY)
            # Expand the original mask slightly to include boundary effects
            kernel = np.ones((5, 5), np.uint8)
            expanded_binary = cv2.dilate(binary_mask, kernel, iterations=3)
            diff_binary = cv2.bitwise_and(diff_binary, expanded_binary)
        
        # Clean up the mask with morphological operations
        kernel_close = np.ones((7, 7), np.uint8)
        diff_binary = cv2.morphologyEx(diff_binary, cv2.MORPH_CLOSE, kernel_close)
        
        # Small opening to remove noise
        kernel_open = np.ones((3, 3), np.uint8)
        diff_binary = cv2.morphologyEx(diff_binary, cv2.MORPH_OPEN, kernel_open)
        
        # Store raw mask for debug
        raw_mask = diff_binary.astype(np.float32) / 255.0
        
        # Create feathered version
        if radius > 0:
            blur_size = radius * 2 + 1
            feathered = cv2.GaussianBlur(raw_mask, (blur_size, blur_size), 0)
        else:
            feathered = raw_mask
        
        return raw_mask, feathered
    
    def harmonize(
        self,
        inpainted_image: Union[Image.Image, np.ndarray],
        original_image: Union[Image.Image, np.ndarray],
        mask: Union[Image.Image, np.ndarray],
        bbox: Optional[Tuple[int, int, int, int]] = None,
        debug_dir: Optional[str] = None,
    ) -> HarmonizationResult:
        """
        Harmonize the inpainted region with the scene.
        
        Args:
            inpainted_image: Image after inpainting
            original_image: Original image before inpainting
            mask: Binary mask of inpainted region (white = inpainted)
            bbox: Optional bounding box (x1, y1, x2, y2)
            debug_dir: Optional directory to save debug visualizations
            
        Returns:
            HarmonizationResult with harmonized image
        """
        # Convert to numpy arrays
        inpainted = self._to_numpy(inpainted_image)
        original = self._to_numpy(original_image)
        mask_np = self._to_mask(mask)
        
        # Get bbox from mask if not provided
        if bbox is None:
            bbox = self._get_bbox_from_mask(mask_np)
        
        adjustments = {}
        
        # Create mask for harmonization
        if self.use_difference_mask:
            # Create mask from pixel differences - more natural object boundaries
            raw_diff_mask, feathered_mask = self._create_difference_mask(
                original, inpainted, mask_np
            )
            adjustments["difference_mask"] = True
            adjustments["difference_threshold"] = self.difference_threshold
            
            # Save debug visualizations
            if debug_dir:
                self._save_debug_masks(
                    debug_dir, raw_diff_mask, feathered_mask, mask_np, original, inpainted
                )
        else:
            # Use traditional feathered mask
            feathered_mask = self._create_feathered_mask(mask_np)
            raw_diff_mask = None
        
        if self.method == HarmonizationMethod.NONE:
            harmonized = inpainted
        elif self.method == HarmonizationMethod.COLOR_TRANSFER:
            harmonized = self._color_transfer(inpainted, original, mask_np, feathered_mask, bbox)
            adjustments["color_transfer"] = True
        elif self.method == HarmonizationMethod.HISTOGRAM_MATCH:
            harmonized = self._histogram_matching(inpainted, original, mask_np, feathered_mask, bbox)
            adjustments["histogram_match"] = True
        elif self.method == HarmonizationMethod.POISSON_BLEND:
            harmonized = self._poisson_blend(inpainted, original, mask_np, feathered_mask, bbox)
            adjustments["poisson_blend"] = True
        elif self.method == HarmonizationMethod.LAB_ADJUST:
            harmonized = self._lab_adjustment(inpainted, original, mask_np, feathered_mask, bbox)
            adjustments["lab_adjust"] = True
        elif self.method == HarmonizationMethod.MULTI_BAND_BLEND:
            harmonized = self._multi_band_blend(inpainted, original, mask_np, feathered_mask, bbox)
            adjustments["multi_band_blend"] = True
        elif self.method == HarmonizationMethod.COMBINED:
            harmonized = self._combined_harmonization(inpainted, original, mask_np, feathered_mask, bbox)
            adjustments["combined"] = True
        else:
            harmonized = inpainted
        
        # Final blend using feathered mask to ensure smooth transition
        # This is the key step that prevents hard edges
        feathered_3d = np.stack([feathered_mask] * 3, axis=-1)
        result = (
            harmonized.astype(np.float32) * feathered_3d +
            inpainted.astype(np.float32) * (1 - feathered_3d)
        ).astype(np.uint8)
        
        adjustments["mask_feather_radius"] = self.mask_feather_radius
        adjustments["use_difference_mask"] = self.use_difference_mask
        
        # Save final harmonized result for debug
        if debug_dir:
            self._save_final_debug(debug_dir, result, harmonized, inpainted)
        
        # Convert back to PIL
        result_pil = Image.fromarray(result)
        
        return HarmonizationResult(
            image=result_pil,
            method=self.method.value,
            adjustments=adjustments,
        )
    
    def _save_debug_masks(
        self,
        debug_dir: str,
        raw_diff_mask: np.ndarray,
        feathered_mask: np.ndarray,
        original_mask: np.ndarray,
        original_image: np.ndarray,
        inpainted_image: np.ndarray,
    ) -> None:
        """Save debug visualizations of masks."""
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Raw difference mask
        raw_vis = (raw_diff_mask * 255).astype(np.uint8)
        Image.fromarray(raw_vis).save(debug_path / "20_difference_mask_raw.png")
        
        # 2. Feathered difference mask
        feathered_vis = (feathered_mask * 255).astype(np.uint8)
        Image.fromarray(feathered_vis).save(debug_path / "21_difference_mask_feathered.png")
        
        # 3. Original binary mask for comparison
        Image.fromarray(original_mask).save(debug_path / "22_original_binary_mask.png")
        
        # 4. Pixel difference heatmap (shows where changes occurred)
        diff = cv2.absdiff(original_image, inpainted_image)
        diff_gray = np.max(diff, axis=2)
        diff_heatmap = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
        diff_heatmap = cv2.cvtColor(diff_heatmap, cv2.COLOR_BGR2RGB)
        Image.fromarray(diff_heatmap).save(debug_path / "23_pixel_difference_heatmap.png")
        
        # 5. Mask overlay on inpainted image
        overlay = inpainted_image.copy()
        mask_colored = np.zeros_like(overlay)
        mask_colored[:, :, 1] = (feathered_mask * 255).astype(np.uint8)  # Green channel
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        Image.fromarray(overlay).save(debug_path / "24_mask_overlay.png")
        
        print(f"  [Harmonizer] Debug masks saved to: {debug_path}")
    
    def _save_final_debug(
        self,
        debug_dir: str,
        final_result: np.ndarray,
        harmonized_before_blend: np.ndarray,
        inpainted: np.ndarray,
    ) -> None:
        """Save final harmonization debug images."""
        debug_path = Path(debug_dir)
        
        # Harmonized before final blend
        Image.fromarray(harmonized_before_blend).save(
            debug_path / "25_harmonized_before_blend.png"
        )
        
        # Final result after blend
        Image.fromarray(final_result).save(
            debug_path / "26_harmonized_final.png"
        )
        
        # Difference between inpainted and harmonized
        diff = cv2.absdiff(inpainted, final_result)
        diff_enhanced = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        Image.fromarray(diff_enhanced).save(
            debug_path / "27_harmonization_effect.png"
        )
    
    def _to_numpy(self, img: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Convert image to numpy array."""
        if isinstance(img, Image.Image):
            return np.array(img)
        return img.copy()
    
    def _to_mask(self, mask: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Convert mask to binary numpy array."""
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        return (mask > 127).astype(np.uint8) * 255
    
    def _get_bbox_from_mask(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Extract bounding box from mask."""
        rows = np.any(mask > 0, axis=1)
        cols = np.any(mask > 0, axis=0)
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        return (int(x1), int(y1), int(x2 + 1), int(y2 + 1))
    
    def _get_context_region(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        mask: np.ndarray,
    ) -> np.ndarray:
        """Get context region around the bbox (excluding masked area)."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Expand bbox significantly for context
        ctx_x1 = max(0, x1 - self.context_radius)
        ctx_y1 = max(0, y1 - self.context_radius)
        ctx_x2 = min(w, x2 + self.context_radius)
        ctx_y2 = min(h, y2 + self.context_radius)
        
        # Get context region
        context = image[ctx_y1:ctx_y2, ctx_x1:ctx_x2].copy()
        context_mask = mask[ctx_y1:ctx_y2, ctx_x1:ctx_x2]
        
        # Return only non-masked pixels
        return context[context_mask == 0]
    
    def _color_transfer(
        self,
        inpainted: np.ndarray,
        original: np.ndarray,
        mask: np.ndarray,
        feathered_mask: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """
        Apply color transfer to match context region statistics.
        Uses LAB color space and feathered mask for smooth blending.
        """
        # Get context pixels from a wider area
        context_pixels = self._get_context_region(original, bbox, mask)
        if len(context_pixels) == 0:
            return inpainted.copy()
        
        # Convert to LAB
        inpainted_lab = cv2.cvtColor(inpainted, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Calculate statistics from context
        context_lab = cv2.cvtColor(
            context_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB
        ).reshape(-1, 3).astype(np.float32)
        
        ctx_mean = np.mean(context_lab, axis=0)
        ctx_std = np.std(context_lab, axis=0) + 1e-6
        
        # Get inpainted region statistics
        inpainted_region = inpainted_lab[mask > 0]
        inp_mean = np.mean(inpainted_region, axis=0)
        inp_std = np.std(inpainted_region, axis=0) + 1e-6
        
        # Transfer colors using feathered mask for smooth blending
        result_lab = inpainted_lab.copy()
        feathered_3d = np.stack([feathered_mask] * 3, axis=-1)
        
        for i in range(3):
            channel = inpainted_lab[:, :, i]
            # Normalize, then denormalize with context stats
            normalized = (channel - inp_mean[i]) / inp_std[i]
            transferred = normalized * ctx_std[i] + ctx_mean[i]
            
            # Blend using feathered mask
            blended = (
                transferred * self.color_transfer_strength +
                channel * (1 - self.color_transfer_strength)
            )
            # Apply with feathered mask
            result_lab[:, :, i] = np.clip(
                blended * feathered_mask + channel * (1 - feathered_mask),
                0, 255
            )
        
        # Convert back to RGB
        result = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        return result
    
    def _histogram_matching(
        self,
        inpainted: np.ndarray,
        original: np.ndarray,
        mask: np.ndarray,
        feathered_mask: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """Match histogram of inpainted region to context with feathered blending."""
        result = inpainted.copy()
        
        # Get context region
        context_pixels = self._get_context_region(original, bbox, mask)
        if len(context_pixels) == 0:
            return result
        
        # Process each channel
        for c in range(3):
            # Get source (inpainted region) and reference (context) pixels
            src_pixels = inpainted[:, :, c][mask > 0]
            ref_pixels = context_pixels[:, c]
            
            if len(src_pixels) == 0 or len(ref_pixels) == 0:
                continue
            
            # Calculate CDFs
            src_hist, _ = np.histogram(src_pixels, bins=256, range=(0, 256))
            ref_hist, _ = np.histogram(ref_pixels, bins=256, range=(0, 256))
            
            src_cdf = np.cumsum(src_hist).astype(np.float32)
            src_cdf /= src_cdf[-1] + 1e-6
            
            ref_cdf = np.cumsum(ref_hist).astype(np.float32)
            ref_cdf /= ref_cdf[-1] + 1e-6
            
            # Create lookup table
            lut = np.zeros(256, dtype=np.uint8)
            j = 0
            for i in range(256):
                while j < 255 and ref_cdf[j] < src_cdf[i]:
                    j += 1
                lut[i] = j
            
            # Apply to channel with feathered blending
            channel = result[:, :, c].astype(np.float32)
            matched = lut[result[:, :, c]].astype(np.float32)
            
            # Blend using feathered mask
            blended = matched * self.blend_strength + channel * (1 - self.blend_strength)
            result[:, :, c] = (
                blended * feathered_mask + channel * (1 - feathered_mask)
            ).astype(np.uint8)
        
        return result
    
    def _poisson_blend(
        self,
        inpainted: np.ndarray,
        original: np.ndarray,
        mask: np.ndarray,
        feathered_mask: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """Apply Poisson blending with feathered mask fallback."""
        # Get center of mask
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        try:
            # Use OpenCV's seamless cloning
            result = cv2.seamlessClone(
                inpainted,
                original,
                mask,
                center,
                cv2.NORMAL_CLONE
            )
            
            # Additional feathered blend for smoother transition
            feathered_3d = np.stack([feathered_mask] * 3, axis=-1)
            result = (
                result.astype(np.float32) * self.blend_strength +
                inpainted.astype(np.float32) * (1 - self.blend_strength)
            )
            result = (
                result * feathered_3d +
                inpainted.astype(np.float32) * (1 - feathered_3d)
            ).astype(np.uint8)
            
            return result
        except cv2.error:
            print("  [Harmonizer] Poisson blending failed, using color transfer")
            return self._color_transfer(inpainted, original, mask, feathered_mask, bbox)
    
    def _lab_adjustment(
        self,
        inpainted: np.ndarray,
        original: np.ndarray,
        mask: np.ndarray,
        feathered_mask: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """Adjust LAB channels for better lighting match with feathered blending."""
        # Get context pixels
        context_pixels = self._get_context_region(original, bbox, mask)
        if len(context_pixels) == 0:
            return inpainted.copy()
        
        # Convert to LAB
        inpainted_lab = cv2.cvtColor(inpainted, cv2.COLOR_RGB2LAB).astype(np.float32)
        context_lab = cv2.cvtColor(
            context_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB
        ).reshape(-1, 3).astype(np.float32)
        
        # Calculate luminance statistics
        ctx_L_mean = np.mean(context_lab[:, 0])
        
        # Get inpainted region L channel stats
        inp_L = inpainted_lab[:, :, 0]
        inp_L_masked = inp_L[mask > 0]
        if len(inp_L_masked) == 0:
            return inpainted.copy()
        inp_L_mean = np.mean(inp_L_masked)
        
        # Adjust luminance with feathered mask
        L_shift = (ctx_L_mean - inp_L_mean) * self.blend_strength
        L_adjusted = inp_L + L_shift * feathered_mask
        
        inpainted_lab[:, :, 0] = np.clip(L_adjusted, 0, 255)
        
        # Convert back
        result = cv2.cvtColor(inpainted_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        return result
    
    def _multi_band_blend(
        self,
        inpainted: np.ndarray,
        original: np.ndarray,
        mask: np.ndarray,
        feathered_mask: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """Multi-band blending for smooth transitions using feathered mask."""
        # Use the already feathered mask
        feathered_3d = np.stack([feathered_mask] * 3, axis=-1)
        
        # Build Gaussian pyramids
        levels = 5  # More levels for smoother blending
        gp_inpainted = [inpainted.astype(np.float32)]
        gp_original = [original.astype(np.float32)]
        gp_mask = [feathered_3d]
        
        for i in range(levels):
            gp_inpainted.append(cv2.pyrDown(gp_inpainted[-1]))
            gp_original.append(cv2.pyrDown(gp_original[-1]))
            gp_mask.append(cv2.pyrDown(gp_mask[-1]))
        
        # Build Laplacian pyramids
        lp_inpainted = [gp_inpainted[-1]]
        lp_original = [gp_original[-1]]
        
        for i in range(levels, 0, -1):
            size = (gp_inpainted[i-1].shape[1], gp_inpainted[i-1].shape[0])
            lap_inp = gp_inpainted[i-1] - cv2.pyrUp(gp_inpainted[i], dstsize=size)
            lap_orig = gp_original[i-1] - cv2.pyrUp(gp_original[i], dstsize=size)
            lp_inpainted.append(lap_inp)
            lp_original.append(lap_orig)
        
        # Blend pyramids
        lp_blended = []
        for i, (lap_inp, lap_orig) in enumerate(zip(lp_inpainted, lp_original)):
            mask_level = gp_mask[levels - i] if i < len(gp_mask) else gp_mask[-1]
            if mask_level.shape[:2] != lap_inp.shape[:2]:
                mask_level = cv2.resize(mask_level, (lap_inp.shape[1], lap_inp.shape[0]))
            blended = lap_inp * mask_level + lap_orig * (1 - mask_level)
            lp_blended.append(blended)
        
        # Reconstruct
        result = lp_blended[0]
        for i in range(1, len(lp_blended)):
            size = (lp_blended[i].shape[1], lp_blended[i].shape[0])
            result = cv2.pyrUp(result, dstsize=size) + lp_blended[i]
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _combined_harmonization(
        self,
        inpainted: np.ndarray,
        original: np.ndarray,
        mask: np.ndarray,
        feathered_mask: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """Apply multiple harmonization techniques in sequence."""
        result = inpainted.copy()
        
        # Step 1: LAB adjustment for luminance matching
        result = self._lab_adjustment(result, original, mask, feathered_mask, bbox)
        
        # Step 2: Color transfer for color consistency
        result = self._color_transfer(result, original, mask, feathered_mask, bbox)
        
        # Step 3: Multi-band blend for smooth edges
        result = self._multi_band_blend(result, original, mask, feathered_mask, bbox)
        
        return result
    
    def __call__(
        self,
        inpainted_image: Union[Image.Image, np.ndarray],
        original_image: Union[Image.Image, np.ndarray],
        mask: Union[Image.Image, np.ndarray],
        bbox: Optional[Tuple[int, int, int, int]] = None,
        debug_dir: Optional[str] = None,
    ) -> HarmonizationResult:
        """Alias for harmonize method."""
        return self.harmonize(inpainted_image, original_image, mask, bbox, debug_dir)


def quick_harmonize(
    inpainted: Image.Image,
    original: Image.Image,
    mask: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    method: str = "combined",
    strength: float = 0.7,
    feather_radius: int = 50,
    use_difference_mask: bool = True,
    debug_dir: Optional[str] = None,
) -> Image.Image:
    """
    Quick harmonization function for simple use cases.
    
    Args:
        inpainted: Inpainted image
        original: Original image
        mask: Binary mask
        bbox: Optional bounding box
        method: Harmonization method
        strength: Blending strength
        feather_radius: Mask feathering radius
        use_difference_mask: Use difference-based mask
        debug_dir: Optional debug directory
        
    Returns:
        Harmonized PIL Image
    """
    harmonizer = ImageHarmonizer(
        method=method,
        blend_strength=strength,
        mask_feather_radius=feather_radius,
        use_difference_mask=use_difference_mask,
    )
    result = harmonizer.harmonize(inpainted, original, mask, bbox, debug_dir)
    return result.image

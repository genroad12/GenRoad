"""
SAM Segmentation Module (Segment Anything Model)
=================================================

Point-prompt segmentation with Meta's Segment Anything Model (SAM).

Features:
  - One-click object segmentation
  - Positive and negative point prompts
  - Mask post-processing and refinement
  - GPU and CPU support

Usage:
    from genroad.models.sam_segmenter import SAMSegmenter
    
    segmenter = SAMSegmenter()
    mask = segmenter.segment_point(image, point=(x, y))
"""

import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SAMSegmenter:
    """
    Interactive segmentation based on the Segment Anything Model (SAM).

    Segments the object selected with a point prompt using Hugging Face
    Transformers.
    """
    
    # SAM model variants from smallest to largest.
    MODEL_CONFIGS = {
        "vit-base": "facebook/sam-vit-base",       # ~375 MB, fast
        "vit-large": "facebook/sam-vit-large",     # ~1.2 GB, balanced
        "vit-huge": "facebook/sam-vit-huge",       # ~2.5 GB, highest quality
    }
    
    def __init__(
        self,
        model_variant: str = "vit-base",
        device: Optional[str] = None,
        dtype: str = "float32",
    ):
        """
        Initialize the SAM segmenter.
        
        Args:
            model_variant: Model size: "vit-base", "vit-large", or "vit-huge".
                "vit-base" is recommended for the GUI.
            device: Compute device ("cuda" or "cpu").
            dtype: Model precision. float32 is recommended for SAM.
        """
        self.model_variant = model_variant
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # SAM is more stable with float32.
        self.dtype = torch.float32
        
        self._model = None
        self._processor = None
        
        logger.info(f"SAMSegmenter initialized: variant={model_variant}, device={self.device}, dtype=float32")
    
    @property
    def model_id(self) -> str:
        """HuggingFace model ID."""
        return self.MODEL_CONFIGS.get(self.model_variant, self.MODEL_CONFIGS["vit-base"])
    
    def load_model(self):
        """Load the model and processor lazily."""
        if self._model is not None:
            return
        
        logger.info(f"Loading SAM model: {self.model_id}")
        
        try:
            from transformers import SamModel, SamProcessor
            
            self._processor = SamProcessor.from_pretrained(self.model_id)
            # SAM should be loaded with float32 for numerical stability.
            self._model = SamModel.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,  # float32 is required for SAM
            ).to(self.device)
            
            # Eval mode for inference
            self._model.eval()
            
            logger.info(f"✓ SAM model loaded: {self.model_variant} (float32)")
            
        except Exception as e:
            logger.error(f"Failed to load SAM model: {e}")
            raise RuntimeError(
                f"Could not load the SAM model: {e}\n"
                "Make sure transformers>=4.35.0 is installed."
            )
    
    def segment_point(
        self,
        image: Union[Image.Image, np.ndarray],
        point: Tuple[int, int],
        point_label: int = 1,
        multimask_output: bool = True,
        return_best: bool = True,
        use_crop: bool = True,
        crop_size: int = 512,
        crop_padding: int = 100,
    ) -> Tuple[np.ndarray, float]:
        """
        Segment an object using a single point prompt.

        When use_crop=True, segmentation runs on a crop around the point and
        maps the mask back to the original image for faster inference.
        
        Args:
            image: PIL Image veya numpy array
            point: Click coordinates (x, y).
            point_label: 1 for foreground and 0 for background.
            multimask_output: Whether to generate multiple masks.
            return_best: Return the highest-scoring mask.
            use_crop: Enable crop-based segmentation.
            crop_size: Crop size in pixels.
            crop_padding: Additional crop padding.
            
        Returns:
            Tuple[mask, score]: Binary mask (HxW, uint8 0/255) and confidence.
        """
        self.load_model()
        
        # Normalize the image format.
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        orig_w, orig_h = image.size
        
        # Crop-based segmentation for speed
        if use_crop and (orig_w > crop_size or orig_h > crop_size):
            return self._segment_point_cropped(
                image, point, point_label, multimask_output, return_best,
                crop_size, crop_padding
            )
        
        # Full image segmentation (for small images or when crop disabled)
        return self._segment_point_full(
            image, point, point_label, multimask_output, return_best
        )
    
    def _segment_point_cropped(
        self,
        image: Image.Image,
        point: Tuple[int, int],
        point_label: int,
        multimask_output: bool,
        return_best: bool,
        crop_size: int,
        crop_padding: int,
    ) -> Tuple[np.ndarray, float]:
        """
        Crop-based segmentation for faster inference.

        Creates a crop around the selected point, runs SAM on the crop, and
        maps the resulting mask back to the original image.
        """
        import time
        start_time = time.time()
        
        orig_w, orig_h = image.size
        px, py = point
        
        # Calculate crop region centered on click point
        half_size = crop_size // 2
        
        # Crop bounds with padding
        crop_x1 = max(0, px - half_size - crop_padding)
        crop_y1 = max(0, py - half_size - crop_padding)
        crop_x2 = min(orig_w, px + half_size + crop_padding)
        crop_y2 = min(orig_h, py + half_size + crop_padding)
        
        # Ensure minimum crop size
        if crop_x2 - crop_x1 < crop_size:
            if crop_x1 == 0:
                crop_x2 = min(orig_w, crop_size)
            else:
                crop_x1 = max(0, crop_x2 - crop_size)
        
        if crop_y2 - crop_y1 < crop_size:
            if crop_y1 == 0:
                crop_y2 = min(orig_h, crop_size)
            else:
                crop_y1 = max(0, crop_y2 - crop_size)
        
        # Create crop
        crop_img = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        crop_w, crop_h = crop_img.size
        
        # Adjust point to crop coordinates
        crop_point = (px - crop_x1, py - crop_y1)
        
        logger.info(f"Crop-based SAM: {orig_w}x{orig_h} -> {crop_w}x{crop_h}, "
                   f"point {point} -> {crop_point}")
        
        try:
            # Run SAM on crop
            input_points = [[[crop_point[0], crop_point[1]]]]
            input_labels = [[point_label]]
            
            inputs = self._processor(
                images=crop_img,
                input_points=input_points,
                input_labels=input_labels,
                return_tensors="pt",
            )
            
            inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._model(**inputs, multimask_output=multimask_output)
            
            # Get masks for crop
            crop_masks = self._processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu()
            )[0]
            
            scores = outputs.iou_scores.squeeze().cpu().numpy()
            crop_masks = crop_masks.squeeze().cpu().numpy()
            
            # Select best mask
            if return_best and len(crop_masks.shape) == 3:
                best_idx = np.argmax(scores)
                crop_mask = crop_masks[best_idx]
                score = float(scores[best_idx])
            else:
                crop_mask = crop_masks[0] if len(crop_masks.shape) == 3 else crop_masks
                score = float(scores[0]) if isinstance(scores, np.ndarray) and scores.size > 1 else float(scores)
            
            # Convert to binary
            crop_mask = (crop_mask > 0.5).astype(np.uint8) * 255
            
            # Map mask back to original image size
            full_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            full_mask[crop_y1:crop_y2, crop_x1:crop_x2] = crop_mask
            
            elapsed = time.time() - start_time
            logger.info(f"Crop-based segmentation: {elapsed:.3f}s, score={score:.3f}")
            
            return full_mask, score
            
        except Exception as e:
            logger.error(f"Crop segmentation error: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _segment_point_full(
        self,
        image: Image.Image,
        point: Tuple[int, int],
        point_label: int,
        multimask_output: bool,
        return_best: bool,
    ) -> Tuple[np.ndarray, float]:
        """Full-image segmentation without cropping."""
        import time
        start_time = time.time()
        
        try:
            input_points = [[[point[0], point[1]]]]
            input_labels = [[point_label]]
            
            inputs = self._processor(
                images=image,
                input_points=input_points,
                input_labels=input_labels,
                return_tensors="pt",
            )
            
            inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._model(**inputs, multimask_output=multimask_output)
            
            masks = self._processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu()
            )[0]
            
            scores = outputs.iou_scores.squeeze().cpu().numpy()
            masks = masks.squeeze().cpu().numpy()
            
            if return_best and len(masks.shape) == 3:
                best_idx = np.argmax(scores)
                mask = masks[best_idx]
                score = float(scores[best_idx])
            else:
                mask = masks[0] if len(masks.shape) == 3 else masks
                score = float(scores[0]) if isinstance(scores, np.ndarray) and scores.size > 1 else float(scores)
            
            mask = (mask > 0.5).astype(np.uint8) * 255
            
            elapsed = time.time() - start_time
            logger.info(f"Full segmentation: {elapsed:.3f}s, score={score:.3f}")
            
            return mask, score
            
        except Exception as e:
            logger.error(f"Segmentation error: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def segment_points(
        self,
        image: Union[Image.Image, np.ndarray],
        positive_points: List[Tuple[int, int]],
        negative_points: Optional[List[Tuple[int, int]]] = None,
        multimask_output: bool = True,
        return_best: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """
        Segment an object using multiple positive and negative points.
        
        Args:
            image: PIL image or NumPy array.
            positive_points: Points belonging to the object.
            negative_points: Points that do not belong to the object.
            multimask_output: Whether to generate multiple masks.
            return_best: Return the highest-scoring mask.
            
        Returns:
            Tuple[mask, score]: Binary mask and confidence score.
        """
        self.load_model()
        
        # Normalize the image format.
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Prepare points and labels
        all_points = [list(p) for p in positive_points]
        all_labels = [1] * len(positive_points)
        
        if negative_points:
            all_points.extend([list(p) for p in negative_points])
            all_labels.extend([0] * len(negative_points))
        
        try:
            inputs = self._processor(
                images=image,
                input_points=[[all_points]],
                input_labels=[all_labels],
                return_tensors="pt",
            )
            
            inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._model(**inputs, multimask_output=multimask_output)
            
            masks = self._processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu()
            )[0]
            
            scores = outputs.iou_scores.squeeze().cpu().numpy()
            masks = masks.squeeze().cpu().numpy()
            
            if return_best and len(masks.shape) == 3:
                best_idx = np.argmax(scores)
                mask = masks[best_idx]
                score = float(scores[best_idx])
            else:
                mask = masks[0] if len(masks.shape) == 3 else masks
                score = float(scores[0]) if isinstance(scores, np.ndarray) and scores.size > 1 else float(scores)
            
            mask = (mask > 0.5).astype(np.uint8) * 255
            
            logger.info(f"Multi-point segmentation: {len(positive_points)} pos, "
                       f"{len(negative_points or [])} neg, score={score:.3f}")
            return mask, score
            
        except Exception as e:
            logger.error(f"Multi-point segmentation error: {e}")
            raise
    
    def segment_bbox(
        self,
        image: Union[Image.Image, np.ndarray],
        bbox: Tuple[int, int, int, int],
        point: Optional[Tuple[int, int]] = None,
        multimask_output: bool = True,
        return_best: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """
        Segment an object using a bounding-box prompt.

        The complete object inside the box is segmented, which is useful for
        multi-part objects such as strollers and bicycles.
        
        Args:
            image: PIL image or NumPy array.
            bbox: Bounding box (x1, y1, x2, y2).
            point: Optional center point combined with the box prompt.
            multimask_output: Whether to generate multiple masks.
            return_best: Return the highest-scoring mask.
            
        Returns:
            Tuple[mask, score]: Binary mask and confidence score.
        """
        self.load_model()
        
        import time
        start_time = time.time()
        
        # Normalize the image format.
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        orig_w, orig_h = image.size
        x1, y1, x2, y2 = bbox
        
        # Crop-based processing for speed
        crop_padding = 50
        crop_x1 = max(0, x1 - crop_padding)
        crop_y1 = max(0, y1 - crop_padding)
        crop_x2 = min(orig_w, x2 + crop_padding)
        crop_y2 = min(orig_h, y2 + crop_padding)
        
        crop_img = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        
        # Adjust bbox to crop coordinates
        crop_bbox = [
            x1 - crop_x1,
            y1 - crop_y1,
            x2 - crop_x1,
            y2 - crop_y1
        ]
        
        # Prepare point if provided (adjust to crop)
        input_points = None
        input_labels = None
        if point:
            crop_point = [point[0] - crop_x1, point[1] - crop_y1]
            input_points = [[[crop_point[0], crop_point[1]]]]
            input_labels = [[1]]
        
        try:
            # SAM input with box prompt
            inputs = self._processor(
                images=crop_img,
                input_boxes=[[crop_bbox]],
                input_points=input_points,
                input_labels=input_labels,
                return_tensors="pt",
            )
            
            inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._model(**inputs, multimask_output=multimask_output)
            
            # Get masks for crop
            crop_masks = self._processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu()
            )[0]
            
            scores = outputs.iou_scores.squeeze().cpu().numpy()
            crop_masks = crop_masks.squeeze().cpu().numpy()
            
            # Select best mask
            if return_best and len(crop_masks.shape) == 3:
                best_idx = np.argmax(scores)
                crop_mask = crop_masks[best_idx]
                score = float(scores[best_idx])
            else:
                crop_mask = crop_masks[0] if len(crop_masks.shape) == 3 else crop_masks
                score = float(scores[0]) if isinstance(scores, np.ndarray) and scores.size > 1 else float(scores)
            
            # Convert to binary
            crop_mask = (crop_mask > 0.5).astype(np.uint8) * 255
            
            # Map back to original size
            full_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            full_mask[crop_y1:crop_y2, crop_x1:crop_x2] = crop_mask
            
            elapsed = time.time() - start_time
            logger.info(f"BBox segmentation: bbox={bbox}, score={score:.3f}, time={elapsed:.3f}s")
            
            return full_mask, score
            
        except Exception as e:
            logger.error(f"BBox segmentation error: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def segment_point_with_bbox(
        self,
        image: Union[Image.Image, np.ndarray],
        point: Tuple[int, int],
        bbox: Optional[Tuple[int, int, int, int]] = None,
        use_bbox_if_available: bool = True,
        fallback_to_point: bool = True,
        **kwargs
    ) -> Tuple[np.ndarray, float]:
        """
        Use a bounding box when available; otherwise use the point prompt.

        A box is useful for multi-part objects such as strollers and bicycles.
        A point is sufficient for single-part objects.
        
        Args:
            image: PIL image or NumPy array.
            point: Click coordinates (x, y).
            bbox: Optional bounding box from inpainting.
            use_bbox_if_available: Use box-guided segmentation when available.
            fallback_to_point: Fall back to the point if box segmentation fails.
            **kwargs: Parameters forwarded to segment_point.
            
        Returns:
            Tuple[mask, score]: Binary mask and confidence score.
        """
        # Use box-guided segmentation when available.
        if bbox and use_bbox_if_available:
            try:
                logger.info(f"Using BBox-guided segmentation: bbox={bbox}, point={point}")
                mask, score = self.segment_bbox(image, bbox, point=point)
                
                # Require a meaningful mask area.
                if np.sum(mask > 0) > 100:
                    return mask, score
                else:
                    logger.warning("BBox mask too small, falling back to point")
            except Exception as e:
                logger.warning(f"BBox segmentation failed: {e}")
                if not fallback_to_point:
                    raise
        
        # Point-based segmentation (fallback or default).
        logger.info(f"Using point-based segmentation: point={point}")
        return self.segment_point(image, point, **kwargs)
    
    def merge_masks(
        self,
        masks: List[np.ndarray],
        mode: str = "union"
    ) -> np.ndarray:
        """
        Merge multiple masks.
        
        Args:
            masks: List of masks.
            mode: "union" (OR), "intersection" (AND)
            
        Returns:
            Merged mask.
        """
        if not masks:
            return None
        
        if len(masks) == 1:
            return masks[0]
        
        result = masks[0].copy()
        
        for mask in masks[1:]:
            if mode == "union":
                result = np.maximum(result, mask)
            elif mode == "intersection":
                result = np.minimum(result, mask)
        
        return result
    
    def refine_mask(
        self,
        mask: np.ndarray,
        kernel_size: int = 5,
        iterations: int = 2,
    ) -> np.ndarray:
        """
        Refine a mask with morphological operations.
        
        Args:
            mask: Binary mask (HxW, uint8).
            kernel_size: Morphological kernel size.
            iterations: Number of iterations.
            
        Returns:
            Refined mask
        """
        import cv2
        
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (kernel_size, kernel_size)
        )
        
        # Close small holes
        refined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        
        # Remove small noise
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return refined
    
    def dilate_mask(
        self,
        mask: np.ndarray,
        pixels: int = 5,
    ) -> np.ndarray:
        """
        Dilate a mask for harmonization.
        
        Args:
            mask: Binary mask.
            pixels: Dilation amount in pixels.
            
        Returns:
            Dilated mask
        """
        import cv2
        
        kernel_size = pixels * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size)
        )
        
        return cv2.dilate(mask, kernel, iterations=1)
    
    def feather_mask(
        self,
        mask: np.ndarray,
        radius: int = 10,
    ) -> np.ndarray:
        """
        Feather mask edges with a Gaussian blur.
        
        Args:
            mask: Binary mask.
            radius: Blur radius.
            
        Returns:
            Feathered mask (float32, 0-1)
        """
        import cv2
        
        blur_size = radius * 2 + 1
        feathered = cv2.GaussianBlur(
            mask.astype(np.float32),
            (blur_size, blur_size),
            0
        )
        
        # Normalize to 0-1
        if feathered.max() > 0:
            feathered = feathered / feathered.max()
        
        return feathered
    
    def get_mask_bbox(
        self,
        mask: np.ndarray,
        padding: int = 0,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Compute the bounding box of a mask.
        
        Args:
            mask: Binary mask.
            padding: Additional padding in pixels.
            
        Returns:
            (x1, y1, x2, y2), or None when the mask is empty.
        """
        # Find non-zero pixels
        coords = np.where(mask > 0)
        
        if len(coords[0]) == 0:
            return None
        
        y1, y2 = coords[0].min(), coords[0].max()
        x1, x2 = coords[1].min(), coords[1].max()
        
        # Add padding
        if padding > 0:
            h, w = mask.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
        
        return (int(x1), int(y1), int(x2), int(y2))
    
    def visualize_mask(
        self,
        image: Union[Image.Image, np.ndarray],
        mask: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        alpha: float = 0.4,
        show_contour: bool = True,
        contour_thickness: int = 2,
    ) -> np.ndarray:
        """
        Visualize a mask over an image.
        
        Args:
            image: Original image.
            mask: Binary mask.
            color: Overlay color (R, G, B).
            alpha: Overlay opacity.
            show_contour: Draw the mask contour.
            contour_thickness: Contour thickness.
            
        Returns:
            Visualization array (RGB)
        """
        import cv2
        
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        vis = image.copy()
        
        # Ensure mask is uint8
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
        
        # Create colored overlay
        overlay = np.zeros_like(vis)
        mask_bool = mask > 127
        overlay[mask_bool] = color
        
        # Blend
        vis = (vis * (1 - alpha) + overlay * alpha).astype(np.uint8)
        
        # Draw contour
        if show_contour:
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(vis, contours, -1, color, contour_thickness)
        
        return vis
    
    def unload_model(self):
        """Release the model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("SAM model unloaded")


# Convenience function
def create_segmenter(
    variant: str = "vit-base",
    device: Optional[str] = None,
) -> SAMSegmenter:
    """
    Create a SAMSegmenter instance.
    
    Args:
        variant: "vit-base", "vit-large", "vit-huge"
        device: "cuda" or "cpu"
        
    Returns:
        SAMSegmenter instance
    """
    return SAMSegmenter(model_variant=variant, device=device)

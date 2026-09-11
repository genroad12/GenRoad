"""
Stable Diffusion Inpainting Prompts for 23 Object Categories
Used for synthetic obstacle generation in autonomous driving scenarios.

Includes:
- Prompts with scene lighting consistency
- Object-specific mask specifications (aspect ratio, size)
- Real-world size constraints for depth-based scaling
"""

from typing import Dict, Any, Optional, Tuple

# Common lighting/scene consistency suffix for all prompts
LIGHTING_SUFFIX = (
    "matching the scene ambient lighting and color temperature, "
    "consistent shadows with surrounding environment, "
    "natural integration with road surface, "
    "realistic ambient occlusion at base"
)

# Enhanced negative prompt for better results
ENHANCED_NEGATIVE_PROMPT = (
    "floating, hovering, unnatural lighting, wrong shadow direction, "
    "overexposed, underexposed, glowing, neon, oversaturated colors, "
    "cartoon, illustration, painting, drawing, artificial looking, "
    "inconsistent lighting, mismatched illumination, harsh edges, "
    "blurry, low quality, distorted, invisible, transparent"
)

OBJECT_PROMPTS = {
    # Temel Nesneler (Basic Objects)
    "cardboard_box": (
        "A large brown cardboard shipping box sitting upright on the asphalt road, "
        "clearly visible corrugated cardboard texture, rectangular box shape with sharp edges, "
        "tan brown color standing out against grey road surface, "
        "casting a distinct shadow on the ground, solid 3D object blocking the road, "
        f"{LIGHTING_SUFFIX}"
    ),
    "rubber_tire": (
        "A single black car wheel tire lying on the road surface, circular rubber tire shape, "
        "clearly visible thick rubber tread pattern and sidewall, "
        "dark black color contrasting with grey asphalt, round tire obstacle on the road, "
        "casting shadow underneath, realistic automotive tire debris, "
        f"{LIGHTING_SUFFIX}"
    ),
    "traffic_cone": (
        "A bright orange traffic cone standing upright on the road, highly visible, "
        "hard plastic texture with slight surface sheen matching ambient lighting, "
        "white reflective stripe wrapping around body, "
        "casting defined shadow on asphalt consistent with scene sun position, "
        f"{LIGHTING_SUFFIX}"
    ),
    "metal_bucket": (
        "A red metal bucket placed upright on the road surface, clearly visible, "
        "metallic paint finish with subtle reflections matching sky and surroundings, "
        "solid cylindrical form with realistic specular highlights, "
        "casting a distinct shadow on asphalt matching scene lighting, "
        f"{LIGHTING_SUFFIX}"
    ),
    "shipping_crate": (
        "A wooden shipping crate positioned on the road, solid 3D object, "
        "brown wood texture with visible grain pattern and weathering, "
        "metal reinforcement bands with subtle reflections, matte wood absorbing light naturally, "
        "realistic shadow cast on pavement matching scene illumination, "
        f"{LIGHTING_SUFFIX}"
    ),
    "plastic_bag": (
        "A white plastic bag placed on the road, crumpled and semi-opaque appearance, "
        "translucent plastic texture with visible creases and folds, "
        "soft diffuse lighting through material, casting soft shadow on asphalt surface, "
        "natural daytime lighting with subtle color pickup from environment, "
        f"{LIGHTING_SUFFIX}"
    ),
    "construction_helmet": (
        "A yellow construction helmet positioned on the road surface, hard plastic shell texture, "
        "glossy finish with realistic highlights matching sky reflection, "
        "subtle environmental color reflection on curved surface, "
        "casting defined shadow on asphalt matching scene light angle, "
        f"{LIGHTING_SUFFIX}"
    ),
    "rock": (
        "A large grey boulder rock sitting on the asphalt road, solid stone obstacle, "
        "rough textured grey stone surface with natural cracks and weathering, "
        "heavy rock clearly visible against the dark road surface, "
        "substantial 3D rock object blocking part of the road, casting shadow, "
        f"{LIGHTING_SUFFIX}"
    ),
    # "broken_bumper": (
    #     "A broken black plastic bumper piece scattered on the road surface, jagged edges visible, "
    #     "realistic matte plastic texture with minor scratches and scuffs, "
    #     "non-reflective damaged automotive material, casting shadow on asphalt, "
    #     "debris appearance with natural wear and dirt, "
    #     f"{LIGHTING_SUFFIX}"
    # ),

    # Wood and construction materials
    "wooden_pallet": (
        "A wooden pallet lying on the road, brown weathered wood texture, "
        "visible horizontal and vertical slats with grain pattern, "
        "matte wood surface absorbing ambient light naturally, metal nails visible, "
        "casting shadow on asphalt matching scene illumination direction, "
        f"{LIGHTING_SUFFIX}"
    ),
    "temporary_road_sign": (
        "A yellow and black temporary road sign standing on the road edge, "
        "plastic or metal construction with appropriate surface reflectivity, "
        "clear reflective text and warning symbol visible, "
        "casting shadow on pavement consistent with scene lighting angle, "
        f"{LIGHTING_SUFFIX}"
    ),

    # Furniture and household items
    "stroller": (
        "A baby stroller positioned on the road, grey or dark colored fabric body, "
        "metal frame with subtle reflections matching environment, "
        "fabric material with appropriate light absorption, wheels grounded on road, "
        "casting realistic shadow on asphalt matching scene illumination, "
        f"{LIGHTING_SUFFIX}"
    ),
    "shopping_cart": (
        "A metal shopping cart placed on the road, silver or grey metallic finish, "
        "wire mesh basket with realistic metal reflections matching sky and surroundings, "
        "wheels and handle detailed, slight rust or wear for realism, "
        "casting shadow on pavement matching scene light direction, "
        f"{LIGHTING_SUFFIX}"
    ),
    "chair": (
        "A single office chair or plastic chair sitting on the asphalt road, "
        "clearly visible four-legged chair with seat and backrest, furniture obstacle on road, "
        "solid chair standing upright on the pavement, blocking the driving lane, "
        "casting shadow on road surface, abandoned chair debris, "
        f"{LIGHTING_SUFFIX}"
    ),

    # Natural obstacles
    "fallen_tree": (
        "A large fallen tree lying across the road, thick trunk with realistic wood grain texture, "
        "bark texture visible with natural weathering and moss, branches extending to sides, "
        "organic material with natural light absorption, leaves with appropriate color, "
        "casting shadow on asphalt matching scene sun position, "
        f"{LIGHTING_SUFFIX}"
    ),

    # Waste and traffic-control equipment
    "dustbin": (
        "A plastic dustbin or garbage can placed on the road, cylindrical or square shape, "
        "typically grey, green, or black color with matte plastic finish, "
        "metallic or plastic lid with appropriate reflectivity, realistic material texture, "
        "casting shadow on asphalt matching scene lighting direction, "
        f"{LIGHTING_SUFFIX}"
    ),
    "delineator": (
        "A flexible plastic delineator post standing on the road, white and red striped pattern, "
        "thin cylindrical form with reflective elements catching ambient light, "
        "typically 30-40cm height, base grounded firmly on road, "
        "casting shadow on pavement matching scene illumination angle, "
        f"{LIGHTING_SUFFIX}"
    ),
    "traffic_drum": (
        "A traffic drum or barrier cylinder standing on the road, bright orange or red color, "
        "plastic construction with slight surface sheen, reflective white stripes, "
        "thick cylindrical form with flat base grounded on road, "
        "casting shadow on asphalt matching scene light direction, "
        f"{LIGHTING_SUFFIX}"
    ),
    "jersey_barrier": (
        "A concrete jersey barrier positioned on the road, grey concrete material, "
        "ribbed profile visible, thick solid construction with rough matte surface, "
        "realistic concrete texture with weathering and subtle stains, "
        "casting long shadow on pavement matching scene sun angle, "
        f"{LIGHTING_SUFFIX}"
    ),
    "plastic_barrier": (
        "A bright orange plastic road barrier standing on the asphalt, "
        "safety barrier with reflective white stripes, construction zone barrier, "
        "plastic barricade clearly visible on the road surface, orange warning color, "
        "traffic safety equipment blocking the lane, casting shadow on ground, "
        f"{LIGHTING_SUFFIX}"
    ),

    # Communication and road-marking equipment
    "arrow_board": (
        "A portable arrow board positioned on the road, large electronic or reflective display, "
        "yellow background with black arrows pointing direction, rectangular form, "
        "metal or plastic stand with appropriate material reflections, "
        "casting shadow on pavement matching scene lighting angle, "
        f"{LIGHTING_SUFFIX}"
    ),
    "message_board": (
        "A traffic control message board placed on the road, electronic display panel, "
        "black or dark background with amber or yellow text/symbols visible, "
        "portable wheeled base or legs grounded on road, metal frame with subtle reflections, "
        "casting shadow matching scene illumination direction, "
        f"{LIGHTING_SUFFIX}"
    ),
    "open_manhole": (
        "An open circular manhole or utility access point on the road surface, "
        "dark cast iron or metal frame with weathered patina, square or circular opening, "
        "shadow cast into the dark opening depth, metal texture with realistic wear, "
        "rim flush with road surface, natural integration with pavement, "
        f"{LIGHTING_SUFFIX}"
    ),
}


# =============================================================================
# Object Mask Specifications
# Defines aspect ratio and relative size for each object type
# =============================================================================
OBJECT_MASK_SPECS: Dict[str, Dict[str, Any]] = {
    # Basic Objects
    "cardboard_box": {
        "aspect_ratio": 1.2,      # slightly wider than tall
        "size_factor": 0.8,       # medium size
        "orientation": "square",
        "description": "Medium cardboard box",
    },
    "rubber_tire": {
        "aspect_ratio": 1.0,      # circular/square when flat
        "size_factor": 1.0,       # standard reference size
        "orientation": "square",
        "description": "Car tire lying flat",
    },
    "traffic_cone": {
        "aspect_ratio": 0.5,      # tall and narrow (height > width)
        "size_factor": 0.4,       # reduced from 0.6 to prevent oversized cones
        "orientation": "vertical",
        "description": "Standing traffic cone",
    },
    "metal_bucket": {
        "aspect_ratio": 0.8,      # slightly taller than wide
        "size_factor": 0.5,       # small object
        "orientation": "vertical",
        "description": "Upright metal bucket",
    },
    "shipping_crate": {
        "aspect_ratio": 1.3,      # wider than tall
        "size_factor": 2.2,       # larger object (increased from 1.5 for better SD results)
        "orientation": "horizontal",
        "description": "Large wooden crate",
    },
    "plastic_bag": {
        "aspect_ratio": 1.0,      # roughly square when crumpled
        "size_factor": 0.4,       # small
        "orientation": "square",
        "description": "Crumpled plastic bag",
    },
    "construction_helmet": {
        "aspect_ratio": 1.2,      # slightly wider
        "size_factor": 0.15,      # reduced from 0.25 to prevent oversized helmets
        "orientation": "square",
        "description": "Hard hat on ground",
    },
    "rock": {
        "aspect_ratio": 1.1,      # slightly irregular
        "size_factor": 0.7,       # medium
        "orientation": "square",
        "description": "Rock or stone",
    },
    "broken_bumper": {
        "aspect_ratio": 2.5,      # long horizontal piece
        "size_factor": 1.0,       # medium-large
        "orientation": "horizontal",
        "description": "Broken car bumper piece",
    },
    
    # Wood and Construction Materials
    "wooden_pallet": {
        "aspect_ratio": 1.2,      # standard pallet ratio
        "size_factor": 1.3,       # large
        "orientation": "horizontal",
        "description": "Wooden pallet",
    },
    "temporary_road_sign": {
        "aspect_ratio": 0.7,      # taller than wide
        "size_factor": 1.0,       # medium-tall
        "orientation": "vertical",
        "description": "Standing road sign",
    },
    
    # Furniture and Household Items
    "stroller": {
        "aspect_ratio": 0.8,      # taller structure
        "size_factor": 1.1,       # medium-large
        "orientation": "vertical",
        "description": "Baby stroller",
    },
    "shopping_cart": {
        "aspect_ratio": 0.9,      # roughly square-ish
        "size_factor": 1.2,       # large
        "orientation": "square",
        "description": "Shopping cart",
    },
    "chair": {
        "aspect_ratio": 0.7,      # taller than wide
        "size_factor": 0.9,       # medium
        "orientation": "vertical",
        "description": "Chair",
    },
    
    # Natural Obstacles
    "fallen_tree": {
        "aspect_ratio": 4.0,      # very wide/long
        "size_factor": 2.5,       # very large
        "orientation": "horizontal",
        "description": "Fallen tree trunk",
    },
    
    # Waste and Control Equipment
    "dustbin": {
        "aspect_ratio": 0.6,      # tall cylinder
        "size_factor": 0.8,       # medium
        "orientation": "vertical",
        "description": "Garbage bin",
    },
    "delineator": {
        "aspect_ratio": 0.25,     # very tall and thin
        "size_factor": 0.15,      # reduced from 0.25 to prevent oversized delineators
        "orientation": "vertical",
        "description": "Flexible delineator post",
    },
    "traffic_drum": {
        "aspect_ratio": 0.7,      # barrel shape
        "size_factor": 0.9,       # medium
        "orientation": "vertical",
        "description": "Traffic barrel",
    },
    "jersey_barrier": {
        "aspect_ratio": 3.0,      # long horizontal barrier
        "size_factor": 2.0,       # large
        "orientation": "horizontal",
        "description": "Concrete barrier",
    },
    "plastic_barrier": {
        "aspect_ratio": 2.0,      # horizontal barrier
        "size_factor": 1.2,       # medium-large
        "orientation": "horizontal",
        "description": "Plastic traffic barrier",
    },
    
    # Communication and Marking Equipment
    "arrow_board": {
        "aspect_ratio": 1.5,      # wide display
        "size_factor": 1.8,       # large
        "orientation": "horizontal",
        "description": "Arrow direction board",
    },
    "message_board": {
        "aspect_ratio": 1.3,      # wide display
        "size_factor": 2.0,       # very large
        "orientation": "horizontal",
        "description": "Electronic message board",
    },
    "open_manhole": {
        "aspect_ratio": 1.0,      # circular
        "size_factor": 0.6,       # medium-small
        "orientation": "square",
        "description": "Open manhole cover",
    },
}


# =============================================================================
# Object Size Constraints
# Real-world dimensions and minimum/maximum pixel limits for depth-based scaling
# IMPORTANT: Minimum pixel sizes must be large enough for SD inpainting to work
# effectively. Recommended minimums: width >= 40px, height >= 50px for most objects.
# =============================================================================
OBJECT_SIZE_CONSTRAINTS: Dict[str, Dict[str, Any]] = {
    # Basic Objects
    "cardboard_box": {
        "real_width_cm": 50,
        "real_height_cm": 40,
        "min_pixel_width": 60,       # Increased for better SD inpainting
        "max_pixel_width": 150,      # Reduced to prevent oversized boxes
        "min_pixel_height": 50,      # Increased for better visibility
        "max_pixel_height": 120,     # Reduced to keep realistic size
    },
    "rubber_tire": {
        "real_width_cm": 70,
        "real_height_cm": 70,
        "min_pixel_width": 70,       # Increased - tires need to be clearly visible
        "max_pixel_width": 160,      # Reduced to prevent oversized tires
        "min_pixel_height": 70,      # Increased for clear visibility
        "max_pixel_height": 160,     # Reduced for realistic size
    },
    "traffic_cone": {
        "real_width_cm": 30,
        "real_height_cm": 75,
        "min_pixel_width": 35,       # Increased from 20 (narrow but visible)
        "max_pixel_width": 70,       # Reduced from 100 - cones are narrow
        "min_pixel_height": 65,      # Increased from 40
        "max_pixel_height": 150,     # Reduced from 200 - prevent oversized
    },
    "metal_bucket": {
        "real_width_cm": 30,
        "real_height_cm": 35,
        "min_pixel_width": 40,       # Increased from 15
        "max_pixel_width": 100,
        "min_pixel_height": 45,      # Increased from 20
        "max_pixel_height": 120,
    },
    "shipping_crate": {
        "real_width_cm": 120,
        "real_height_cm": 80,
        "min_pixel_width": 150,      # Increased from 80 - needs larger mask for SD
        "max_pixel_width": 400,      # Increased from 350
        "min_pixel_height": 100,     # Increased from 55 - needs larger mask for SD
        "max_pixel_height": 300,     # Increased from 250
    },
    "plastic_bag": {
        "real_width_cm": 40,
        "real_height_cm": 40,
        "min_pixel_width": 40,       # Increased from 15
        "max_pixel_width": 100,
        "min_pixel_height": 40,      # Increased from 15
        "max_pixel_height": 100,
    },
    "construction_helmet": {
        "real_width_cm": 25,
        "real_height_cm": 15,
        "min_pixel_width": 40,       # Increased from 15
        "max_pixel_width": 80,
        "min_pixel_height": 30,      # Increased from 10
        "max_pixel_height": 60,
    },
    "rock": {
        "real_width_cm": 50,
        "real_height_cm": 40,
        "min_pixel_width": 60,       # Increased for clear visibility
        "max_pixel_width": 140,      # Reduced to prevent oversized rocks
        "min_pixel_height": 50,      # Increased for clear visibility
        "max_pixel_height": 110,     # Reduced for realistic size
    },
    "broken_bumper": {
        "real_width_cm": 100,
        "real_height_cm": 30,
        "min_pixel_width": 70,       # Increased from 50
        "max_pixel_width": 300,
        "min_pixel_height": 35,      # Increased from 15
        "max_pixel_height": 100,
    },
    
    # Wood and Construction Materials
    "wooden_pallet": {
        "real_width_cm": 120,
        "real_height_cm": 100,
        "min_pixel_width": 70,       # Increased from 50
        "max_pixel_width": 300,
        "min_pixel_height": 60,      # Increased from 40
        "max_pixel_height": 250,
    },
    "temporary_road_sign": {
        "real_width_cm": 60,
        "real_height_cm": 90,
        "min_pixel_width": 45,       # Increased from 30
        "max_pixel_width": 180,
        "min_pixel_height": 65,      # Increased from 50
        "max_pixel_height": 250,
    },
    
    # Furniture and Household Items
    "stroller": {
        "real_width_cm": 60,
        "real_height_cm": 100,
        "min_pixel_width": 50,       # Increased from 30
        "max_pixel_width": 200,
        "min_pixel_height": 70,      # Increased from 50
        "max_pixel_height": 280,
    },
    "shopping_cart": {
        "real_width_cm": 90,
        "real_height_cm": 100,
        "min_pixel_width": 60,       # Increased from 45
        "max_pixel_width": 280,
        "min_pixel_height": 70,      # Increased from 50
        "max_pixel_height": 300,
    },
    "chair": {
        "real_width_cm": 50,
        "real_height_cm": 80,
        "min_pixel_width": 45,       # Increased from 25
        "max_pixel_width": 160,
        "min_pixel_height": 60,      # Increased from 40
        "max_pixel_height": 220,
    },
    
    # Natural Obstacles
    "fallen_tree": {
        "real_width_cm": 400,
        "real_height_cm": 100,
        "min_pixel_width": 180,      # Increased from 150
        "max_pixel_width": 800,
        "min_pixel_height": 55,      # Increased from 40
        "max_pixel_height": 200,
    },
    
    # Waste and Control Equipment
    "dustbin": {
        "real_width_cm": 50,
        "real_height_cm": 90,
        "min_pixel_width": 40,       # Increased from 25
        "max_pixel_width": 150,
        "min_pixel_height": 65,      # Increased from 45
        "max_pixel_height": 250,
    },
    "delineator": {
        "real_width_cm": 15,
        "real_height_cm": 75,
        "min_pixel_width": 12,       # Reduced - delineators are very thin
        "max_pixel_width": 25,       # Reduced from 40 - keep thin
        "min_pixel_height": 45,      # Reduced from 60
        "max_pixel_height": 90,      # Reduced from 150 - prevent oversized
    },
    "traffic_drum": {
        "real_width_cm": 45,
        "real_height_cm": 90,
        "min_pixel_width": 45,       # Increased from 25
        "max_pixel_width": 100,      # Reduced from 140 - drums are compact
        "min_pixel_height": 70,      # Increased from 50
        "max_pixel_height": 180,     # Reduced from 250 - prevent oversized
    },
    "jersey_barrier": {
        "real_width_cm": 300,
        "real_height_cm": 80,
        "min_pixel_width": 120,      # Increased from 100
        "max_pixel_width": 600,
        "min_pixel_height": 50,      # Increased from 30
        "max_pixel_height": 180,
    },
    "plastic_barrier": {
        "real_width_cm": 150,
        "real_height_cm": 80,
        "min_pixel_width": 80,       # Increased from 60
        "max_pixel_width": 400,
        "min_pixel_height": 50,      # Increased from 30
        "max_pixel_height": 200,
    },
    
    # Communication and Marking Equipment
    "arrow_board": {
        "real_width_cm": 150,
        "real_height_cm": 100,
        "min_pixel_width": 100,      # Increased from 80
        "max_pixel_width": 400,
        "min_pixel_height": 70,      # Increased from 50
        "max_pixel_height": 280,
    },
    "message_board": {
        "real_width_cm": 180,
        "real_height_cm": 140,
        "min_pixel_width": 120,      # Increased from 100
        "max_pixel_width": 500,
        "min_pixel_height": 95,      # Increased from 80
        "max_pixel_height": 380,
    },
    "open_manhole": {
        "real_width_cm": 60,
        "real_height_cm": 60,
        "min_pixel_width": 50,       # Increased from 30
        "max_pixel_width": 180,
        "min_pixel_height": 50,      # Increased from 30
        "max_pixel_height": 180,
    },
}


def get_prompt(object_key: str) -> str:
    """Get the inpainting prompt for a specific object."""
    return OBJECT_PROMPTS.get(object_key, "")


def get_all_object_keys() -> list[str]:
    """Get all available object keys."""
    return list(OBJECT_PROMPTS.keys())


def get_negative_prompt() -> str:
    """Get the enhanced negative prompt for better results."""
    return ENHANCED_NEGATIVE_PROMPT


def get_lighting_suffix() -> str:
    """Get the lighting consistency suffix."""
    return LIGHTING_SUFFIX


def get_mask_specs(object_key: str) -> Optional[Dict[str, Any]]:
    """
    Get mask specifications for a specific object type.
    
    Args:
        object_key: Object type key
        
    Returns:
        Dictionary with aspect_ratio, size_factor, orientation, description
        or None if not found
    """
    return OBJECT_MASK_SPECS.get(object_key)


def get_size_constraints(object_key: str) -> Optional[Dict[str, Any]]:
    """
    Get size constraints for a specific object type.
    
    Args:
        object_key: Object type key
        
    Returns:
        Dictionary with real-world dimensions and pixel limits
        or None if not found
    """
    return OBJECT_SIZE_CONSTRAINTS.get(object_key)


def calculate_object_dimensions(
    object_key: str,
    base_width: int = 80,
    base_height: int = 60,
    scale: float = 1.0,
    depth: float = None,
    max_scale_override: float = None,
) -> Tuple[int, int]:
    """
    Calculate object dimensions based on type-specific specifications.
    
    Args:
        object_key: Object type key
        base_width: Base width in pixels
        base_height: Base height in pixels
        scale: Depth-based scale factor
        depth: Optional depth value for additional size constraints
        max_scale_override: Optional maximum scale to apply
        
    Returns:
        Tuple of (width, height) in pixels
    """
    specs = get_mask_specs(object_key)
    constraints = get_size_constraints(object_key)
    
    # Apply maximum scale limit to prevent oversized objects
    # Objects far away (high depth) should be smaller
    if max_scale_override is not None:
        scale = min(scale, max_scale_override)
    elif depth is not None:
        # Depth-aware scaling: more aggressive limit for nearby objects (low depth)
        # to prevent unnaturally large objects at close range
        # At depth 5m, max scale = 0.6; at depth 15m, max scale = 1.0
        depth_max_scale = 0.6 + (depth - 5.0) * 0.04  # 0.6 at 5m, ~1.0 at 15m
        depth_max_scale = max(0.5, min(depth_max_scale, 1.2))
        scale = min(scale, depth_max_scale)
    
    if specs is None:
        # Fallback to base dimensions
        return int(base_width * scale), int(base_height * scale)
    
    # Apply size factor
    size_factor = specs.get("size_factor", 1.0)
    aspect_ratio = specs.get("aspect_ratio", 1.0)
    
    # Calculate dimensions
    width = int(base_width * size_factor * scale)
    height = int(width / aspect_ratio)
    
    # Apply constraints if available
    if constraints:
        width = max(constraints["min_pixel_width"], 
                   min(width, constraints["max_pixel_width"]))
        height = max(constraints["min_pixel_height"], 
                    min(height, constraints["max_pixel_height"]))
    
    return width, height


def get_object_orientation(object_key: str) -> str:
    """
    Get the preferred orientation for an object.
    
    Args:
        object_key: Object type key
        
    Returns:
        Orientation string: 'horizontal', 'vertical', or 'square'
    """
    specs = get_mask_specs(object_key)
    if specs:
        return specs.get("orientation", "square")
    return "square"

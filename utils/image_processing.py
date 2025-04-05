"""
Image processing utilities for the anime character evolution system.
"""

import os
import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter, ImageOps
import logging
import io
import base64
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)

def resize_image(
    image: Image.Image, 
    width: int = None, 
    height: int = None, 
    maintain_aspect: bool = True
) -> Image.Image:
    """
    Resize an image to the specified dimensions.
    
    Args:
        image: PIL Image to resize
        width: Target width (if None, will be calculated from height)
        height: Target height (if None, will be calculated from width)
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized PIL Image
    """
    if image is None:
        raise ValueError("No image provided")
        
    if width is None and height is None:
        return image.copy()
        
    orig_width, orig_height = image.size
    
    if maintain_aspect:
        if width is None:
            # Calculate width from height
            width = int(orig_width * (height / orig_height))
        elif height is None:
            # Calculate height from width
            height = int(orig_height * (width / orig_width))
        else:
            # Both dimensions provided, maintain aspect by fitting within bounds
            ratio = min(width / orig_width, height / orig_height)
            width = int(orig_width * ratio)
            height = int(orig_height * ratio)
    else:
        # Use provided dimensions or original if not provided
        if width is None:
            width = orig_width
        if height is None:
            height = orig_height
    
    return image.resize((width, height), Image.LANCZOS)

def crop_image(
    image: Image.Image,
    crop_type: str = 'center',
    width_ratio: float = 1.0,
    height_ratio: float = 1.0
) -> Image.Image:
    """
    Crop an image based on various strategies.
    
    Args:
        image: PIL Image to crop
        crop_type: Type of crop ('center', 'face', 'square', 'portrait', 'landscape')
        width_ratio: Width ratio for custom crop (0.0-1.0)
        height_ratio: Height ratio for custom crop (0.0-1.0)
        
    Returns:
        Cropped PIL Image
    """
    if image is None:
        raise ValueError("No image provided")
        
    orig_width, orig_height = image.size
    
    if crop_type == 'center':
        # Determine the minimum dimension and create a square crop
        min_dim = min(orig_width, orig_height)
        left = (orig_width - min_dim) // 2
        top = (orig_height - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim
        
    elif crop_type == 'square':
        # Similar to center but explicitly creating a square
        min_dim = min(orig_width, orig_height)
        left = (orig_width - min_dim) // 2
        top = (orig_height - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim
        
    elif crop_type == 'portrait':
        # Create a portrait-oriented crop (3:4 aspect ratio)
        if orig_width / orig_height > 0.75:  # If image is wider than 3:4
            new_width = int(orig_height * 0.75)
            left = (orig_width - new_width) // 2
            top = 0
            right = left + new_width
            bottom = orig_height
        else:  # Image is already narrower than 3:4
            return image.copy()
            
    elif crop_type == 'landscape':
        # Create a landscape-oriented crop (4:3 aspect ratio)
        if orig_width / orig_height < 1.3333:  # If image is taller than 4:3
            new_height = int(orig_width * 0.75)
            left = 0
            top = (orig_height - new_height) // 2
            right = orig_width
            bottom = top + new_height
        else:  # Image is already wider than 4:3
            return image.copy()
            
    elif crop_type == 'face':
        # This would ideally use face detection, but for now just crop the upper part
        # In a real system, this would use a proper face detection model
        left = int(orig_width * 0.2)
        top = int(orig_height * 0.1)
        right = int(orig_width * 0.8)
        bottom = int(orig_height * 0.6)
        
    else:  # Custom crop based on ratios
        # width_ratio and height_ratio determine the size of the crop
        # 1.0 means full width/height, 0.5 means half width/height
        crop_width = int(orig_width * width_ratio)
        crop_height = int(orig_height * height_ratio)
        
        # Center the crop
        left = (orig_width - crop_width) // 2
        top = (orig_height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height
    
    return image.crop((left, top, right, bottom))

def add_overlay_text(
    image: Image.Image,
    text: str,
    position: str = 'bottom',
    font_size: int = 20,
    font_color: Tuple[int, int, int] = (255, 255, 255),
    background_color: Optional[Tuple[int, int, int, int]] = (0, 0, 0, 128),
    padding: int = 10
) -> Image.Image:
    """
    Add text overlay to an image.
    
    Args:
        image: PIL Image to add text to
        text: Text to overlay
        position: Text position ('top', 'bottom', 'center', 'top-left', etc.)
        font_size: Font size in pixels
        font_color: RGB tuple for font color
        background_color: RGBA tuple for text background (None for transparent)
        padding: Padding around text in pixels
        
    Returns:
        PIL Image with text overlay
    """
    if image is None:
        raise ValueError("No image provided")
        
    if not text:
        return image.copy()
        
    # Create a copy to avoid modifying the original
    result = image.copy()
    width, height = result.size
    
    # Create a draw object
    draw = ImageDraw.Draw(result)
    
    # Try to find a font, use default if not found
    try:
        font_path = os.path.join("data", "fonts", "arial.ttf")
        font = ImageFont.truetype(font_path, size=font_size)
    except Exception:
        font = ImageFont.load_default()
    
    # Calculate text size
    if hasattr(draw, 'textbbox'):
        # Newer PIL versions
        _, _, text_width, text_height = draw.textbbox((0, 0), text, font=font)
    else:
        # Older PIL versions
        text_width, text_height = draw.textsize(text, font=font)
    
    # Determine position
    if position == 'top':
        x = (width - text_width) // 2
        y = padding
    elif position == 'bottom':
        x = (width - text_width) // 2
        y = height - text_height - padding
    elif position == 'center':
        x = (width - text_width) // 2
        y = (height - text_height) // 2
    elif position == 'top-left':
        x = padding
        y = padding
    elif position == 'top-right':
        x = width - text_width - padding
        y = padding
    elif position == 'bottom-left':
        x = padding
        y = height - text_height - padding
    elif position == 'bottom-right':
        x = width - text_width - padding
        y = height - text_height - padding
    else:
        # Default to bottom
        x = (width - text_width) // 2
        y = height - text_height - padding
    
    # Draw background box if background color is provided
    if background_color:
        draw.rectangle(
            [(x - padding, y - padding), (x + text_width + padding, y + text_height + padding)],
            fill=background_color
        )
    
    # Draw text
    draw.text((x, y), text, font=font, fill=font_color)
    
    return result

def create_image_grid(
    images: List[Image.Image],
    rows: int = None,
    cols: int = None,
    padding: int = 5,
    background_color: Tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    """
    Create a grid of images.
    
    Args:
        images: List of PIL Images to arrange in a grid
        rows: Number of rows (calculated if None)
        cols: Number of columns (calculated if None)
        padding: Padding between images in pixels
        background_color: RGB tuple for background color
        
    Returns:
        PIL Image containing the grid
    """
    if not images:
        raise ValueError("No images provided")
        
    num_images = len(images)
    
    # Determine grid dimensions if not provided
    if rows is None and cols is None:
        # Default to square-ish grid
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))
    elif rows is None:
        # Calculate rows from columns
        rows = int(np.ceil(num_images / cols))
    elif cols is None:
        # Calculate columns from rows
        cols = int(np.ceil(num_images / rows))
    
    # Determine cell size (use maximum dimensions)
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)
    
    # Create the grid image
    grid_width = cols * max_width + (cols - 1) * padding
    grid_height = rows * max_height + (rows - 1) * padding
    
    grid = Image.new('RGB', (grid_width, grid_height), background_color)
    
    # Place images in the grid
    for i, img in enumerate(images):
        if i >= rows * cols:
            break
            
        row = i // cols
        col = i % cols
        
        x = col * (max_width + padding)
        y = row * (max_height + padding)
        
        # Center the image in its cell if smaller than max dimensions
        x_offset = (max_width - img.width) // 2
        y_offset = (max_height - img.height) // 2
        
        grid.paste(img, (x + x_offset, y + y_offset))
    
    return grid

def apply_image_effects(
    image: Image.Image,
    effects: List[str] = None,
    params: Dict[str, Any] = None
) -> Image.Image:
    """
    Apply a series of effects to an image.
    
    Args:
        image: PIL Image to process
        effects: List of effect names to apply
        params: Dictionary of effect parameters
        
    Returns:
        Processed PIL Image
    """
    if image is None:
        raise ValueError("No image provided")
        
    if not effects:
        return image.copy()
        
    # Initialize parameters dictionary
    if params is None:
        params = {}
        
    # Create a copy to avoid modifying the original
    result = image.copy()
    
    # Apply each effect in sequence
    for effect in effects:
        if effect == 'grayscale':
            result = ImageOps.grayscale(result)
            result = result.convert('RGB')  # Convert back to RGB
            
        elif effect == 'sepia':
            # Create sepia effect
            sepia_data = np.array([
                [ 0.393, 0.769, 0.189],
                [ 0.349, 0.686, 0.168],
                [ 0.272, 0.534, 0.131]
            ])
            
            # Convert to numpy array
            data = np.array(result)
            
            # Apply sepia matrix
            sepia_img = np.dot(data, sepia_data.T)
            
            # Clip values to valid range
            sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
            
            # Convert back to PIL Image
            result = Image.fromarray(sepia_img)
            
        elif effect == 'invert':
            result = ImageOps.invert(result)
            
        elif effect == 'blur':
            radius = params.get('blur_radius', 2)
            result = result.filter(ImageFilter.GaussianBlur(radius=radius))
            
        elif effect == 'sharpen':
            factor = params.get('sharpen_factor', 2)
            # Apply multiple passes for stronger effect
            for _ in range(factor):
                result = result.filter(ImageFilter.SHARPEN)
                
        elif effect == 'brightness':
            factor = params.get('brightness_factor', 1.5)
            result = ImageEnhance.Brightness(result).enhance(factor)
            
        elif effect == 'contrast':
            factor = params.get('contrast_factor', 1.5)
            result = ImageEnhance.Contrast(result).enhance(factor)
            
        elif effect == 'color':
            factor = params.get('color_factor', 1.5)
            result = ImageEnhance.Color(result).enhance(factor)
            
        elif effect == 'edge_enhance':
            result = result.filter(ImageFilter.EDGE_ENHANCE)
            
        elif effect == 'posterize':
            bits = params.get('posterize_bits', 4)
            result = ImageOps.posterize(result, bits)
            
        elif effect == 'solarize':
            threshold = params.get('solarize_threshold', 128)
            result = ImageOps.solarize(result, threshold)
            
        elif effect == 'equalize':
            result = ImageOps.equalize(result)
            
        elif effect == 'auto_contrast':
            cutoff = params.get('auto_contrast_cutoff', 0)
            result = ImageOps.autocontrast(result, cutoff)
    
    return result

def image_to_base64(image: Image.Image, format: str = 'PNG') -> str:
    """
    Convert a PIL Image to a base64 string.
    
    Args:
        image: PIL Image to convert
        format: Image format to use
        
    Returns:
        Base64-encoded string representation
    """
    if image is None:
        raise ValueError("No image provided")
        
    # Convert image to bytes
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return f"data:image/{format.lower()};base64,{img_str}"

def base64_to_image(base64_str: str) -> Image.Image:
    """
    Convert a base64 string to a PIL Image.
    
    Args:
        base64_str: Base64-encoded image string
        
    Returns:
        PIL Image
    """
    if not base64_str:
        raise ValueError("No base64 string provided")
        
    # Remove header if present
    if ',' in base64_str:
        base64_str = base64_str.split(',', 1)[1]
        
    # Decode base64 to bytes
    img_data = base64.b64decode(base64_str)
    
    # Convert bytes to PIL Image
    buffered = io.BytesIO(img_data)
    img = Image.open(buffered)
    
    return img

def enhance_image_quality(
    image: Image.Image,
    sharpen: bool = True,
    contrast: bool = True,
    color: bool = True
) -> Image.Image:
    """
    Enhance image quality through various adjustments.
    
    Args:
        image: PIL Image to enhance
        sharpen: Whether to apply sharpening
        contrast: Whether to enhance contrast
        color: Whether to enhance color
        
    Returns:
        Enhanced PIL Image
    """
    if image is None:
        raise ValueError("No image provided")
        
    # Create a copy to avoid modifying the original
    result = image.copy()
    
    # Apply enhancements
    if sharpen:
        result = result.filter(ImageFilter.SHARPEN)
        
    if contrast:
        result = ImageEnhance.Contrast(result).enhance(1.2)
        
    if color:
        result = ImageEnhance.Color(result).enhance(1.2)
    
    return result

def add_watermark(
    image: Image.Image,
    text: str = "Anime Evolution",
    position: str = 'bottom-right',
    opacity: float = 0.5,
    font_size: int = 20
) -> Image.Image:
    """
    Add a watermark to an image.
    
    Args:
        image: PIL Image to watermark
        text: Watermark text
        position: Watermark position
        opacity: Watermark opacity (0-1)
        font_size: Font size for text watermark
        
    Returns:
        Watermarked PIL Image
    """
    if image is None:
        raise ValueError("No image provided")
        
    if not text:
        return image.copy()
        
    # Create a copy to avoid modifying the original
    result = image.copy()
    width, height = result.size
    
    # Create a transparent overlay for the watermark
    overlay = Image.new('RGBA', result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Try to find a font, use default if not found
    try:
        font_path = os.path.join("data", "fonts", "arial.ttf")
        font = ImageFont.truetype(font_path, size=font_size)
    except Exception:
        font = ImageFont.load_default()
    
    # Calculate text size
    if hasattr(draw, 'textbbox'):
        # Newer PIL versions
        _, _, text_width, text_height = draw.textbbox((0, 0), text, font=font)
    else:
        # Older PIL versions
        text_width, text_height = draw.textsize(text, font=font)
    
    # Determine position
    padding = 10
    if position == 'top-left':
        x, y = padding, padding
    elif position == 'top-right':
        x, y = width - text_width - padding, padding
    elif position == 'bottom-left':
        x, y = padding, height - text_height - padding
    elif position == 'bottom-right':
        x, y = width - text_width - padding, height - text_height - padding
    elif position == 'center':
        x, y = (width - text_width) // 2, (height - text_height) // 2
    else:
        # Default to bottom-right
        x, y = width - text_width - padding, height - text_height - padding
    
    # Draw text with shadow for better visibility
    draw.text((x+2, y+2), text, font=font, fill=(0, 0, 0, int(255 * opacity)))
    draw.text((x, y), text, font=font, fill=(255, 255, 255, int(255 * opacity)))
    
    # Composite the text overlay with the image
    if result.mode != 'RGBA':
        result = result.convert('RGBA')
        
    result = Image.alpha_composite(result, overlay)
    result = result.convert('RGB')  # Convert back to RGB
    
    return result

def save_image_with_metadata(
    image: Image.Image,
    filepath: str,
    metadata: Dict[str, Any],
    format: str = None
) -> bool:
    """
    Save an image with metadata.
    
    Args:
        image: PIL Image to save
        filepath: Path to save the image
        metadata: Dictionary of metadata to embed
        format: Image format override (determined from extension if None)
        
    Returns:
        True if successful, False otherwise
    """
    if image is None:
        raise ValueError("No image provided")
        
    if not filepath:
        raise ValueError("No filepath provided")
        
    try:
        # Determine format from extension if not provided
        if format is None:
            format = os.path.splitext(filepath)[1][1:].upper()
            if not format:
                format = 'PNG'
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Create a copy to avoid modifying the original
        result = image.copy()
        
        # Add metadata
        if hasattr(result, 'info'):
            # Convert all metadata values to strings
            for key, value in metadata.items():
                if isinstance(value, (dict, list)):
                    import json
                    result.info[key] = json.dumps(value)
                else:
                    result.info[key] = str(value)
        
        # Save the image
        result.save(filepath, format=format)
        return True
    except Exception as e:
        logger.error(f"Error saving image with metadata: {e}")
        return False
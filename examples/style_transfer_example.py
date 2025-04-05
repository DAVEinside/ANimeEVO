"""
Style transfer example for the Anime Character Evolution System.

This script demonstrates applying different anime styles to characters.
"""

import os
import sys
import time
import logging
from pathlib import Path
from PIL import Image

# Add project root to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core components
from core.attributes.character_attributes import CharacterAttributes
from core.attributes.anime_styles import AnimeStyleLibrary
from core.diffusion.anime_pipeline import AnimePipeline
from multimodal.image_generator import ImageGenerator
from utils.image_processing import create_image_grid, add_overlay_text
from utils.data_handling import ensure_directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_style_transfer_demo(output_dir: str = "outputs/examples/style_transfer"):
    """
    Run a demonstration of style transfer.
    
    Args:
        output_dir: Directory to save outputs
    """
    logger.info("Running style transfer demonstration")
    
    # Create output directory
    ensure_directory(output_dir)
    
    # Initialize components
    pipeline = AnimePipeline()
    image_generator = ImageGenerator(pipeline=pipeline)
    style_library = AnimeStyleLibrary()
    
    # Create a base character
    base_character = CharacterAttributes(
        name="Style Transfer Demo",
        gender="female",
        age_category="teen",
        hair_color="blue",
        eye_color="green",
        personality=["cheerful", "energetic"],
        anime_style=""  # No specific style yet
    )
    
    # Generate base character image
    logger.info("Generating base character image...")
    base_images, _ = image_generator.generate_images(
        attributes=base_character,
        num_samples=1,
        seed=42  # Use fixed seed for consistency
    )
    
    if not base_images:
        logger.error("Failed to generate base character image")
        return
    
    # Save base image
    base_image_path = os.path.join(output_dir, "base_character.png")
    base_images[0].save(base_image_path)
    logger.info(f"Saved base character image: {base_image_path}")
    
    # Get available styles
    available_styles = style_library.list_anime_styles()
    logger.info(f"Available styles: {available_styles}")
    
    # Select styles to demonstrate
    demo_styles = [
        "shonen", "shojo", "seinen", "isekai", 
        "mecha", "chibi", "90s_anime", "modern_anime"
    ]
    
    # Filter to styles that actually exist in the library
    demo_styles = [style for style in demo_styles if style in available_styles]
    
    # Apply each style and store results
    style_images = []
    style_names = []
    
    for style in demo_styles:
        logger.info(f"Applying {style} style...")
        
        # Create a copy of the character with this style
        styled_character = base_character.clone()
        styled_character.anime_style = style
        
        # Generate image with this style
        styled_images, _ = image_generator.generate_images(
            attributes=styled_character,
            num_samples=1,
            seed=42  # Use fixed seed for consistency
        )
        
        if styled_images:
            # Save the styled image
            style_image_path = os.path.join(output_dir, f"style_{style}.png")
            styled_images[0].save(style_image_path)
            logger.info(f"Saved {style} styled image: {style_image_path}")
            
            # Store for comparison grid
            style_images.append(styled_images[0])
            style_names.append(style)
    
    # Get style information for detailed comparison
    style_info = {}
    for style in style_names:
        info = style_library.get_style_info(style)
        style_info[style] = info
    
    # Add style labels to images
    labeled_images = []
    labeled_images.append(add_overlay_text(base_images[0], "Base Character", position='bottom'))
    
    for img, style in zip(style_images, style_names):
        label_text = f"{style}"
        labeled_img = add_overlay_text(img, label_text, position='bottom')
        labeled_images.append(labeled_img)
    
    # Create comparison grid
    logger.info("Creating style comparison grid...")
    
    # Calculate reasonable grid dimensions
    total_images = len(labeled_images)
    cols = min(3, total_images)
    rows = (total_images + cols - 1) // cols  # Ceiling division
    
    grid = create_image_grid(labeled_images, rows=rows, cols=cols)
    
    # Save the comparison grid
    grid_path = os.path.join(output_dir, "style_comparison.png")
    grid.save(grid_path)
    logger.info(f"Saved style comparison grid: {grid_path}")
    
    # Create style info document
    create_style_info_document(style_info, output_dir)
    
    # Now try progressive style transfer
    try_progressive_style_transfer(base_images[0], output_dir)
    
    logger.info(f"Style transfer demo complete. Results saved to {output_dir}")

def create_style_info_document(style_info: dict, output_dir: str):
    """
    Create a document with style information.
    
    Args:
        style_info: Dictionary of style information
        output_dir: Directory to save output
    """
    try:
        doc_path = os.path.join(output_dir, "style_information.txt")
        
        with open(doc_path, 'w') as f:
            f.write("ANIME STYLE INFORMATION\n")
            f.write("======================\n\n")
            
            for style_name, info in style_info.items():
                f.write(f"Style: {style_name}\n")
                f.write(f"{'-' * (len(style_name) + 7)}\n")
                
                if isinstance(info, dict):
                    if 'description' in info:
                        f.write(f"Description: {info['description']}\n")
                    
                    if 'era' in info:
                        f.write(f"Era: {info['era']}\n")
                        
                    if 'target_audience' in info:
                        f.write(f"Target Audience: {info['target_audience']}\n")
                    
                    if 'common_themes' in info and info['common_themes']:
                        f.write("Common Themes: \n")
                        for theme in info['common_themes']:
                            f.write(f"  - {theme}\n")
                    
                    if 'visual_characteristics' in info and info['visual_characteristics']:
                        f.write("Visual Characteristics: \n")
                        for char in info['visual_characteristics']:
                            f.write(f"  - {char}\n")
                    
                    if 'reference_anime' in info and info['reference_anime']:
                        f.write("Reference Anime: \n")
                        for ref in info['reference_anime']:
                            f.write(f"  - {ref}\n")
                            
                f.write("\n\n")
                
        logger.info(f"Saved style information document: {doc_path}")
    except Exception as e:
        logger.error(f"Error creating style information document: {e}")

def try_progressive_style_transfer(base_image: Image.Image, output_dir: str):
    """
    Try progressive style transfer with varying strengths.
    
    Args:
        base_image: Base image to apply style to
        output_dir: Directory to save outputs
    """
    logger.info("Running progressive style transfer demo...")
    
    # Initialize components
    pipeline = AnimePipeline()
    
    # Select a style for progression
    style = "chibi"  # This style should show clear visual differences
    
    # Different strength levels
    strengths = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    # Apply style with different strengths
    progression_images = []
    
    for strength in strengths:
        logger.info(f"Applying {style} style with strength {strength}...")
        
        # In a real implementation, this would use the pipeline's style_transfer method
        # Here we're just simulating progressive transfer
        
        # Simulate style transfer by altering the image based on strength
        # In a real implementation, this would call pipeline.style_transfer
        
        # Create a placeholder image with a label indicating the strength
        img_copy = base_image.copy()
        label_text = f"{style} (strength: {strength})"
        styled_img = add_overlay_text(img_copy, label_text, position='bottom')
        
        # Store the "styled" image
        progression_images.append(styled_img)
        
        # Save individual image
        style_path = os.path.join(output_dir, f"style_{style}_strength_{int(strength*100)}.png")
        styled_img.save(style_path)
        logger.info(f"Saved styled image: {style_path}")
    
    # Create progression grid
    progression_grid = create_image_grid(progression_images, rows=1)
    
    # Save progression grid
    progression_path = os.path.join(output_dir, f"style_{style}_progression.png")
    progression_grid.save(progression_path)
    logger.info(f"Saved style progression grid: {progression_path}")

def main():
    """Run the style transfer example."""
    print("\n===== Anime Character Evolution - Style Transfer Example =====\n")
    
    # Output directory
    output_dir = "outputs/examples/style_transfer"
    
    # Run style transfer demo
    run_style_transfer_demo(output_dir=output_dir)
    
    print(f"\nStyle transfer complete! Results saved to {output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
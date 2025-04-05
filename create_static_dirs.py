"""
Script to create static directory structure for the web interface
"""

import os
from pathlib import Path

def create_static_dirs():
    """Create static directory structure for the web interface."""
    print("Creating static directory structure...")
    
    # Create static directories
    static_dirs = [
        "interface/static",
        "interface/static/css",
        "interface/static/js",
        "interface/static/img",
        "interface/static/fonts"
    ]
    
    for dir_path in static_dirs:
        path = Path(dir_path)
        if not path.exists():
            print(f"Creating directory: {path}")
            path.mkdir(parents=True, exist_ok=True)
    
    # Create placeholder files if needed
    placeholder_img = Path("interface/static/img/hero-image.png")
    if not placeholder_img.exists():
        print(f"Creating placeholder hero image: {placeholder_img}")
        # Create a simple colored rectangle as placeholder
        try:
            from PIL import Image, ImageDraw
            img = Image.new('RGB', (800, 400), color=(106, 66, 193))
            draw = ImageDraw.Draw(img)
            draw.text((400, 200), "Anime Character\nEvolution System", fill=(255, 255, 255), anchor="mm")
            img.save(placeholder_img)
        except ImportError:
            print("PIL not available, creating empty file")
            placeholder_img.touch()
    
    print("Static directory structure created!")

if __name__ == "__main__":
    create_static_dirs()
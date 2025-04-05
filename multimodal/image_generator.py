"""
Image generator for anime characters using diffusion models.
Handles generating, saving, and processing character images.
"""

import os
import uuid
import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Any
from pathlib import Path
import logging
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance
import yaml

from core.diffusion.anime_pipeline import AnimePipeline
from core.diffusion.diffusion_model import DiffusionOutput
from core.attributes.character_attributes import CharacterAttributes

# Setup logging
logger = logging.getLogger(__name__)

class ImageGenerator:
    """
    Generates and manages anime character images.
    """
    
    def __init__(
        self,
        config_path: str = "config/config.yaml",
        pipeline: AnimePipeline = None,
        output_dir: str = None
    ):
        """
        Initialize the image generator.
        
        Args:
            config_path: Path to configuration file
            pipeline: Existing AnimePipeline instance (optional)
            output_dir: Directory for saving outputs (overrides config)
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set output directory
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = self.config['paths']['output_dir']
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.image_dir = os.path.join(self.output_dir, "images")
        Path(self.image_dir).mkdir(exist_ok=True)
        
        self.history_dir = os.path.join(self.output_dir, "history")
        Path(self.history_dir).mkdir(exist_ok=True)
        
        # Initialize pipeline
        self.pipeline = pipeline
        if self.pipeline is None:
            logger.info("Initializing new AnimePipeline")
            self.pipeline = AnimePipeline(config_path=config_path)
            
        # Image generation settings
        self.image_settings = self.config['output']['image']
        
        # Load available styles
        self.available_styles = []
        if 'style_transfer' in self.config:
            styles = self.config['style_transfer'].get('styles', [])
            self.available_styles = [style['name'] for style in styles]
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.warning(f"Error loading config from {config_path}: {e}")
            logger.warning("Using default configuration.")
            return {
                'output': {
                    'image': {
                        'formats': ['png'],
                        'resolution': [512, 512],
                        'high_res_multiple': 2,
                        'samples_per_character': 4,
                    }
                },
                'paths': {
                    'output_dir': './outputs',
                }
            }
    
    def generate_images(
        self,
        attributes: CharacterAttributes,
        num_samples: int = None,
        template_key: str = "default",
        custom_prompt: str = None,
        custom_negative_prompt: str = None,
        apply_style: str = None,
        seed: int = -1,
        **kwargs
    ) -> Tuple[List[Image.Image], DiffusionOutput]:
        """
        Generate images for a character.
        
        Args:
            attributes: Character attributes
            num_samples: Number of images to generate
            template_key: Prompt template to use
            custom_prompt: Optional custom prompt
            custom_negative_prompt: Optional custom negative prompt
            apply_style: Style to apply (if any)
            seed: Random seed for generation
            **kwargs: Additional parameters for diffusion
            
        Returns:
            Tuple of (list of images, diffusion output object)
        """
        # Set number of samples if not provided
        if num_samples is None:
            num_samples = self.image_settings['samples_per_character']
            
        # Set resolution from config if not in kwargs
        if 'width' not in kwargs and 'height' not in kwargs:
            width, height = self.image_settings['resolution']
            kwargs['width'] = width
            kwargs['height'] = height
            
        logger.info(f"Generating {num_samples} images for character {attributes.id}")
        
        # Check if style is valid
        if apply_style and apply_style not in self.available_styles:
            logger.warning(f"Style '{apply_style}' not found. Using default style.")
            apply_style = None
            
        # Generate images
        output = self.pipeline.generate(
            attributes=attributes,
            template_key=template_key,
            custom_prompt=custom_prompt,
            custom_negative_prompt=custom_negative_prompt,
            num_images=num_samples,
            apply_style=apply_style,
            seed=seed,
            **kwargs
        )
        
        return output.images, output
    
    def save_images(
        self,
        images: List[Image.Image],
        character: CharacterAttributes,
        output: DiffusionOutput = None,
        prefix: str = "character",
        add_metadata: bool = True,
        add_caption: bool = False
    ) -> List[str]:
        """
        Save generated images to disk.
        
        Args:
            images: List of generated images
            character: Character attributes
            output: DiffusionOutput object with generation metadata
            prefix: Filename prefix
            add_metadata: Whether to add generation metadata to images
            add_caption: Whether to add character caption to images
            
        Returns:
            List of saved file paths
        """
        saved_paths = []
        
        # Create character-specific directory
        char_name = character.name.lower().replace(" ", "_") if character.name else "unnamed"
        char_dir = os.path.join(self.image_dir, f"{char_name}_{character.id[:8]}")
        Path(char_dir).mkdir(exist_ok=True)
        
        # Get image format
        image_format = self.image_settings['formats'][0]  # Use first format
        
        # Save each image
        for i, img in enumerate(images):
            # Create a copy to avoid modifying original
            image = img.copy()
            
            # Add caption if requested
            if add_caption:
                image = self._add_caption(image, character)
                
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{character.id[:8]}_{timestamp}_{i}.{image_format}"
            filepath = os.path.join(char_dir, filename)
            
            # Add metadata if requested
            if add_metadata and output:
                image = self._add_metadata(image, character, output)
                
            # Save the image
            image.save(filepath)
            saved_paths.append(filepath)
            
            logger.info(f"Saved image to {filepath}")
            
        return saved_paths
    
    def _add_metadata(self, image: Image.Image, character: CharacterAttributes, output: DiffusionOutput) -> Image.Image:
        """Add generation metadata to image EXIF/metadata."""
        try:
            # Create metadata dict
            metadata = {
                "prompt": output.prompt,
                "negative_prompt": output.negative_prompt,
                "character_id": character.id,
                "seed": output.seed,
                "steps": output.parameters.get("steps", "unknown"),
                "sampler": output.parameters.get("sampler", "unknown"),
                "guidance_scale": output.parameters.get("guidance_scale", "unknown"),
                "model": output.parameters.get("model_id", "unknown"),
            }
            
            # Add character attributes metadata
            for attr in ["name", "gender", "hair_color", "eye_color", "anime_style"]:
                if hasattr(character, attr):
                    value = getattr(character, attr)
                    if value:
                        metadata[f"character_{attr}"] = value
            
            # Convert to EXIF-compatible format
            exif_data = image.getexif() if hasattr(image, "getexif") else {}
            
            # Add metadata to EXIF UserComment
            try:
                import json
                exif_data[0x9286] = json.dumps(metadata).encode()
            except:
                pass
                
            # Set EXIF data
            if hasattr(image, "info"):
                for k, v in metadata.items():
                    if isinstance(v, (str, int, float)):
                        image.info[f"anime:{k}"] = str(v)
            
            return image
        except Exception as e:
            logger.error(f"Error adding metadata: {e}")
            return image
    
    def _add_caption(self, image: Image.Image, character: CharacterAttributes) -> Image.Image:
        """Add character information caption to the image."""
        try:
            # Create a new image with extra space for caption
            width, height = image.size
            caption_height = int(height * 0.1)  # 10% of image height for caption
            new_img = Image.new("RGB", (width, height + caption_height), (255, 255, 255))
            new_img.paste(image, (0, 0))
            
            # Create caption text
            caption = []
            if character.name:
                caption.append(character.name)
            caption.append(f"{character.gender}, {character.age_category}")
            caption.append(f"{character.hair_color} hair, {character.eye_color} eyes")
            if character.anime_style:
                caption.append(f"{character.anime_style} style")
                
            caption_text = " | ".join(caption)
            
            # Draw the caption
            draw = ImageDraw.Draw(new_img)
            
            # Try to find a font, use default if not found
            try:
                font_path = os.path.join("data", "fonts", "arial.ttf")
                font = ImageFont.truetype(font_path, size=caption_height // 3)
            except:
                font = ImageFont.load_default()
                
            # Draw text with shadow for readability
            text_x = width // 2
            text_y = height + (caption_height // 2)
            
            # Draw shadow
            draw.text((text_x+1, text_y+1), caption_text, fill=(0, 0, 0), font=font, anchor="mm")
            # Draw text
            draw.text((text_x, text_y), caption_text, fill=(255, 255, 255), font=font, anchor="mm")
            
            return new_img
            
        except Exception as e:
            logger.error(f"Error adding caption: {e}")
            return image
    
    def generate_high_res(
        self,
        character: CharacterAttributes,
        reference_image: Optional[Image.Image] = None,
        scale_factor: float = None,
        **kwargs
    ) -> Tuple[Image.Image, DiffusionOutput]:
        """
        Generate a high-resolution version of a character.
        
        Args:
            character: Character attributes
            reference_image: Optional reference image for img2img
            scale_factor: Resolution scale factor (default from config)
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (high-res image, diffusion output)
        """
        # Set scale factor from config if not provided
        if scale_factor is None:
            scale_factor = self.image_settings.get('high_res_multiple', 2)
            
        # Get base resolution
        base_width, base_height = self.image_settings['resolution']
        
        # Calculate high-res dimensions
        high_res_width = int(base_width * scale_factor)
        high_res_height = int(base_height * scale_factor)
        
        logger.info(f"Generating high-res image ({high_res_width}x{high_res_height}) for character {character.id}")
        
        if reference_image:
            # Use img2img for better consistency with reference
            if hasattr(self.pipeline.diffusion.pipeline, "img2img"):
                # Scale up reference image
                ref_scaled = reference_image.resize((high_res_width, high_res_height), Image.LANCZOS)
                
                # Generate prompt from character
                prompt = self.pipeline._build_prompt(character)
                
                # Use img2img with low strength to preserve reference details
                output = self.pipeline.diffusion.pipeline.img2img(
                    prompt=prompt,
                    negative_prompt=self.pipeline.negative_prompt,
                    image=ref_scaled,
                    strength=0.3,  # Low strength to preserve details
                    width=high_res_width,
                    height=high_res_height,
                    **kwargs
                )
                
                return output.images[0], output
            else:
                logger.warning("img2img not available, falling back to direct high-res generation")
        
        # Direct high-res generation
        output = self.pipeline.generate(
            attributes=character,
            num_images=1,
            width=high_res_width,
            height=high_res_height,
            **kwargs
        )
        
        return output.images[0], output
    
    def apply_style_transfer(
        self,
        image: Image.Image,
        character: CharacterAttributes,
        style_name: str,
        strength: float = 0.75,
        **kwargs
    ) -> Tuple[Image.Image, DiffusionOutput]:
        """
        Apply style transfer to an existing character image.
        
        Args:
            image: Input image
            character: Character attributes
            style_name: Name of the style to apply
            strength: Strength of the style transfer
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (styled image, diffusion output)
        """
        if style_name not in self.available_styles:
            logger.warning(f"Style '{style_name}' not found. Available styles: {self.available_styles}")
            return image, None
            
        logger.info(f"Applying {style_name} style to character {character.id} with strength {strength}")
        
        # Use the pipeline's style_transfer method
        output = self.pipeline.style_transfer(
            image=image,
            style_name=style_name,
            strength=strength,
            **kwargs
        )
        
        return output.images[0], output
    
    def create_character_sheet(
        self,
        character: CharacterAttributes,
        images: List[Image.Image],
        include_info: bool = True,
        output_path: str = None
    ) -> str:
        """
        Create a character sheet with multiple images and info.
        
        Args:
            character: Character attributes
            images: List of character images to include
            include_info: Whether to include character info
            output_path: Custom output path (optional)
            
        Returns:
            Path to saved character sheet
        """
        if not images:
            logger.warning("No images provided for character sheet")
            return None
            
        # Limit to 6 images maximum
        if len(images) > 6:
            images = images[:6]
            
        # Calculate sheet layout
        num_images = len(images)
        rows = 2 if num_images > 3 else 1
        cols = min(3, (num_images + rows - 1) // rows)  # Ceiling division
        
        # Get image size
        img_width, img_height = images[0].size
        
        # Create character sheet dimensions
        if include_info:
            info_height = img_height // 2
            sheet_width = img_width * cols
            sheet_height = (img_height * rows) + info_height
            info_area = (0, img_height * rows, sheet_width, sheet_height)
        else:
            sheet_width = img_width * cols
            sheet_height = img_height * rows
            info_area = None
            
        # Create the character sheet image
        sheet = Image.new("RGB", (sheet_width, sheet_height), (255, 255, 255))
        
        # Place images in grid
        for i, img in enumerate(images):
            if i >= rows * cols:
                break
                
            row = i // cols
            col = i % cols
            
            x = col * img_width
            y = row * img_height
            
            sheet.paste(img, (x, y))
            
        # Add character info if requested
        if include_info and info_area:
            self._add_character_info(sheet, character, info_area)
            
        # Save the sheet
        if output_path is None:
            char_name = character.name.lower().replace(" ", "_") if character.name else "unnamed"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"character_sheet_{char_name}_{character.id[:8]}_{timestamp}.png"
            output_path = os.path.join(self.image_dir, filename)
            
        sheet.save(output_path)
        logger.info(f"Saved character sheet to {output_path}")
        
        return output_path
    
    def _add_character_info(
        self,
        sheet: Image.Image,
        character: CharacterAttributes,
        info_area: Tuple[int, int, int, int]
    ):
        """Add character information to the sheet."""
        try:
            # Create a draw object
            draw = ImageDraw.Draw(sheet)
            
            # Try to find a font, use default if not found
            try:
                font_path = os.path.join("data", "fonts", "arial.ttf")
                title_font = ImageFont.truetype(font_path, size=36)
                regular_font = ImageFont.truetype(font_path, size=24)
            except:
                title_font = ImageFont.load_default()
                regular_font = ImageFont.load_default()
                
            # Draw background for text area
            x1, y1, x2, y2 = info_area
            draw.rectangle(info_area, fill=(240, 240, 240))
            
            # Draw character name
            name = character.name if character.name else "Unnamed Character"
            draw.text((x1 + 20, y1 + 20), name, fill=(0, 0, 0), font=title_font)
            
            # Gather character info
            info_lines = []
            info_lines.append(f"Gender: {character.gender}")
            info_lines.append(f"Age: {character.age_category}")
            
            # Physical appearance
            appearance = []
            if character.hair_color:
                appearance.append(f"{character.hair_color} hair")
            if character.eye_color:
                appearance.append(f"{character.eye_color} eyes")
            if character.height:
                appearance.append(character.height)
            if character.body_type:
                appearance.append(character.body_type)
                
            if appearance:
                info_lines.append(f"Appearance: {', '.join(appearance)}")
                
            # Personality
            if character.personality:
                info_lines.append(f"Personality: {', '.join(character.personality)}")
                
            # Style
            style_info = []
            if character.anime_style:
                style_info.append(character.anime_style)
            if character.art_style:
                style_info.append(character.art_style)
            if character.era:
                style_info.append(character.era)
                
            if style_info:
                info_lines.append(f"Style: {', '.join(style_info)}")
                
            # Distinctive features
            if character.distinctive_features:
                info_lines.append(f"Distinctive features: {', '.join(character.distinctive_features)}")
                
            # Draw info lines
            y_offset = y1 + 80
            for line in info_lines:
                draw.text((x1 + 20, y_offset), line, fill=(0, 0, 0), font=regular_font)
                y_offset += 40
                
            # Draw evolution info
            if character.generation > 0:
                gen_text = f"Generation: {character.generation}"
                if character.parent_ids:
                    parent_text = f"Parents: {', '.join([pid[:8] for pid in character.parent_ids])}"
                else:
                    parent_text = "No parent information"
                    
                # Draw at bottom right
                gen_width, _ = draw.textsize(gen_text, font=regular_font)
                parent_width, _ = draw.textsize(parent_text, font=regular_font)
                
                draw.text((x2 - gen_width - 20, y2 - 80), gen_text, fill=(0, 0, 0), font=regular_font)
                draw.text((x2 - parent_width - 20, y2 - 40), parent_text, fill=(0, 0, 0), font=regular_font)
                
        except Exception as e:
            logger.error(f"Error adding character info: {e}")
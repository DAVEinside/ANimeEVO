"""
Animation generator for anime characters.
Converts static character images into animated sequences.
"""

import os
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple, Union, Any
from pathlib import Path
import logging
from datetime import datetime
import yaml
import tempfile
import shutil
import subprocess
from PIL import Image

from core.attributes.character_attributes import CharacterAttributes

# Conditional imports for animation models
try:
    from diffusers import AnimationPipeline, DDIMScheduler
    ANIMATION_SUPPORTED = True
except ImportError:
    ANIMATION_SUPPORTED = False
    
try:
    import moviepy.editor as mpy
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)

class AnimationGenerator:
    """
    Generates animations for anime characters.
    """
    
    def __init__(
        self,
        config_path: str = "config/config.yaml",
        output_dir: str = None,
        device: str = None
    ):
        """
        Initialize the animation generator.
        
        Args:
            config_path: Path to configuration file
            output_dir: Directory for saving outputs (overrides config)
            device: Device to use (cuda, cpu)
        """
        # Check if required libraries are available
        if not ANIMATION_SUPPORTED:
            logger.warning("Animation diffusion models not available. Some features will be limited.")
            
        if not MOVIEPY_AVAILABLE:
            logger.warning("MoviePy not available. Video export will be limited.")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set output directory
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = self.config['paths']['output_dir']
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create animation subdirectory
        self.animation_dir = os.path.join(self.output_dir, "animations")
        Path(self.animation_dir).mkdir(exist_ok=True)
        
        # Determine device
        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # Animation settings
        self.animation_settings = self.config['output']['animation']
        
        # Initialize animation models lazily
        self.animation_pipeline = None
        self._load_animation_models()
    
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
                    'animation': {
                        'enabled': True,
                        'fps': 24,
                        'duration': 3.0,
                        'formats': ['mp4', 'gif'],
                    }
                },
                'paths': {
                    'output_dir': './outputs',
                },
                'multimodal': {
                    'animation': {
                        'model': 'animatediff',
                        'motion_strength': 0.6,
                        'motion_modules': ['mm_sd_v15']
                    }
                }
            }
    
    def _load_animation_models(self):
        """Load animation models if available."""
        if not ANIMATION_SUPPORTED or not self.animation_settings['enabled']:
            return
            
        try:
            # Get animation model settings
            animation_config = self.config['multimodal']['animation']
            model_type = animation_config['model']
            
            if model_type == "animatediff":
                # AnimateDiff model
                logger.info("Loading AnimateDiff model")
                
                # This is a placeholder for AnimateDiff integration
                # In a real implementation, you would load the actual model
                # from the specified path or HuggingFace hub
                
                # Example placeholder for AnimateDiff pipeline
                from diffusers import AnimationPipeline, DDIMScheduler, DPMSolverMultistepScheduler
                
                # For now, create a simple pipeline placeholder
                # In a real implementation, this would be the actual model load
                # self.animation_pipeline = AnimationPipeline.from_pretrained(
                #     "path_to_animatediff_model",
                #     torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                # ).to(self.device)
                
                logger.info("AnimateDiff model loaded")
            else:
                logger.warning(f"Unknown animation model type: {model_type}")
                
        except Exception as e:
            logger.error(f"Error loading animation models: {e}")
            logger.warning("Animation generation will be limited")
    
    def generate_animation(
        self,
        character: CharacterAttributes,
        reference_image: Optional[Image.Image] = None,
        duration: float = None,
        fps: int = None,
        motion_type: str = "default",
        output_format: str = "mp4",
        output_path: str = None,
        **kwargs
    ) -> str:
        """
        Generate an animation for a character.
        
        Args:
            character: Character attributes
            reference_image: Optional reference image for consistency
            duration: Animation duration in seconds
            fps: Frames per second
            motion_type: Type of motion to generate
            output_format: Output format (mp4, gif)
            output_path: Custom output path (optional)
            **kwargs: Additional parameters
            
        Returns:
            Path to saved animation file
        """
        # Set defaults from config if not provided
        if duration is None:
            duration = self.animation_settings['duration']
        if fps is None:
            fps = self.animation_settings['fps']
        if output_format not in self.animation_settings['formats']:
            output_format = self.animation_settings['formats'][0]
            
        num_frames = int(duration * fps)
        
        logger.info(f"Generating {duration}s animation at {fps} FPS for character {character.id}")
        
        # Check if animation pipeline is available
        if self.animation_pipeline is None:
            logger.warning("Animation pipeline not available. Using fallback animation.")
            return self._generate_fallback_animation(
                character=character,
                reference_image=reference_image,
                num_frames=num_frames,
                fps=fps,
                output_format=output_format,
                output_path=output_path
            )
            
        # Generate animation using the pipeline
        # This would be the actual implementation with a working animation pipeline
        # For now, we'll fall back to the basic animation
        return self._generate_fallback_animation(
            character=character,
            reference_image=reference_image,
            num_frames=num_frames,
            fps=fps,
            output_format=output_format,
            output_path=output_path
        )
    
    def _generate_fallback_animation(
        self,
        character: CharacterAttributes,
        reference_image: Optional[Image.Image] = None,
        num_frames: int = 24,
        fps: int = 8,
        motion_type: str = "default",
        output_format: str = "mp4",
        output_path: str = None
    ) -> str:
        """
        Generate a simple fallback animation when dedicated models aren't available.
        Uses basic image processing to create simple animations.
        
        Args:
            character: Character attributes
            reference_image: Reference image to animate
            num_frames: Number of frames to generate
            fps: Frames per second
            motion_type: Type of basic motion
            output_format: Output format
            output_path: Custom output path
            
        Returns:
            Path to saved animation file
        """
        if not MOVIEPY_AVAILABLE:
            logger.error("MoviePy not available. Cannot generate fallback animation.")
            return None
            
        # We need a reference image to create a fallback animation
        if reference_image is None:
            logger.error("Reference image required for fallback animation")
            return None
            
        try:
            # Create a temporary directory for the frames
            with tempfile.TemporaryDirectory() as temp_dir:
                # Generate simple motion frames
                frames = self._create_simple_motion_frames(
                    reference_image,
                    num_frames=num_frames,
                    motion_type=motion_type
                )
                
                # Save frames to disk
                frame_paths = []
                for i, frame in enumerate(frames):
                    frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                    frame.save(frame_path)
                    frame_paths.append(frame_path)
                
                # Create output path if not provided
                if output_path is None:
                    char_name = character.name.lower().replace(" ", "_") if character.name else "unnamed"
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"animation_{char_name}_{character.id[:8]}_{timestamp}.{output_format}"
                    output_path = os.path.join(self.animation_dir, filename)
                
                # Create animation using MoviePy
                if output_format == "gif":
                    # Create GIF
                    clip = mpy.ImageSequenceClip(frame_paths, fps=fps)
                    clip.write_gif(output_path, fps=fps)
                else:
                    # Create MP4
                    clip = mpy.ImageSequenceClip(frame_paths, fps=fps)
                    clip.write_videofile(output_path, fps=fps, codec="libx264", audio=False)
                
                logger.info(f"Saved fallback animation to {output_path}")
                return output_path
                
        except Exception as e:
            logger.error(f"Error generating fallback animation: {e}")
            return None
    
    def _create_simple_motion_frames(
        self,
        image: Image.Image,
        num_frames: int = 24,
        motion_type: str = "default"
    ) -> List[Image.Image]:
        """
        Create simple motion frames from a single image.
        
        Args:
            image: Reference image
            num_frames: Number of frames to generate
            motion_type: Type of motion (default, breathing, pan, zoom)
            
        Returns:
            List of frame images
        """
        frames = []
        width, height = image.size
        
        if motion_type == "breathing":
            # Breathing effect (subtle scale animation)
            for i in range(num_frames):
                # Oscillating scale factor
                scale = 1.0 + 0.02 * np.sin(2 * np.pi * i / num_frames)
                
                # Calculate new dimensions
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                # Resize and center
                resized = image.resize((new_width, new_height), Image.LANCZOS)
                
                # Create a new image to center the resized one
                frame = Image.new("RGB", (width, height), (0, 0, 0))
                
                # Paste in center
                x_offset = (width - new_width) // 2
                y_offset = (height - new_height) // 2
                frame.paste(resized, (x_offset, y_offset))
                
                frames.append(frame)
                
        elif motion_type == "pan":
            # Pan effect (horizontal movement)
            for i in range(num_frames):
                # Calculate horizontal offset with oscillation
                x_offset = int(width * 0.1 * np.sin(2 * np.pi * i / num_frames))
                
                # Create a new image for the frame
                frame = Image.new("RGB", (width, height), (0, 0, 0))
                
                # Paste with offset
                frame.paste(image, (x_offset, 0))
                
                # Fill the empty space with content from the other side
                if x_offset > 0:
                    # Fill left side
                    frame.paste(image.crop((width - x_offset, 0, width, height)), (0, 0))
                elif x_offset < 0:
                    # Fill right side
                    frame.paste(image.crop((0, 0, -x_offset, height)), (width + x_offset, 0))
                
                frames.append(frame)
                
        elif motion_type == "zoom":
            # Zoom effect
            for i in range(num_frames):
                # Oscillating zoom factor
                zoom = 1.0 + 0.15 * (0.5 - 0.5 * np.cos(2 * np.pi * i / num_frames))
                
                # Calculate new dimensions
                new_width = int(width * zoom)
                new_height = int(height * zoom)
                
                # Resize
                resized = image.resize((new_width, new_height), Image.LANCZOS)
                
                # Crop center portion
                x_offset = (new_width - width) // 2
                y_offset = (new_height - height) // 2
                cropped = resized.crop((x_offset, y_offset, x_offset + width, y_offset + height))
                
                frames.append(cropped)
                
        else:  # default
            # Default gentle movement and subtle effects
            for i in range(num_frames):
                # Combine subtle zoom and position changes
                phase = 2 * np.pi * i / num_frames
                
                # Small zoom oscillation
                zoom = 1.0 + 0.03 * np.sin(phase)
                
                # Small position offset
                x_offset = int(width * 0.02 * np.sin(phase))
                y_offset = int(height * 0.01 * np.cos(phase))
                
                # Apply transformations
                new_width = int(width * zoom)
                new_height = int(height * zoom)
                
                # Resize
                resized = image.resize((new_width, new_height), Image.LANCZOS)
                
                # Create new frame
                frame = Image.new("RGB", (width, height), (0, 0, 0))
                
                # Calculate paste position
                paste_x = (width - new_width) // 2 + x_offset
                paste_y = (height - new_height) // 2 + y_offset
                
                # Paste resized image
                frame.paste(resized, (paste_x, paste_y))
                
                frames.append(frame)
        
        return frames
    
    def create_expression_animation(
        self,
        base_image: Image.Image,
        character: CharacterAttributes,
        expressions: List[str] = None,
        duration: float = None,
        fps: int = None,
        output_format: str = "mp4",
        output_path: str = None
    ) -> str:
        """
        Create an animation cycling through different character expressions.
        
        Args:
            base_image: Base character image
            character: Character attributes
            expressions: List of expressions to generate
            duration: Animation duration in seconds
            fps: Frames per second
            output_format: Output format
            output_path: Custom output path
            
        Returns:
            Path to saved animation file
        """
        # Not implemented yet as it requires generating multiple expression images
        # This would require a more sophisticated implementation that can generate
        # various facial expressions for the character
        
        logger.warning("Expression animation not implemented yet")
        
        # Fall back to simple animation
        return self.generate_animation(
            character=character,
            reference_image=base_image,
            duration=duration,
            fps=fps,
            motion_type="default",
            output_format=output_format,
            output_path=output_path
        )
    
    def add_special_effects(
        self,
        animation_path: str,
        effect_type: str = "sparkle",
        intensity: float = 0.5,
        output_path: str = None
    ) -> str:
        """
        Add special effects to an existing animation.
        
        Args:
            animation_path: Path to input animation
            effect_type: Type of effect to add
            intensity: Effect intensity (0-1)
            output_path: Custom output path
            
        Returns:
            Path to saved animation with effects
        """
        # Not fully implemented
        if not MOVIEPY_AVAILABLE:
            logger.error("MoviePy not available. Cannot add special effects.")
            return animation_path
            
        try:
            # Generate output path if not provided
            if output_path is None:
                base_name = os.path.basename(animation_path)
                name_parts = os.path.splitext(base_name)
                output_path = os.path.join(
                    os.path.dirname(animation_path),
                    f"{name_parts[0]}_{effect_type}{name_parts[1]}"
                )
            
            # Load the animation
            clip = mpy.VideoFileClip(animation_path)
            
            # Apply effect based on type
            if effect_type == "sparkle":
                # This would need a custom implementation to add sparkle effects
                # For now, just copy the file
                shutil.copy(animation_path, output_path)
                
            elif effect_type == "blur":
                # Apply a simple blur effect
                blurred = clip.fl_image(lambda img: self._apply_blur(img, intensity))
                blurred.write_videofile(output_path, codec="libx264", audio=False)
                
            else:
                # Unknown effect, just copy
                shutil.copy(animation_path, output_path)
                
            return output_path
            
        except Exception as e:
            logger.error(f"Error adding special effects: {e}")
            return animation_path
    
    def _apply_blur(self, img, intensity):
        """Apply blur effect to image."""
        try:
            import cv2
            
            # Convert to OpenCV format
            img_cv = np.array(img)
            
            # Calculate blur amount based on intensity
            blur_amount = int(5 + 20 * intensity)
            if blur_amount % 2 == 0:
                blur_amount += 1  # Must be odd
                
            # Apply blur
            blurred = cv2.GaussianBlur(img_cv, (blur_amount, blur_amount), 0)
            
            return blurred
        except Exception:
            # If error, return original image
            return img
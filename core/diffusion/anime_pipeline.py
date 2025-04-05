"""
Anime-specific diffusion pipeline for character generation and evolution.
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import logging
from PIL import Image

from .diffusion_model import DiffusionModel, DiffusionOutput
from ..attributes.character_attributes import CharacterAttributes

# Setup logging
logger = logging.getLogger(__name__)

class AnimePipeline:
    """
    Specialized pipeline for anime character generation.
    Extends the base diffusion model with anime-specific functionality.
    """
    
    def __init__(
        self,
        diffusion_model: DiffusionModel = None,
        model_id: str = None,
        config_path: str = "config/model_config.yaml",
        device: str = None,
        load_style_models: bool = True,
        **kwargs
    ):
        """
        Initialize the anime pipeline.
        
        Args:
            diffusion_model: Existing diffusion model to use
            model_id: Model ID or path to load if diffusion_model not provided
            config_path: Path to model configuration file
            device: Device to use (cuda, cpu)
            load_style_models: Whether to load style-specific models
            **kwargs: Additional arguments for the diffusion model
        """
        # Use provided diffusion model or create a new one
        if diffusion_model is not None:
            self.diffusion = diffusion_model
        else:
            self.diffusion = DiffusionModel(
                model_id=model_id,
                config_path=config_path,
                device=device,
                **kwargs
            )
        
        # Load config
        self.config_path = config_path
        try:
            with open(config_path, 'r') as f:
                import yaml
                self.config = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Error loading config from {config_path}: {e}")
            logger.warning("Using default configuration.")
            # Use a simple default configuration
            self.config = {
                'paths': {'models_dir': './models'},
                'style_transfer': {'enabled': False, 'styles': []},
                'attribute_conditioning': {'use_textual_inversion': False, 'embedding_dir': './embeddings'}
            }
        
        # Store device
        self.device = self.diffusion.device
        
        # Load anime-specific LoRA models and textual inversions
        if load_style_models:
            self._load_anime_models()
            
        # Initialize prompt templates
        self._initialize_prompt_templates()
        
        # Initialize style transfer models
        self.style_models = {}
        if load_style_models and self.config['style_transfer']['enabled']:
            self._load_style_models()
            
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Create an AnimePipeline from a pretrained model."""
        try:
            return cls(model_id=model_path, **kwargs)
        except Exception as e:
            logger.error(f"Error creating AnimePipeline from pretrained model: {e}")
            # Use default parameters as fallback
            logger.warning("Falling back to default model parameters")
            return cls(model_id="models/stable-diffusion-v1-5", **kwargs)
        
    def _load_anime_models(self):
        """Load anime-specific LoRA models and textual inversions."""
        try:
            # Load default anime LoRA if specified
            anime_lora = self.config['diffusion'].get('anime_lora')
            if anime_lora:
                lora_path = os.path.join(self.config['paths']['models_dir'], anime_lora)
                if os.path.exists(lora_path):
                    self.diffusion.load_lora(lora_path)
                else:
                    logger.warning(f"Anime LoRA model not found at {lora_path}")
            
            # Load attribute embeddings
            if self.config['attribute_conditioning']['use_textual_inversion']:
                embedding_dir = Path(self.config['attribute_conditioning']['embedding_dir'])
                
                if embedding_dir.exists():
                    # Load hair color embeddings
                    for hair_color in self.config['attribute_conditioning']['embeddings']['hair_colors']:
                        embedding_path = embedding_dir / f"{hair_color}.pt"
                        if embedding_path.exists():
                            self.diffusion.load_textual_inversion(str(embedding_path))
                        
                    # Load eye color embeddings
                    for eye_color in self.config['attribute_conditioning']['embeddings']['eye_colors']:
                        embedding_path = embedding_dir / f"{eye_color}.pt"
                        if embedding_path.exists():
                            self.diffusion.load_textual_inversion(str(embedding_path))
                            
                    # Load character type embeddings
                    for char_type in self.config['attribute_conditioning']['embeddings']['character_types']:
                        embedding_path = embedding_dir / f"{char_type}.pt"
                        if embedding_path.exists():
                            self.diffusion.load_textual_inversion(str(embedding_path))
                else:
                    logger.warning(f"Embedding directory not found at {embedding_dir}")
            
            logger.info("Loaded anime-specific models")
        except Exception as e:
            logger.error(f"Error loading anime models: {e}")
            
    def _initialize_prompt_templates(self):
        """Initialize prompt templates for character generation."""
        # Default anime character prompt template
        self.prompt_templates = {
            "default": "anime character, {gender}, {age}, {hair_color} hair, {eye_color} eyes, {style_desc}, {personality_desc}, {distinctive_features}, highly detailed, best quality",
            "portrait": "portrait of anime character, {gender}, {age}, {hair_color} hair, {eye_color} eyes, {style_desc}, {personality_desc}, {distinctive_features}, highly detailed, best quality",
            "full_body": "full body shot of anime character, {gender}, {age}, {hair_color} hair, {eye_color} eyes, {style_desc}, {personality_desc}, {distinctive_features}, highly detailed, best quality",
            "action": "anime character in action, {gender}, {age}, {hair_color} hair, {eye_color} eyes, {style_desc}, {personality_desc}, {distinctive_features}, action pose, dynamic, highly detailed, best quality",
            "emotion": "anime character showing {emotion}, {gender}, {age}, {hair_color} hair, {eye_color} eyes, {style_desc}, {personality_desc}, {distinctive_features}, expressive, highly detailed, best quality"
        }
        
        # Default negative prompt
        self.negative_prompt = "low quality, worst quality, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, bad proportions, deformed, mutated"
            
    def _load_style_models(self):
        """Load style transfer models for different anime styles."""
        style_dir = Path(self.config['style_transfer']['style_reference_dir'])
        
        if not style_dir.exists():
            logger.warning(f"Style reference directory not found at {style_dir}")
            return
            
        # Load reference images for styles
        for style in self.config['style_transfer']['styles']:
            style_name = style['name']
            reference_path = style_dir / style['reference']
            
            if reference_path.exists():
                try:
                    reference_img = Image.open(reference_path)
                    self.style_models[style_name] = {
                        "reference": reference_img,
                        "strength": style['strength']
                    }
                    logger.info(f"Loaded style reference for {style_name}")
                except Exception as e:
                    logger.error(f"Error loading style reference for {style_name}: {e}")
            else:
                logger.warning(f"Style reference for {style_name} not found at {reference_path}")
                
    def _build_prompt(self, attributes: CharacterAttributes, template_key: str = "default") -> str:
        """
        Build a generation prompt from character attributes.
        
        Args:
            attributes: Character attributes to include in the prompt
            template_key: Which prompt template to use
            
        Returns:
            Formatted prompt string
        """
        # Get template or use default
        template = self.prompt_templates.get(template_key, self.prompt_templates["default"])
        
        # Map attributes to prompt components
        prompt_components = {}
        
        # Basic attributes
        prompt_components["gender"] = attributes.gender
        prompt_components["age"] = attributes.age_category
        prompt_components["hair_color"] = attributes.hair_color
        prompt_components["eye_color"] = attributes.eye_color
        
        # Handle style description
        style_desc = []
        if attributes.anime_style:
            style_desc.append(f"{attributes.anime_style} style anime")
        if attributes.art_style:
            style_desc.append(attributes.art_style)
        prompt_components["style_desc"] = ", ".join(style_desc) if style_desc else "anime style"
        
        # Handle personality
        personality_desc = []
        if attributes.personality:
            personality_desc.extend(attributes.personality)
        if attributes.archetype:
            personality_desc.append(attributes.archetype)
        prompt_components["personality_desc"] = ", ".join(personality_desc) if personality_desc else ""
        
        # Handle distinctive features
        distinctive = []
        if attributes.distinctive_features:
            distinctive.extend(attributes.distinctive_features)
        prompt_components["distinctive_features"] = ", ".join(distinctive) if distinctive else ""
        
        # Handle emotion for emotion template
        if template_key == "emotion" and hasattr(attributes, "emotion"):
            prompt_components["emotion"] = attributes.emotion
        else:
            prompt_components["emotion"] = "neutral expression"
            
        # Format the template with available components
        prompt = template.format(**prompt_components)
        
        # Remove any empty placeholders that might remain
        prompt = prompt.replace(",,", ",").replace(", ,", ",")
        
        return prompt
        
    def generate(
        self, 
        attributes: CharacterAttributes,
        template_key: str = "default",
        custom_prompt: str = None,
        custom_negative_prompt: str = None,
        num_images: int = 1,
        apply_style: str = None,
        **kwargs
    ) -> DiffusionOutput:
        """
        Generate anime character images based on attributes.
        
        Args:
            attributes: Character attributes
            template_key: Prompt template to use
            custom_prompt: Optional custom prompt (overrides template)
            custom_negative_prompt: Optional custom negative prompt
            num_images: Number of images to generate
            apply_style: Optional style to apply (from style_models)
            **kwargs: Additional parameters for diffusion model
            
        Returns:
            DiffusionOutput containing generated images and metadata
        """
        # Build prompt from attributes or use custom prompt
        prompt = custom_prompt if custom_prompt else self._build_prompt(attributes, template_key)
        
        # Use custom negative prompt or default
        negative_prompt = custom_negative_prompt if custom_negative_prompt else self.negative_prompt
        
        # Apply style if specified
        if apply_style and apply_style in self.style_models:
            style_info = self.style_models[apply_style]
            
            # Adjust prompt to include style
            if not custom_prompt:
                prompt = f"{apply_style} style, {prompt}"
                
            # Use reference-based generation if available
            if hasattr(self.diffusion.pipeline, "image_reference"):
                # This would be implemented for IP-Adapter or other reference-based methods
                return self.diffusion.generate(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_images=num_images,
                    image_reference=style_info["reference"],
                    reference_strength=style_info["strength"],
                    **kwargs
                )
        
        # Standard generation without style reference
        return self.diffusion.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images=num_images,
            **kwargs
        )
    
    def evolve_image(
        self,
        image: Image.Image,
        attributes: CharacterAttributes,
        strength: float = 0.75,
        **kwargs
    ) -> DiffusionOutput:
        """
        Evolve an existing character image by guiding it with new attributes.
        
        Args:
            image: Base image to evolve
            attributes: New attributes to apply
            strength: Strength of the evolution (0-1)
            **kwargs: Additional parameters for diffusion
            
        Returns:
            DiffusionOutput with evolved images
        """
        # Build prompt from attributes
        prompt = self._build_prompt(attributes)
        
        # Use img2img pipeline for evolution
        if not hasattr(self.diffusion.pipeline, "img2img"):
            logger.warning("Pipeline doesn't support img2img, falling back to txt2img")
            return self.generate(attributes, **kwargs)
        
        # Call the img2img pipeline with the image and new prompt
        output = self.diffusion.pipeline.img2img(
            prompt=prompt,
            negative_prompt=self.negative_prompt,
            image=image,
            strength=strength,
            **kwargs
        )
        
        # Get parameters for metadata
        parameters = {
            'strength': strength,
            'base_image': 'evolved',
            'prompt': prompt,
            **kwargs
        }
        
        return DiffusionOutput(
            images=output.images,
            parameters=parameters,
            prompt=prompt,
            negative_prompt=self.negative_prompt,
            seed=kwargs.get('seed', -1),
            metadata={'evolution_strength': strength}
        )
    
    def style_transfer(
        self,
        image: Image.Image,
        style_name: str,
        strength: float = None,
        **kwargs
    ) -> DiffusionOutput:
        """
        Apply a specific anime style to an existing character image.
        
        Args:
            image: Character image to modify
            style_name: Name of the style to apply
            strength: Override style strength (0-1)
            **kwargs: Additional parameters
            
        Returns:
            DiffusionOutput with styled images
        """
        if style_name not in self.style_models:
            raise ValueError(f"Style '{style_name}' not found. Available styles: {list(self.style_models.keys())}")
            
        style_info = self.style_models[style_name]
        style_strength = strength if strength is not None else style_info["strength"]
        
        # Build a style-specific prompt
        prompt = f"{style_name} style anime character, {style_name} anime"
        
        # Use img2img pipeline for style transfer
        if not hasattr(self.diffusion.pipeline, "img2img"):
            logger.warning("Pipeline doesn't support img2img, falling back to txt2img")
            return self.diffusion.generate(prompt=prompt, **kwargs)
        
        # Call the img2img pipeline with the image and style prompt
        output = self.diffusion.pipeline.img2img(
            prompt=prompt,
            negative_prompt=self.negative_prompt,
            image=image,
            strength=style_strength,
            **kwargs
        )
        
        parameters = {
            'style': style_name,
            'strength': style_strength,
            'prompt': prompt,
            **kwargs
        }
        
        return DiffusionOutput(
            images=output.images,
            parameters=parameters,
            prompt=prompt,
            negative_prompt=self.negative_prompt,
            seed=kwargs.get('seed', -1),
            metadata={'style_transfer': style_name, 'strength': style_strength}
        )
    
    def batch_generate(
        self,
        attributes_list: List[CharacterAttributes],
        **kwargs
    ) -> List[DiffusionOutput]:
        """
        Generate multiple characters in batch.
        
        Args:
            attributes_list: List of character attributes
            **kwargs: Additional parameters for generation
            
        Returns:
            List of DiffusionOutput objects
        """
        results = []
        for attributes in attributes_list:
            result = self.generate(attributes, **kwargs)
            results.append(result)
        return results
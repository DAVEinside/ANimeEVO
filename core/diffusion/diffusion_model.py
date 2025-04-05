"""
Diffusion model implementation for anime character generation.
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from diffusers import (
    StableDiffusionPipeline, 
    DiffusionPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DDIMScheduler,
    DPMSolverMultistepScheduler
)
from diffusers.loaders import LoraLoaderMixin
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image

import yaml
import logging
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class DiffusionOutput:
    """Container for diffusion model outputs."""
    images: List[Image.Image]
    parameters: Dict[str, any]
    prompt: str
    negative_prompt: str
    seed: int
    metadata: Dict[str, any] = None


class DiffusionModel:
    """Base class for diffusion models used in anime character generation."""
    
    def __init__(
        self, 
        model_id: str = None,
        config_path: str = "config/model_config.yaml",
        device: str = None,
        **kwargs
    ):
        """
        Initialize the diffusion model.
        
        Args:
            model_id: Model ID or path to load
            config_path: Path to model configuration file
            device: Device to use (cuda, cpu, etc.)
            **kwargs: Additional arguments to pass to the model
        """
        self.config = self._load_config(config_path)
        
        # Determine device
        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Model ID default from config if not provided
        if model_id is None:
            try:
                model_id = self.config['diffusion']['anime_model']
            except (KeyError, TypeError):
                # Use default if 'diffusion' key is missing
                model_id = "models/stable-diffusion-v1-5"
                logger.warning(f"'diffusion' section not found in config, using default model: {model_id}")
            
        self.model_id = model_id
        self.pipeline = None
        self.lora_weights = {}
        self.textual_inversions = {}
        
        # Load model based on config
        self._load_model(**kwargs)
        
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
                'diffusion': {
                    'base_model': 'runwayml/stable-diffusion-v1-5',
                    'anime_model': 'animefull-latest',
                    'inference': {
                        'steps': 30,
                        'guidance_scale': 7.5,
                        'width': 512,
                        'height': 512,
                        'sampler': 'DDIM',
                        'clip_skip': 1,
                        'seed': -1
                    }
                }
            }
    

    def _load_model(self, **kwargs):
        """Load the diffusion model."""
        logger.info(f"Loading diffusion model {self.model_id} on {self.device}")
        
        try:
            # Check if the model path is a local file path or a model ID
            is_local_path = os.path.exists(self.model_id) or (
                self.model_id.startswith('./') or 
                self.model_id.startswith('/') or 
                self.model_id.startswith('models/')
            )
            
            # Get inference parameters from config
            inference_config = self.config.get('diffusion', {}).get('inference', {})
            scheduler_type = inference_config.get('sampler', 'DDIM')
            
            try:
                # For local model paths that exist
                if is_local_path and os.path.exists(self.model_id):
                    logger.info(f"Loading model from local path: {self.model_id}")
                    
                    # Try to load model directly from local path
                    try:
                        from diffusers import StableDiffusionPipeline
                        self.pipeline = StableDiffusionPipeline.from_single_file(
                            self.model_id,
                            safety_checker=None,
                            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                            **kwargs
                        )
                    except Exception as e:
                        logger.warning(f"Couldn't load from single file, trying standard loading: {e}")
                        # Fall back to standard loading
                        self.pipeline = StableDiffusionPipeline.from_pretrained(
                            self.model_id,
                            safety_checker=None,
                            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                            **kwargs
                        )
                else:
                    # For huggingface model IDs or paths that don't exist locally
                    logger.info(f"Loading model from HuggingFace or registry: {self.model_id}")
                    
                    # Load model with appropriate scheduler
                    if scheduler_type.lower() == 'ddim':
                        scheduler = DDIMScheduler.from_pretrained(
                            self.model_id,
                            subfolder="scheduler"
                        )
                    else:
                        scheduler = DPMSolverMultistepScheduler.from_pretrained(
                            self.model_id,
                            subfolder="scheduler",
                            algorithm_type="dpmsolver++",
                            solver_order=2
                        )
                    
                    # Initialize pipeline with appropriate settings
                    self.pipeline = StableDiffusionPipeline.from_pretrained(
                        self.model_id,
                        scheduler=scheduler,
                        safety_checker=None,  # Disable safety checker for anime content
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        **kwargs
                    )
            except Exception as model_error:
                logger.error(f"Error loading specified model: {model_error}")
                logger.warning("Falling back to default Stable Diffusion model")
                
                # Fall back to default Stable Diffusion model
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    safety_checker=None,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    **kwargs
                )
            
            # Apply clip skip if specified
            clip_skip = inference_config.get('clip_skip', 1)
            if clip_skip > 1:
                self._apply_clip_skip(clip_skip)
                
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            
            # Enable memory optimization if available
            if hasattr(self.pipeline, "enable_xformers_memory_efficient_attention"):
                self.pipeline.enable_xformers_memory_efficient_attention()
            
            logger.info(f"Successfully loaded model {self.model_id}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_id}: {e}")
            raise
    
    def _apply_clip_skip(self, clip_skip: int):
        """Apply CLIP skip to the model."""
        if not hasattr(self.pipeline, "text_encoder"):
            logger.warning("Pipeline does not have text_encoder, skipping clip skip")
            return
            
        # Get original text encoder
        text_encoder = self.pipeline.text_encoder
        
        # Check if the model is compatible with clip skip
        if not isinstance(text_encoder, CLIPTextModel):
            logger.warning(f"Text encoder is {type(text_encoder)}, not CLIPTextModel, skipping clip skip")
            return
            
        # Slice layers to apply clip skip
        layers = text_encoder.text_model.encoder.layers
        if clip_skip > len(layers):
            logger.warning(f"Requested clip_skip {clip_skip} > number of layers {len(layers)}")
            clip_skip = len(layers)
            
        # Create a custom forward method to skip the final clip_skip layers
        def forward_with_clip_skip(text_input_ids, attention_mask=None):
            # Get original embeddings
            inputs_embeds = text_encoder.text_model.embeddings(text_input_ids)
            
            # Run through transformer blocks, skipping final clip_skip layers
            output = inputs_embeds
            for layer in layers[:-clip_skip]:
                output = layer(output, attention_mask=attention_mask)[0]
                
            # Apply final layer norm and get pooled output
            output = text_encoder.text_model.final_layer_norm(output)
            pooled_output = text_encoder.text_model.pooler(output)
            
            return (output, pooled_output)
            
        # Replace text encoder's forward method
        text_encoder.forward = forward_with_clip_skip
        logger.info(f"Applied CLIP skip {clip_skip}")
    
    def load_lora(self, lora_path: str, alpha: float = 0.75):
        """Load LoRA weights for fine-tuning."""
        try:
            logger.info(f"Loading LoRA from {lora_path} with alpha {alpha}")
            if not hasattr(self.pipeline, "load_lora_weights"):
                raise AttributeError("Pipeline does not support LoRA weights.")
                
            self.pipeline.load_lora_weights(lora_path)
            self.lora_weights[lora_path] = alpha
            self.pipeline.fuse_lora(lora_scale=alpha)
            logger.info(f"Successfully loaded LoRA from {lora_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load LoRA from {lora_path}: {e}")
            return False
    
    def load_textual_inversion(self, embedding_path: str):
        """Load textual inversion embeddings."""
        try:
            logger.info(f"Loading textual inversion from {embedding_path}")
            token_name = os.path.basename(embedding_path).split('.')[0]
            self.pipeline.load_textual_inversion(embedding_path, token=token_name)
            self.textual_inversions[token_name] = embedding_path
            logger.info(f"Successfully loaded textual inversion {token_name}")
            return token_name
        except Exception as e:
            logger.error(f"Failed to load textual inversion from {embedding_path}: {e}")
            return None
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_images: int = 1,
        seed: int = -1,
        steps: int = None,
        guidance_scale: float = None,
        width: int = None,
        height: int = None,
        callback=None,
        **kwargs
    ) -> DiffusionOutput:
        """
        Generate images using the diffusion model.
        
        Args:
            prompt: Prompt text for generation
            negative_prompt: Negative prompt text
            num_images: Number of images to generate
            seed: Random seed (-1 for random)
            steps: Number of sampling steps
            guidance_scale: Guidance scale for classifier-free guidance
            width: Output width
            height: Output height
            callback: Callback function for progress updates
            **kwargs: Additional arguments for the pipeline
            
        Returns:
            DiffusionOutput object containing generated images and metadata
        """
        if self.pipeline is None:
            raise ValueError("Model not loaded. Call _load_model first.")
        
        # Get parameters from config if not provided
        inference_config = self.config['diffusion']['inference']
        if steps is None:
            steps = inference_config.get('steps', 30)
        if guidance_scale is None:
            guidance_scale = inference_config.get('guidance_scale', 7.5)
        if width is None:
            width = inference_config.get('width', 512)
        if height is None:
            height = inference_config.get('height', 512)
        
        # Set seed
        if seed == -1:
            seed = np.random.randint(0, 2**32 - 1)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        logger.info(f"Generating {num_images} images with prompt: {prompt}")
        
        # Call the pipeline
        output = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
            callback=callback,
            **kwargs
        )
        
        # Get parameters for metadata
        parameters = {
            'steps': steps,
            'guidance_scale': guidance_scale,
            'width': width,
            'height': height,
            'sampler': inference_config.get('sampler', 'DDIM'),
            'model_id': self.model_id,
        }
        
        return DiffusionOutput(
            images=output.images,
            parameters=parameters,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            metadata={
                'lora_weights': self.lora_weights,
                'textual_inversions': self.textual_inversions
            }
        )
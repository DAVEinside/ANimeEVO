"""
Diffusion module for the Anime Character Evolution System.

This module provides integration with diffusion models for image generation:
- DiffusionModel: Base class for diffusion model integration
- AnimePipeline: Specialized pipeline for anime character generation
- DiffusionOutput: Container for diffusion generation results
"""

from .diffusion_model import DiffusionModel, DiffusionOutput
from .anime_pipeline import AnimePipeline

__all__ = [
    'DiffusionModel',
    'AnimePipeline',
    'DiffusionOutput'
]